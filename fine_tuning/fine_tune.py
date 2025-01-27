import os
import glob
import time
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    GaussianBlur,
    Resize,
    ToTensor,
    RandomRotation,
    RandomAffine,
    Normalize,
    functional,
    RandomCrop,
    CenterCrop
)
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

# import albumentations as A  # Uncomment if needed.

warnings.filterwarnings("ignore")

print(f"Torch-Version {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


# ------------------------------------------------------------------------
# Transforms for Training and Validation
# ------------------------------------------------------------------------
r_crop = RandomResizedCrop(224)
r_fliph = RandomHorizontalFlip()
ttensor = ToTensor()

custom_transform_train = Compose([
    r_crop,
    r_fliph,
    ttensor,
])

resize = Resize(256)
ccrop = CenterCrop(224)
ttensor = ToTensor()

custom_transform_val = Compose([
    resize,
    ccrop,
    ttensor,
])


# ------------------------------------------------------------------------
# Datasets and Dataloaders
# ------------------------------------------------------------------------
"""
Using ImageFolder for training and validation. Update paths if needed.
"""

train2_ds = ImageFolder(
    root="/home/s13mchop/HybridML/data/aptos/train",
    transform=custom_transform_train
)
print(f"Number of training samples: {len(train2_ds)}")

valid2_ds = ImageFolder(
    root="/home/s13mchop/HybridML/data/aptos/test",
    transform=custom_transform_val
)
print(f"Number of validation samples: {len(valid2_ds)}")

# Optionally limit dataset size (uncomment if desired)
# train_size = int(0.5 * len(train2_ds))
# valid_size = int(0.5 * len(valid2_ds))
# train2_ds, _ = torch.utils.data.random_split(train2_ds, [train_size, len(train2_ds) - train_size])
# valid2_ds, _ = torch.utils.data.random_split(valid2_ds, [valid_size, len(valid2_ds) - valid_size])

nu_classes = 5
BATCH_SIZE = 256

train_dl = DataLoader(
    train2_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

valid_dl = DataLoader(
    valid2_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


# ------------------------------------------------------------------------
# SiCoVa Definition
# ------------------------------------------------------------------------
class SiCoVa(nn.Module):
    """
    A neural network that implements SiCoVa with a ResNet50 encoder
    and an MLP expander. This model is primarily used for self-supervised
    learning and can be finetuned or used for feature extraction.
    """
    def __init__(self):
        """
        Initialize the SiCoVa with a pretrained ResNet50 backbone
        truncated before the final fully connected layer, then flatten
        the output, and finally an MLP expansion layer.
        """
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        # Remove the original fully connected layer and add flatten
        self.encoder = nn.Sequential(
            *(list(self.encoder.children())[:-1]),
            nn.Flatten()
        )
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192)
        )

    def forward(self, x):
        """
        Forward pass through the ResNet50 encoder and MLP expander.

        Args:
            x (torch.Tensor): Image batch of shape [batch_size, 3, height, width]

        Returns:
            torch.Tensor: Expanded embedding of shape [batch_size, 8192]
        """
        _repr = self.encoder(x)
        _embeds = self.expander(_repr)
        return _embeds


# ------------------------------------------------------------------------
# Identity Layer
# ------------------------------------------------------------------------
class Identity(nn.Module):
    """
    An identity layer that simply returns its input. Useful for replacing
    parts of a model (e.g., projector head) that are not required during
    finetuning or linear evaluation.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that does nothing but return the input unchanged.

        Args:
            x (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: Same tensor as input
        """
        return x


# ------------------------------------------------------------------------
# LinearEvaluation Model
# ------------------------------------------------------------------------
class LinearEvaluation(nn.Module):
    """
    A classification head on top of a pretrained SiCoVa encoder for
    downstream tasks. Optionally, you can allow finetuning of the encoder
    by setting requires_grad=True for its parameters.
    """

    def __init__(self, model, nu_classes):
        """
        Args:
            model (SiCoVa): The pretrained SiCoVa model to wrap.
            nu_classes (int): Number of classes for classification.
        """
        super().__init__()
        # Store the pretrained SiCoVa model
        SiCoVa = model

        # Indicate that we are in linear_eval (or finetune) mode
        SiCoVa.linear_eval = True

        # Replace the projector (expander) with Identity if you only
        # want to use the 2048-dim features directly from the ResNet encoder.
        SiCoVa.projector = Identity()

        self.SiCoVa = SiCoVa

        # By default, allow gradient updates on entire model (encoder + linear)
        for param in self.SiCoVa.parameters():
            param.requires_grad = True

        # A linear layer from the 2048-dim encoder output to the number of classes
        self.linear = nn.Linear(2048, nu_classes)

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Image batch [batch_size, 3, height, width]

        Returns:
            torch.Tensor: Logits of shape [batch_size, nu_classes]
        """
        features = self.SiCoVa.encoder(x)
        features = torch.squeeze(features)
        pred = self.linear(features)
        return pred


# ------------------------------------------------------------------------
# Instantiate and Prepare Model
# ------------------------------------------------------------------------
model = SiCoVa()
SiCoVa_model = model.to(DEVICE)

# Load pretrained checkpoint
pretrained_path = "/home/s13mchop/HybridML/experiments/pretrain/VR1/SiCoVa_resnet50_200"
SiCoVa_model.load_state_dict(torch.load(pretrained_path))
print(f"Loaded SiCoVa model from: {pretrained_path}")

# Create linear evaluation model
eval_model = LinearEvaluation(SiCoVa_model, nu_classes).to(DEVICE)

# Set up criterion, optimizer, scheduler
criterion_eval = nn.CrossEntropyLoss().to(DEVICE)
optimizer_eval = torch.optim.SGD(eval_model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)
scheduler_eval = optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, T_max=1000)

print("Model architecture:")
print(eval_model)

# Example of enumerating child layers
child_count = 0
for child in eval_model.children():
    child_count += 1
    print(child)

# Optional code to freeze specific layers:
# ct = 0
# for child in eval_model.children():
#     ct += 1
#     if ct < 2:  # Freeze the encoder, train only the linear head
#         for param in child.parameters():
#             param.requires_grad = False


# ------------------------------------------------------------------------
# Training + Validation Loop
# ------------------------------------------------------------------------
def train_and_validate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device=DEVICE,
    epochs=201,
    checkpoint_interval=20,
    checkpoint_prefix="VR1_SiCoVa_Downstream"
):
    """
    Train and validate the given model for a specified number of epochs.

    Args:
        model (nn.Module): The model to train (e.g., LinearEvaluation).
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler instance.
        device (torch.device): Device for computation (CPU/CUDA).
        epochs (int): Number of epochs to train. Default is 201.
        checkpoint_interval (int): Frequency of checkpoint saving. Default is 20.
        checkpoint_prefix (str): Prefix for checkpoint filenames. Default is 'VR1_SiCoVa_Downstream'.

    Returns:
        None
    """
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_accuracies = []

        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            accuracy = y.eq(logits.detach().argmax(dim=1)).float().mean()
            train_accuracies.append(accuracy)

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = f"{checkpoint_prefix}_{epoch+1}"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at epoch {epoch+1} to {ckpt_path}")

        print(f"[Epoch {epoch+1}/{epochs}] Training Loss: {torch.tensor(train_losses).mean():.5f}, "
              f"Training Acc: {torch.tensor(train_accuracies).mean():.5f}")

        # Validation phase
        model.eval()
        val_losses = []
        val_accuracies = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Valid Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

                accuracy = y.eq(logits.detach().argmax(dim=1)).float().mean()
                val_accuracies.append(accuracy)

        scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}] Validation Loss: {torch.tensor(val_losses).mean():.5f}, "
              f"Validation Acc: {torch.tensor(val_accuracies).mean():.5f}\n")


# ------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_validate(
        model=eval_model,
        train_loader=train_dl,
        val_loader=valid_dl,
        criterion=criterion_eval,
        optimizer=optimizer_eval,
        scheduler=scheduler_eval,
        device=DEVICE,
        epochs=201,
        checkpoint_interval=20,
        checkpoint_prefix="VR1_SiCoVa_Downstream"
    )

    # --------------------------------------------------------------------
    # Additional Evaluation (Commented)
    # --------------------------------------------------------------------
    # eval_model.load_state_dict(torch.load('VR1_SiCoVa_Downstream_90'))
    # correct = 0
    # total = 0
    # preds = []
    # labels_list = []
    # with torch.no_grad():
    #     for i, element in enumerate(tqdm(valid_dl)):
    #         image, label = element
    #         image = image.to(DEVICE)
    #         label = label.to(DEVICE)
    #         outputs = eval_model(image)
    #         _, predicted = torch.max(outputs.data, 1)
    #         preds += predicted.cpu().numpy().tolist()
    #         labels_list += label.cpu().numpy().tolist()
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()
    #
    # print(f"Accuracy: {100 * correct / total:.2f}%")
    #
    # target_names = ["Class0", "Class1", "Class2", "Class3", "Class4"]
    # print(classification_report(labels_list, preds, target_names=target_names))
    # print(cohen_kappa_score(labels_list, preds, weights='quadratic'))
    #
    # # Top-k Evaluation Example
    # correct_k = 0
    # total_k = 0
    # with torch.no_grad():
    #     for i, element in enumerate(tqdm(valid_dl)):
    #         k = 5
    #         image, label = element
    #         image = image.to(DEVICE)
    #         label = label.to(DEVICE)
    #         batch_size = label.size(0)
    #         outputs = eval_model(image)
    #         _, pred = outputs.topk(k=k, dim=1)
    #         pred = pred.t()
    #         correct = pred.eq(label.view(1, -1).expand_as(pred))
    #         correct_k += correct[:k].reshape(-1).float().sum(0, keepdim=True)
    #         total_k += batch_size
    #
    # print(f"Top-5 Accuracy: {correct_k.mul_(100.0 / total_k).item():.2f}%")
