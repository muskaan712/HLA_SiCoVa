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
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    CenterCrop,
    RandomCrop,
    Resize,
    ToTensor,
    ColorJitter,
    RandomGrayscale,
    RandomVerticalFlip,
    RandomApply,
    GaussianBlur,
    RandomRotation,
    RandomAffine,
    Normalize,
    functional
)

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

# import albumentations as A  # Uncomment if needed.

warnings.filterwarnings('ignore')

print(f"Torch-Version {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


# -------------------------------------------------------------------------
# Transforms (train and validation)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Datasets and Dataloaders
# -------------------------------------------------------------------------
"""
Using ImageFolder for training and validation. Adjust root paths as needed.
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


# -------------------------------------------------------------------------
# SiCoVa
# -------------------------------------------------------------------------
class SiCoVaNet(nn.Module):
    """
    A neural network that implements SiCoVa with a ResNet50 encoder and an MLP expander.
    Primarily used for self-supervised pretraining before linear evaluation.
    """

    def __init__(self):
        """
        Initializes the SiCoVaNet with a pretrained ResNet50 backbone truncated 
        before the final fully connected layer, followed by a flatten operation and 
        an MLP expansion.
        """
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        # Truncate the final layer and add flatten
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
        Forward pass through the truncated ResNet50 and the MLP expander.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: The expanded embedding of shape [batch_size, 8192].
        """
        _repr = self.encoder(x)
        _embeds = self.expander(_repr)
        return _embeds


# -------------------------------------------------------------------------
# Identity Layer
# -------------------------------------------------------------------------
class Identity(nn.Module):
    """
    An identity layer that simply returns its input. Useful for 
    replacing parts of a network architecture (e.g., projector) 
    that are not needed during inference or linear evaluation.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input without any modification.

        Args:
            x (torch.Tensor): Any input tensor.

        Returns:
            torch.Tensor: Same as input.
        """
        return x


# -------------------------------------------------------------------------
# Linear Evaluation Model
# -------------------------------------------------------------------------
class LinearEvaluation(nn.Module):
    """
    Wraps a pretrained SiCoVa model and adds a linear classifier on top 
    for downstream classification tasks.
    """

    def __init__(self, model, nu_classes):
        """
        Args:
            model (SiCoVaNet): The pretrained SiCoVa model.
            nu_classes (int): Number of classes for the linear classifier.
        """
        super().__init__()
        SiCoVa = model

        # Indicate a linear evaluation mode (optional flag)
        SiCoVa.linear_eval = True

        # Replace the SiCoVa's 'projector' (expander) with an Identity
        # if you only want to use the encoder's outputs for classification
        SiCoVa.projector = Identity()

        # Store and freeze the main SiCoVa model
        self.SiCoVa = SiCoVa
        for param in self.SiCoVa.parameters():
            param.requires_grad = False

        # Add a new linear classifier on top of ResNet's 2048-dim output
        self.linear = nn.Linear(2048, nu_classes)

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: Logits of shape [batch_size, nu_classes].
        """
        # Pass input through SiCoVa's encoder
        t = self.SiCoVa.encoder(x)
        t = torch.squeeze(t)
        # Classification layer
        pred = self.linear(t)
        return pred


# -------------------------------------------------------------------------
# Instantiate and Load Pretrained Model
# -------------------------------------------------------------------------
model = SiCoVaNet().to(DEVICE)
model_ckpt = "/home/s13mchop/HybridML/experiments/pretrain/VR1/SiCoVa_resnet50_200"
model.load_state_dict(torch.load(model_ckpt))
print(f"Loaded pretrained model from: {model_ckpt}")

eval_model = LinearEvaluation(model, nu_classes).to(DEVICE)
criterion_eval = nn.CrossEntropyLoss().to(DEVICE)
optimizer_eval = torch.optim.SGD(eval_model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)
scheduler_eval = optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, T_max=1000)

print("Model architecture:")
print(eval_model)

# Example of freezing certain children layers
cntr = 0
for child in eval_model.children():
    cntr += 1
    print(child)

# Freeze SiCoVa model parameters while allowing the linear layer to train
ct = 0
for child in eval_model.children():
    ct += 1
    if ct < 2:  # typically the first child is the SiCoVa portion
        for param in child.parameters():
            param.requires_grad = False


# -------------------------------------------------------------------------
# Training Loop for Linear Evaluation
# -------------------------------------------------------------------------
def train_linear_evaluation(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    epochs=201,
    device=DEVICE,
    checkpoint_step=20,
    checkpoint_prefix="VR1_SiCoVa_Downstream"
):
    """
    Train the linear classifier on top of the pretrained SiCoVa encoder.

    Args:
        model (nn.Module): The linear evaluation model (includes encoder + linear head).
        criterion (nn.Module): Loss function (e.g., CrossEntropy).
        optimizer (torch.optim.Optimizer): Optimizer for the linear head.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training dataset.
        valid_loader (DataLoader): DataLoader for validation dataset.
        epochs (int): Number of training epochs. Default is 201.
        device (torch.device): Device to use ('cpu' or 'cuda'). Default is DEVICE.
        checkpoint_step (int): Save model checkpoint every this many epochs. Default is 20.
        checkpoint_prefix (str): Filename prefix for saved checkpoints. Default is 'VR1_SiCoVa_Downstream'.

    Returns:
        None
    """
    for epoch in range(epochs):
        # Training
        model.train()
        train_accuracies = []
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = y.eq(logits.detach().argmax(dim=1)).float().mean()
            train_accuracies.append(acc)

        scheduler.step()

        # Checkpointing
        if (epoch + 1) % checkpoint_step == 0:
            ckpt_path = f"{checkpoint_prefix}_{epoch+1}"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at epoch {epoch + 1}: {ckpt_path}")

        print(f"[Epoch {epoch+1} / {epochs}]")
        print(f" -> Training loss: {torch.tensor(train_losses).mean():.5f}")
        print(f" -> Training accuracy: {torch.tensor(train_accuracies).mean():.5f}\n")

        # Validation
        model.eval()
        val_accuracies = []
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

                acc = y.eq(logits.detach().argmax(dim=1)).float().mean()
                val_accuracies.append(acc)

        scheduler.step()
        print(f"[Epoch {epoch+1} / {epochs}]")
        print(f" -> Validation loss: {torch.tensor(val_losses).mean():.5f}")
        print(f" -> Validation accuracy: {torch.tensor(val_accuracies).mean():.5f}\n")


# -------------------------------------------------------------------------
# Run Training
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_linear_evaluation(
        model=eval_model,
        criterion=criterion_eval,
        optimizer=optimizer_eval,
        scheduler=scheduler_eval,
        train_loader=train_dl,
        valid_loader=valid_dl,
        epochs=201,
        device=DEVICE,
        checkpoint_step=20,
        checkpoint_prefix="VR1_SiCoVa_Downstream"
    )

    # Uncomment to evaluate final performance or load a saved checkpoint:
    # eval_model.load_state_dict(torch.load('VR1_SiCoVa_Downstream_90'))
    # correct = 0
    # total = 0
    # preds = []
    # labels = []
    # with torch.no_grad():
    #     for i, element in enumerate(tqdm(valid_dl)):
    #         image, label = element
    #         image = image.to(DEVICE)
    #         label = label.to(DEVICE)
    #         outputs = eval_model(image)
    #         _, predicted = torch.max(outputs.data, 1)
    #         preds += predicted.cpu().numpy().tolist()
    #         labels += label.cpu().numpy().tolist()
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()

    # print(f"Accuracy: {100 * correct / total:.2f}%")

    # target_names = ["Class0", "Class1", "Class2", "Class3", "Class4"]
    # print(classification_report(labels, preds, target_names=target_names))
    # print(cohen_kappa_score(labels, preds, weights='quadratic'))

    # # Example of top-5 evaluation
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
    # print(f"Top-5 Accuracy: {correct_k.mul_(100.0 / total_k).item():.2f}%")
