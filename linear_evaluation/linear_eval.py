import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as tr
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    Resize,
    ToTensor,
    RandomRotation,
    RandomAffine,
    Normalize,
    functional
)
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import time
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#import albumentations as A
from torchvision.transforms import (
    RandomCrop,
    Resize,
    CenterCrop)
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score


print(f'Torch-Version {torch.__version__}')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')


r_crop = RandomResizedCrop(224)
r_fliph = RandomHorizontalFlip()
# ccrop = CenterCrop(224)
ttensor = ToTensor()

# transforms train

r_crop = RandomResizedCrop(224)
r_fliph = RandomHorizontalFlip()
# ccrop = CenterCrop(224)
ttensor = ToTensor()

custom_transform_train = Compose([
    r_crop,
    r_fliph,
    ttensor,
])

# transforms val

resize = Resize(256)
ccrop = CenterCrop(224)
# ccrop = CenterCrop(224)
ttensor = ToTensor()

custom_transform_val = Compose([
    resize,
    ccrop,
    ttensor,
])

train2_ds = ImageFolder(
    root="/home/s13mchop/HybridML/data/aptos/train",
    transform=custom_transform_train
)
len(train2_ds)

valid2_ds = ImageFolder(
    root="/home/s13mchop/HybridML/data/aptos/test",
    transform=custom_transform_val
)
len(valid2_ds)

# Use only 10% of the original dataset
# train_size = int(0.5 * len(train2_ds))
# valid_size = int(0.5 * len(valid2_ds))

# # Split the dataset
# train2_ds, _ = torch.utils.data.random_split(train2_ds, [train_size, len(train2_ds) - train_size])
# valid2_ds, _ = torch.utils.data.random_split(valid2_ds, [valid_size, len(valid2_ds) - valid_size])


nu_classes = 5

BATCH_SIZE = 256

# train_size = int(0.8 * len(ds))
# valid_size = len(ds) - train_size
# train_ds, valid_ds = torch.utils.data.random_split(ds, [train_size, valid_size])

# print(len(train_ds))
# print(len(valid_ds))

# Building the data loader
train_dl = torch.utils.data.DataLoader(
    train2_ds,

    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

valid_dl = torch.utils.data.DataLoader(
    valid2_ds,
    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


class VICRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]),
                                           nn.Flatten())
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192))

    def forward(self, x):
        _repr = self.encoder(x)
        _embeds = self.expander(_repr)
        return _embeds


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


# classifier
class LinearEvaluation(nn.Module):
    def __init__(self, model, nu_classes):
        super().__init__()
        vicreg = model
        vicreg.linear_eval = True
        vicreg.projector = Identity()
        self.vicreg = vicreg
        for param in self.vicreg.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(2048, nu_classes)


    def forward(self, x):
        t = self.vicreg.encoder(x)
#         print(t.shape, type(t), len(t))
#         encoding, _ = self.barlowTwins(x)
        t = torch.squeeze(t)
#         print(t.shape, type(t), len(t))
        pred = self.linear(t)
        return pred

model = VICRegNet()
vicreg_model=model.to(DEVICE)
vicreg_model.load_state_dict(torch.load('/home/s13mchop/HybridML/experiments/pretrain/VR1/VICReg_resnet50_200')) #load_tar
eval_model = LinearEvaluation(vicreg_model, nu_classes).to(DEVICE)
criterion_eval = nn.CrossEntropyLoss().to(DEVICE)
optimizer_eval = torch.optim.SGD(eval_model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)
scheduler_eval = optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, T_max=1000)

print(eval_model)

cntr=0
for child in eval_model.children():
    cntr+=1
    print(child)

eval_model.children()

#Linear_Evaluation
ct = 0
for child in eval_model.children():
    ct += 1
    if ct < 2:
        for param in child.parameters():
            param.requires_grad = False


epochs = 201
for epoch in range(epochs):
    accuracies = list()
    class_losses = list()
    eval_model.train()
    for class_batch in tqdm(train_dl):
        x, y = class_batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logit = eval_model(x)
        classification_loss = criterion_eval(logit, y)
        class_losses.append(classification_loss.item())

        optimizer_eval.zero_grad()
        classification_loss.backward()
        optimizer_eval.step()
        accuracies.append(y.eq(logit.detach().argmax(dim =1)).float().mean())
    scheduler_eval.step()
    if (epoch+1)%20==0:
        torch.save(eval_model.state_dict(), f'VR1_VICReg_Downstream_{epoch+1}')
        print(f"saved checkpoint for epoch {epoch + 1}")

    print(f'Epoch {epoch + 1}')
    print(f'classification training loss: {torch.tensor(class_losses).mean():.5f}')
    print(f'classification training accuracy: {torch.tensor(accuracies).mean():.5f}',
          end ='\n\n')

    for class_batch in tqdm(valid_dl):
        x, y = class_batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logit = eval_model(x)

        classification_loss = criterion_eval(logit, y)
        class_losses.append(classification_loss.item())

        optimizer_eval.zero_grad()
        classification_loss.backward()
        optimizer_eval.step()
        accuracies.append(y.eq(logit.detach().argmax(dim =1)).float().mean())
    scheduler_eval.step()

    print(f'Epoch {epoch + 1}')
    print(f'classification valid loss: {torch.tensor(class_losses).mean():.5f}')
    print(f'classification valid accuracy: {torch.tensor(accuracies).mean():.5f}',
          end ='\n\n')

    

# eval_model.load_state_dict(torch.load('VR1_VICReg_Downstream_90'))    
# correct = 0
# total = 0
# preds = []
# labels = []
# with torch.no_grad():
#     for i, element in enumerate(tqdm(valid_dl)):
#         image, label = element
#         image = image.to(DEVICE)
#         label = label.to(DEVICE)
#         print('lables',label.shape)
#         outputs = eval_model(image)
#         _, predicted = torch.max(outputs.data, 1)
#         preds += predicted.cpu().numpy().tolist()
#         labels += label.cpu().numpy().tolist()
#         total += label.size(0)
#         correct += (predicted == label).sum().item()

# print(f'Accuracy: {100 * correct / total} %')

# target_names=["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# # labels=['0','1','2','3','4']

# print(classification_report(labels, preds, target_names=target_names))

# print(cohen_kappa_score(labels, preds, weights='quadratic'))

#Top-5
# correct = 0
# total = 0
# preds = []
# labels = []
# with torch.no_grad():
#     for i, element in enumerate(tqdm(valid_dl)):
#         k=5
#         image, label = element
#         image = image.to(DEVICE)
#         label = label.to(DEVICE)
#         batch_size = label.size(0)
#         print('labels',label.shape)
#         outputs = eval_model(image)
#         _, pred = outputs.topk(k=k, dim=1)
#         pred = pred.t()
#         correct = pred.eq(label.view(1, -1).expand_as(pred))
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

# print(f'Accuracy: {correct_k.mul_(100.0 / batch_size).item()}')






















