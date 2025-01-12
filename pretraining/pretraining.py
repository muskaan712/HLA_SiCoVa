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
import math
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


print(f'Torch-Version {torch.__version__}')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')



def get_complete_transform(output_shape, kernel_size, s=1.0):
    """
    The color distortion transform.
    
    Args:
        s: Strength parameter.
    
    Returns:
        A color distortion transform.
    """
    rnd_crop = RandomResizedCrop(224,(0.08,0.1))    # random crop
    rnd_flip = RandomHorizontalFlip(p=0.5)     # random flip
#     rnd_vflip= RandomVerticalFlip(p=0.5)
#     rnd_rotate = RandomRotation(degrees=(-90, 90), fill=(0,))
    color_jitter = ColorJitter(0.4*s, 0.4*s, 0.2*s, 0.1*s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)      # random color jitter
#    rnd_aff= RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2))
    rnd_gray = RandomGrayscale(p=0.2)    # random grayscale
    gblur =GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0))
    norm= Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
#    solarize=tr.RandomSolarize(16, p=0.1)
    to_tensor = ToTensor()
    image_transform = Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_color_jitter,
        rnd_gray,
        gblur,
#         solarize,
        norm
    ])
#     image_transforms.ToPILImage()(torch.randint(0, 256, (1, 224, 224), dtype=torch.uint8))
    return image_transform


# generate two views for an image
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.base_transform(x) for i in range(self.n_views)]
        return views


class CustomDataset(Dataset):

    def __init__(self, list_images, transform=None):
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.list_images[idx]
#         print(img_name)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image

output_shape = [224, 224]
kernel_size = [23,23] # 10% of the output_shape

base_transforms = get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)
custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms)

trainn_ds = CustomDataset(
    list_images=glob.glob("/home/s13mchop/HybridML/data/eyepacs/train/**/*.jpeg",recursive = True),
    transform=custom_transform
)
len(trainn_ds)

trainn_dl = torch.utils.data.DataLoader(
    trainn_ds,
    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def var_loss(x, y, epsilon=1e-3):
    bs = x.size(0)
    emb = x.size(1)
    x0 = x - x.mean(dim=0)
    y0 = y - y.mean(dim=0)
    std_x = torch.sqrt(x0.var(dim=0) + epsilon)
    std_y = torch.sqrt(y0.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    
    return var_loss

def invar_loss(x,y):
    invar_loss = F.mse_loss(x, y)
    
    return invar_loss

def cov_loss(x,y):
    bs = x.size(0)
    emb = x.size(1)
    x1 = x-x.mean(0)
    y1 = y-y.mean(0)
    cov_x = (x1.T @ x1) / (bs - 1)
    cov_y = (y1.T @ y1) / (bs - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(emb) + off_diagonal(cov_y).pow_(2).sum().div(emb)
    
    return cov_loss

def cross_corr_loss(x,y,lmbda=5e-3):
    bs = x.size(0)
    emb = x.size(1)
    xNorm = (x - x.mean(0)) / x.std(0)
    yNorm = (y - y.mean(0)) / y.std(0)
    crossCorMat = (xNorm.T@yNorm) / bs
    cross_loss = (crossCorMat*lmbda - torch.eye(emb, device=torch.device('cuda'))*lmbda).pow(2).sum()
    
    return cross_loss


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


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=0.2,
            weight_decay=1e-6,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def adjust_learning_rate(optimizer, loader, step):
    epoch=100
    batch_size=2048
    max_steps = epoch * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = 0.2 * batch_size / 256
    if step < warmup_steps:
        lr = 0.2 * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * base_lr 


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 2048
offset_bs = 256
base_lr = 0.2

lr = base_lr*batch_size/offset_bs

model = VICRegNet()

vicreg_model=model.to(device)

def exclude_bias_and_norm(p):
    return p.ndim == 1

params = model.parameters()

optimizer = LARS(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

vicreg_model
vicreg_model.load_state_dict(torch.load('/home/s13mchop/HybridML/experiments/pretrain/VR1/VICReg_resnet50_81')) #load_tar


# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
def train_loop(model, optimizer, trainn_dl, var_loss,invar_loss,cov_loss, cross_corr_loss, device):
    tk0 = tqdm(trainn_dl)
    train_loss = []

    lmbd=25
    u=25
    v=1

    accumulation_steps = 8  # Number of steps to accumulate gradients before updating weights (final batch size 2048 with base batch size 256)
    optimizer.zero_grad()  # Reset gradients initially

    for i, (x, x1) in enumerate(tk0):

        adjust_learning_rate(optimizer, trainn_dl, 11)

        x = x.to(device)
        x1 = x1.to(device)

        fx = model(x)
        fx1 = model(x1)

        variance_loss = var_loss(fx, fx1)
        invariance_loss = invar_loss(fx,fx1)
        covariance_loss = cov_loss(fx,fx1)
        cross_correlation_loss = cross_corr_loss(fx,fx1)
        loss = lmbd*variance_loss + u*invariance_loss + v*covariance_loss + cross_correlation_loss
        train_loss.append(loss.item())
        
        # Backward pass (accumulate gradients)
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()
        
        # Update weights every `accumulation_steps`
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients after updating

    #scheduler.step()

    if(epoch % 20 == 0) :
        global checkpoint
        print(f'EPOCH: {epoch+1} ')
        # Checkpoint
        torch.save(vicreg_model.state_dict(), f'VICReg_resnet50_{epoch+1}')
    print(f'Epoch {epoch + 81}')
    print(f'Total Loss {torch.tensor(loss).mean():.5f}')
    print(f'Variance Loss {torch.tensor(variance_loss).mean():.5f}')
    print(f'Invariance Loss {torch.tensor(invariance_loss).mean():.5f}')
    print(f'Covariance Loss {torch.tensor(covariance_loss).mean():.5f}')
    print(f'Cross Correlation Loss {torch.tensor(cross_correlation_loss).mean():.5f}')

    return train_loss

epochs=121

device=torch.device('cuda')

for epoch in range(epochs):
    train_loss = train_loop(model, optimizer, trainn_dl, var_loss,invar_loss,cov_loss, cross_corr_loss, device)
#     print(np.mean(train_loss))
