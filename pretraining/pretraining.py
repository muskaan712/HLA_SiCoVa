import os
import glob
import math
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    functional
)
from tqdm import tqdm

# import albumentations as A  # Uncomment if needed.

warnings.filterwarnings('ignore')

print(f"Torch-Version {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def get_complete_transform(output_shape, kernel_size, s=1.0):
    """
    Create and return a color distortion transform for an image.
    
    This transform applies random resized cropping, random horizontal flipping,
    color jitter, random grayscale, Gaussian blur, and normalization.

    Args:
        output_shape (List[int]): The desired output shape of the image (e.g., [224, 224]).
        kernel_size (List[int] or Tuple[int, int]): The kernel size for GaussianBlur (e.g., [23, 23]).
        s (float): Strength of the color jitter. Default is 1.0.

    Returns:
        torchvision.transforms.Compose: The composed transform that can be applied to images.
    """
    rnd_crop = RandomResizedCrop(224, (0.08, 0.1))  # random crop
    rnd_flip = RandomHorizontalFlip(p=0.5)          # random flip
    # rnd_vflip = RandomVerticalFlip(p=0.5)
    # rnd_rotate = RandomRotation(degrees=(-90, 90), fill=(0,))

    color_jitter = ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)  # random color jitter
    
    # rnd_aff = RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2))

    rnd_gray = RandomGrayscale(p=0.2)    # random grayscale
    gblur = GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0))
    norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # solarize = tr.RandomSolarize(16, p=0.1)

    to_tensor = ToTensor()

    image_transform = Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_color_jitter,
        rnd_gray,
        gblur,
        # solarize,
        norm
    ])
    # image_transforms.ToPILImage()(torch.randint(0, 256, (1, 224, 224), dtype=torch.uint8))

    return image_transform


class ContrastiveLearningViewGenerator(object):
    """
    Generate multiple views of a single image for contrastive learning.

    Attributes:
        base_transform (callable): A transform function that will be applied to the image.
        n_views (int): The number of transformed views to generate per image.
    """

    def __init__(self, base_transform, n_views=2):
        """
        Initialize the view generator with a base transform and the number of views.

        Args:
            base_transform (callable): The transform to generate views (e.g., color distortion).
            n_views (int): How many views to generate per sample. Default is 2.
        """
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        """
        Generate n_views transforms of the same image.

        Args:
            x (PIL.Image or np.ndarray): The input image.

        Returns:
            list: A list of n_views transformed images.
        """
        return [self.base_transform(x) for _ in range(self.n_views)]


class CustomDataset(Dataset):
    """
    A custom Dataset for loading images and applying transforms.

    Attributes:
        list_images (list): A list of file paths to images.
        transform (callable): A transform to apply to each image (optional).
    """

    def __init__(self, list_images, transform=None):
        """
        Initialize the dataset.

        Args:
            list_images (list): A list of file paths to images.
            transform (callable, optional): A transform to apply to each image. Defaults to None.
        """
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.list_images)

    def __getitem__(self, idx):
        """
        Get the image at index idx and apply transform if provided.

        Args:
            idx (int): The index of the image in the dataset.

        Returns:
            Any: The transformed image (tensor, by default).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.list_images[idx]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image


def off_diagonal(x):
    """
    Return all elements of a square matrix x except the diagonal elements.

    Args:
        x (torch.Tensor): A 2D square tensor.

    Returns:
        torch.Tensor: Off-diagonal elements flattened.
    """
    n, m = x.shape
    assert n == m, "Input must be a square matrix."
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def var_loss(x, y, epsilon=1e-3):
    """
    Compute variance loss term from VICReg.

    Ensures that the standard deviation of each embedding dimension is above a threshold.

    Args:
        x (torch.Tensor): Embeddings from view 1, shape [batch_size, embedding_dim].
        y (torch.Tensor): Embeddings from view 2, shape [batch_size, embedding_dim].
        epsilon (float): Small constant to prevent division by zero. Defaults to 1e-3.

    Returns:
        torch.Tensor: The variance loss value.
    """
    x0 = x - x.mean(dim=0)
    y0 = y - y.mean(dim=0)
    std_x = torch.sqrt(x0.var(dim=0) + epsilon)
    std_y = torch.sqrt(y0.var(dim=0) + epsilon)
    var_l = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    return var_l


def invar_loss(x, y):
    """
    Compute the invariance loss term (mean squared error) for VICReg.

    Args:
        x (torch.Tensor): Embeddings from view 1, shape [batch_size, embedding_dim].
        y (torch.Tensor): Embeddings from view 2, shape [batch_size, embedding_dim].

    Returns:
        torch.Tensor: The invariance loss value.
    """
    return F.mse_loss(x, y)


def cov_loss(x, y):
    """
    Compute the covariance loss term for VICReg.

    Penalizes non-diagonal entries of covariance matrices for each view's embeddings.

    Args:
        x (torch.Tensor): Embeddings from view 1, shape [batch_size, embedding_dim].
        y (torch.Tensor): Embeddings from view 2, shape [batch_size, embedding_dim].

    Returns:
        torch.Tensor: The covariance loss value.
    """
    bs = x.size(0)
    emb = x.size(1)

    x1 = x - x.mean(0)
    y1 = y - y.mean(0)

    cov_x = (x1.T @ x1) / (bs - 1)
    cov_y = (y1.T @ y1) / (bs - 1)

    cov_l = off_diagonal(cov_x).pow_(2).sum().div(emb) + off_diagonal(cov_y).pow_(2).sum().div(emb)
    return cov_l


def cross_corr_loss(x, y, lmbda=5e-3):
    """
    Compute cross-correlation loss between two sets of embeddings.

    Args:
        x (torch.Tensor): Embeddings from view 1, shape [batch_size, embedding_dim].
        y (torch.Tensor): Embeddings from view 2, shape [batch_size, embedding_dim].
        lmbda (float): Weighting factor for the cross-correlation term. Defaults to 5e-3.

    Returns:
        torch.Tensor: The cross-correlation loss value.
    """
    bs = x.size(0)
    emb = x.size(1)

    x_norm = (x - x.mean(0)) / x.std(0)
    y_norm = (y - y.mean(0)) / y.std(0)

    cross_cor_mat = (x_norm.T @ y_norm) / bs

    cross_l = (cross_cor_mat * lmbda - torch.eye(emb, device=x.device) * lmbda).pow(2).sum()
    return cross_l


class SiCoVa(nn.Module):
    """
    A neural network that implements SiCoVa with a ResNet50 encoder and a MLP expander.

    Attributes:
        encoder (nn.Sequential): Truncated ResNet50 for feature extraction.
        expander (nn.Sequential): MLP to further process embeddings.
    """

    def __init__(self):
        """
        Initialize the SiCoVa with a pretrained ResNet50 backbone and MLP expander.
        """
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
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
        Forward pass of the SiCoVa network.

        Args:
            x (torch.Tensor): Input image batch, shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: The expanded embedding, shape [batch_size, 8192].
        """
        _repr = self.encoder(x)
        _embeds = self.expander(_repr)
        return _embeds


class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    """

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
        """
        Initialize the LARS optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            momentum (float): Momentum factor. Default is 0.9.
            eta (float): LARS coefficient. Default is 0.001.
            weight_decay_filter (callable, optional): Function that returns True if a parameter
                                                      should not be decayed.
            lars_adaptation_filter (callable, optional): Function that returns True if a parameter
                                                        should not be adaptive scaled.
        """
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
        """
        Perform a single optimization step using LARS.
        """
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one),
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
    """
    Adjust the learning rate during training.

    Implements a warm-up followed by a cosine annealing schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        loader (DataLoader): The DataLoader for the training set.
        step (int): The current global step (e.g., batch iteration).
    """
    epoch = 100
    batch_size = 2048
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

    optimizer.param_groups[0]["lr"] = lr * base_lr


def exclude_bias_and_norm(p):
    """
    Check if a parameter is a bias or normalization parameter.

    Args:
        p (nn.Parameter): A parameter tensor.

    Returns:
        bool: True if the parameter dimension suggests it's bias/norm, False otherwise.
    """
    return p.ndim == 1


def train_loop(model, 
               optimizer, 
               trainn_dl, 
               var_loss_fn, 
               invar_loss_fn, 
               cov_loss_fn, 
               cross_corr_loss_fn, 
               device, 
               epoch):
    """
    One training epoch loop for SiCoVa.

    Args:
        model (nn.Module): The SiCoVa model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use (e.g., LARS).
        trainn_dl (DataLoader): DataLoader for training data.
        var_loss_fn (callable): Variance loss function.
        invar_loss_fn (callable): Invariance loss function.
        cov_loss_fn (callable): Covariance loss function.
        cross_corr_loss_fn (callable): Cross-correlation loss function.
        device (torch.device): Device to move tensors (CPU or GPU).
        epoch (int): Current epoch number.

    Returns:
        list: A list of loss values (floats) for each iteration in the epoch.
    """
    tk0 = tqdm(trainn_dl, desc=f"Epoch {epoch+1}")
    train_loss = []

    # SiCoVa weights
    lmbd = 25
    u = 25
    v = 1

    # Gradient Accumulation
    accumulation_steps = 8  # Accumulate gradients to effectively increase batch size (256 * 8 = 2048)
    optimizer.zero_grad()  # Reset gradients

    for i, (x, x1) in enumerate(tk0):
        # Adjust LR (pass a dummy step here; replace 11 with actual global step if needed)
        adjust_learning_rate(optimizer, trainn_dl, 11)

        x, x1 = x.to(device), x1.to(device)

        fx = model(x)
        fx1 = model(x1)

        variance_loss = var_loss_fn(fx, fx1)
        invariance_loss = invar_loss_fn(fx, fx1)
        covariance_loss = cov_loss_fn(fx, fx1)
        cross_correlation_loss = cross_corr_loss_fn(fx, fx1)

        loss = (lmbd * variance_loss
                + u * invariance_loss
                + v * covariance_loss
                + cross_correlation_loss)

        train_loss.append(loss.item())

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Checkpoint at EPOCH: {epoch+1}")
        torch.save(model.state_dict(), f"SiCoVa_resnet50_{epoch+1}")

    print(f"Completed Epoch {epoch + 1}")
    print(f"Final Loss {torch.tensor(train_loss).mean():.5f}")
    print(f"Variance Loss {variance_loss.mean():.5f}")
    print(f"Invariance Loss {invariance_loss.mean():.5f}")
    print(f"Covariance Loss {covariance_loss.mean():.5f}")
    print(f"Cross Correlation Loss {cross_correlation_loss.mean():.5f}")

    return train_loss


if __name__ == "__main__":
    # Set up parameters for transformations
    output_shape = [224, 224]
    kernel_size = [23, 23]  # ~10% of the output_shape

    base_transforms = get_complete_transform(
        output_shape=output_shape, 
        kernel_size=kernel_size, 
        s=1.0
    )
    custom_transform = ContrastiveLearningViewGenerator(
        base_transform=base_transforms
    )

    # Create dataset and dataloader
    trainn_ds = CustomDataset(
        list_images=glob.glob("/home/s13mchop/HybridML/data/eyepacs/train/**/*.jpeg", recursive=True),
        transform=custom_transform
    )
    print(f"Number of training samples: {len(trainn_ds)}")

    trainn_dl = DataLoader(
        trainn_ds,
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
        pin_memory=True,
    )

    # Initialize model, optimizer
    batch_size = 2048
    offset_bs = 256
    base_lr = 0.2
    lr = base_lr * batch_size / offset_bs

    model = SiCoVa().to(DEVICE)
    optimizer = LARS(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    # (Optional) Load previously saved model state
    # model.load_state_dict(torch.load("/path/to/checkpoint"))

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        train_loss_values = train_loop(
            model=model,
            optimizer=optimizer,
            trainn_dl=trainn_dl,
            var_loss_fn=var_loss,
            invar_loss_fn=invar_loss,
            cov_loss_fn=cov_loss,
            cross_corr_loss_fn=cross_corr_loss,
            device=DEVICE,
            epoch=epoch
        )
