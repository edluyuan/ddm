"""Datasets and dataloaders used throughout the project."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class GMM2D(torch.utils.data.IterableDataset):
    """Two-Gaussian mixture dataset used in Section 6.1."""

    def __init__(self, mu1=(3.0, 3.0), mu2=(-3.0, 3.0), sigma: float = 0.5, seed: int = 0):
        super().__init__()
        self.mu1 = torch.tensor(mu1, dtype=torch.float32)
        self.mu2 = torch.tensor(mu2, dtype=torch.float32)
        self.sigma = float(sigma)
        self.rng = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            which = torch.bernoulli(torch.tensor(0.5)).item()
            mu = self.mu1 if which > 0.5 else self.mu2
            x = torch.randn(2, generator=self.rng) * self.sigma + mu
            yield x


def sample_gmm(
    batch: int,
    mu1=(3.0, 3.0),
    mu2=(-3.0, 3.0),
    sigma: float = 0.5,
    device: str = "cpu",
) -> torch.Tensor:
    b1 = torch.bernoulli(0.5 * torch.ones(batch, device=device))
    mu = torch.stack(
        [torch.tensor(mu1, device=device), torch.tensor(mu2, device=device)], dim=0
    )
    pick = mu[b1.long()]
    return pick + sigma * torch.randn(batch, 2, device=device)



class GMM(nn.Module):
    """
    2D Gaussian Mixture Model (GMM)

    This model generates data from a mixture of Gaussians and can compute
    the log-probability of a sample.
    """

    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1,
                 seed=0, n_test_set_samples=1000, device="cpu"):
        """
        Initialize the GMM.

        Args:
            dim (int): Data dimensionality.
            n_mixes (int): Number of mixture components.
            loc_scaling (float): Scale factor for component means.
            log_var_scaling (float): Scaling factor for log variance.
            seed (int): Random seed for reproducibility.
            n_test_set_samples (int): Number of test samples.
            device (str or torch.device): Device for computations.
        """
        super(GMM, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        # Randomly initialize component means scaled by loc_scaling.
        mean = (torch.rand((n_mixes, dim)) - 0.5) * 2 * loc_scaling
        # Fixed log variance for each component.
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        # Uniform mixture weights.
        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        # Use softplus to ensure positive scale parameters.
        self.register_buffer("scale_trils",
                             torch.diag_embed(F.softplus(log_var)))
        self.device = device
        # Move the model's buffers to the specified device.
        self.to(self.device)

    def to(self, device):
        """
        Override the to() method to move all buffers to the specified device.

        Args:
            device (torch.device or str): The device to move the model to.

        Returns:
            self: The model on the new device.
        """
        super().to(device)
        self.device = device
        return self

    @property
    def distribution(self):
        """
        Return the MixtureSameFamily distribution representing the GMM.

        All parameters are explicitly moved to the current device.
        """
        # Ensure device is a torch.device
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        cat_probs = self.cat_probs.to(device)
        locs = self.locs.to(device)
        scale_trils = self.scale_trils.to(device)
        mix = torch.distributions.Categorical(cat_probs)
        comp = torch.distributions.MultivariateNormal(
            locs, scale_tril=scale_trils, validate_args=False
        )
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=comp,
            validate_args=False
        )

    @property
    def test_set(self):
        """
        Generate a test set of samples from the GMM.
        """
        return self.sample((self.n_test_set_samples,))

    def log_prob(self, x: torch.Tensor, **kwargs):
        """
        Compute the log-probability of x under the GMM.

        Ensures that x is moved to the correct device before computation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Log-probabilities.
        """
        x = x.to(self.device)
        log_prob = self.distribution.log_prob(x)
        # Clip very low log-probabilities for numerical stability.
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = -float("inf")
        return log_prob + mask

    def sample(self, shape=(1,)):
        """
        Sample from the GMM.

        Args:
            shape (tuple): Desired shape for the samples.

        Returns:
            torch.Tensor: Generated samples.
        """
        return self.distribution.sample(shape)


@dataclass
class CIFAR10DataConfig:
    """Configuration for CIFAR-10 dataloaders.

    Attributes:
        data_dir: Root directory where the dataset is stored.
        batch_size: Number of images per batch.
        num_workers: Number of worker processes for loading data.
        image_size: Target spatial size for the images.
        augment: Whether to apply standard CIFAR-10 augmentations.
        download: Whether to download the dataset if not present.
        drop_last: Whether to drop the last incomplete batch from the training loader.
        pin_memory: Whether to pin memory in the training loader (useful for GPUs).
    """

    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 32
    augment: bool = True
    download: bool = True
    drop_last: bool = True
    pin_memory: bool = True


def _build_cifar10_transforms(config: CIFAR10DataConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Construct the training and evaluation transforms for CIFAR-10."""

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    resize = []
    if config.image_size != 32:
        resize.append(transforms.Resize(config.image_size))

    train_tfms = []
    if config.augment:
        train_tfms.extend(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_tfms.extend(resize)
    train_tfms.extend([transforms.ToTensor(), normalize])

    eval_tfms = resize + [transforms.ToTensor(), normalize]

    return transforms.Compose(train_tfms), transforms.Compose(eval_tfms)


def build_cifar10_dataloaders(
    config: CIFAR10DataConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders for CIFAR-10 using a shared configuration."""

    train_tfms, eval_tfms = _build_cifar10_transforms(config)

    train_set = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=config.download,
        transform=train_tfms,
    )
    test_set = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=config.download,
        transform=eval_tfms,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader

def plot_contours(log_prob_func, samples=None, ax=None,
                  bounds=(-25.0, 25.0), grid_width_n_points=100,
                  n_contour_levels=None, log_prob_min=-1000.0,
                  device='cpu', plot_marginal_dims=[0, 1],
                  s=2, alpha=0.6, title=None, plt_show=True, xy_tick=True):
    r"""
    Plot contours of a log-probability function over a 2D grid.

    Useful for visualizing the true data density \(\hat{p}(x)\) alongside generated samples.

    Args:
        log_prob_func (callable): Function computing log probability.
        samples (torch.Tensor, optional): Samples to overlay.
        ax (matplotlib.axes.Axes, optional): Plot axes.
        bounds (tuple): (min, max) bounds for each axis.
        grid_width_n_points (int): Number of grid points per axis.
        n_contour_levels (int, optional): Number of contour levels.
        log_prob_min (float): Minimum log probability value.
        device (str): Device for computation.
        plot_marginal_dims (list): Dimensions to plot.
        s (int): Marker size.
        alpha (float): Marker transparency.
        title (str, optional): Plot title.
        plt_show (bool): Whether to display the plot.
        xy_tick (bool): Whether to set custom ticks.
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    x_points = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    grid_points = torch.tensor(list(itertools.product(x_points, x_points)),
                               device=device)
    log_p_x = log_prob_func(grid_points).cpu().detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))

    x1 = grid_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    x2 = grid_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()

    if n_contour_levels:
        ax.contour(x1, x2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x1, x2, log_p_x)

    if samples is not None:
        samples = np.clip(samples.detach().cpu(), bounds[0], bounds[1])
        ax.scatter(samples[:, plot_marginal_dims[0]],
                   samples[:, plot_marginal_dims[1]],
                   s=s, alpha=alpha)
        if xy_tick:
            ax.set_xticks([bounds[0], 0, bounds[1]])
            ax.set_yticks([bounds[0], 0, bounds[1]])
        ax.tick_params(axis='both', which='major', labelsize=15)

    if title:
        ax.set_title(title, fontsize=15)
    if plt_show:
        plt.show()


def plot_MoG40(log_prob_function, samples, file_name=None, title=None):
    """
    Plot GMM density contours with overlaid generated samples.

    Args:
        log_prob_function (callable): Function computing log probability.
        samples (torch.Tensor): Samples from the model.
        file_name (str, optional): Path to save the plot.
        title (str, optional): Plot title.
    """
    if file_name is None:
        plot_contours(log_prob_function, samples=samples.detach().cpu(),
                      bounds=(-45, 45), n_contour_levels=30,
                      grid_width_n_points=200, device="cpu", title=title, plt_show=True)
    else:
        plot_contours(log_prob_function, samples=samples.detach().cpu(),
                      bounds=(-45, 45), n_contour_levels=30,
                      grid_width_n_points=200, device="cpu", title=title, plt_show=False)
        plt.savefig(file_name)
        plt.close()
