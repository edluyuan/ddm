"""Evaluation metrics for DDDM models."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


def _extract_images(batch: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Return the tensor of images from a dataloader batch."""

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


class InceptionEmbedding(nn.Module):
    """Utility module that returns pool3 activations from Inception-v3."""

    def __init__(self, resize_input: bool = True) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        net.fc = nn.Identity()
        for param in net.parameters():
            param.requires_grad_(False)
        self.inception = net.eval()
        self.resize_input = resize_input
        mean = torch.tensor(weights.meta["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(weights.meta["std"]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("Expecting images of shape [B, 3, H, W]")
        x = torch.clamp(images, -1.0, 1.0)
        x = (x + 1.0) / 2.0
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.inception(x)


@torch.no_grad()
def compute_activation_statistics(
    loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    embedder: InceptionEmbedding,
    device: torch.device | str,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and covariance of Inception activations from a loader."""

    features: list[torch.Tensor] = []
    seen = 0
    device = torch.device(device)
    embedder = embedder.to(device)

    for batch in loader:
        images = _extract_images(batch).to(device)
        activations = embedder(images)
        features.append(activations)
        seen += activations.size(0)
        if max_items is not None and seen >= max_items:
            break

    if not features:
        raise ValueError("No activations collected from the provided loader")

    feats = torch.cat(features, dim=0)
    if max_items is not None and feats.size(0) > max_items:
        feats = feats[:max_items]

    if feats.size(0) < 2:
        raise ValueError("Need at least two samples to compute covariance")

    mu = feats.mean(dim=0)
    diff = feats - mu
    cov = diff.T @ diff / (feats.size(0) - 1)
    return mu, cov


def _matrix_sqrt_psd(mat: torch.Tensor) -> torch.Tensor:
    mat = (mat + mat.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals.unsqueeze(0)) @ eigvecs.T


def frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the Fréchet distance between two Gaussian statistics."""

    if mu1.ndim != 1 or mu2.ndim != 1:
        raise ValueError("Means must be vectors")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Covariance matrices must have matching shapes")

    offset = mu1 - mu2
    eye = torch.eye(sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype)
    sigma1_eps = sigma1 + eps * eye
    sigma2_eps = sigma2 + eps * eye
    sqrt_sigma1 = _matrix_sqrt_psd(sigma1_eps)
    cov_prod = sqrt_sigma1 @ sigma2_eps @ sqrt_sigma1
    cov_mean = _matrix_sqrt_psd(cov_prod)
    trace_term = torch.trace(sigma1_eps + sigma2_eps - 2.0 * cov_mean)
    distance = offset.dot(offset) + trace_term
    return distance.clamp_min(0.0)


@torch.no_grad()
def compute_fid(
    real_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    fake_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    device: torch.device | str,
    max_items: Optional[int] = None,
    embedder: Optional[InceptionEmbedding] = None,
) -> torch.Tensor:
    """Compute the Fréchet Inception Distance (FID) between two loaders."""

    if embedder is None:
        embedder = InceptionEmbedding()
    mu_r, sigma_r = compute_activation_statistics(real_loader, embedder, device, max_items)
    mu_f, sigma_f = compute_activation_statistics(fake_loader, embedder, device, max_items)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased MMD^2 with the RBF kernel (paper eq. (9b)/(10))."""

    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(-1).unsqueeze(-1)
        b2 = (b * b).sum(-1).unsqueeze(0)
        return a2 + b2 - 2.0 * (a @ b.T)

    n = x.size(0)
    m = y.size(0)
    if n < 2 or m < 2:
        raise ValueError("Need at least two samples per set to compute MMD")

    gamma = 1.0 / (2.0 * sigma**2)
    dxx = pdist2(x, x)
    dyy = pdist2(y, y)
    dxy = pdist2(x, y)

    mask_x = ~torch.eye(n, dtype=torch.bool, device=x.device)
    mask_y = ~torch.eye(m, dtype=torch.bool, device=x.device)
    kxx = torch.exp(-gamma * dxx)[mask_x].mean()
    kyy = torch.exp(-gamma * dyy)[mask_y].mean()
    kxy = torch.exp(-gamma * dxy).mean()
    return kxx + kyy - 2.0 * kxy


@torch.no_grad()
def compute_image_mmd(
    fake_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    real_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    device: torch.device | str,
    sigma: float = 1.0,
    max_items: Optional[int] = None,
) -> torch.Tensor:
    """Compute MMD between generated and real images using flattened pixels."""

    device = torch.device(device)

    def gather(
        loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    ) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        seen = 0
        for batch in loader:
            images = _extract_images(batch)
            if images.ndim > 2:
                images = images.view(images.size(0), -1)
            images = images.to(device)
            batches.append(images)
            seen += images.size(0)
            if max_items is not None and seen >= max_items:
                break
        if not batches:
            raise ValueError("No samples provided for MMD computation")
        tensor = torch.cat(batches, dim=0)
        if max_items is not None and tensor.size(0) > max_items:
            tensor = tensor[:max_items]
        return tensor

    fake = gather(fake_loader)
    real = gather(real_loader)
    n = min(fake.size(0), real.size(0))
    fake = fake[:n]
    real = real[:n]
    return rbf_mmd2(fake, real, sigma=sigma)


class KernelMMDLoss(nn.Module):
    """Multi-kernel MMD loss with input sanitation for stability."""

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 1, fix_sigma: float | None = None) -> None:
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        kernel_mul: float = 2.0,
        kernel_num: int = 1,
        fix_sigma: float | None = None,
    ) -> torch.Tensor:
        n_samples = int(source.size(0) + target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            denominator = n_samples**2 - n_samples
            bandwidth = torch.sum(l2_distance).div(max(denominator, 1)).clamp(min=1e-6)

        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernels = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernels)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        source = source.view(source.size(0), -1)
        target = target.view(target.size(0), -1)
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        batch_size = int(source.size(0))
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        loss = torch.mean(xx + yy - xy - yx)
        return loss


# Backwards compatibility alias
MMD_loss = KernelMMDLoss

