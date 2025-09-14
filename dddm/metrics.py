import torch


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased MMD^2 with RBF kernel, σ fixed."""

    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(-1).unsqueeze(-1)
        b2 = (b * b).sum(-1).unsqueeze(0)
        return a2 + b2 - 2.0 * (a @ b.T)

    k = lambda d2: torch.exp(-d2 / (2.0 * (sigma**2)))

    n = x.size(0)
    m = y.size(0)
    dxx = pdist2(x, x)
    dyy = pdist2(y, y)
    dxy = pdist2(x, y)

    mask_x = ~torch.eye(n, dtype=torch.bool, device=x.device)
    mask_y = ~torch.eye(m, dtype=torch.bool, device=x.device)
    kxx = k(dxx)[mask_x].mean()
    kyy = k(dyy)[mask_y].mean()
    kxy = k(dxy).mean()
    return kxx + kyy - 2.0 * kxy
