from typing import Sequence

import torch

from .schedules import gaussian_bridge_mu_sigma


@torch.no_grad()
def sample_dddm(
    model: torch.nn.Module,
    n_samples: int = 4096,
    steps: int = 20,
    eps_churn: float = 1.0,
    device: str = "cpu",
    data_shape: Sequence[int] | torch.Size | None = None,
) -> torch.Tensor:
    """Implements Algorithm 2 with coarse grid t_0=0<...<t_N=1."""
    model = model.to(device).eval()
    B = n_samples
    t_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if data_shape is None:
        data_shape = (2,)
    x = torch.randn((B, *tuple(data_shape)), device=device)
    for k in reversed(range(steps)):
        s = t_grid[k]
        t = t_grid[k + 1]
        xi = torch.randn_like(x)
        xhat0 = model(x, t.repeat(B), xi)
        mu, std = gaussian_bridge_mu_sigma(s, t, xhat0, x, eps_churn=eps_churn)
        z = torch.randn_like(x)
        x = mu + std * z
    return x
