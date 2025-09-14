import torch
from typing import Tuple


def generalized_energy_terms(
    x0hats: torch.Tensor, x0: torch.Tensor, beta: float, lam: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized energy score terms for one minibatch."""
    B, m, _ = x0hats.shape
    diff = x0[:, None, :] - x0hats
    if beta == 2.0:
        conf = (diff.pow(2).sum(-1)).mean()
    else:
        conf = (diff.pow(2).sum(-1) + 1e-12).pow(beta / 2.0).mean()

    x_i = x0hats[:, :, None, :]
    x_j = x0hats[:, None, :, :]
    pd2 = (x_i - x_j).pow(2).sum(-1)
    mask = ~torch.eye(m, dtype=torch.bool, device=x0.device).unsqueeze(0).expand(B, m, m)
    pd2 = pd2[mask].view(B, m, m - 1)
    if beta == 2.0:
        inter = pd2.mean()
    else:
        inter = (pd2 + 1e-12).pow(beta / 2.0).mean()
    return conf, inter


def sigmoid_weight(t: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    """w(t) = 1 / (1 + exp(b - log(α(t)^2 / σ(t)^2)))."""
    from .schedules import alpha_sigma

    a, s = alpha_sigma(t)
    ratio = (a * a) / (s * s + 1e-12)
    z = torch.log(ratio + 1e-12)
    return torch.sigmoid(z - bias)
