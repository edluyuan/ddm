"""Loss components for the Distributional Diffusion objective."""

from __future__ import annotations

import torch


def generalized_energy_terms(
    x0hats: torch.Tensor, x0: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample generalized energy terms from eq. (12).

    Args:
        x0hats: Tensor shaped ``[B, m, D]`` containing ``m`` distributional
            samples for each element in the minibatch.
        x0: Tensor shaped ``[B, D]`` with ground-truth samples.
        beta: Exponent for the generalized energy score.

    Returns:
        Tuple ``(conf, inter)`` each shaped ``[B]`` corresponding to the
        *confinement* and *interaction* contributions of eq. (12). These can be
        combined with the weighting scheme from eq. (14) before averaging.
    """

    if x0hats.dim() != 3:
        raise ValueError("Expected x0hats with shape [B, m, D]")
    if x0.dim() != 2:
        raise ValueError("Expected x0 with shape [B, D]")

    B, m, _ = x0hats.shape
    diff = x0[:, None, :] - x0hats
    sq = diff.pow(2).sum(-1)
    if beta == 2.0:
        conf = sq.mean(dim=1)
    else:
        conf = (sq + 1e-12).pow(beta / 2.0).mean(dim=1)

    x_i = x0hats[:, :, None, :]
    x_j = x0hats[:, None, :, :]
    pd2 = (x_i - x_j).pow(2).sum(-1)
    mask = ~torch.eye(m, dtype=torch.bool, device=x0hats.device)
    pd2 = pd2[:, mask].view(B, m, m - 1)
    if beta == 2.0:
        inter = pd2.mean(dim=(1, 2))
    else:
        inter = (pd2 + 1e-12).pow(beta / 2.0).mean(dim=(1, 2))
    return conf, inter


def sigmoid_weight(t: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    """Logistic weight ``w(t)`` from eq. (14)."""
    from .schedules import alpha_sigma

    a, s = alpha_sigma(t)
    ratio = (a * a) / (s * s + 1e-12)
    z = torch.log(ratio + 1e-12)
    return torch.sigmoid(z - bias)
