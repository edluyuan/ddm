"""Noise schedules and bridge dynamics mirroring the paper's notation."""

from __future__ import annotations

import torch


def alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow-matching noise schedule (paper eq. (3))."""

    return 1.0 - t, t


def forward_marginal_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Sample ``x_t`` from the forward process ``p(x_t | x_0)`` (eq. (2))."""

    alpha_t, sigma_t = alpha_sigma(t)
    while eps.ndim < x0.ndim:
        eps = eps.unsqueeze(-1)
    while alpha_t.ndim < x0.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    return alpha_t * x0 + sigma_t * eps


def gaussian_bridge_mu_sigma(
    s: torch.Tensor,
    t: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    eps_churn: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bridge parameters ``(μ_{s,t}, Σ_{s,t})`` from eq. (4).

    ``eps_churn`` implements the stochasticity factor ``ε`` in Algorithm 2.
    The helper returns the mean and isotropic standard deviation of the
    Gaussian bridge used during sampling.
    """

    a_s, sig_s = alpha_sigma(s)
    a_t, sig_t = alpha_sigma(t)
    eps = torch.finfo(sig_s.dtype).eps
    sig_ratio = (sig_s**2) / (sig_t**2 + eps)
    a_ratio = a_t / (a_s + eps)
    r11 = a_ratio * sig_ratio
    r12 = a_ratio * sig_ratio**2
    r21 = (a_ratio**2) * sig_ratio
    r22 = (a_ratio**2) * sig_ratio**2
    r01 = sig_ratio
    e2 = eps_churn**2

    def bcast(x: torch.Tensor) -> torch.Tensor:
        while x.ndim < x0.ndim:
            x = x.unsqueeze(-1)
        return x

    mu = (
        e2 * bcast(r12) + (1.0 - e2) * bcast(r01)
    ) * xt + bcast(a_s) * (1.0 - e2 * bcast(r22) - (1.0 - e2) * bcast(r21)) * x0

    inner = e2 * r11 + (1.0 - e2)
    var = (sig_s**2) * (1.0 - inner**2).clamp(min=0.0)
    std = var.clamp(min=0.0).sqrt()
    std = bcast(std)
    return mu, std
