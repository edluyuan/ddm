import torch

# ---------- Utilities: schedule α(t), σ(t) per eq. (3) ----------

def alpha_sigma(t: torch.Tensor):
    """Flow-matching noise schedule (paper eq. (3)).

    Args:
        t: shape [B] or []

    Returns:
        α, σ broadcast to t.shape
    """
    return 1.0 - t, t


def forward_marginal_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
    """x_t = α_t x_0 + σ_t ε, with ε ~ N(0,I) (paper eq. (2))."""
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
):
    """Bridge transition parameters μ_{s,t}, Σ_{s,t} (eq. (4)).

    Args:
        s, t: scalars or [B] with ``0 <= s < t <= 1``
        x0, xt: [..., d]
        eps_churn ε in [0,1]

    Returns:
        μ [..., d], std [..., 1] where Σ = std^2 * I
    """
    a_s, sig_s = alpha_sigma(s)
    a_t, sig_t = alpha_sigma(t)
    eps = 1e-8  # avoid division by zero
    r11 = (a_t / (a_s + eps)) * ((sig_s**2) / (sig_t**2 + eps))
    r12 = (a_t / (a_s + eps)) * ((sig_s**2) / (sig_t**2 + eps))**2
    r21 = (a_t / (a_s + eps)) ** 2 * ((sig_s**2) / (sig_t**2 + eps))
    r22 = (a_t / (a_s + eps)) ** 2 * ((sig_s**2) / (sig_t**2 + eps))**2
    r01 = (sig_s**2) / (sig_t**2 + eps)
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
