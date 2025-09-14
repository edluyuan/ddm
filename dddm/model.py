import math
import torch
import torch.nn as nn


class TimeFeat(nn.Module):
    """Fourier time features."""

    def __init__(self, n: int = 16) -> None:
        super().__init__()
        self.freq = nn.Parameter(torch.linspace(1.0, n, n), requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        f = self.freq[None, :] * (2.0 * math.pi) * t[:, None]
        return torch.cat([torch.sin(f), torch.cos(f)], dim=-1)


class DDDMMLP(nn.Module):
    """Distributional denoiser `\hat{x}_θ(t, x_t, ξ)`.

    Small MLP that takes `[x_t (2), ξ (2), time-features]` and outputs a 2D
    sample `\hat{x}_0`.
    """

    def __init__(self, time_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.tfeat = TimeFeat(n=time_dim // 2)
        inp = 2 + 2 + time_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        tf = self.tfeat(t)
        h = torch.cat([xt, xi, tf], dim=-1)
        return self.net(h)
