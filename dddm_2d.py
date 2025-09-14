
import math, os, random, time, json
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utilities: schedule α(t), σ(t) per eq. (3) ----------

def alpha_sigma(t: torch.Tensor):
    """
    Flow-matching noise schedule (paper eq. (3)):
        α(t) = 1 - t
        σ(t) = t
    Args:
        t: shape [B] or []
    Returns:
        α, σ broadcast to t.shape
    """
    return 1.0 - t, t


# ---------- Forward noising (eq. (2)) ----------

def forward_marginal_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
    """
    x_t = α_t x_0 + σ_t ε, with ε ~ N(0,I)  (paper eq. (2))
    """
    alpha_t, sigma_t = alpha_sigma(t)
    while eps.ndim < x0.ndim:
        eps = eps.unsqueeze(-1)
    return alpha_t[..., None] * x0 + sigma_t[..., None] * eps


# ---------- Bridge transition parameters μ_{s,t}, Σ_{s,t} (eq. (4)) ----------

def gaussian_bridge_mu_sigma(s: torch.Tensor, t: torch.Tensor,
                             x0: torch.Tensor, xt: torch.Tensor, eps_churn: float = 1.0):
    """
    Implements paper eq. (4) exactly for 2D (works in any d).
      r_{i,j}(s,t) = (α_t/α_s)^i * (σ_s^2 / σ_t^2)^j
      μ_{s,t}(x0, xt) = (ε^2 r_{1,2} + (1-ε^2) r_{0,1}) xt
                        + α_s (1 - ε^2 r_{2,2} - (1-ε^2) r_{1,1}) x0
      Σ_{s,t} = σ_s^2 [1 - (ε^2 r_{1,1} + (1-ε^2))^2] I
    Args:
      s, t: scalars or [B] with 0 <= s < t <= 1
      x0, xt: [..., d]
      eps_churn ε in [0,1]
    Returns: μ [..., d], std [..., 1] where Σ = std^2 * I
    """
    a_s, sig_s = alpha_sigma(s)
    a_t, sig_t = alpha_sigma(t)
    # Avoid division by zero
    eps = 1e-8
    r11 = (a_t / (a_s + eps)) * ( (sig_s**2) / (sig_t**2 + eps) )
    r12 = (a_t / (a_s + eps)) * ( (sig_s**2) / (sig_t**2 + eps) )**2
    r21 = (a_t / (a_s + eps))**2 * ( (sig_s**2) / (sig_t**2 + eps) )
    r22 = (a_t / (a_s + eps))**2 * ( (sig_s**2) / (sig_t**2 + eps) )**2
    r01 = ((sig_s**2) / (sig_t**2 + eps))
    e2 = eps_churn**2

    # Broadcast dims: ensure s,t shape aligns with x tensors
    def bcast(x):
        while x.ndim < x0.ndim:
            x = x.unsqueeze(-1)
        return x

    mu = (e2 * bcast(r12) + (1.0 - e2) * bcast(r01)) * xt \
         + bcast(a_s) * (1.0 - e2 * bcast(r22) - (1.0 - e2) * bcast(r21)) * x0

    inner = e2 * r11 + (1.0 - e2)
    var = (sig_s**2) * (1.0 - inner**2).clamp(min=0.0)
    std = var.clamp(min=0.0).sqrt()
    std = bcast(std)

    return mu, std


# ---------- Fourier time features ----------

class TimeFeat(nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.freq = nn.Parameter(torch.linspace(1.0, n, n), requires_grad=False)

    def forward(self, t):
        # t in [0,1], shape [B]
        f = self.freq[None, :] * (2.0 * math.pi) * t[:, None]
        return torch.cat([torch.sin(f), torch.cos(f)], dim=-1)  # [B, 2n]


# ---------- Distributional denoiser x̂_θ(t, x_t, ξ) ----------

class DDDMMLP(nn.Module):
    """
    Small MLP that takes [x_t (2), ξ (2), time-features] and outputs a 2D sample x̂_0.
    Matches distributional generator in Section 3 (sampling ξ ~ N(0,I) to sample x̂).
    """
    def __init__(self, time_dim=32, hidden=128):
        super().__init__()
        self.tfeat = TimeFeat(n=time_dim//2)
        inp = 2 + 2 + time_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, xt, t, xi):
        """
        xt: [B,2], t: [B], xi: [B,2]
        returns: x0_hat [B,2]
        """
        tf = self.tfeat(t)
        h = torch.cat([xt, xi, tf], dim=-1)
        return self.net(h)


# ---------- Energy-score loss (eqs. (12)-(14)) ----------

def generalized_energy_terms(x0hats: torch.Tensor, x0: torch.Tensor, beta: float, lam: float):
    """
    Compute the empirical conditional generalized energy score terms for one minibatch.
      x0hats: [B, m, 2] distributional samples x̂_θ(t, x_t, ξ_j)
      x0    : [B, 2] ground-truth clean
      beta  : β in (0,2], lam: λ in [0,1]
    Returns:
      confinement: mean_j ||x0 - x0hat_j||^β  (per-batch mean)
      interaction: mean_{j≠j'} ||x0hat_j - x0hat_{j'}||^β (per-batch mean)
    Full loss per eq. (14): L = mean_b w_t [ confinement - lam/(2(m-1)) * interaction ]
    """
    B, m, d = x0hats.shape
    # confinement
    diff = (x0[:, None, :] - x0hats)  # [B, m, 2]
    if beta == 2.0:
        conf = (diff.pow(2).sum(-1)).mean()  # ||·||^2
    else:
        conf = (diff.pow(2).sum(-1) + 1e-12).pow(beta/2.0).mean()

    # interaction (pairwise j != j')
    # Compute pairwise distances inside each batch element
    x = x0hats  # [B, m, 2]
    # [B, m, m, 2]
    x_i = x[:, :, None, :]
    x_j = x[:, None, :, :]
    pd2 = (x_i - x_j).pow(2).sum(-1)  # [B, m, m]
    # new: expand mask across the batch dimension
    mask = ~torch.eye(m, dtype=torch.bool, device=x.device).unsqueeze(0).expand(B, m, m)
    pd2 = pd2[mask].view(B, m, m - 1)  # remove diagonal
    if beta == 2.0:
        inter = pd2.mean()
    else:
        inter = (pd2 + 1e-12).pow(beta/2.0).mean()
    return conf, inter


def sigmoid_weight(t: torch.Tensor, bias: float = 0.0):
    """
    w(t) = 1 / (1 + exp(b - log(α(t)^2 / σ(t)^2)))  (Section 4.2; Kingma et al., 2021)
    """
    a, s = alpha_sigma(t)
    ratio = (a*a) / (s*s + 1e-12)
    z = torch.log(ratio + 1e-12)
    return torch.sigmoid(z - bias)


@dataclass
class TrainConfig:
    beta: float = 0.1      # β in (0,2], β=2 reduces to MSE
    lam: float = 1.0       # λ in [0,1]
    m: int = 8             # population per (x0, t, x_t)
    w_bias: float = 0.0    # sigmoid weight bias b
    lr: float = 2e-3
    epochs: int = 2000
    batch: int = 512
    device: str = "cpu"
    seed: int = 0


# ---------- Toy data: 2-Gaussian mixture (Section 6.1) ----------

class GMM2D(torch.utils.data.IterableDataset):
    def __init__(self, mu1=(3.0, 3.0), mu2=(-3.0, 3.0), sigma=0.5, seed=0):
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

def sample_gmm(batch, mu1=(3.0,3.0), mu2=(-3.0,3.0), sigma=0.5, device="cpu"):
    b1 = torch.bernoulli(0.5*torch.ones(batch, device=device))
    mu = torch.stack([torch.tensor(mu1, device=device), torch.tensor(mu2, device=device)], dim=0)  # [2,2]
    pick = mu[b1.long()]  # [B,2]
    return pick + sigma * torch.randn(batch, 2, device=device)


# ---------- Training loop (Algorithm 1) ----------

def train_dddm(config: TrainConfig, outdir="./out"):
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(outdir, exist_ok=True)

    model = DDDMMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    for it in range(1, config.epochs+1):
        # Sample minibatch x0 ~ p0
        x0 = sample_gmm(config.batch, device=device)

        # Sample t ~ U[0,1], and forward-noise to get x_t (eq. (2))
        t = torch.rand(config.batch, device=device)
        eps = torch.randn(config.batch, 2, device=device)
        xt = forward_marginal_sample(x0, t, eps)  # [B,2]

        # Population noise ξ_j ~ N(0,I) to sample from x̂_θ(t,x_t,ξ) (Section 3)
        xi = torch.randn(config.batch, config.m, 2, device=device)
        t_rep = t[:, None].expand(-1, config.m).reshape(-1)
        xt_rep = xt[:, None, :].expand(-1, config.m, -1).reshape(-1, 2)
        xi_flat = xi.reshape(-1, 2)

        x0hat_flat = model(xt_rep, t_rep, xi_flat)    # [B*m,2]
        x0hat = x0hat_flat.view(config.batch, config.m, 2)

        # Energy-score terms (eqs. (12)-(14))
        conf, inter = generalized_energy_terms(x0hat, x0, beta=config.beta, lam=config.lam)
        w = sigmoid_weight(t, bias=config.w_bias).mean()

        loss = w * (conf - (config.lam / (2.0*(config.m-1))) * inter)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 200 == 0 or it == 1:
            print(f"[{it:05d}] loss={loss.item():.4f}  conf={conf.item():.4f}  inter={inter.item():.4f}  w~{w.item():.3f}")
    # save
    torch.save({"model": model.state_dict(), "config": config.__dict__}, os.path.join(outdir, "model.pt"))
    return model


# ---------- Sampling (Algorithm 2) ----------

@torch.no_grad()
def sample_dddm(model: DDDMMLP, n_samples=4096, steps=20, eps_churn=1.0, device="cpu"):
    """
    Implements Algorithm 2 with coarse grid t_0=0<...<t_N=1.
    Returns x0 samples [n_samples,2].
    """
    model = model.to(device).eval()
    B = n_samples
    # time grid
    t_grid = torch.linspace(0.0, 1.0, steps+1, device=device)
    # start at t_N=1 with x ~ N(0,I)
    x = torch.randn(B, 2, device=device)
    for k in reversed(range(steps)):
        s = t_grid[k]
        t = t_grid[k+1]
        # distributional posterior sample X̂0
        xi = torch.randn(B, 2, device=device)
        xhat0 = model(x, t.repeat(B), xi)
        # transition p(x_s | x_t, x_0) via eq. (4)
        mu, std = gaussian_bridge_mu_sigma(s, t, xhat0, x, eps_churn=eps_churn)
        z = torch.randn_like(x)
        x = mu + std * z
    return x  # at s=t_0=0, this is approximate p0


# ---------- Evaluation: MMD^2 with RBF kernel (eq. (9b), (10)) ----------

def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma=1.0):
    """
    Unbiased MMD^2 with RBF kernel, σ fixed (as in paper's Fig.3 left).
    """
    def pdist2(a, b):
        a2 = (a*a).sum(-1).unsqueeze(-1)
        b2 = (b*b).sum(-1).unsqueeze(0)
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


# ---------- Convenience plotting ----------

def save_scatter(points: torch.Tensor, path: str, lim=8.0):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    pts = points.detach().cpu().numpy()
    plt.scatter(pts[:,0], pts[:,1], s=3)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--w-bias", type=float, default=0.0, dest="w_bias")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./out")
    args = parser.parse_args()

    cfg = TrainConfig(beta=args.beta, lam=args.lam, m=args.m, w_bias=args.w_bias,
                      epochs=args.epochs, batch=args.batch, device=args.device, seed=args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    model = train_dddm(cfg, outdir=args.outdir)

    # Sample and evaluate
    xgen = sample_dddm(model, n_samples=8192, steps=args.steps, device=cfg.device)
    xref = sample_gmm(8192, device=cfg.device)
    mmd2 = rbf_mmd2(xgen, xref, sigma=1.0).item()

    save_scatter(xgen, os.path.join(args.outdir, "gen.png"))
    save_scatter(xref, os.path.join(args.outdir, "ref.png"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"mmd2_rbf_sigma1": mmd2}, f, indent=2)
    print(f"MMD^2 (rbf σ=1) = {mmd2:.4f}")
    print(f"Saved samples and metrics in {args.outdir}")

if __name__ == "__main__":
    main()
