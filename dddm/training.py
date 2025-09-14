import os
from dataclasses import dataclass

import torch

from .data import sample_gmm
from .losses import generalized_energy_terms, sigmoid_weight
from .model import DDDMMLP
from .schedules import forward_marginal_sample


@dataclass
class TrainConfig:
    beta: float = 0.1
    lam: float = 1.0
    m: int = 8
    w_bias: float = 0.0
    lr: float = 2e-3
    epochs: int = 2000
    batch: int = 512
    device: str = "cpu"
    seed: int = 0


def train_dddm(config: TrainConfig, outdir: str = "./out") -> DDDMMLP:
    """Train the distributional diffusion model."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(outdir, exist_ok=True)

    model = DDDMMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    for it in range(1, config.epochs + 1):
        x0 = sample_gmm(config.batch, device=device)
        t = torch.rand(config.batch, device=device)
        eps = torch.randn(config.batch, 2, device=device)
        xt = forward_marginal_sample(x0, t, eps)

        xi = torch.randn(config.batch, config.m, 2, device=device)
        t_rep = t[:, None].expand(-1, config.m).reshape(-1)
        xt_rep = xt[:, None, :].expand(-1, config.m, -1).reshape(-1, 2)
        xi_flat = xi.reshape(-1, 2)

        x0hat_flat = model(xt_rep, t_rep, xi_flat)
        x0hat = x0hat_flat.view(config.batch, config.m, 2)

        conf, inter = generalized_energy_terms(x0hat, x0, beta=config.beta, lam=config.lam)
        w = sigmoid_weight(t, bias=config.w_bias).mean()
        loss = w * (conf - (config.lam / (2.0 * (config.m - 1))) * inter)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 200 == 0 or it == 1:
            print(
                f"[{it:05d}] loss={loss.item():.4f}  conf={conf.item():.4f}  inter={inter.item():.4f}  w~{w.item():.3f}"
            )

    torch.save({"model": model.state_dict(), "config": config.__dict__}, os.path.join(outdir, "model.pt"))
    return model
