import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from tqdm.auto import tqdm

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
    use_wandb: bool = False
    wandb_project: str = "dddm"
    wandb_run_name: Optional[str] = None


def distributional_training_step(
    model: torch.nn.Module,
    x0: torch.Tensor,
    *,
    m: int,
    beta: float,
    lam: float,
    w_bias: float,
    t: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the generalized energy training loss (paper eqs. (12)–(14)).

    The helper encapsulates the mechanics shared by both the toy 2D setup and
    the CIFAR-10 script:

    * forward marginals ``x_t = α_t x_0 + σ_t ε`` (eq. (2)) via
      :func:`forward_marginal_sample`;
    * distributional denoiser ``\hat{x}_θ(t, x_t, ξ)`` queried ``m`` times;
    * confidence/interaction terms of the conditional generalized energy score
      (eq. (12)) and the minibatch weighting ``w(t)`` (eq. (14)).

    Returns the differentiable loss tensor along with detached scalar metrics so
    the caller can log them without duplicating math-heavy code.
    """

    if m < 2:
        raise ValueError("m must be >= 2 to form interaction pairs")

    device = x0.device
    dtype = x0.dtype
    batch = x0.shape[0]

    if t is None:
        t = torch.rand(batch, device=device, dtype=dtype)
    eps = torch.randn_like(x0)
    xt = forward_marginal_sample(x0, t, eps)

    xi = torch.randn((batch, m, *x0.shape[1:]), device=device, dtype=dtype)
    xt_rep = xt.unsqueeze(1).expand(-1, m, *xt.shape[1:]).reshape(batch * m, *xt.shape[1:])
    xi_flat = xi.reshape(batch * m, *xt.shape[1:])
    t_rep = t.repeat_interleave(m)

    x0hat = model(xt_rep, t_rep, xi_flat)
    x0hat = x0hat.view(batch, m, *x0.shape[1:])

    conf, inter = generalized_energy_terms(
        x0hat.view(batch, m, -1),
        x0.view(batch, -1),
        beta=beta,
        lam=lam,
    )

    weight = sigmoid_weight(t, bias=w_bias).mean()
    loss = weight * (conf - (lam / (2.0 * (m - 1))) * inter)

    metrics = {
        "loss": float(loss.detach().cpu()),
        "confidence": float(conf.detach().cpu()),
        "interaction": float(inter.detach().cpu()),
        "weight": float(weight.detach().cpu()),
    }
    return loss, metrics


def train_dddm(config: TrainConfig, outdir: str = "./out") -> DDDMMLP:
    """Train the distributional diffusion model."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(outdir, exist_ok=True)

    model = DDDMMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - defensive import guard
            raise RuntimeError(
                "Weights & Biases is not installed but `use_wandb` was set to True."
            ) from exc

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )

    progress = tqdm(
        range(1, config.epochs + 1),
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        x0 = sample_gmm(config.batch, device=device)

        loss, metrics = distributional_training_step(
            model,
            x0,
            m=config.m,
            beta=config.beta,
            lam=config.lam,
            w_bias=config.w_bias,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if wandb_run is not None:
            wandb_run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

        progress.set_postfix(
            {
                "loss": f"{metrics['loss']:.4f}",
                "conf": f"{metrics['confidence']:.4f}",
                "inter": f"{metrics['interaction']:.4f}",
                "w~": f"{metrics['weight']:.3f}",
            },
            refresh=False,
        )

    torch.save({"model": model.state_dict(), "config": config.__dict__}, os.path.join(outdir, "model.pt"))
    if wandb_run is not None:
        wandb_run.finish()
    return model
