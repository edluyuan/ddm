import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from tqdm.auto import tqdm

from .data import sample_gmm
from .losses import generalized_energy_terms, sigmoid_weight
from .model import DDDMMLP
from .schedules import forward_marginal_sample
from .utils import maybe_init_wandb


@dataclass
class TrainConfig:
    """Hyper-parameters for Algorithm 1 of the paper.

    The defaults mirror the settings in Section 6.1: ``beta`` and ``lam``
    parameterise the conditional generalised energy score (eq. (12)) while
    ``m`` controls the number of denoiser samples per data point. ``w_bias``
    shifts the sigmoid weighting scheme discussed in Section 4.2. The remaining
    values describe optimisation and logging behaviour.
    """

    beta: float = 0.1
    lam: float = 1.0
    m: int = 8
    w_bias: float = 0.0
    lr: float = 2e-3
    epochs: int = 2000
    batch: int = 512
    device: str = "cpu"
    seed: int = 0
    log_interval: int = 200
    use_wandb: bool = False
    wandb_project: str = "dddm"
    wandb_run_name: Optional[str] = None


def train_dddm(config: TrainConfig, outdir: str = "./out") -> DDDMMLP:
    """Train the 2D denoiser following Algorithm 1 / eq. (14)."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(outdir, exist_ok=True)

    model = DDDMMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    wandb_run = maybe_init_wandb(
        config.use_wandb,
        project=config.wandb_project,
        run_name=config.wandb_run_name,
        config=asdict(config),
        import_error_message=(
            "Weights & Biases is not installed but `use_wandb` was set to True."
        ),
    )

    progress = tqdm(range(1, config.epochs + 1), desc="Training", unit="step")
    for it in progress:
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

        conf, inter = generalized_energy_terms(x0hat, x0, beta=config.beta)
        # Algorithm 1 / eq. (14): weight the generalised energy score components.
        w = sigmoid_weight(t, bias=config.w_bias).mean()
        loss = w * (conf - (config.lam / (2.0 * (config.m - 1))) * inter)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        metrics = {
            "loss": loss.item(),
            "confidence": conf.item(),
            "interaction": inter.item(),
            "weight": w.item(),
        }

        if wandb_run is not None:
            wandb_run.log(metrics, step=it)

        if it % config.log_interval == 0 or it == 1:
            progress.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "conf": f"{metrics['confidence']:.4f}",
                    "inter": f"{metrics['interaction']:.4f}",
                    "w~": f"{metrics['weight']:.3f}",
                }
            )
            progress.write(
                f"[{it:05d}] loss={metrics['loss']:.4f}  conf={metrics['confidence']:.4f}  "
                f"inter={metrics['interaction']:.4f}  w~{metrics['weight']:.3f}"
            )

    torch.save({"model": model.state_dict(), "config": config.__dict__}, os.path.join(outdir, "model.pt"))
    if wandb_run is not None:
        wandb_run.finish()
    progress.close()
    return model
