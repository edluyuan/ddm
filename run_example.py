"""Minimal example to train & sample on 2D GMM, matching Section 6.1."""
import argparse
import json
import os
from typing import Any, Dict

import torch
import yaml

from dddm import (
    TrainConfig,
    train_dddm,
    sample_dddm,
    sample_gmm,
    rbf_mmd2,
    save_scatter,
)


def _load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--w-bias", type=float, default=0.0, dest="w_bias")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="./out")
    p.add_argument("--log-interval", type=int, default=200, dest="log_interval")
    p.add_argument("--wandb", action="store_true", dest="use_wandb")
    p.add_argument("--wandb-project", type=str, default="dddm")
    p.add_argument("--wandb-name", type=str, default=None)
    preliminary_args, _ = p.parse_known_args()
    if preliminary_args.config:
        cfg_defaults = _load_yaml_config(preliminary_args.config)
        valid_keys = {action.dest for action in p._actions if action.dest != argparse.SUPPRESS}
        filtered = {k: v for k, v in cfg_defaults.items() if k in valid_keys}
        p.set_defaults(**filtered)
    args = p.parse_args()

    cfg = TrainConfig(
        beta=args.beta,
        lam=args.lam,
        m=args.m,
        w_bias=args.w_bias,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
    )
    os.makedirs(args.out, exist_ok=True)

    model = train_dddm(cfg, outdir=args.out)

    xgen = sample_dddm(model, n_samples=4096, steps=args.steps, device=cfg.device)
    xref = sample_gmm(4096, device=cfg.device)
    mmd2 = rbf_mmd2(xgen, xref, sigma=1.0).item()

    save_scatter(xgen, os.path.join(args.out, "gen.png"))
    save_scatter(xref, os.path.join(args.out, "ref.png"))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"mmd2_rbf_sigma1": mmd2}, f, indent=2)
    print(f"MMD^2 (rbf σ=1) = {mmd2:.4f}")
    print(f"Saved samples and metrics in {args.out}")


if __name__ == "__main__":
    main()
