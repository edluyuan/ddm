"""Minimal example to train & sample on 2D GMM, matching Section 6.1."""
import argparse
import json
import os
from typing import Any

import torch

from dddm import (
    TrainConfig,
    train_dddm,
    sample_dddm,
    sample_gmm,
    rbf_mmd2,
    save_scatter,
)


def _load_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "PyYAML is required to load configuration files but is not installed."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping of parameters.")
    return data


def _serialize_history(history: dict[str, list[float | int]]) -> dict[str, list[int] | list[float]]:
    result: dict[str, list[int] | list[float]] = {}
    for key, values in history.items():
        if key == "step":
            result[key] = [int(v) for v in values]
        else:
            result[key] = [float(v) for v in values]
    return result


def _apply_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.config is None:
        return

    config_data = _load_config(args.config)
    for key, value in config_data.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown config key '{key}' in {args.config}")
        default = parser.get_default(key)
        current = getattr(args, key)
        if current == default:
            setattr(args, key, value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Optional YAML config")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--w-bias", type=float, default=0.0, dest="w_bias")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="./out")
    p.add_argument("--wandb", action="store_true", dest="use_wandb")
    p.add_argument("--wandb-project", type=str, default="dddm")
    p.add_argument("--wandb-name", type=str, default=None)
    args = p.parse_args()
    _apply_config(p, args)

    cfg = TrainConfig(
        beta=args.beta,
        lam=args.lam,
        m=args.m,
        w_bias=args.w_bias,
        lr=args.lr,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
    )
    os.makedirs(args.out, exist_ok=True)

    result = train_dddm(cfg, outdir=args.out, return_history=True)
    model, history = result

    xgen = sample_dddm(model, n_samples=4096, steps=args.steps, device=cfg.device)
    xref = sample_gmm(4096, device=cfg.device)
    mmd2 = rbf_mmd2(xgen, xref, sigma=1.0).item()

    save_scatter(xgen, os.path.join(args.out, "gen.png"))
    save_scatter(xref, os.path.join(args.out, "ref.png"))

    payload: dict[str, Any] = {"mmd2_rbf_sigma1": mmd2}
    payload["training"] = _serialize_history(history)

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"MMD^2 (rbf σ=1) = {mmd2:.4f}")
    print(f"Saved samples and metrics in {args.out}")


if __name__ == "__main__":
    main()
