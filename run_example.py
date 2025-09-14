"""Minimal example to train & sample on 2D GMM, matching Section 6.1."""
import argparse
import json
import os

import torch

from dddm import (
    TrainConfig,
    train_dddm,
    sample_dddm,
    sample_gmm,
    rbf_mmd2,
    save_scatter,
)


def main() -> None:
    p = argparse.ArgumentParser()
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
