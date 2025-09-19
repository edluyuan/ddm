"""Train a DiT-backed Distributional Diffusion Model on CIFAR-10."""

from collections import defaultdict
import argparse
import json
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import utils as tv_utils
from tqdm.auto import tqdm

from dddm.data import CIFAR10DataConfig, build_cifar10_dataloaders
from dddm.metrics import (
    InceptionEmbedding,
    compute_activation_statistics,
    compute_image_mmd,
    frechet_distance,
)
from dddm.model import DDDMDiT
from dddm.sampling import sample_dddm
from dddm.training import distributional_training_step


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, args: argparse.Namespace, outdir: str, name: str) -> None:
    payload = {
        "model": model.state_dict(),
        "config": vars(args),
    }
    torch.save(payload, os.path.join(outdir, name))


def maybe_init_wandb(args: argparse.Namespace):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "Weights & Biases is not installed but `--wandb` was provided."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data_config = CIFAR10DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch,
        num_workers=args.workers,
        image_size=args.image_size,
        augment=not args.no_augment,
        download=True,
        pin_memory=device.type == "cuda",
    )
    train_loader, eval_loader = build_cifar10_dataloaders(data_config)
    channels, image_size = 3, args.image_size

    model = DDDMDiT(
        img_size=image_size,
        patch_size=args.patch_size,
        in_channels=channels * 2,
        out_channels=channels,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        time_embed_dim=args.time_embed,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    global_step = 0
    fid_embedder: InceptionEmbedding | None = None
    fid_stats: tuple[torch.Tensor, torch.Tensor] | None = None

    wandb_run = maybe_init_wandb(args)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_sums: Dict[str, float] = defaultdict(float)
        num_batches = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for x0, _ in progress:
            x0 = x0.to(device)

            # Generalized energy score (eqs. (12)–(14)) shared with the toy setup.
            loss, metrics = distributional_training_step(
                model,
                x0,
                m=args.m,
                beta=args.beta,
                lam=args.lam,
                w_bias=args.w_bias,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            global_step += 1
            num_batches += 1
            for key, value in metrics.items():
                epoch_sums[key] += value

            progress.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "conf": f"{metrics['confidence']:.4f}",
                    "inter": f"{metrics['interaction']:.4f}",
                    "w~": f"{metrics['weight']:.3f}",
                },
                refresh=False,
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/lr": opt.param_groups[0]["lr"],
                        **{f"train/{k}": v for k, v in metrics.items()},
                    },
                    step=global_step,
                )

        epoch_avg = {k: epoch_sums[k] / max(num_batches, 1) for k in epoch_sums}
        summary = " ".join(f"{k}={epoch_avg[k]:.4f}" for k in sorted(epoch_avg))
        print(f"[epoch {epoch:03d}] {summary}")

        if wandb_run is not None:
            wandb_run.log({f"epoch/{k}": v for k, v in epoch_avg.items()}, step=epoch)

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ckpt_name = f"model_epoch{epoch:03d}.pt"
            save_checkpoint(model, args, args.out, ckpt_name)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            if fid_embedder is None:
                fid_embedder = InceptionEmbedding()
            if fid_stats is None:
                fid_stats = compute_activation_statistics(
                    eval_loader,
                    fid_embedder,
                    device=device,
                    max_items=args.fid_samples,
                )
            metrics = evaluate(model, args, eval_loader, fid_embedder, fid_stats)
            print(
                f"[epoch {epoch:03d}] FID={metrics['fid']:.3f} "
                f"MMD={metrics['mmd']:.6f}"
            )
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=epoch)

    save_checkpoint(model, args, args.out, "model_final.pt")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.sample_batch > 0:
        model.eval()
        with torch.no_grad():
            samples = sample_dddm(
                model,
                n_samples=args.sample_batch,
                steps=args.sample_steps,
                eps_churn=args.eps_churn,
                device=args.device,
                data_shape=(channels, image_size, image_size),
            )
        samples = samples.clamp(-1.0, 1.0).cpu()
        grid_rows = int(args.sample_batch**0.5)
        if grid_rows * grid_rows < args.sample_batch:
            grid_rows += 1
        grid = tv_utils.make_grid((samples + 1.0) / 2.0, nrow=grid_rows)
        tv_utils.save_image(grid, os.path.join(args.out, "samples.png"))
        print(f"Saved samples and checkpoints to {args.out}")

    if wandb_run is not None:
        wandb_run.finish()


def evaluate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    eval_loader: DataLoader,
    embedder: InceptionEmbedding,
    real_stats: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    device = torch.device(args.device)
    model.eval()
    with torch.no_grad():
        samples = sample_dddm(
            model,
            n_samples=args.eval_samples,
            steps=args.sample_steps,
            eps_churn=args.eps_churn,
            device=args.device,
            data_shape=(3, args.image_size, args.image_size),
        )
    samples = samples.clamp(-1.0, 1.0).cpu()
    fake_loader = DataLoader(
        TensorDataset(samples),
        batch_size=args.eval_batch,
        shuffle=False,
    )

    mu_r, sigma_r = real_stats
    mu_f, sigma_f = compute_activation_statistics(
        fake_loader,
        embedder,
        device=device,
        max_items=args.fid_samples,
    )
    fid = frechet_distance(mu_r, sigma_r, mu_f, sigma_f).item()
    mmd = compute_image_mmd(
        fake_loader,
        eval_loader,
        device=device,
        sigma=args.mmd_sigma,
        max_items=args.mmd_samples,
    ).item()
    return {"fid": fid, "mmd": mmd}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="./cifar10_dit_out")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--w-bias", type=float, default=0.0, dest="w_bias")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--time-embed", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-batch", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--eps-churn", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N epochs (0 disables)")
    parser.add_argument("--eval-batch", type=int, default=256, help="Batch size for evaluation loaders")
    parser.add_argument("--eval-samples", type=int, default=1024, help="Number of samples to draw for evaluation")
    parser.add_argument("--fid-samples", type=int, default=10000, help="Number of real/fake images for FID")
    parser.add_argument("--mmd-samples", type=int, default=2048, help="Number of images used for MMD")
    parser.add_argument("--mmd-sigma", type=float, default=1.0, help="RBF kernel bandwidth for MMD")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="dddm")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    if args.m < 2:
        parser.error("m must be >= 2 for the generalized energy score")
    if args.eval_every > 0 and args.eval_samples <= 0:
        parser.error("--eval-samples must be positive when evaluation is enabled")
    if args.eval_batch <= 0:
        parser.error("--eval-batch must be positive")

    train(args)


if __name__ == "__main__":
    main()
