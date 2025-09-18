"""Train a DiT-backed Distributional Diffusion Model on CIFAR-10."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import utils as tv_utils
from tqdm.auto import tqdm

from dddm.data import CIFAR10DataConfig, build_cifar10_dataloaders
from dddm.losses import generalized_energy_terms, sigmoid_weight
from dddm.metrics import (
    InceptionEmbedding,
    compute_activation_statistics,
    compute_image_mmd,
    frechet_distance,
)
from dddm.model import DDDMDiT
from dddm.schedules import forward_marginal_sample
from dddm.sampling import sample_dddm
from dddm.utils import apply_config_overrides, maybe_init_wandb


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, args: argparse.Namespace, outdir: str, name: str) -> None:
    payload = {
        "model": model.state_dict(),
        "config": vars(args),
    }
    torch.save(payload, os.path.join(outdir, name))


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

    wandb_run = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        run_name=args.wandb_name,
        config=vars(args),
        import_error_message="Weights & Biases is not installed but `--wandb` was set.",
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    progress = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(1, args.epochs + 1):
        progress.set_description(f"Epoch {epoch}/{args.epochs}")
        model.train()
        for epoch_step, (x0, _) in enumerate(train_loader, start=1):
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.rand(B, device=device)
            eps = torch.randn_like(x0)
            xt = forward_marginal_sample(x0, t, eps)

            xi = torch.randn(B, args.m, *x0.shape[1:], device=device)
            xt_rep = xt.unsqueeze(1).expand(-1, args.m, -1, -1, -1)
            xt_rep = xt_rep.reshape(-1, *x0.shape[1:])
            xi_flat = xi.reshape(-1, *x0.shape[1:])
            t_rep = t.repeat_interleave(args.m)

            x0hat = model(xt_rep, t_rep, xi_flat)
            x0hat = x0hat.view(B, args.m, *x0.shape[1:])

            conf, inter = generalized_energy_terms(
                x0hat.view(B, args.m, -1),
                x0.view(B, -1),
                beta=args.beta,
            )
            # Algorithm 1 / eq. (14): generalised energy score with sigmoid weighting.
            w = sigmoid_weight(t, bias=args.w_bias).mean()
            loss = w * (conf - (args.lam / (2.0 * (args.m - 1))) * inter)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            global_step += 1
            progress.update(1)

            metrics = {
                "train/loss": loss.item(),
                "train/confidence": conf.item(),
                "train/interaction": inter.item(),
                "train/weight": w.item(),
                "epoch": epoch,
            }
            if wandb_run is not None:
                wandb_run.log({**metrics, "train/lr": opt.param_groups[0]["lr"]}, step=global_step)

            if global_step % args.log_every == 0 or epoch_step == 1:
                progress.set_postfix(
                    {
                        "loss": f"{metrics['train/loss']:.4f}",
                        "conf": f"{metrics['train/confidence']:.4f}",
                        "inter": f"{metrics['train/interaction']:.4f}",
                        "w~": f"{metrics['train/weight']:.3f}",
                    }
                )
                progress.write(
                    f"[epoch {epoch:03d} step {global_step:06d}] "
                    f"loss={metrics['train/loss']:.4f} "
                    f"conf={metrics['train/confidence']:.4f} "
                    f"inter={metrics['train/interaction']:.4f} "
                    f"w~{metrics['train/weight']:.3f}"
                )

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
            eval_metrics = evaluate(model, args, eval_loader, fid_embedder, fid_stats)
            progress.write(
                f"[epoch {epoch:03d}] FID={eval_metrics['fid']:.3f} "
                f"MMD={eval_metrics['mmd']:.6f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/fid": eval_metrics["fid"],
                        "eval/mmd": eval_metrics["mmd"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

    save_checkpoint(model, args, args.out, "model_final.pt")
    progress.close()

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
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
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
    parser.add_argument("--log-every", type=int, default=100)
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
    parser.add_argument("--wandb", action="store_true", dest="use_wandb", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="dddm")
    parser.add_argument("--wandb-name", type=str, default=None)

    preliminary_args, _ = parser.parse_known_args()
    if preliminary_args.config:
        overrides = apply_config_overrides(parser, preliminary_args.config)
        if overrides:
            print(
                f"Loaded {len(overrides)} setting(s) from {preliminary_args.config}: "
                + ", ".join(sorted(overrides.keys()))
            )
    args = parser.parse_args()

    if args.m < 2:
        parser.error("m must be >= 2 for the generalized energy score")
    if args.eval_every > 0 and args.eval_samples <= 0:
        parser.error("--eval-samples must be positive when evaluation is enabled")
    if args.eval_batch <= 0:
        parser.error("--eval-batch must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")

    train(args)


if __name__ == "__main__":
    main()
