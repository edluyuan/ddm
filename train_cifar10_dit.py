"""Train a DiT-backed Distributional Diffusion Model on CIFAR-10."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as tv_utils

from dddm.losses import generalized_energy_terms, sigmoid_weight
from dddm.model import DDDMDiT
from dddm.schedules import forward_marginal_sample
from dddm.sampling import sample_dddm


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(data_dir: str, batch: int, workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )


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

    loader = build_dataloader(args.data_dir, args.batch, args.workers)
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x0, _ in loader:
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
                lam=args.lam,
            )
            w = sigmoid_weight(t, bias=args.w_bias).mean()
            loss = w * (conf - (args.lam / (2.0 * (args.m - 1))) * inter)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"[epoch {epoch:03d} step {global_step:06d}] loss={loss.item():.4f} "
                    f"conf={conf.item():.4f} inter={inter.item():.4f} w~{w.item():.3f}"
                )

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ckpt_name = f"model_epoch{epoch:03d}.pt"
            save_checkpoint(model, args, args.out, ckpt_name)

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
    args = parser.parse_args()

    if args.m < 2:
        parser.error("m must be >= 2 for the generalized energy score")

    train(args)


if __name__ == "__main__":
    main()
