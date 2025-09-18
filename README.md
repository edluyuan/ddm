# Distributional Diffusion Models (DDDM)

This repository contains a compact, well-documented PyTorch implementation of
**Distributional Diffusion Models with Scoring Rules** (De Bortoli et al.,
2025). The codebase reproduces the mathematics of the paper for both the 2D toy
experiments (Section 6.1) and an image-scale DiT variant for CIFAR-10. Logging
is handled uniformly via `tqdm` progress bars and optional Weights & Biases
tracking.

## Repository layout

- `dddm/`
  - `training.py` – Algorithm&nbsp;1 for the toy denoiser, with the conditional
    generalised energy score from eqs. (12)–(14) and sigmoid weighting from
    Section&nbsp;4.2.
  - `schedules.py` – forward marginals (eq. (2)), schedule (eq. (3)), and bridge
    dynamics (eq. (4)).
  - `sampling.py` – Algorithm&nbsp;2 for ancestral sampling with optional churn.
  - `losses.py` – generalised energy score helpers and Section&nbsp;4.2 weighting.
  - `model.py` – the 2D MLP denoiser and a DiT-style backbone for images.
  - `metrics.py` – MMD (eq. (9b)/(10)), FID utilities, and Inception features.
  - `data.py` – toy Gaussian mixture samples and CIFAR-10 dataloaders.
  - `utils.py` – shared helpers for YAML configs, logging, and plotting.
- `run_example.py` – trains the toy model and reports RBF MMD just like the
  paper.
- `train_cifar10_dit.py` – trains the DiT-based model on CIFAR-10 with logging,
  evaluation, and checkpointing.
- `configs/` – ready-to-run YAML defaults for both entry-points.
- `scripts/` – shell wrappers that pin the repository root and load the configs.

## Quick start

Install the dependencies and run either experiment:

```bash
pip install -r requirements.txt

# Toy 2D mixture (Section 6.1)
python run_example.py --config configs/toy.yaml

# CIFAR-10 with the DiT backbone
python train_cifar10_dit.py --config configs/cifar10.yaml
```

Both scripts accept `--config <yaml>` and command-line flags. YAML values are
validated against the CLI, so unknown keys are ignored and typos are caught by
`argparse`. To keep things reproducible and easy to debug, the out directory is
created automatically and checkpoints/metrics are written there.

## Logging and monitoring

- `tqdm` progress bars surface the current epoch, loss, and the components of
  the generalised energy score.
- Pass `--wandb` (or `use_wandb: true` in the YAML) to enable Weights & Biases
  logging on both entry-points. The helper in `dddm.utils.maybe_init_wandb`
  gracefully reports missing installations.
- Toy runs save scatter plots of generated vs. reference samples, while CIFAR
  runs optionally dump sample grids and FID/MMD metrics every `--eval-every`
  epochs.

## How the implementation matches the paper

- **Forward process** – `dddm.schedules.forward_marginal_sample` implements
  eq. (2) using the schedule from eq. (3).
- **Bridge dynamics** – `dddm.schedules.gaussian_bridge_mu_sigma` follows
  eq. (4) and exposes the churn parameter ε from Algorithm&nbsp;2.
- **Training objective** – the confinement and interaction terms from eq. (12)
  are computed in `dddm.losses.generalized_energy_terms`; Algorithm&nbsp;1 /
  eq. (14) is realised in `dddm.training.train_dddm` and mirrored in the CIFAR
  training loop.
- **Sampling** – `dddm.sampling.sample_dddm` executes Algorithm&nbsp;2 on a coarse
  time grid with optional stochasticity.
- **Metrics** – the RBF MMD from eqs. (9b)/(10) and the FID utilities enable a
  faithful quantitative comparison against the reference distributions.

## Tips

- Use the provided shell scripts if you prefer reproducible entry-points:
  `scripts/train_toy.sh` and `scripts/train_cifar10.sh` automatically pin the
  repository root and load the matching YAML.
- Override any hyper-parameter on the command line. For example
  `scripts/train_toy.sh --epochs 2000 --wandb` keeps the configuration file in
  sync but tweaks the requested options.
- The CIFAR-10 script caches Inception statistics once and reuses them across
  evaluations to keep runs efficient.
