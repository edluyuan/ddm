# Distributional Diffusion on 2D (faithful to the paper)

This minimal PyTorch package implements the **Distributional Diffusion Model (DDDM)**
from *"Distributional Diffusion Models with Scoring Rules"* (De Bortoli et al., 2025)
for both the 2D toy setting and a CIFAR-10 DiT backbone. It adheres to the paper's math
and algorithms, and is designed to be easy to read, experiment with, and extend.

## What’s inside
- `dddm/` — self‑contained implementation:
  - Forward schedule **α_t = 1−t**, **σ_t = t** (paper eq. (3)).
  - Transitional Gaussian **p(x_s|x_0,x_t)=N(μ_{s,t}, Σ_{s,t})** with **μ, Σ** from eq. (4).
  - Training Algorithm 1 (distributional denoiser `x̂_θ(t, x_t, ξ)`), loss eqs. **(12)–(14)**.
  - Sampling Algorithm 2 (coarse reverse with distributional `x̂_θ` plugged into eq. (4)).
  - Optional classical diffusion baseline (β=2, λ=0) via same training loop.
  - MMD^2 evaluation (rbf kernel), matching Section 6.1 setup.
- `run_example.py` — tiny script to train on a 2‑Gaussian mixture and sample.
- `train_cifar10_dit.py` — training loop for a DiT backbone on CIFAR‑10 with
  evaluation hooks (FID & pixel MMD) and optional sampling.
- `configs/` — YAML defaults for both entrypoints. Use them via `--config path.yaml`.
- `scripts/` — convenience launchers (`train_toy.sh`, `train_cifar10.sh`) that load
  the default configs and forward extra CLI arguments.
- Logging is handled uniformly with `tqdm` progress bars and optional Weights & Biases
  tracking (pass `--wandb` to either entrypoint).
- `LICENSE` — MIT.

## Quick start
```bash
pip install -r requirements.txt

# 2D toy example (matches Section 6.1)
python run_example.py --epochs 2000 --batch 512 --beta 0.1 --lam 1.0 --m 8 --steps 20
# or: ./scripts/train_toy.sh --config configs/toy.yaml --epochs 4000

# CIFAR-10 DiT
python train_cifar10_dit.py --epochs 10 --batch 128 --wandb
# or: ./scripts/train_cifar10.sh --config configs/cifar10.yaml --wandb

# samples, checkpoints and metrics are written into the configured output folder
```
*Tip:* Start with fewer epochs to verify end‑to‑end, then increase.

### Configuration + logging
- `--config path/to.yaml` injects defaults into the CLI (only known flags are
  applied). The helper lives in `dddm.utils.apply_yaml_config_defaults` and is shared
  across both entrypoints, so one place controls config parsing behaviour.
- Progress bars display running losses; every `log_interval` / `--log-every` steps we
  also emit a human-readable line with the same metrics. Passing `--wandb` turns on
  Weights & Biases logging with the exact same metric names.

## How this code respects the math (equation mapping)
- **Forward marginals** (eq. **(2)**) are used to corrupt data:
  `x_t = α_t x_0 + σ_t ε`, with schedule **(3)**: `α(t) = 1−t`, `σ(t)=t`.
- **Bridge transition** (eq. **(4)**) implemented exactly in `gaussian_bridge_mu_sigma(...)`:
  - `r_{i,j}(s,t) = (α_t/α_s)^i * (σ_s^2/σ_t^2)^j`
  - `μ_{s,t}(x_0,x_t) = (ε^2 r_{1,2} + (1−ε^2) r_{0,1}) x_t + α_s(1−ε^2 r_{2,2} − (1−ε^2) r_{1,1}) x_0`
  - `Σ_{s,t} = σ_s^2 [1 − (ε^2 r_{1,1} + (1−ε^2))^2] I`
- **Distributional denoiser** `x̂_θ(t,x_t,ξ)` (Section 3) is a small MLP that inputs `x_t`,
  `t` (Fourier‑time features) and `ξ ~ N(0,I_2)` and outputs a *sample* of **x_0|x_t**.
- **Conditional generalized energy score** (eq. **(12)**) with hyper‑params `(β, λ)`:
  - *Confinement* term `‖x̂−x_0‖^β` and *interaction* term over pairs of `{x̂}` samples.
  - We compute these terms per data point, then apply the **sigmoid weighting**
    `w(t)` from eq. **(14)** element-wise before taking the minibatch mean. This keeps
    the implementation faithful to the expectation in the paper.
- **Weighting** `w_t`: we use the *sigmoid* scheme from the paper’s discussion
  (Section 4.2; citing Kingma et al., 2021):
  `w(t) = 1 / (1 + exp(b − log(α(t)^2 / σ(t)^2)))`, tunable via `--w-bias`.
- **Sampling** (Algorithm **2**): for a coarse grid `{t_k}` we:
  1) Draw `ξ, Z ~ N(0,I)`
  2) Set `X̂_0 = x̂_θ(t_{k+1}, x_{t_{k+1}}, ξ)`
  3) Compute `(μ_{t_k,t_{k+1}}, Σ_{t_k,t_{k+1}})` via eq. **(4)**
  4) Sample `x_{t_k} = μ + Σ^{1/2} Z` (with churn `ε∈[0,1]`, default `1.0`)
- **MMD^2** (eq. **(10)** with rbf kernel eq. **(9b)**) reports quality vs target.

## Notes
- Setting **β→2, λ→0** recovers the standard diffusion MSE loss (eq. **(11)** links to eq. (6)).
- The default example matches Section 6.1 (two‑Gaussian mixture).
- CIFAR‑10 training mirrors Algorithm 1 with a DiT denoiser and adds FID / MMD
  evaluation utilities for convenience.

Enjoy hacking!
