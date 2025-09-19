# Distributional Diffusion on 2D Toy Data and CIFAR-10

This repository hosts a minimal yet faithful implementation of the
**Distributional Diffusion Model (DDDM)** from
*“Distributional Diffusion Models with Scoring Rules”* (De Bortoli et al., 2025).
It provides the lightweight 2D experiments from Section 6.1 as well as a
DiT-based CIFAR-10 training script so the full workflow can be inspected,
extended, and logged.

## Repository layout
- `dddm/training.py` – contains the `TrainConfig` dataclass, the 2D training
  loop, and a shared helper `distributional_training_step` that evaluates the
  conditional generalized energy score exactly as in eqs. (12)–(14) while
  relying on the forward marginals from eq. (2).
- `dddm/schedules.py` – implements the linear schedule `α(t) = 1 − t` and
  `σ(t) = t` (eq. (3)) together with the Gaussian bridge parameters from eq. (4).
- `dddm/sampling.py` – Algorithm 2 for drawing reverse-time samples using the
  distributional denoiser.
- `run_example.py` – trains the 2D model on the bimodal mixture, producing
  scatter plots and MMD metrics.
- `train_cifar10_dit.py` – end-to-end CIFAR-10 training with a DiT backbone,
  epoch-wise metrics, optional FID/MMD evaluation, and W&B/TQDM logging.

## Quick start: 2D toy problem
```bash
pip install -r requirements.txt
python run_example.py --epochs 2000 --batch 512 --beta 0.1 --lam 1.0 --m 8 --steps 20
```
The command shows a live `tqdm` progress bar, saves checkpoints under
`./out`, and optionally logs to Weights & Biases with `--wandb` (no extra print
spam – the progress bar and W&B dashboards carry the metrics).

## Training on CIFAR-10
```bash
python train_cifar10_dit.py \
    --data-dir ./data \
    --out ./cifar10_dit_out \
    --epochs 400 \
    --wandb --wandb-project dddm-cifar
```
Key features:
- `tqdm` tracks every minibatch so the console stays tidy; epoch summaries are
  printed once per cycle.
- Enable W&B logging with `--wandb` (project/name configurable via
  `--wandb-project` and `--wandb-name`).
- `--eval-every` performs FID/MMD evaluation, reporting and logging the scores.
- Reproducibility through `--seed`, gradient clipping via `--grad-clip`, and
  on-the-fly sampling with `--sample-batch` & `--sample-steps`.

## Faithfulness to the paper
- **Forward corruption** – `dddm.schedules.forward_marginal_sample` applies
  `x_t = α_t x_0 + σ_t ε` with the linear schedule from eq. (3), exactly matching
  eq. (2).
- **Bridge transitions** – `dddm.schedules.gaussian_bridge_mu_sigma` computes the
  closed-form mean and variance from eq. (4), which are then reused during
  sampling.
- **Distributional denoiser** – both the MLP (`dddm.model.DDDMMLP`) and the DiT
  (`dddm.model.DDDMDiT`) implement the function `\hat{x}_θ(t, x_t, ξ)` described
  in Section 3.
- **Generalized energy score** – `distributional_training_step` expands each
  minibatch into `m` denoiser queries, forms the confinement and interaction
  terms from eq. (12), and combines them exactly as eq. (14) specifies (including
  the factor `λ/(2(m−1))` and the logistic weight `w(t)`). Inline comments in the
  code point back to the governing equations.
- **Sampling** – `dddm.sampling.sample_dddm` follows Algorithm 2, repeatedly
  plugging the denoiser into the bridge parameters of eq. (4) and drawing
  Gaussian samples.

## Logging and monitoring
- **Progress bars** – `tqdm.auto.tqdm` keeps both the toy and CIFAR training
  loops informative without flooding the console.
- **Weights & Biases** – both training entry points expose a flag to enable W&B
  logging. Per-step training metrics use the `train/*` namespace, epoch summaries
  land under `epoch/*`, and evaluation statistics under `eval/*`.

## License
MIT – see `LICENSE`.
