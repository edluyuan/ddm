# Distributional Diffusion on 2D and CIFAR10

This minimal PyTorch package implements the **Distributional Diffusion Model (DDDM)**
from *"Distributional Diffusion Models with Scoring Rules"* (De Bortoli et al., 2025)
for 2D toy data. It adheres to the paper's math and algorithms, and is designed to be
easy to read & extend.

## What’s inside
- `dddm_2d.py` — self‑contained implementation:
  - Forward schedule **α_t = 1−t**, **σ_t = t** (paper eq. (3)).
  - Transitional Gaussian **p(x_s|x_0,x_t)=N(μ_{s,t}, Σ_{s,t})** with **μ, Σ** from eq. (4).
  - Training Algorithm 1 (distributional denoiser `x̂_θ(t, x_t, ξ)`), loss eqs. **(12)–(14)**.
  - Sampling Algorithm 2 (coarse reverse with distributional `x̂_θ` plugged into eq. (4)).
  - Optional classical diffusion baseline (β=2, λ=0) via same training loop.
  - MMD^2 evaluation (rbf kernel), matching Section 6.1 setup.
- `run_example.py` — tiny script to train on a 2‑Gaussian mixture and sample.
- `LICENSE` — MIT.

## Quick start
```bash
pip install torch matplotlib
python run_example.py --epochs 2000 --batch 512 --beta 0.1 --lam 1.0 --m 8 --steps 20
# samples saved to ./out/*.png and checkpoint to ./out/model.pt
```
*Tip:* Start with fewer epochs to verify end‑to‑end, then increase.

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
  - The **empirical energy diffusion loss** (eq. **(14)**) is exactly implemented,
    including factor `λ/(2(m−1))` and minibatch weight `w_t`.
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

Enjoy hacking!
