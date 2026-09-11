# Three training-time ablations for `sirentv`, and what "spatial gradient of a PDF" means

This note explains, for an audience that has not been following the day-to-day debugging, why we
made three specific changes to how the SIREN model (`sirentv`) is trained, what each one actually
computes, and — since it comes up in the third ablation and is not obvious — what it means to take
a "spatial gradient" of a multi-channel output like a PCA coefficient vector or a quantile function.

## 1. Background: what the model predicts and where it struggles

For every (voxel position, PMT) pair, the true simulation produces:

- a **visibility** `v` — the fraction of photons emitted at that voxel that land on that PMT,
- a **t0** `t0` - the first non-zero bin of that PMT's waveform from that voxel,
- an **arrival-time distribution** for the photons that do land, represented either as a
**quantile function** `Q(u)` (512 samples of "time by which fraction `u` of the photons have
arrived", `u ∈ (0,1)`) or as a **PCA-compressed** version of the same curve (50 coefficients
in a shared basis fit across all voxels/PMTs).

`sirentv` is a SIREN — a coordinate network `f(x): R^3 → outputs` with periodic (sine) activations
— trained to reproduce `v`, `t0`, and either the quantile values or the
PCA coefficients, at every voxel and every PMT, given only 3D position as input.

Two systematic error patterns motivated this round of changes:

1. **Visibility bias is worse in regions with sharp near/far-field structure** (close to a PMT,
  where visibility falls off steeply) than in smooth regions far from any PMT. Farthest from a PMT where the visibility is low, the siren predicts high-frequency oscillatory artifacts.
2. **The early quantile bins** (small `u`, i.e. the leading edge of the arrival-time distribution)
  carry more error than the late bins, for both the PCA and non-PCA models — we confirmed
   through several rounds of diagnostics that this reflects genuine structure in the target data
   (steep local curvature, low photon counts) rather than a metric artifact.

Both patterns point at the same root cause: **the network is not being told, directly, how
steeply its own output should vary in space** — it only ever sees pointwise value targets, never
slope targets. The three ablations below all attack this from a different angle.

---



## 2. Ablation A — Poisson NLL loss for visibility

**File:** `sirentv/loss/poisson.py` (`WeightedPoissonNLLLoss`)

### Why not L2 / smooth-L1?

The visibility target `t` at a voxel is itself an *estimate* from a finite Monte Carlo photon
count — if `n` photons were launched from a voxel and `Y` landed on a PMT, the recorded visibility
is `t = Y/n`. `Y` is a Poisson-distributed count with mean `λ = n·p`, where `p` is the (unknown)
true visibility. An L2 or smooth-L1 loss on `t` implicitly assumes the noise on `t` has *constant*
variance everywhere. It does not: `Var(t) = p(1-p)/n ≈ p/n` for small `p`, so voxels with tiny
true visibility have *proportionally* noisier targets than high-visibility voxels. Training on raw
squared error over-weights exactly the low-visibility (typically far-from-PMT, but also
shadowed/reflective) regions where the target itself is least trustworthy.

### The derivation

The Poisson negative log-likelihood of observing count `Y` given rate `λ` is

$$
\mathrm{NLL}(\lambda) = \lambda - Y \log \lambda \quad (\text{dropping the } Y\text{-only, } \lambda\text{-independent term } \log Y!).
$$

Substituting the model's predicted visibility `p` for the rate via `λ = n·p`, and the target
count via `Y = n·t`:

$$
\mathrm{NLL}(p) = np - nt\log(np) = np - nt\log n - nt\log p.
$$

The middle term `n t log n` does not depend on `p`, so it contributes nothing to the gradient
w.r.t. the network's output and can be dropped. What remains is the loss actually implemented:

$$
\mathcal{L}_{\text{Poisson}}(p, t) = n\left(p - t\log p\right).
$$

This is exactly the count-domain Poisson NLL up to an additive, prediction-independent constant —
so minimizing it is *identical* to maximum-likelihood fitting of the true photon counts, while
still operating on the visibility fraction `p` the network already outputs. The `n` (number of
photons launched per voxel, `n_photon` in the config) rescales the loss back into a count-like
scale so its magnitude is comparable to the other terms; it also makes the effective per-voxel
weighting `∝ n`, and the implicit per-target-value curvature `∝ 1/p` — i.e. the loss automatically
penalizes relative (not absolute) errors more heavily at small `p`, which is the physically
correct thing given `Var(t) ∝ p/n`.

`p` in the formula above is always the model's output **after inverting whatever domain transform**
`transform_vis` **applies** (e.g. `sin`, log/vmax rescaling) — `WeightedPoissonNLLLoss` is handed the
inverse transform (`inv_xform`) precisely so the Poisson NLL is evaluated in true-visibility space,
not in the network's internal (transformed) output space.

### What this ablation tests

Whether shaping the *loss curvature* to match the true noise model of the visibility target (heavier
penalty on relative error at low visibility, matching the Poisson floor) reduces the observed
low-visibility-region bias, without needing any change to the network or to spatial supervision.

Config: `train_sirentv_81_dualpca_poisson.yaml`.

---



## 3. Ablation B — density-weighted reconstructed-quantile loss

**File:** `sirentv/loss/quantile_recon.py` (`WeightedReconstructedQuantileLoss`)

### Why not plain L2 on PCA coefficients (or quantile bins)?

The dual-branch PCA model (`DualPcaSiren`) predicts 50 PCA coefficients per (voxel, PMT). Plain L2
loss on those coefficients treats every principal component as equally important, and — after
reconstructing the actual quantile function via `Q = mean + coeffs · components` — treats every
point on the *time axis* as equally important too. But a fixed *fraction* step `Δu` corresponds to
a very different *time* interval `ΔQ` depending on where you are on the curve: where the quantile
function `Q(u)` is steep (density is high, i.e. many photons arrive in a narrow time window), a
given `Δu` maps to a small `ΔQ`; where `Q(u)` is flat (density is low, photons trickle in slowly),
the same `Δu` maps to a large `ΔQ`. Supervising uniformly in bin-index space over- or
under-weights different parts of the physical time axis depending on this local density, and does
so inconsistently across voxels with different overall pulse shapes.

### The fix: reconstruct, then weight by local density

`WeightedReconstructedQuantileLoss` first reconstructs the actual quantile-function values from
whatever the model predicts (denormalizing and un-PCA-projecting if needed):

$$
\hat{Q} = \mu + \hat{c} V^\top, \qquad Q = \mu + c V^\top
$$

(`μ` = PCA mean curve, `V` = PCA components, `ĉ`/`c` = predicted/target coefficients — or, for the
non-PCA quantile model, `Q̂`/`Q` are the network's direct bin outputs and this step is a no-op).

It then estimates the **local probability density** at each bin from the *target* curve's own bin
spacing. For quantile fraction `u`, the density of arrival times near `Q(u)` is, by the standard
change-of-variables relation between a CDF and its inverse (the quantile function),

$$
f\big(Q(u)\big) = \frac{1}{Q'(u)} \approx \frac{\Delta u}{\Delta Q(u)} \Big|_{\text{finite difference across adjacent bins}}.
$$

i.e. `density = 1 / local_bin_spacing`, computed with a simple finite difference between adjacent
target bins (edge bins reuse their single neighbor). This gives a per-bin weight

$$
w(u) = f\big(Q(u)\big)^{\gamma}, \qquad \text{normalized so } \langle w \rangle_u = 1 \text{ per curve},
$$

where `γ` (`power` in the config, e.g. `weight_power: 3.0`) controls how aggressively bins are
up-weighted where the physical arrival-time density is high. The final loss is a weighted smooth
squared error directly on reconstructed *time* values:

$$
\mathcal{L}_{\text{recon}} = \Big\langle w(u)\cdot\big(\hat{Q}(u) - Q(u)\big)^2 \Big\rangle_u .
$$

This is a **physical-importance weighting**, not a **statistical-confidence weighting** — it is
deliberately *not* the order-statistic sampling-variance formula `u(1-u) / (N f(Q(u))^2)` that
governs how noisy the *empirical* quantile at `u` is for `N` samples. That formula answers "how
much do I trust this target value," which is a different question from "how much does this part of
the curve matter physically," and the two are not algebraically combinable (one lives in bin
space, the other would need to be transported through the PCA basis, which is not diagonal in
either representation). We chose the physical-importance interpretation: statistical confidence in
`v` is already handled by Ablation A's Poisson weighting, so this loss's job is purely to make the
*shape* fit spend its capacity where the pulse actually has structure.

Two important implementation details, both due to the PCA representation:

- If `coeff_mean`/`coeff_std` are supplied (i.e. training used `normalize_coeffs: true`), the
predicted/target coefficients are **denormalized before reconstruction** — reconstructing
directly from normalized coefficients would silently apply the wrong affine offset to the
entire curve.
- Any externally supplied per-sample `weight` tensor is used only if its last dimension matches
the *reconstructed* (bin-space) shape; a component-space weight (e.g. per-PCA-coefficient
variance-based weighting) is a different, non-diagonal basis and is deliberately ignored rather
than silently misapplied to the wrong axis.



### What this ablation tests

Whether reconstructing to physical time and weighting by local photon density — rather than
supervising uniformly in coefficient or bin-index space — reduces the early-quantile-bin bias,
i.e. whether that bias is a *loss-weighting* problem rather than a *network capacity* problem.
Per explicit design choice, this ablation supervises **either** on coefficients **or** on
reconstructed quantiles, never both at once, to keep the ablation clean.

Configs: `train_sirentv_81_dualpca_quantile_recon.yaml` (PCA model, reconstruction path),
`train_sirentv_81_quantile_density_weighted.yaml` (direct quantile model, no reconstruction step
needed).

---



## 4. Ablation C — spatial-gradient (Jacobian) supervision

This is the most involved change, and the one that needs the conceptual background below.

### 4.1 What "spatial gradient" means for a multi-channel output

For an ordinary scalar function of position, `g(x): R^3 → R`, the gradient `∇g = (∂g/∂x, ∂g/∂y, ∂g/∂z)` is a single 3-vector at each point: it tells you, for a tiny step in any direction, how
much `g` changes.

Our network does not output one number per voxel — it outputs a **vector** of `K` numbers per
(voxel, PMT): `K=1` for visibility, `K=50` for PCA coefficients, `K=512` for direct quantile bins.
Write this as `f(x): R^3 → R^K`. There is no longer a single "the gradient" — instead, **each of
the** `K` **output channels has its own gradient vector**, $∇f_k = (\partial f_k/\partial x,\ \partial f_k/\partial y,\ \partial f_k/\partial z)$. Stacking all `K` of these gradient rows on top of each
other gives the **Jacobian matrix**:

$$
J(x) \in \mathbb{R}^{K\times 3}, \qquad J_{k,d}(x) = \frac{\partial f_k(x)}{\partial x_d}.
$$

Concretely: if `f` predicts the 50 PCA coefficients, `J` is a 50×3 matrix at every voxel. Its
5th row is "how does coefficient 5 change as I nudge the voxel in x, y, or z"; its 5th column
(restricted to one row) tells you how much coefficient 5 alone reacts to a step in `z`. 

### 4.2 Collapsing the Jacobian to one number: the Frobenius norm

We don't want to match the full `K×3` matrix element-by-element (expensive, and not needed for our
purposes) — we want **one scalar per (voxel, PMT)** here, to compare against the same scalar computed from the true data. The natural
"overall size" of a matrix is its **Frobenius norm** — the square root of the sum of squares of
every entry:

$$
J_F = \sqrt{\sum*{k=1}^{K}\sum_{d=1}^{3} J_{k,d}^2} = \sqrt{\sum_{k=1}^K \nabla f_k^2}.
$$

This is exactly "root-sum-of-squares of the ordinary gradient magnitude of every output channel" —
a single roughness score that grows if *any* channel is changing steeply in *any* direction. This
scalar, $‖J‖_F$, is what both sides of the new loss actually compute and compare — under the key
`f"{key}_grad_frob"` (e.g. `v_grad_frob`, `coeffs_grad_frob`) in both the prediction and the
target dictionaries.

### 4.3 The problem: computing the full Jacobian is expensive

The textbook way to get `‖J‖_F` from an autodiff framework is to backpropagate each output channel
separately: `K` calls to `torch.autograd.grad`, one per channel, each giving one row of `J`. For
`K=1` (visibility) that's fine. For `K=50` (PCA coefficients) it is 50 backward passes *per
training step, per PMT* — with 81 PMTs in a batch, 81×50 = 4050 backward passes per step. This is
what caused repeated out-of-memory failures when first implemented literally.

### 4.4 The fix: the Hutchinson trace estimator

There is a classical randomized-linear-algebra trick (Hutchinson's estimator, normally used to
estimate matrix traces) that gets an **unbiased estimate of** `‖J‖_F^2` using a **single** random
projection and a **single** backward pass, regardless of `K`.

Draw a random vector $v \in R^K$ whose entries are i.i.d. **Rademacher** (each `±1` with probability
½ — so $\mathbb{E}[v_k] = 0$ and $\mathbb{E}[v_k v_{k'}] = \delta_{k,k'}$, i.e. mean zero, unit variance, uncorrelated
across components). Backpropagate the single **scalar** $s = v^\top f(x) = \sum_k v_k f_k(x)$
through the network once. By linearity of differentiation, the resulting gradient w.r.t. position is

$$
g = \nabla_x s = \sum_{k=1}^K v_k \nabla f_k(x) = v^\top J(x) \ \in \mathbb{R}^3.
$$

This one backward pass gives us `g`, a 3-vector that mixes together all `K` channels' gradients
weighted by the random signs `v_k`. Now look at its squared norm:

$$
g^2 = \sum_{d=1}^3 g_d^2 = \sum_{d=1}^3\Big(\sum_k v_k J_{k,d}\Big)^2
= \sum_{d=1}^3\sum_{k,k'} v_k v_{k'} J_{k,d}J_{k',d}.
$$

Taking the expectation over the random draw of `v`, and using $\mathbb{E}[v_k v_{k'}] = \delta_{k,k'}$:

$$
\mathbb{E}\big[g^2\big] = \sum_{d=1}^3\sum_{k} J_{k,d}^2 = \sum_k \nabla f_k^2 = J_F^2.
$$

So `‖g‖²` from **one** random-projected backward pass is an **unbiased estimator** of the exact
quantity we wanted, `‖J‖_F^2`, no matter how large `K` is. This is the entire trick: trade an exact
answer requiring `K` backward passes for a cheap, unbiased, single-pass estimate whose cost is
independent of `K`. (In expectation over many voxels/PMTs in a batch/epoch, the estimator is
correct; per-sample it has variance, but that noise plays the same role as ordinary
minibatch-gradient noise SGD already tolerates.)

This is implemented in `compute_grad_frob_hutchinson` (`sirentv/utils/misc.py`): for each PMT it
draws one Rademacher vector, computes the scalar projection, calls
`torch.autograd.grad(..., create_graph=True, retain_graph=True)` once, and takes the norm.
`create_graph=True` is essential — it keeps the computation attached to the autograd graph so that
a *loss* built on top of this gradient can itself be backpropagated (a second-order,
double-backward pattern, the same one used in WGAN-GP gradient penalties). An earlier, now-removed
implementation used `create_graph=False` and `.detach()`, silently producing a *numerical* value
with **no gradient at all** — any loss built on it was training on a constant, contributing exactly
zero to the network's weight updates. That bug is the most likely explanation for why gradient
supervision had appeared not to help in earlier attempts.

### 4.5 Aggregating over PMTs too (memory fallback)

Even the per-PMT Hutchinson estimator (1 backward pass per PMT instead of `K`) OOM'd once both `v`
and `coeffs` were supervised together (81 PMTs × 2 keys per step). We apply the *same* trick a
second time, along the PMT axis: instead of one projection per PMT, form a single visibility-weighted
combined scalar across **all** PMTs in one shot,

$$
s_{\text{agg}} = \sum_{\text{pmt}} \sqrt{w_{\text{pmt}}} v^{(\text{pmt})\top} f^{(\text{pmt})}(x),
$$

where `w_pmt` is the (detached) predicted visibility of that PMT — used so that PMTs the voxel
barely sees contribute little to the aggregate roughness score, matching how much they'd matter
physically. One backward pass now yields an unbiased estimate of the visibility-weighted sum
$\sum_{\text{pmt}} w_{\text{pmt}} \cdot ‖J_{\text{pmt}}‖_F^2$ over *all* PMTs at once (`compute_grad_frob_hutchinson_aggregate`).
The target side is aggregated with the identical weights so the two sides of the loss are
comparable (`_grad_aggregate` path in `CompressedPLibDataset`/`QuantilePLibDataset`).

### 4.7 Warmup: don't supervise gradients before the value fit has converged

Early in training, the value prediction itself is far from converged, so its spatial derivative is
not yet meaningful — pushing the network to match gradients of a still-wrong function can fight the
value loss rather than help it. `train.py` supports a **per-key warmup**: each entry in
`grad_supervision_keys` (e.g. `{v: 0, coeffs: 100}`) is a "don't activate this key's gradient
computation until warmup progress ≥ N" threshold, where progress is counted either in epochs or
iterations (`grad_supervision_warmup_unit`). Visibility's own value fit converges quickly, so its
gradient term can be on from step 0; PCA-coefficient gradients are only turned on after the value
branch has had time to settle (100 epochs in the full config). Below the warmup threshold, the
model simply does not compute `{key}_grad_frob` at all, and `WeightedGradFrobLoss` returns exactly
`0` when the key is absent — a genuine no-op rather than a fudged zero-gradient term.

### What this ablation tests

Whether directly matching the *local rate of change* of the predicted field (visibility, and
separately, PCA coefficients / quantile shape) to the true finite-difference rate of change,
weighted toward regions with real spatial curvature, closes the bias gap in exactly those
high-curvature regions (near PMTs, steep-CDF regions) that the value-only losses (Ablations A, B)
cannot directly see.

Config: `train_sirentv_81_dualpca_grad_supervision.yaml` (and its `smoke_...` variant for a
1-epoch sanity run).

---



## 5. Summary table


| Ablation                           | Loss / mechanism                                                                                                    | Targets the...                                                               | Key hyperparameters                                                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| A. Poisson NLL                     | `n(p - t log p)`                                                                                                    | statistical noise floor of `v`, correct relative weighting at low visibility | `n_photon`                                                                                                                   |
| B. Density-weighted quantile recon | reconstruct → weight by `1/local_bin_spacing` → weighted MSE in time-space                                          | physical importance of different parts of the pulse shape                    | `weight_power (γ)`, `normalize_coeffs`                                                                                       |
| C. Gradient (Jacobian) supervision | Hutchinson-estimated `‖J‖_F` (pred, `create_graph=True`) vs. finite-difference `‖J‖_F` (target, coarse+PMT-refined) | local sharpness/curvature that pointwise losses can't express                | `grad_supervision_keys` + warmup, `grad_supervision_aggregate`, `grad_target_agg_chunk_size`, `grad_target_refine_radius_mm` |


All three are independent, individually switchable ablations against the same `DualPcaSiren`
baseline (trained from scratch, no checkpoint warm-start), so their effects can be attributed
separately before considering combining them.