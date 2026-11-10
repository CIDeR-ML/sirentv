# SirenTV

A neural photon visibility model for [DUNE](https://www.dunescience.org/): from a 3D position, SirenTV predicts, per PMT, the visibility, the first-photon time t0 and the photon arrival-time distribution, using [SIREN](https://arxiv.org/abs/2006.09661) (sinusoidal representation) networks. It models one detector half.

## Installation

Python >= 3.10. SirenTV needs two packages that are not on PyPI:
[`photonlib`](https://github.com/CIDeR-ML/photonlib) (photon library I/O, voxel metadata) and
`slar` from [`siren-lartpc`](https://github.com/CIDeR-ML/siren-lartpc) (SIREN layers, optimizers, visibility transforms).

**Own environment (pip):**

```bash
git clone https://github.com/CIDeR-ML/photonlib.git
git clone https://github.com/CIDeR-ML/siren-lartpc.git
git clone https://github.com/CIDeR-ML/sirentv.git
pip install -e photonlib -e siren-lartpc -e sirentv   # sirentv also pulls torch, numpy, h5py, pyyaml, tqdm, wandb, matplotlib, scipy
```

This also installs the `sirentv-train` and `sirentv-eval` commands (same flags as `python -m sirentv.train` / `sirentv/eval/eval.py`).

## Quick start

The reference configs are `config/{train,eval}_q3_t0align_{s3df,nersc}.yaml`; every other config is kept under `config/archive/`.

```bash
# Single GPU (CSV logging); add --wandb for Weights & Biases
python -m sirentv.train --config config/train_q3_t0align_s3df.yaml

# Archived configs by name
python -m sirentv.train --config abl_q_w120_d6_s3df.yaml --archive

# Multi-GPU
torchrun --nproc_per_node=4 -m sirentv.train --config config/train_q3_t0align_s3df.yaml --wandb

# Evaluation (routes on data.dataset.type)
torchrun --nproc_per_node=4 sirentv/eval/eval.py --config config/eval_q3_t0align_s3df.yaml \
    --output eval-q3_t0align.pt --pmt-ids 0 40 80
```

## Running jobs on NERSC and S3DF

`sirentv/scripts/submit_nersc.sh` (Perlmutter) and `sirentv/scripts/submit_s3df.sh` are minimal Slurm scripts: one task per GPU on 2 nodes x 4 GPUs, the 8-rank shape the reference configs assume (global batch 4096, memory shard per rank). They run inside the `deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim` image (CUDA 12.1, torch 2.5.1). Install the three packages once into your user site from inside that image, from the directory holding the three clones (the image already has torch, so pip adds only what is missing, such as wandb):

```bash
# NERSC
shifterimg pull deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim
shifter --image=deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim \
    pip install --user -e photonlib -e siren-lartpc -e sirentv

# S3DF: build a .sif once, then point IMAGE at it when submitting
singularity pull larcv2.sif docker://deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim
singularity exec larcv2.sif pip install --user -e photonlib -e siren-lartpc -e sirentv
export IMAGE=$PWD/larcv2.sif
```

Set `#SBATCH --account` in the script first, then submit from the repository root. Everything after the config is passed to the module, and `MODULE=sirentv.eval.eval` runs the evaluator instead of training.

```bash
# NERSC Perlmutter
sbatch sirentv/scripts/submit_nersc.sh config/train_q3_t0align_nersc.yaml --wandb
sbatch sirentv/scripts/submit_nersc.sh config/train_q3_t0align_nersc.yaml --wandb --resume
MODULE=sirentv.eval.eval sbatch sirentv/scripts/submit_nersc.sh config/eval_q3_t0align_nersc.yaml \
    --output eval-q3_t0align.pt --pmt-ids 0 40 80

# S3DF
sbatch sirentv/scripts/submit_s3df.sh config/train_q3_t0align_s3df.yaml --wandb
MODULE=sirentv.eval.eval sbatch sirentv/scripts/submit_s3df.sh config/eval_q3_t0align_s3df.yaml \
    --output eval-q3_t0align.pt --pmt-ids 0 40 80

# sbatch options override the script's, e.g. a short run in another QOS
sbatch -q premium -t 02:00:00 sirentv/scripts/submit_nersc.sh config/train_q3_t0align_nersc.yaml
```

`train.max_epochs` counts epochs per job: to continue a run, resubmit with `--resume` and `model.ckpt_file` pointing at its latest checkpoint.

## Representations

The time distribution is learned in one of two compressed forms, each with its own LUT and dataset:

| Representation | Dataset | LUT section | Output key | Evaluator |
|---|---|---|---|---|
| Quantile function (u-grid levels) | `QuantilePLibDataset` | `photonlib` / `quantile_plib` | `quantiles` | `eval/eval_quantile.py` |
| PCA coefficients | `CompressedPLibDataset` | `compressed_plib` | `coeffs` | `eval/eval_pca.py` |

Targets: `v = xform_vis(vis / n_photon)` (a log transform), `t0 = log(t0 [ns])`, and the representation. With `photonlib.mode: log_quantile` the quantiles are `log10(Q + C)`; with `align_quantiles_to_t0` the onset is removed first, so they are delays after t0.

## Models

`model.network.type`, wrapped by `SirenTV` (`sirentv/models/model.py`), which normalizes positions, tiles them against the PMT positions, and masks points outside the volume.

| Network | Description |
|---|---|
| `MultiBranchSiren` | Any split of the outputs into independent SIREN branches, each with its own width, depth and omega_0 |
| `PcaSiren`, `QuantileSiren` | One trunk for `[v, t0, coeffs]` / `[v, t0, quantiles]` (legacy checkpoint names) |
| `DualPcaSiren` | `vt0_net` for `[v, t0]`, `coeff_net` for `[coeffs]` (legacy checkpoint names) |

```yaml
model:
  mode: quantile_cdf
  network:
    type: MultiBranchSiren
    in_features: 6
    n_quantile: 512
    branches:
    - {name: v_net, keys: [v], hidden_features: 256, hidden_layers: 9, first_omega_0: 30.0, hidden_omega_0: 30.0}
    - {name: t0_net, keys: [t0], hidden_features: 256, hidden_layers: 9, first_omega_0: 15.0, hidden_omega_0: 15.0}
    - {name: quantile_net, keys: [quantiles], hidden_features: 512, hidden_layers: 6, first_omega_0: 120.0, hidden_omega_0: 120.0}
```

## Configuration notes

- `data.target_keys` limits the targets (and the LUT fields read). `data.x_min` [mm] keeps only voxels with x >= x_min, e.g. -2160 for the active volume. `data.subsample_factor` / `data.max_len` thin the voxels.
- `data.memory_shard: true` keeps each rank's 1/world_size of the voxels in RAM (quantile dataset).
- `train.clip_grad`: a float (global) or `{branch_name: max_norm}`.
- `train.grad_supervision_keys` (`[v, coeffs]` or `{coeffs: <start epoch>}`) adds `WeightedGradFrobLoss` targets. Fill `train.grad_target_cache_file` once with `python -m sirentv.scripts.precompute_grad_frob_targets --config ...`.
- `logger.analysis`: a list of `{func: bias|abs_bias, suffix, key, threshold[, signed]}`, logged every `logger.log_every_nsteps` steps.
- `train.reduction`: `mean`, `sum` or `geometric_mean` over the loss terms. `train.seed` seeds the weight init.

## Losses

| Loss | Description |
|---|---|
| `WeightedL2Loss` | L2 with optional per-sample weights and `mask_key` |
| `SmoothL1Loss` / `WeightedSmoothL1Loss` | Huber, unweighted / weighted |
| `WeightedPoissonNLLLoss` | Poisson NLL in count units (n_photon and the inverse xform are injected) |
| `WeightedReconstructedQuantileLoss` | L2 in quantile space, weighted by the target's local density |
| `WeightedGradFrobLoss` | Matches the squared spatial-gradient Frobenius norm (gradient supervision) |
| `WeightedCosineDissimilarity`, `JSDivergenceLoss` | Kept, unused by current configs |

The production configs use the Poisson NLL on `v`; the loss-ablation notes are in `docs/archive/loss_ablations.md`.

## Evaluation output

`eval.py` writes a `.pt` with overall / per-PMT / per-quantile-bin bias and absolute error, the error in units of the local quantile spacing with its Monte Carlo Poisson floor, t0 error, per-voxel arrays, and with `--pmt-ids` spatial-slice diagnostics and (kx, ky, kz) power spectra (`--no-spectra` to skip). For PCA runs, `--compare` selects pred_vs_compressed / pred_vs_raw / compressed_vs_raw.

## Extending

Components are registry-based (mmcv-style); none need changes to the training loop:

- model: `@MODELS.register_module()` in `sirentv/models/`
- dataset: `@DATASETS.register_module()` in `sirentv/data/` (import it in `data/builder.py`)
- loss: `@LOSSES.register_module()` in `sirentv/loss/`
- inference/plotting: `@INFER_FNS.register_module()` in `sirentv/infer.py`

## Tests

**Run the tests before every push to the remote.**

```bash
tests/run_tests.sh       # import/compile + unit tests, in the active environment (needs photonlib, slar)
tests/submit_tests.sh    # S3DF: the same as a CPU job inside the container -> ./tests-<jobid>.out
```

`tests/test_imports.py` compiles and imports every module; the other `tests/test_*.py` are fast unit tests (no LUT, no GPU).

For a change that should not alter numerics (a refactor), also run the bitwise regression suite: `tests/bitwise/` trains `q3_t0align` and a gradient-supervised PCA config for a few epochs and evaluates two production checkpoints, then compares two runs tensor by tensor:

```bash
tests/bitwise/submit.sh <checkout A> <out A>    # S3DF, 4 GPUs, inside the container (~1 h)
tests/bitwise/submit.sh <checkout B> <out B>
tests/bitwise/compare.sh <out A> <out B>        # PASS / FAIL report in <out B>/compare-*.out
```

## Project structure

```
sirentv/
├── config/              # reference configs; archive/ holds the rest
├── docs/                # archive/ superseded notes
├── tests/               # import/unit tests (run_tests.sh); bitwise/ regression suite
├── sirentv/
│   ├── models/          # networks + SirenTV wrapper
│   ├── loss/            # losses (registry)
│   ├── data/            # LUT readers, datasets, loaders (registry)
│   ├── eval/            # eval CLI, PCA / quantile evaluators, shared metrics
│   ├── training/        # loss/logger/clipping helpers
│   ├── utils/           # logging, gradient estimators, spectra, DDP, registry
│   ├── scripts/         # Slurm submission examples (NERSC, S3DF); offline gradient-target precompute
│   ├── infer.py         # training-time plots (registry)
│   ├── analysis.py      # bias metrics for logger.analysis
│   └── train.py         # training loop + CLI
└── setup.py
```
