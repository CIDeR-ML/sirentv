# SirenTV

A neural network-based photon visibility model for [DUNE](https://www.dunescience.org/) (Deep Underground Neutrino Experiment). SirenTV learns to predict photon time distributions (PDF/CDF) and visibility for photomultiplier tubes (PMTs) from 3D spatial positions using [SIREN](https://arxiv.org/abs/2006.09661) (Sinusoidal Representation Networks) architectures.

## Installation

```bash
git clone https://github.com/CIDeR-ML/sirentv.git
cd sirentv
pip install -e .
```

**Dependencies**: `torch`, `numpy`, `pyyaml`, `tqdm`, `wandb`, `hist`

**External packages** (install separately):
- [`photonlib`](https://github.com/CIDeR-ML/photonlib) — Photon library I/O and voxel metadata
- [`slar`](https://github.com/CIDeR-ML/slar) — SIREN base layers, optimizers, visibility transforms

## Quick Start

```bash
# Single GPU
python -m sirentv.train --config config/train_sirentv_81_branched.yaml

# Multi-GPU (via torchrun)
torchrun --nproc_per_node=4 -m sirentv.train --config config/train_sirentv_81_branched.yaml

# Enable Weights & Biases logging (default: CSV)
python -m sirentv.train --config config/train_sirentv_81_pca.yaml --wandb
```

## Architecture

### Models

SirenTV supports multiple network architectures, selected via the `model.network.type` config key:

| Architecture | Config Type | Description |
|---|---|---|
| `BranchedSiren` | Waveform | Shared encoder + separate visibility and timing decoders |
| `ParallelSiren` | Waveform | Two independent SIREN networks for visibility and timing |
| `ConditionalSiren` | Waveform | Conditional architecture with position/time/PMT embeddings |
| `PcaSiren` | PCA | Single SIREN outputting PCA coefficients + visibility + log(t0) |
| `DualPcaSiren` | PCA | Two branches: one for vis+t0, one for PCA coefficients |

All models are wrapped by `SirenTV` (`sirentv/models/model.py`), which handles coordinate normalization, visibility transforms, and masking.

### Training Modes

The training loop is fully generic — mode-specific behavior is resolved at setup time through config-driven registries:

**Waveform mode**: Trains on full photon library waveforms (PDF or CDF). The model predicts per-PMT visibility and time distributions directly.

```yaml
data:
  dataset:
    type: PLibDataset
model:
  mode: cdf        # or "pdf"
  network:
    type: BranchedSiren
```

**PCA mode**: Trains on compressed photon libraries where waveforms are represented as PCA coefficients + first-photon time (t0). The model predicts PCA coefficients, visibility, and log(t0).

```yaml
data:
  dataset:
    type: CompressedPLibDataset
model:
  mode: pca_cdf
  network:
    type: PcaSiren
```

### First Photon Time (t0)

t0 represents the earliest photon arrival time at each PMT and is handled differently depending on the training mode:

**Waveform mode**: The model predicts t0 in units of time ticks. During training, if `load_pos: true` in the data config, the training loop computes target t0 as the geometric time-of-flight from each voxel position to each PMT:

```
target_t0 = distance(position, pmt_position) / speed_of_light_in_LAr
```

The predicted t0 is scaled by `tick_size` (ns per tick) before comparison. t0 is used to apply a sigmoid ramp mask (`t0_mask()`) that zeros out the timing distribution before the predicted first photon arrival.

**PCA mode**: The compressed photon library stores t0 directly in nanoseconds. The dataset applies a log transform (`log(t0_raw)`) and the model predicts log(t0) via its `t0` output head. The loss operates in log-space, which handles the large dynamic range of arrival times across PMTs. During inference, the predicted log(t0) is exponentiated to recover t0 in nanoseconds for CDF reconstruction and alignment.

### Data Flow

1. Dataset loads photon library data (full waveform or compressed PCA)
2. Positions are normalized to [-1, 1] via `AABox.norm_coord()`
3. Visibility targets are log-transformed: `log(v + eps)` normalized to `[0, vmax]`
4. Network predicts visibility (`v`), timing (`t` or `coeffs`), and optionally `t0` per PMT
5. Loss is computed in the transformed domain; metrics in the linear domain

## Configuration

Training is fully driven by YAML config files. Key sections:

```yaml
# Data source
photonlib:                          # Waveform mode
  filepath: path/to/plib.h5
compressed_plib:                    # PCA mode
  filepath: path/to/compressed.h5
  n_components: 50

# Dataset (registry-based)
data:
  dataset:
    type: PLibDataset               # or CompressedPLibDataset
  n_pmt: 81
  loader:
    batch_size: 2048
    num_workers: 8
    shuffle: true
  weight:
    v:
      enable: true
      factor: 5000.0
      threshold: 1.0e-08

# Model
model:
  mode: cdf                         # pdf, cdf, or pca_cdf
  network:
    type: BranchedSiren
    in_features: 6
    hidden_features: [512, 256, 1024]
    hidden_layers: [3, 3, 3]

# Visibility transform
transform_vis:
  vmax: 1.0
  eps: 1.e-8

# Training
train:
  max_epochs: 500
  optimizer_class: AdamW
  optimizer_param:
    lr: 2.0e-05
  scheduler_class: ReduceLROnPlateau
  reduction: mean                   # mean, sum, or geometric_mean
  amp: false                        # mixed precision
  clip_grad: null                   # gradient clipping max norm
  loss:
    - type: SmoothL1Loss
      key: t
      weight: 1.0
    - type: SmoothL1Loss
      key: v
      weight: 1.0

# Logging
logger:
  type: wandb                       # or csv
  project: my-project
  infer_fn:
    type: WaveformInfer             # or PCAInfer
  grad_norm:
    enabled: false
    log_per_layer: false
```

### Available Losses

All losses use the registry pattern (`type` key in config):

| Loss | Description |
|---|---|
| `WeightedL2Loss` / `L2Loss` | L2 (MSE) with optional sample weighting |
| `SmoothL1Loss` / `WeightedSmoothL1Loss` | Huber loss with optional masking |
| `WeightedPoissonNLLLoss` | Poisson NLL for count data |
| `WeightedCosineDissimilarity` | Cosine dissimilarity loss |
| `JSDivergenceLoss` | Jensen-Shannon divergence |
| `VisibilityGradient_L2Loss` | Gradient magnitude supervision (L2) |
| `VisibilityGradient_SmoothL1Loss` | Gradient magnitude supervision (Huber) |

Regularizers: `L1Regularization`, `L2Regularization`, `LNRegularization`

## Evaluation

```bash
python -m sirentv.eval --config config/eval_sirentv_81_branched.yaml --output results.pt
```

Computes per-PMT visibility bias, per-tick timing bias, and stores predictions/targets/errors to a `.pt` file.

## Distributed Training

SirenTV supports multi-GPU training via PyTorch DDP:

```bash
torchrun --nproc_per_node=4 -m sirentv.train --config config/train.yaml --wandb
```

For SLURM clusters, see example scripts in `scripts/`:

```bash
sbatch scripts/run_sirentv.sh
```

## Extending SirenTV

The codebase uses mmcv-style registries. Adding a new component requires no changes to the training loop:

**New model**: Register with `@MODELS.register_module()` in `sirentv/models/`, specify `type` in config.

**New dataset**: Register with `@DATASETS.register_module()` in `sirentv/data/`, return the [standardized batch format](#data-flow), specify `type` in config.

**New loss**: Register with `@LOSSES.register_module()` in `sirentv/loss/`, specify `type` and `key` in config.

**New inference/plotting**: Register with `@INFER_FNS.register_module()` in `sirentv/infer.py`, specify `type` in config.

## Project Structure

```
sirentv/
├── config/              # YAML training/eval configs
├── scripts/             # SLURM job scripts
├── sirentv/
│   ├── models/          # Network architectures + SirenTV wrapper
│   ├── loss/            # Loss functions (registry-based)
│   ├── data/            # Datasets + dataloaders (registry-based)
│   ├── training/        # Shared training utilities
│   ├── utils/           # Logging, transforms, DDP, misc
│   ├── infer.py         # Inference/plotting (registry-based)
│   ├── analysis.py      # Post-training analysis
│   ├── train.py         # Unified training loop + CLI
│   └── eval.py          # Evaluation loop + CLI
└── setup.py
```
