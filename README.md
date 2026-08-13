<div align="center">

# SirenTV

### Neural photon-visibility and timing surrogates for liquid-argon TPCs

SirenTV learns a continuous map from a 3D detector position to the photon
visibility and arrival-time distribution seen by every photomultiplier tube
(PMT). It uses [sinusoidal representation networks
(SIRENs)](https://arxiv.org/abs/2006.09661) as compact, differentiable
surrogates for voxelized photon libraries.

</div>

SirenTV provides:

- full-waveform PDF/CDF and PCA-compressed training paths;
- shared-encoder and parallel SIREN architectures;
- lazy HDF5 loading for datasets that do not fit in memory;
- single-process and DistributedDataParallel training;
- CSV logging by default, with optional Weights & Biases integration; and
- portable checkpoint loading for checkpoints created by version 0.1.0 or
  newer.

## Installation

SirenTV requires Python 3.10 or newer and PyTorch 2.x. Install the package from
a clone:

```bash
git clone https://github.com/CIDeR-ML/sirentv.git
cd sirentv
python -m pip install .
```

The package metadata installs the core dependencies, including compatible
PhotonLib and slar revisions. Plotting and W&B are optional:

```bash
python -m pip install ".[wandb]"
```

Until PhotonLib and slar publish their 0.2 releases, SirenTV pins the exact
release-candidate commits below for reproducible installs:

| Dependency | Revision |
|---|---|
| [PhotonLib](https://github.com/CIDeR-ML/photonlib) | `5f734f43218b65951001c7c02cc2f95767056da8` |
| [slar](https://github.com/CIDeR-ML/siren-lartpc) | `d20eaa0076c9c953169fec6eb119bee8c16591a8` |

## Portable quick start

The repository includes a generator for an eight-voxel, two-PMT photon
library. It is intended as an installation and training smoke test, not a
physics sample:

```bash
python examples/create_tiny_photonlib.py tiny_photonlib.h5
python -m sirentv.train --config config/example_tiny.yaml
```

The example performs two CPU-compatible optimizer steps and writes a CSV log
under `tiny-logs/`. Its model has deliberately small hidden layers, and the
configuration disables `torch.compile` so the first run starts quickly.

For a real detector library, copy the closest recipe and update its data and
output paths:

```bash
cp config/train_sirentv_81_branched.yaml run.yaml
# Edit photonlib.filepath and logger.dir_name in run.yaml.
python -m sirentv.train --config run.yaml
```

Passing `--wandb` preserves a configured W&B logger. Without that flag, the
training CLI uses local CSV logging.

## Supported paths

| Goal | Dataset | Model | Starting recipe |
|---|---|---|---|
| Full timing PDF/CDF | `PLibDataset` | `BranchedSiren` | [`train_sirentv_81_branched.yaml`](config/train_sirentv_81_branched.yaml) |
| Independent timing and visibility branches | `PLibDataset` | `ParallelSiren` | [`train_sirentv_81_parallel.yaml`](config/train_sirentv_81_parallel.yaml) |
| Visibility-gradient supervision | `PLibDataset` | `BranchedSiren` | [`train_sirentv_81_branched_hardmask_L1.yaml`](config/train_sirentv_81_branched_hardmask_L1.yaml) |
| PCA-compressed timing | `CompressedPLibDataset` | `PcaSiren` | [`train_sirentv_81_pca.yaml`](config/train_sirentv_81_pca.yaml) |
| Two-branch PCA model | `CompressedPLibDataset` | `DualPcaSiren` | [`train_sirentv_81_dualpca.yaml`](config/train_sirentv_81_dualpca.yaml) |

`ConditionalSiren` remains experimental. It is registered for development but
is not part of the tested wrapper contract, so new runs should use one of the
models above. Older files under `config/` record historical experiments and
may use the pre-registry schema.

## Data and model contract

Waveform mode reads an HDF5 photon library through PhotonLib. PCA mode reads a
`CompressedPLib` file containing coefficients, visibility, and first-photon
time. Both datasets return the same top-level sample structure:

```python
{
    "position": position,                    # (3,)
    "target": {"v": visibility, ...},       # values consumed by losses
    "meta": {"v_linear": visibility, ...}, # linear-domain diagnostics
}
```

For a batch of `B` positions, `P` PMTs, `T` timing bins, and `K` PCA
components, supported models expose these shapes:

| Value | Shape |
|---|---|
| Input positions | `(B, 3)` |
| Visibility | `(B, P)` |
| Timing PDF/CDF | `(B, P, T)` |
| First-arrival time | `(B, P)` |
| PCA coefficients | `(B, P, K)` |

The `SirenTV` wrapper normalizes source coordinates with the photon library's
axis-aligned box, masks positions outside its volume, and optionally
concatenates normalized PMT coordinates. Set
`data.loader.load_pos: true` for the latter and use six network input features;
otherwise use three.

Waveform timing heads reserve their first output for `t0`, so
`model.network.out_features[1]` must be one plus the number of timing bins.
PCA recipes instead set `model.network.n_components` to match the compressed
library.

## Configuration essentials

Training is controlled by YAML. These fields should be checked together:

```yaml
photonlib:
  filepath: /path/to/library.h5
  lazy: true
  time_tick_size: 0.1

data:
  n_photon: 15000000
  n_pmt: 81
  dataset:
    type: PLibDataset
  loader:
    batch_size: 2048
    load_pos: true
    num_workers: 8
    shuffle: true

model:
  ckpt_file: null
  mode: cdf
  network:
    type: BranchedSiren
    in_features: 6
    hidden_features: [512, 256, 1024]
    hidden_layers: [3, 3, 3]
    out_features: [1, 1001]

train:
  compile: false
  max_epochs: 500
  optimizer_class: AdamW
  optimizer_param:
    lr: 2.0e-5
  scheduler_class: ReduceLROnPlateau
  scheduler_param:
    patience: 10
    factor: 0.5
  loss:
    - type: SmoothL1Loss
      key: t
      weight: 1.0
    - type: SmoothL1Loss
      key: v
      weight: 1.0
```

`train.compile` defaults to `false`; enable it after validating an uncompiled
run on the target PyTorch installation. To resume the model, optimizer,
scheduler, epoch, and exact iteration, set `model.ckpt_file` and
`train.resume: true`. New checkpoints include the geometry and data/model
configuration needed for standalone inference.

## Evaluation and inference

Evaluate a waveform checkpoint with:

```bash
python -m sirentv.eval \
  --config config/eval_sirentv_81_branched.yaml \
  --output eval_results.pt
```

Update both `photonlib.filepath` and `model.ckpt_file` first. The output stores
overall and per-PMT visibility bias, per-time-bin timing bias, all evaluated
predictions and targets, and visibility/timing error correlations.

Checkpoints produced by this release can also reconstruct a model without the
original YAML:

```python
import torch

from sirentv.models import SirenTV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SirenTV.load("iteration-000100-epoch-0001.ckpt").to(device).eval()
positions = torch.tensor([[0.0, 0.0, 0.0]], device=device)

with torch.inference_mode():
    outputs = model(positions)
    waveforms = model.visibility(positions, return_type="pdf")
```

For older checkpoints without embedded data configuration, pass a complete
configuration with `model.ckpt_file` set.

## Distributed training

Launch one process per GPU with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m sirentv.train --config run.yaml --wandb
```

`data.loader.batch_size` is per process. A distributed sampler partitions the
voxels, while log and checkpoint writes are restricted to rank zero.

## Development

Install the development dependencies and run the regression suite:

```bash
python -m pip install ".[dev]"
python -m pytest -q
python -m build
```

The tests generate their own tiny HDF5 library and exercise dense and lazy
loading, supported model forward/backward paths, analytical-gradient
supervision, scheduler stepping, checkpoint reconstruction, a training step,
and evaluation. CI runs the suite on Python 3.10 and 3.12.

The codebase uses lightweight registries for datasets, models, losses,
regularizers, and training-time inference hooks. A new dataset should preserve
the sample contract above; a configured loss key must exist in both the
selected model output and dataset target.

## Repository layout

```text
.
├── config/                  # Training and evaluation recipes
├── examples/                # Portable data-generation examples
├── sirentv/
│   ├── data/                # Photon-library datasets and loaders
│   ├── loss/                # Registered objectives and regularizers
│   ├── models/              # SIREN architectures and wrapper
│   ├── training/            # Shared training helpers
│   ├── utils/               # Logging, transforms, DDP, and metrics
│   ├── eval.py              # Evaluation CLI
│   ├── infer.py             # Inference hooks and PCA reconstruction
│   └── train.py             # Training CLI
├── tests/                   # Unit and integration tests
└── pyproject.toml           # Build and dependency metadata
```

## Contact

Sam Young (`youngsam@stanford.edu`) and Junjie Xia
(`junjiex@stanford.edu`).
