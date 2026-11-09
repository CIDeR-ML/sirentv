# Handoff: NERSC 100-epoch bitwise check of the refactor

For the agent (or person) running this on NERSC Perlmutter. Read all of it before acting.

## Context

Branch `ablations` was refactored and pruned on S3DF (2026-11-09): dead code removed (waveform
line, weighting/, layers/, regularizer hook, mirroring, ...), duplicates merged
(`eval/utils.py`, `data/common.py`), docstrings trimmed, configs moved to `config/archive/`.
The pre-refactor reference is commit `846ec07`.

The S3DF regression suite (`tests/bitwise/`) gave **PASS: bitwise identical** for
`846ec07` vs the refactor: q3_t0align training (2-GPU DDP, torch.compile, memory shard),
PCA training with aggregate gradient supervision, and both evaluators.
Intended differences: eval outputs no longer contain `visibility_per_pmt["std"]`;
the CSV logger writes a row every `logger.log_every_nsteps` (`%.4e`); `--debug-timing`
and `train.regularization` are gone; `--archive` reads a config from `config/archive/`.

## Goal

Show that the refactored code reproduces the original production run exactly over a long,
multi-node run: train q3_t0align for 100 epochs from scratch and compare its epoch-50 and
epoch-100 checkpoints bit for bit with the original run's.

- Original checkpoints: `/pscratch/sd/j/junjiex/SIREN/logs/logs_train_sirentv_81_q3_t0align/version-00/`
  `iteration-019450-epoch-0050.ckpt`, `iteration-038900-epoch-0100.ckpt`
  (if `/pscratch` purged them: the CFS archive under `/global/cfs/cdirs/m5238/users/junjiex/CIDER-ML/logs/`).
- Config: `tests/bitwise/configs/q3_t0align_100ep_nersc.yaml` (= `config/train_q3_t0align_nersc.yaml`
  with 100 epochs, a checkpoint every 50, `resume: false`, log dir `logs_bitwise_q3_t0align`).

## Constraints that decide whether a match is possible

1. **8 ranks (2 nodes x 4 A100), the site default.** The original ran 389 iterations/epoch,
   which is 8 ranks; any other rank count changes the shards and the DDP average.
2. **Provenance of version-00 is unknown.** It was trained by the NERSC checkout before the
   S3DF commits, possibly with uncommitted changes. Capture that state before pulling (step 1).
3. **The container tag is not pinned** (`deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim`).
   If the image moved, even the old code may not reproduce, hence the control run.
4. Production runs do not enable deterministic algorithms; on S3DF the q3 training path hit no
   nondeterministic ops, so this should not matter.

## Rules

- Run Python only inside the container (`shifter --module gpu --image=$CONTAINER_IMAGE ...`).
- Do not change the training algorithm or checkpoint cadence to fit the infrastructure; if a
  constraint blocks the run, report it and let the user choose.
- Never delete or overwrite anything under the production log dirs; write only to `logs_bitwise_q3_t0align`.
- Ask before deleting code or committing/pushing.

## Steps

Paths: `REPO=/global/cfs/cdirs/m5238/users/junjiex/CIDER-ML/sirentv` (REPO_DIR in `site_env.sh`);
`SCRIPTS=<CIDER-ML>/scripts/slurm_job_sirentv` (outside git; verify its location).

1. Freeze the code that trained version-00, then update:
   ```bash
   cd $REPO && git log -1 && git status && git diff > ~/sirentv_pre_refactor.diff
   # local changes? -> git checkout -b nersc-pre-refactor && git commit -am "state that trained q3_t0align v00"
   git worktree add ../sirentv-baseline HEAD
   git fetch origin && git checkout ablations && git pull --ff-only
   ```
2. Unit tests in the container (must print `52 passed` or more):
   ```bash
   source $SCRIPTS/site_env.sh
   shifter --module gpu --image=$CONTAINER_IMAGE bash -c "export PYTHONPATH=$SIBLING_PATHS; $REPO/tests/run_tests.sh"
   ```
3. Train, refactor and control (old code), same job shape; set TIME from version-00's epoch rate:
   ```bash
   CFG=$REPO/tests/bitwise/configs/q3_t0align_100ep_nersc.yaml
   CHAIN=0 TIME=<hh:mm:ss> $SCRIPTS/submit_train.sh --wandb-offline $CFG
   REPO_DIR=$(dirname $REPO)/sirentv-baseline CHAIN=0 TIME=<hh:mm:ss> $SCRIPTS/submit_train.sh --wandb-offline $CFG
   ```
   Outputs: `/pscratch/sd/j/junjiex/SIREN/logs/logs_bitwise_q3_t0align/version-NN` (numbered in
   start order; each job log prints its folder). Check each log shows 389 iterations/epoch.
4. Compare epoch 50 and 100, refactor and control each against the original:
   ```bash
   O=/pscratch/sd/j/junjiex/SIREN/logs/logs_train_sirentv_81_q3_t0align/version-00
   N=/pscratch/sd/j/junjiex/SIREN/logs/logs_bitwise_q3_t0align/version-<NN>
   for e in iteration-019450-epoch-0050 iteration-038900-epoch-0100; do
     shifter --image=$CONTAINER_IMAGE python3 $REPO/tests/bitwise/compare.py $O/$e.ckpt $N/$e.ckpt; done
   ```

## Reading the result

| refactor vs original | control vs original | Meaning |
|---|---|---|
| PASS | (any) | Done: the refactor reproduces the original run. |
| FAIL | PASS | The refactor changed numerics: report the compare output (first differing tensors). |
| FAIL | FAIL | The original is not reproducible here (code/image drift): compare refactor vs control. |

Report the table row, the compare outputs, the version-00 provenance from step 1, and the
container image digest (`shifterimg lookup $CONTAINER_IMAGE`).
