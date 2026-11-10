#!/bin/bash
#SBATCH --account=<account>
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=sirentv
#SBATCH --output=%x-%j.out
#
# Minimal Perlmutter submission: one task per GPU, DDP via srun inside shifter.
# Needs photonlib, siren-lartpc and sirentv pip-installed into the image per the README.
# Submit from the sirentv checkout; everything after the config goes to the module.
#
#   sbatch sirentv/scripts/submit_nersc.sh config/train_q3_t0align_nersc.yaml --wandb
#   sbatch sirentv/scripts/submit_nersc.sh config/train_q3_t0align_nersc.yaml --wandb --resume
#   MODULE=sirentv.eval.eval sbatch sirentv/scripts/submit_nersc.sh \
#       config/eval_q3_t0align_nersc.yaml --output eval_q3.pt --spectra --pmt-ids 0 40 80
#
# 2 nodes x 4 GPUs = 8 ranks is the production shape: batch_size 512 per rank gives the
# 4096 global batch (389 iterations/epoch), and the memory_shard (1/8 of the LUT,
# ~31 GiB per rank in host RAM) fits 4 ranks per 256 GB node. Other node counts change the
# shards and the DDP average. Flags on the sbatch line override the #SBATCH ones, e.g.
# `sbatch -q premium -t 02:00:00 ...`.

set -euo pipefail

export MODULE="${MODULE:-sirentv.train}"
export REPO="${REPO:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

# srun does not inherit --cpus-per-task from sbatch, so pass it; RANK/LOCAL_RANK/WORLD_SIZE
# are what train.py and eval.py read.
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    shifter --module gpu --image="$IMAGE" bash -c '
        export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
        export OMP_NUM_THREADS=8
        cd "$REPO" && exec python3 -m "$MODULE" --config "$@"
    ' bash "$@"
