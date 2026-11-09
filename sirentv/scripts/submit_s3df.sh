#!/bin/bash
# Set your S3DF account (facility:repo); the reference runs were submitted under neutrino:cider-nu.
#SBATCH --account=<account>
#SBATCH --partition=ampere
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=sirentv
#SBATCH --output=%x-%j.out
#
# Minimal S3DF submission: one task per GPU, DDP via srun inside singularity.
# Submit from the sirentv checkout; everything after the config goes to the module.
#
#   sbatch sirentv/scripts/submit_s3df.sh config/train_q3_t0align_s3df.yaml --wandb
#   sbatch sirentv/scripts/submit_s3df.sh config/train_q3_t0align_s3df.yaml --wandb --resume
#   MODULE=sirentv.eval.eval sbatch sirentv/scripts/submit_s3df.sh \
#       config/eval_q3_t0align_s3df.yaml --output eval_q3.pt --spectra --pmt-ids 0 40 80
#
# 2 nodes x 4 GPUs = 8 ranks, the production shape: batch_size 512 per rank gives the
# 4096 global batch. Do not drop to 1 node with memory_shard on: 4 ranks would each hold
# 1/4 of the LUT (~62 GiB), ~250 GB per node, past --mem=256G once training runs.
# A config without memory_shard runs on 1 node (`sbatch -N 1 ...`, batch_size 1024 to keep
# the global batch). Flags on the sbatch line override the #SBATCH ones.

set -euo pipefail

export MODULE="${MODULE:-sirentv.train}"
export REPO="${REPO:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$(readlink -f /sdf/home/j/junjie/sdf-sig/develop.sif)}"
# slar (siren-lartpc) and photonlib are sibling checkouts
export PYPATH="${PYPATH:-$REPO:$REPO/../siren-lartpc:$REPO/../photonlib}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

# srun does not inherit --cpus-per-task from sbatch, so pass it; RANK/LOCAL_RANK/WORLD_SIZE
# are what train.py and eval.py read. PYTHONPATH is replaced inside the container so the
# host's site-packages cannot shadow the image's torch.
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    singularity exec --nv --bind /sdf,/fs,/sdf/scratch,/lscratch "$IMAGE" bash -c '
        export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
        export PYTHONPATH="$PYPATH" OMP_NUM_THREADS=8
        cd "$REPO" && exec python3 -m "$MODULE" --config "$@"
    ' bash "$@"
