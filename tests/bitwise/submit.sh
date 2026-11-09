#!/bin/bash
# Submit the bitwise suite on S3DF:  submit.sh REPO OUT [extra sbatch args]
# then compare two finished runs:   compare.sh OUT_A OUT_B   (see README.md)
set -euo pipefail
REPO=$(readlink -f "$1")
OUT=$2
shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
{ git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short; } > "$OUT/commit.txt"
# site_env.sh provides container_exec and SIBLING_PATHS (slar, photonlib)
SITE_ENV=/sdf/data/neutrino/junjie/SIREN/cider-ml/scripts/slurm_job_sirentv/site_env.sh

sbatch --parsable --job-name=bitwise --partition=ampere --account=neutrino:cider-nu \
    --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=32 --mem=256G --time=4:00:00 \
    --output="$OUT/slurm-%j.out" "$@" \
    --wrap="source '$SITE_ENV' && container_exec bash '$HERE/run_suite.sh' '$REPO' '$OUT'"
