#!/bin/bash
# Compare two finished suite runs on a CPU node:  compare.sh OUT_A OUT_B [extra sbatch args]
# The report lands in OUT_B/compare-<jobid>.out.
set -euo pipefail
A=$(readlink -f "$1")
B=$(readlink -f "$2")
shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SITE_ENV=/sdf/data/neutrino/junjie/SIREN/cider-ml/scripts/slurm_job_sirentv/site_env.sh

sbatch --parsable --job-name=bitwise-cmp --partition=turing --account=neutrino:cider-nu \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=1:00:00 \
    --output="$B/compare-%j.out" "$@" \
    --wrap="source '$SITE_ENV' && container_exec python3 '$HERE/compare.py' '$A' '$B'"
