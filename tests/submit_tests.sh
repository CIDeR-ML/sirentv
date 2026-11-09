#!/bin/bash
# tests/run_tests.sh as an S3DF CPU job inside the container; the result is in tests-<jobid>.out.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SITE_ENV=/sdf/data/neutrino/junjie/SIREN/cider-ml/scripts/slurm_job_sirentv/site_env.sh
sbatch --parsable --job-name=sirentv-tests --partition=turing --account=neutrino:cider-nu \
    --ntasks=1 --cpus-per-task=2 --mem=16G --time=0:20:00 --output="$PWD/tests-%j.out" "$@" \
    --wrap="source '$SITE_ENV' && container_exec bash -c 'export PYTHONPATH=\$SIBLING_PATHS && bash $HERE/run_tests.sh'"
