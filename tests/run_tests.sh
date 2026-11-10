#!/bin/bash
# Import/compile + unit tests, in whatever Python environment is active:  tests/run_tests.sh
# (on S3DF, inside the container: tests/submit_tests.sh). Run before every push.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
python3 -m pytest -q -p no:cacheprovider tests "$@"
