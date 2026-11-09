#!/bin/bash
# Bitwise regression suite, run INSIDE the container on a 4-GPU node (submit.sh does that).
#   run_suite.sh REPO OUT
# REPO: the sirentv checkout under test. OUT: fresh dir for every artifact.
# Smoke:  2-GPU DDP training of q3_t0align and p3_omega030 (grad supervision on), side by side.
# Eval:   4-GPU eval of each production checkpoint (q3 with --x-min, p3 with all comparisons).
# Unit:   the package's own pytest tests (pass/fail only, not compared).
set -uo pipefail
REPO=$(readlink -f "$1")
OUT=$(readlink -f "$2")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CACHE=/sdf/data/neutrino/junjie/SIREN/cider-ml/bitwise/grad_cache/dualpca_n50_grad_frob.h5
Q3_CKPT=/sdf/data/neutrino/junjie/SIREN/cider-ml/logs/logs_train_sirentv_81_q3_t0align/version-15/iteration-2334000-epoch-6000.ckpt
P3_CKPT=/sdf/data/neutrino/junjie/SIREN/cider-ml/logs/logs_stage1_gs_p3_omega030/version-02/iteration-1050300-epoch-2700.ckpt

export PYTHONPATH="${REPO}:${SIBLING_PATHS}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MPLBACKEND=Agg
mkdir -p "$OUT/cfg"
cd "$REPO" || exit 1

python3 -c "import sirentv, sys; f = sirentv.__file__; print('sirentv from', f); sys.exit(not f.startswith('${REPO}/'))" \
    || { echo "sirentv is not imported from ${REPO}" >&2; exit 1; }

# q3 trains under the wandb logger (offline: exercises plots, grad-norm and analysis
# logging), p3 under the CSV logger (its per-step losses land in log.csv and get compared).
export WANDB_MODE=offline WANDB_DIR="$OUT"
common=(train.resume=false model.ckpt_file=null "logger.run_dir=$OUT"
        data.subsample_factor=40 data.loader.num_workers=4 train.save_every_epochs=1
        train.save_every_iterations=-1)
python3 "$HERE/make_cfg.py" "$HERE/configs/q3_t0align.yaml" "$OUT/cfg/train_q3.yaml" "${common[@]}" \
    logger.dir_name=train_q3 train.max_epochs=3 logger.plot_every_nsteps=10
python3 "$HERE/make_cfg.py" "$HERE/configs/p3_omega030.yaml" "$OUT/cfg/train_p3.yaml" "${common[@]}" \
    logger.dir_name=train_p3 logger.log_every_nsteps=5 train.max_epochs=2 data.loader.batch_size=512 \
    "train.grad_supervision_keys={coeffs: 0}" "train.grad_target_cache_file=$CACHE"
python3 "$HERE/make_cfg.py" "$HERE/configs/q3_t0align.yaml" "$OUT/cfg/eval_q3.yaml" model.ckpt_file=null
python3 "$HERE/make_cfg.py" "$HERE/configs/p3_omega030.yaml" "$OUT/cfg/eval_p3.yaml" model.ckpt_file=null \
    "train.grad_target_cache_file=$CACHE"

status=0
run() {  # run NAME GPUS PORT SCRIPT ARGS...
    local name=$1 gpus=$2 port=$3; shift 3
    local n=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
    echo "[suite] $name on GPUs $gpus: started $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nnodes=1 --nproc_per_node=$n \
        --master_port=$port "$HERE/det_run.py" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "[suite] $name: exit $rc at $(date +%T)"
    return $rc
}

run train_q3 0,1 29611 sirentv/train.py --config "$OUT/cfg/train_q3.yaml" --wandb & pid_q3=$!
run train_p3 2,3 29612 sirentv/train.py --config "$OUT/cfg/train_p3.yaml" & pid_p3=$!
wait $pid_q3 || status=1
wait $pid_p3 || status=1

run eval_q3 0,1,2,3 29613 sirentv/eval/eval.py --config "$OUT/cfg/eval_q3.yaml" --ckpt "$Q3_CKPT" \
    --output "$OUT/eval_q3.pt" --pmt-ids 0 40 80 --x-min -2160 || status=1
run eval_p3 0,1,2,3 29614 sirentv/eval/eval.py --config "$OUT/cfg/eval_p3.yaml" --ckpt "$P3_CKPT" \
    --output "$OUT/eval_p3.pt" --pmt-ids 40 || status=1

# the example config must hold exactly the production values the suite trains with
# (ignoring run-specific keys and the removed debug_timing / train.regularization)
EXAMPLE="$REPO/config/train_q3_t0align_s3df.yaml"
if [[ -f "$EXAMPLE" ]]; then
    python3 - "$EXAMPLE" "$HERE/configs/q3_t0align.yaml" <<'PY' || status=1
import sys, yaml
a, b = (yaml.safe_load(open(p)) for p in sys.argv[1:3])
for c in (a, b):
    c["logger"].pop("run_dir", None); c["model"].pop("ckpt_file", None)
    c.pop("debug_timing", None); c["train"].pop("regularization", None)  # removed options
print("[suite] example config", "matches" if a == b else "DIFFERS from", "the frozen q3 config")
sys.exit(a != b)
PY
fi

UNIT="$REPO/tests"; [[ -d "$UNIT" ]] || UNIT="$REPO/sirentv"  # baseline kept its test under sirentv/
python3 -m pytest -q -p no:cacheprovider "$UNIT" > "$OUT/pytest.log" 2>&1 || status=1
echo "[suite] pytest: $(tail -n1 "$OUT/pytest.log")"

grep -h "deterministic" "$OUT"/*.log | sort | uniq -c > "$OUT/nondeterministic_ops.txt"
echo "[suite] done, status $status"
exit $status
