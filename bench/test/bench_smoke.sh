#!/usr/bin/env bash
# bench/test/bench_smoke.sh — the dev.cajeta.xgboost benchmark smoke
# (cajeta agents/benchmark-reference-plan.md 17.1.2 and 17.1.3): the old
# ~/code/ml/xgbbench cases as schema-v2 rows, checked against upstream XGBoost.
#
# Runs ./bench.sh --competitors in smoke mode (CAJETA_BENCH_SMOKE=1: warmup 1,
# trials 3, 1e4 x 20 instead of 1e6 x 50, 10 rounds instead of 100) and asserts,
# against the compiler repo's bench/ml/XGBoost.md:
#   * `--list` names exactly ml/xgb-hist-fit, ml/xgb-gpu-fit, ml/xgb-predict;
#   * each id has a Cajeta row and an upstream-XGBoost row;
#   * xgb-hist-fit is ok on both sides with the training logloss within the
#     recorded tolerance of upstream `hist`'s on the same data;
#   * xgb-predict is ok on both sides with the margins within epsilon of
#     upstream's for the same (upstream-trained) model;
#   * xgb-gpu-fit is skipped on both sides while the GPU is held
#     (CAJETA_BENCH_NO_GPU=1, the default here; export 0 to measure it);
#   * the exact schema-v2 header, every row tagged smoke, verify-fidelity.py.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench="$(cd "$here/.." && pwd)"

export CAJETA_BENCH_SMOKE=1
export CAJETA_BENCH_NO_GPU="${CAJETA_BENCH_NO_GPU:-1}"

t0=$(date +%s)
echo ">> bench_smoke (dev.cajeta.xgboost): start $(date +%H:%M:%S), NO_GPU=$CAJETA_BENCH_NO_GPU"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if [[ ! -x "$bench/bench.sh" ]]; then
    echo "FAIL: $bench/bench.sh is missing"
    exit 1
fi
"$bench/bench.sh" --competitors 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
run_dir="$(sed -n 's/^bench: results in //p' "$log" | tail -1)"
if [[ -z "$run_dir" || ! -f "$run_dir/results.csv" ]]; then
    echo "FAIL: bench.sh (exit $rc) wrote no results.csv"
    exit 1
fi

bin="$(sed -n 's/^bench: binary //p' "$log" | tail -1)"
list_out="$("$bin" --list 2>&1)" || { echo "FAIL: $bin --list exited non-zero"; exit 1; }

PY="${BENCH_PYTHON:-$bench/../../../cpp/cajeta/bench/competitors/python/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" "$here/check_results.py" "$run_dir" "$list_out"
crc=$?

VERIFY="${VERIFY_FIDELITY:-$bench/../../../cpp/cajeta-bench/tools/verify-fidelity.py}"
if [[ ! -f "$VERIFY" ]]; then
    echo "FAIL: verify-fidelity.py not found at $VERIFY (set VERIFY_FIDELITY)"
    exit 1
fi
"$PY" "$VERIFY" "$run_dir/results.csv"
vrc=$?

t1=$(date +%s)
echo ">> bench_smoke (dev.cajeta.xgboost): bench.sh exit $rc, checks exit $crc, verify-fidelity exit $vrc, $((t1 - t0)) s"
if [[ $rc -ne 0 || $crc -ne 0 || $vrc -ne 0 ]]; then
    echo "FAIL"
    exit 1
fi
echo "PASS"
