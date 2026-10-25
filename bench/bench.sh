#!/usr/bin/env bash
# bench.sh — measure dev.cajeta.xgboost against upstream XGBoost.
#
#   ./bench.sh                              # Cajeta rows only
#   ./bench.sh --competitors                # and the upstream XGBoost rows
#   ./bench.sh --bench xgb-predict --competitors # one id, both sides
#
# Steps, each timed on stdout:
#   1. data   competitors/python/gen_data.py writes every input and reference
#             result once, from its recorded seed, into data/<smoke|full>/
#             (skipped when the manifest is newer than the generator)
#   2. build  `cajeta release` → build/xgb-bench (NO_BUILD=1 reuses it)
#   3. cajeta build/xgb-bench --run → cajeta.csv + env.csv
#   4. python competitors/python/xgb_bench.py → python.csv (with --competitors)
#   5. merge  results.csv = both, one schema-v2 header
# into results/<machine>/<run-id>/, run-id `<UTC basic>-<cajeta version>`.
#
# Environment: CAJETA_BENCH_SMOKE=1 (reduced sizes, warmup 1, trials 3, rows
# tagged smoke), CAJETA_BENCH_NO_GPU=1 (device rows skipped on both sides; the
# GPU is not touched), CAJETA_BENCH_MACHINE, CAJETA_BENCH_RUN_ID, CAJETA
# (compiler, default `cajeta` on PATH else ../../../cpp/cajeta/build/src/cajeta),
# BENCH_PYTHON (default the compiler repo's competitor venv).
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

bench=""
competitors=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench) bench="${2:-}"; shift 2 ;;
        --competitors) competitors=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "bench.sh: unknown argument '$1'"; exit 2 ;;
    esac
done

CAJETA="${CAJETA:-$(command -v cajeta || echo "$here/../../../cpp/cajeta/build/src/cajeta")}"
PY="${BENCH_PYTHON:-$here/../../../cpp/cajeta/bench/competitors/python/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

smoke=0
case "${CAJETA_BENCH_SMOKE:-}" in 1|true|yes) smoke=1 ;; esac
scale=full; [[ $smoke -eq 1 ]] && scale=smoke
data="$here/data/$scale"
export CAJETA_XGB_BENCH_DATA="$data"

# The probed SFU captures, as ../run-tests.sh exports them (gitignored, built
# by tools/fdividef and tools/expf): without RCP_TABLE the split finder's
# __fdividef model falls back and the row says "SFU tables fallback".
[[ -z "${RCP_TABLE:-}" && -f "$here/../tools/fdividef/rcp_mantissa.npy" ]] && export RCP_TABLE="$here/../tools/fdividef/rcp_mantissa.npy"
[[ -z "${EX2_TABLE:-}" && -f "$here/../tools/expf/ex2_table.npy" ]] && export EX2_TABLE="$here/../tools/expf/ex2_table.npy"
# NVPTX device code needs ptxas newer than CUDA 12.0 (cajeta CLAUDE.md).
[[ -z "${CUDA_PATH:-}" && -x /usr/local/cuda/bin/ptxas ]] && export CUDA_PATH=/usr/local/cuda

version="$("$CAJETA" --version 2>/dev/null | awk 'NR==1{print $2}')"
machine="${CAJETA_BENCH_MACHINE:-$(hostname | tr '[:upper:]' '[:lower:]')}"
run_id="${CAJETA_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%MZ)-${version:-unknown}}"
export CAJETA_BENCH_MACHINE="$machine" CAJETA_BENCH_RUN_ID="$run_id"
export CAJETA_BENCH_CAJETA_VERSION="$version" CAJETA_BENCH_RUN_TS="$(date +%s)"
export CAJETA_BENCH_FLAGS="--release --opt=O3 --cpu=native"
run_dir="$here/results/$machine/$run_id"
mkdir -p "$run_dir"

step() { echo ">> $1 $(date +%H:%M:%S)"; }
secs() { echo $(( $(date +%s) - $1 )); }

# 1. data
t=$(date +%s)
if [[ ! -f "$data/manifest.tsv" || "$here/competitors/python/gen_data.py" -nt "$data/manifest.tsv" ]]; then
    step "data: generating $scale inputs into $data"
    ( cd "$here/competitors/python" && "$PY" gen_data.py "$data" ) || { echo "bench.sh: gen_data.py failed"; exit 1; }
else
    step "data: $data is current"
fi
echo "   data: $(secs $t) s"

# 2. build
bin="$here/build/xgb-bench"
t=$(date +%s)
if [[ -n "${NO_BUILD:-}" && -x "$bin" ]]; then
    step "build: reusing $bin (NO_BUILD)"
else
    step "build: cajeta release"
    "$CAJETA" release > "$here/build-release.log" 2>&1 || {
        echo "bench.sh: build failed"; grep -v 'wildcard-crosses-hot-boundary' "$here/build-release.log" | tail -20; exit 1; }
    mkdir -p "$here/build" && mv "$here/build-release.log" "$here/build/release.log"
fi
echo "   build: $(secs $t) s"
echo "bench: binary $bin"

# 3. the Cajeta rows
t=$(date +%s)
step "cajeta: $bin --run${bench:+ --bench $bench}"
"$bin" --run --out "$run_dir/results.csv" ${bench:+--bench "$bench"}
crc=$?
mv "$run_dir/results.csv" "$run_dir/cajeta.csv"
echo "   cajeta: $(secs $t) s (exit $crc)"

# 4. the competitor rows
prc=0
if [[ $competitors -eq 1 ]]; then
    t=$(date +%s)
    step "python: xgb_bench.py${bench:+ --bench $bench}"
    ( cd "$here/competitors/python" && "$PY" xgb_bench.py --data "$data" --out "$run_dir/python.csv" ${bench:+--bench "$bench"} )
    prc=$?
    echo "   python: $(secs $t) s (exit $prc)"
fi

# 5. merge, and the competitor toolchain + quiet-box facts into env.csv
{
    cat "$run_dir/cajeta.csv"
    [[ -f "$run_dir/python.csv" ]] && tail -n +2 "$run_dir/python.csv"
} > "$run_dir/results.csv"
{
    echo "python,$("$PY" -c 'import platform; print(platform.python_version())')"
    "$PY" -c 'import numpy, xgboost; print(f"numpy,{numpy.__version__}\nxgboost,{xgboost.__version__}")'
    echo "dev.cajeta.xgboost,0.7.0"
    echo "loadavg,\"$(cut -d' ' -f1-3 /proc/loadavg)\""
    if [[ "${CAJETA_BENCH_NO_GPU:-}" =~ ^(1|true|yes)$ ]]; then
        echo "gpu,not probed (CAJETA_BENCH_NO_GPU=1: held by another session)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu,\"$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv,noheader | head -1)\""
    fi
    echo "data_seed,$(sed -n 's/.*seed \([0-9]*\).*/\1/p' "$data/manifest.tsv" | head -1)"
    echo "data_manifest_sha256,$(sha256sum "$data/manifest.tsv" | cut -d' ' -f1)"
    echo "scale,$scale"
} >> "$run_dir/env.csv"
cp "$data/manifest.tsv" "$run_dir/data-manifest.tsv"
echo "bench: results in $run_dir"
[[ $crc -eq 0 && $prc -eq 0 ]]
