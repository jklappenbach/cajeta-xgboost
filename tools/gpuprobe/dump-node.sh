#!/usr/bin/env bash
# Build + run the NodeDumpMain driver: writes one node's split-evaluation
# inputs (gpu-numeric-fidelity 2.2.1) to tools/gpuprobe/<name>/ as npys,
# consumed by probe_evaluate_gain.cu on the GPU runner. The dump is tiny and
# COMMITTED — the runner has no cajeta toolchain, so it can't produce it.
#
#   tools/gpuprobe/dump-node.sh [nid] [outname]     # default: 14, node14
#
# Env (as run-tests.sh): CAJETA, UNIT_REPO / UNIT_CJA.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
CAJETA="${CAJETA:-cajeta}"
UNIT_REPO="${UNIT_REPO:-$root/../cajeta-unit}"
nid="${1:-14}"
outname="${2:-node14}"

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

unit_cja="${UNIT_CJA:-}"
if [[ -z "$unit_cja" && -d "$UNIT_REPO" ]]; then
    echo ">> building cajeta-unit from checkout ($UNIT_REPO)"
    ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null )
    unit_cja="$(ls -t "$UNIT_REPO"/build/archive/dev.cajeta.unit-*.cja 2>/dev/null | head -1)"
fi
[[ -f "$unit_cja" ]] || { echo "could not resolve a dev.cajeta.unit archive (set UNIT_CJA)" >&2; exit 1; }

echo ">> building xgboost library .cja"
"$CAJETA" --emit=cja -o "$out/xgboost.cja" \
    dev.cajeta.xgboost.XGBoost.run "$root/src/main/cajeta" "$out" >/dev/null

echo ">> building the dump driver"
"$CAJETA" --emit=exe --profile=test \
    --classpath="$out/xgboost.cja,$unit_cja" \
    -o "$out/nodedump" \
    dev.cajeta.xgboost.selftest.NodeDumpMain.run "$root/src/test/cajeta" "$out" >/dev/null

mkdir -p "$here/$outname"
XGBOOST_FIXTURES="$root/tools/fixtures" \
XGBOOST_NODE_DUMP="$here/$outname" \
XGBOOST_NODE_DUMP_NID="$nid" \
"$out/nodedump"
echo ">> dumped node $nid to $here/$outname"
ls -la "$here/$outname"
