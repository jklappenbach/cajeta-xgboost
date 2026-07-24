#!/usr/bin/env bash
# Build + run the cajeta-xgboost unit tests.
#
# The suite lives under src/test/cajeta and is driven by cajeta-unit's reflective
# @Test discovery (dev.cajeta.unit.Runner). It compiles ONLY the test sources into
# an executable, with the xgboost library and cajeta-unit supplied as .cja
# classpath dependencies — the compiler links their bitcode into the test binary.
#
# Override paths via env:
#   CAJETA    — compiler binary (default: cajeta on PATH)
#   UNIT_REPO — path to the cajeta-unit checkout (default: ../cajeta-unit)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"
UNIT_REPO="${UNIT_REPO:-$here/../cajeta-unit}"

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

unit_cja="$UNIT_REPO/build/archive/dev.cajeta.unit-0.1.0.cja"
if [[ ! -f "$unit_cja" ]]; then
    echo ">> building cajeta-unit .cja ($UNIT_REPO)"
    ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null )
fi

echo ">> building xgboost library .cja"
"$CAJETA" --emit=cja -o "$out/xgboost.cja" \
    dev.cajeta.xgboost.XGBoost.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> building + running the test binary"
"$CAJETA" --emit=exe --profile=test \
    --classpath="$out/xgboost.cja,$unit_cja" \
    -o "$out/xgboosttests" \
    dev.cajeta.xgboost.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

"$out/xgboosttests"
