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

# cajeta-unit resolution (the cajeta-logging pattern), in order:
#   1. $UNIT_CJA        — explicit archive path, used verbatim
#   2. $UNIT_REPO       — sibling checkout when it exists: build it and use
#                         whatever version it emits (local dev, unit HEAD)
#   3. $OLLA_HOME store — an installed dev.cajeta.unit at the version pinned
#                         in cajeta.json's dev-dependencies
#   4. Olla registry    — /v2/resolve + /v2/blob (the toolchain's own fetch
#                         protocol), sha256-verified, cached under build/.
#                         The CI flow: bare runners have no checkout.
OLLA_HOME="${OLLA_HOME:-$HOME/.olla}"
OLLA_URL="${OLLA_URL:-https://olla.cajeta.dev}"
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1;
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
unit_cja="${UNIT_CJA:-}"
if [[ -z "$unit_cja" && -d "$UNIT_REPO" ]]; then
    echo ">> building cajeta-unit from checkout ($UNIT_REPO)"
    ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null )
    unit_cja="$(ls -t "$UNIT_REPO"/build/archive/dev.cajeta.unit-*.cja 2>/dev/null | head -1)"
fi
if [[ -z "$unit_cja" ]]; then
    UNIT_VER="$(sed -n 's/.*"dev\.cajeta\.unit"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$here/cajeta.json" | head -1)"
    [[ -n "$UNIT_VER" ]] || { echo "no dev.cajeta.unit pin in cajeta.json" >&2; exit 1; }
    store_cja="$OLLA_HOME/dev.cajeta.unit/$UNIT_VER/dev.cajeta.unit-$UNIT_VER.cja"
    cache_cja="$here/build/.unit-cache/dev.cajeta.unit-$UNIT_VER.cja"
    if [[ -f "$store_cja" ]]; then unit_cja="$store_cja"
    elif [[ -f "$cache_cja" ]]; then unit_cja="$cache_cja"
    else
        echo ">> fetching dev.cajeta.unit $UNIT_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.unit&version=$UNIT_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_cja")"
        curl -fsS -o "$cache_cja" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$cache_cja")"
        [[ "$got" == "$sha" ]] || { rm -f "$cache_cja"; echo "sha256 mismatch fetching unit" >&2; exit 1; }
        unit_cja="$cache_cja"
    fi
fi
[[ -f "$unit_cja" ]] || { echo "could not resolve a dev.cajeta.unit archive" >&2; exit 1; }
echo ">> cajeta-unit: $unit_cja"

echo ">> building xgboost library .cja"
"$CAJETA" --emit=cja -o "$out/xgboost.cja" \
    dev.cajeta.xgboost.XGBoost.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> building + running the test binary"
"$CAJETA" --emit=exe --profile=test \
    --classpath="$out/xgboost.cja,$unit_cja" \
    -o "$out/xgboosttests" \
    dev.cajeta.xgboost.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

# Parity tests load golden fixtures from tools/fixtures via this env var.
export XGBOOST_FIXTURES="$here/tools/fixtures"
# Probed SFU reciprocal table for bit-exact __fdividef gain scoring (FastMath).
# Absent (e.g. CI without the capture) → FastMath uses a correctly-rounded fallback.
rcp="$here/tools/fdividef/rcp_mantissa.npy"
[ -f "$rcp" ] && export RCP_TABLE="$rcp"
# Probed SFU exp2 table for bit-exact device expf (Logistic/Softmax gradients).
# Same deal: absent → FastMath falls back to correctly-rounded exp2.
ex2="$here/tools/expf/ex2_table.npy"
[ -f "$ex2" ] && export EX2_TABLE="$ex2"
# Device expf sweep over [-90, 90] — the ground truth FastMathExpfTest compares
# against bit-for-bit. Absent → that comparison self-skips; the plumbing
# assertions, which need no capture, still run.
sweep="$here/tools/expf/expf_sweep.npy"
[ -f "$sweep" ] && export EXPF_SWEEP="$sweep"
"$out/xgboosttests"
