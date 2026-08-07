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

# dev.cajeta.ml resolution — same ladder as cajeta-unit. The library proper
# depends on it (the XGBRegressor Predictor conformer, settings.dependencies),
# so it is threaded through BOTH the library and the test classpaths:
#   1. $ML_CJA      — explicit archive path, used verbatim
#   2. $ML_REPO     — sibling checkout (default ../cajeta-ml): build and use it
#   3. $OLLA_HOME   — installed dev.cajeta.ml at the cajeta.json pin
#   4. Olla registry — sha256-verified fetch, cached under build/.ml-cache
ML_REPO="${ML_REPO:-$here/../cajeta-ml}"
ml_cja="${ML_CJA:-}"
if [[ -z "$ml_cja" && -d "$ML_REPO" ]]; then
    echo ">> building cajeta-ml from checkout ($ML_REPO)"
    ( cd "$ML_REPO" && "$CAJETA" build >/dev/null )
    ml_cja="$(ls -t "$ML_REPO"/build/archive/dev.cajeta.ml-*.cja 2>/dev/null | head -1)"
fi
if [[ -z "$ml_cja" ]]; then
    ML_VER="$(sed -n 's/.*"dev\.cajeta\.ml"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$here/cajeta.json" | head -1)"
    [[ -n "$ML_VER" ]] || { echo "no dev.cajeta.ml pin in cajeta.json" >&2; exit 1; }
    store_ml="$OLLA_HOME/dev.cajeta.ml/$ML_VER/dev.cajeta.ml-$ML_VER.cja"
    cache_ml="$here/build/.ml-cache/dev.cajeta.ml-$ML_VER.cja"
    if [[ -f "$store_ml" ]]; then ml_cja="$store_ml"
    elif [[ -f "$cache_ml" ]]; then ml_cja="$cache_ml"
    else
        echo ">> fetching dev.cajeta.ml $ML_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.ml&version=$ML_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_ml")"
        curl -fsS -o "$cache_ml" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$cache_ml")"
        [[ "$got" == "$sha" ]] || { rm -f "$cache_ml"; echo "sha256 mismatch fetching ml" >&2; exit 1; }
        ml_cja="$cache_ml"
    fi
fi
[[ -f "$ml_cja" ]] || { echo "could not resolve a dev.cajeta.ml archive" >&2; exit 1; }
echo ">> cajeta-ml: $ml_cja"

echo ">> building xgboost library .cja"
XPU_BACKENDS="${XPU_BACKENDS:-nvptx,amdgpu,vulkan,cpu}"
"$CAJETA" --emit=cja --xpu-backend="$XPU_BACKENDS" -o "$out/xgboost.cja" \
    --classpath="$ml_cja" \
    dev.cajeta.xgboost.XGBoost.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> building + running the test binary"
# @Kernel device codegen: bundle every backend the toolchain can emit —
# the runtime picks CUDA -> HIP -> Vulkan -> CPU at first device touch
# (CAJETA_XPU_BACKEND overrides), so one binary runs the GPU histogram on
# phoenix-wsl (nvptx), the Strix Halo dev box (amdgpu/vulkan), and hosted
# CI (cpu emulation) alike.
"$CAJETA" --emit=exe --profile=test --xpu-backend="$XPU_BACKENDS" \
    --classpath="$out/xgboost.cja,$unit_cja,$ml_cja" \
    -o "$out/xgboosttests" \
    dev.cajeta.xgboost.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

# Parity tests load golden fixtures from tools/fixtures via this env var.
export XGBOOST_FIXTURES="$here/tools/fixtures"
# Probed SFU capture tables (RCP for __fdividef gain scoring, EX2 for device
# expf in Logistic/Softmax). Gitignored (32 MB each); when the local capture
# is absent, fetch the sha256-pinned copy published as a v0.3.0 release
# asset — so CI asserts the capture-dependent parity tests instead of
# self-skipping. If the fetch fails (offline), FastMath's correctly-rounded
# fallback keeps the rest of the suite meaningful.
CAPTURE_RELEASE="v0.3.0"
fetch_capture() {
    # $1 = local path, $2 = asset name, $3 = pinned sha256
    if [[ -f "$1" ]]; then printf '%s' "$1"; return 0; fi
    local cache="$here/build/.capture-cache/$2"
    if [[ ! -f "$cache" ]]; then
        mkdir -p "$(dirname "$cache")"
        echo ">> fetching $2 ($CAPTURE_RELEASE release asset)" >&2
        curl -fsSL -o "$cache.tmp"             "https://github.com/jklappenbach/cajeta-xgboost/releases/download/$CAPTURE_RELEASE/$2"             || { rm -f "$cache.tmp"; echo ">> fetch failed — running without $2" >&2; return 1; }
        local got
        got="$(sha256_of "$cache.tmp")"
        [[ "$got" == "$3" ]] || { rm -f "$cache.tmp"; echo ">> sha256 mismatch for $2 — refusing it" >&2; return 1; }
        mv "$cache.tmp" "$cache"
    fi
    printf '%s' "$cache"
    return 0
}
rcp="$(fetch_capture "$here/tools/fdividef/rcp_mantissa.npy" rcp_mantissa.npy     234796e0057823853e8f8843556bfc23a362e76cb0871b9ae082a612568f61ae)"     && export RCP_TABLE="$rcp"
ex2="$(fetch_capture "$here/tools/expf/ex2_table.npy" ex2_table.npy     fe6720ac0a3685e6925f4d739cdbfe462184df30e1b2e8e4ed5b026dae680c64)"     && export EX2_TABLE="$ex2"
# Device expf sweep over [-90, 90] — the ground truth FastMathExpfTest compares
# against bit-for-bit. Absent → that comparison self-skips; the plumbing
# assertions, which need no capture, still run.
sweep="$here/tools/expf/expf_sweep.npy"
[ -f "$sweep" ] && export EXPF_SWEEP="$sweep"
"$out/xgboosttests"
