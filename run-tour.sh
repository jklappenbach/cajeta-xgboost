#!/usr/bin/env bash
# Build the library .cja, compile the tour against it, run it.
# The tour is self-checking: non-zero exit means a demonstrated claim failed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"

echo ">> building dev.cajeta.xgboost"
"$CAJETA" build >/dev/null
art="$(ls -t "$here"/build/archive/dev.cajeta.xgboost-*.cja | head -1)"

# dev.cajeta.ml for the protocol section (run-tests.sh's ladder, compact):
# ML_CJA verbatim → sibling checkout → olla store → sha256-verified fetch.
OLLA_HOME="${OLLA_HOME:-$HOME/.olla}"
OLLA_URL="${OLLA_URL:-https://olla.cajeta.dev}"
ML_REPO="${ML_REPO:-$here/../cajeta-ml}"
ml_cja="${ML_CJA:-}"
if [[ -z "$ml_cja" && -d "$ML_REPO" ]]; then
    ( cd "$ML_REPO" && "$CAJETA" build >/dev/null )
    ml_cja="$(ls -t "$ML_REPO"/build/archive/dev.cajeta.ml-*.cja 2>/dev/null | head -1)"
fi
if [[ -z "$ml_cja" ]]; then
    ML_VER="$(sed -n 's/.*"dev\.cajeta\.ml"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$here/cajeta.json" | head -1)"
    store_ml="$OLLA_HOME/dev.cajeta.ml/$ML_VER/dev.cajeta.ml-$ML_VER.cja"
    cache_ml="$here/build/.ml-cache/dev.cajeta.ml-$ML_VER.cja"
    if [[ -f "$store_ml" ]]; then ml_cja="$store_ml"
    elif [[ -f "$cache_ml" ]]; then ml_cja="$cache_ml"
    else
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.ml&version=$ML_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256 for dev.cajeta.ml" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_ml")"
        curl -fsS -o "$cache_ml" "$OLLA_URL/v2/blob/$sha"
        if command -v sha256sum >/dev/null 2>&1; then got="$(sha256sum "$cache_ml" | cut -d' ' -f1)";
        else got="$(shasum -a 256 "$cache_ml" | cut -d' ' -f1)"; fi
        [[ "$got" == "$sha" ]] || { rm -f "$cache_ml"; echo "sha256 mismatch fetching ml" >&2; exit 1; }
        ml_cja="$cache_ml"
    fi
fi
[[ -f "$ml_cja" ]] || { echo "could not resolve a dev.cajeta.ml archive" >&2; exit 1; }

echo ">> compiling the tour"
mkdir -p build/tour
"$CAJETA" --emit=exe --classpath="$art,$ml_cja" \
    -o build/tour/xgboost-tour \
    dev.cajeta.xgboost.tour.Tour.main "$here/tour/src" build/tour >/dev/null

echo ">> running"
exec ./build/tour/xgboost-tour
