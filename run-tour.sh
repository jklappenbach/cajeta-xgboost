#!/usr/bin/env bash
# Build the library .cja, compile the tour against it, run it.
# The tour is self-checking: non-zero exit means a demonstrated claim failed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"


# --- artifact discovery -------------------------------------------------
# Where a checkout's .cja is. Prefers `cajeta artifact-path`, which reads
# that project's OWN manifest -- so a project that moves its artifacts with
# settings.output is followed rather than guessed, and the version comes
# from details.version instead of whichever file happens to be newest.
#
# Falls back to the historical build/archive glob only when the toolchain
# does not HAVE the verb (it lands after 0.24.0), so this keeps working on
# an older cajeta and starts using the verb as soon as a newer one is on
# PATH -- no flag day.
#
# The gate is the CAPABILITY, not the outcome. A fallback keyed on "the
# verb failed" would silently mask a verb that ran and answered wrongly,
# which is the very failure this replaces; keyed on "the verb is absent",
# it cannot. An empty result still means "not in this checkout", exactly
# as the glob did, so callers' registry fallbacks are unchanged.
cajeta_artifact_path() {
    local dir="$1" name="$2"
    local cj="${CAJETA:-${CAJETA_BIN:-cajeta}}"
    if [[ -z "${_cajeta_has_ap:-}" ]]; then
        if "$cj" artifact-path --help 2>/dev/null \
                | grep -q 'artifact-path \[options\]'; then
            _cajeta_has_ap=yes
        else
            _cajeta_has_ap=no
        fi
    fi
    if [[ "$_cajeta_has_ap" == yes ]]; then
        # Only report a path that EXISTS. The verb answers where the
        # artifact would be even when nothing has built it, but the glob
        # this replaces returned empty in that case, and every caller
        # reads empty as "not in this checkout" and falls back to the
        # registry. Handing back a path to a missing file instead would
        # turn that into a confusing compile failure.
        local p
        p=$( cd "$dir" 2>/dev/null && "$cj" artifact-path 2>/dev/null ) || return 0
        [[ -n "$p" && -f "$p" ]] && printf '%s\n' "$p"
        return 0
    else
        ls -t "$dir"/build/archive/"$name"-*.cja 2>/dev/null | head -1
    fi
}

echo ">> building dev.cajeta.xgboost"
"$CAJETA" build >/dev/null
art="$(cajeta_artifact_path "$here" dev.cajeta.xgboost)"

# dev.cajeta.ml for the protocol section (run-tests.sh's ladder, compact):
# ML_CJA verbatim → sibling checkout → olla store → sha256-verified fetch.
OLLA_HOME="${OLLA_HOME:-$HOME/.olla}"
OLLA_URL="${OLLA_URL:-https://olla.cajeta.dev}"
ML_REPO="${ML_REPO:-$here/../cajeta-ml}"
ml_cja="${ML_CJA:-}"
# A sibling checkout is a local-dev convenience, not a requirement: it may be
# mid-work or need a newer toolchain than this one. Try it, but never let its
# build failure sink the tour — fall through to the pinned published archive,
# which is what a real consumer resolves anyway.
if [[ -z "$ml_cja" && -d "$ML_REPO" ]]; then
    if ( cd "$ML_REPO" && "$CAJETA" build >/dev/null 2>&1 ); then
        ml_cja="$(cajeta_artifact_path "$ML_REPO" dev.cajeta.ml 2>/dev/null)"
    else
        echo ">> sibling cajeta-ml did not build with this toolchain; using the pinned release"
    fi
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
# Device codegen happens at EXE emission (the library's @Kernels lower here),
# so the tour binary bundles the same backends as the test suite.
XPU_BACKENDS="${XPU_BACKENDS:-nvptx,amdgpu,vulkan,cpu}"
"$CAJETA" --emit=exe --xpu-backend="$XPU_BACKENDS" --classpath="$art,$ml_cja" \
    -o build/tour/xgboost-tour \
    dev.cajeta.xgboost.tour.Tour.main "$here/tour/src" build/tour >/dev/null

# The GPU demos want the probed SFU captures (GpuSplitFinder requires the
# RCP table); resolve or fetch them exactly as the test suite does.
source "$here/scripts/fetch-captures.sh" "$here"

echo ">> running"
exec ./build/tour/xgboost-tour
