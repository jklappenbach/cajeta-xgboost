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

# The tour is an executable, and the compiler re-checks an archive's sources when it
# emits one: dev.cajeta.ml's dependency (the kernel package) and the package's own
# (dev.cajeta.unit) must be on the classpath too. Same ladders as run-tests.sh.
kernels_cja="${KERNELS_CJA:-}"
if [[ -z "$kernels_cja" && -d "$ML_REPO/kernels" ]]; then
    if ( cd "$ML_REPO/kernels" && "$CAJETA" build >/dev/null 2>&1 ); then
        kernels_cja="$(cajeta_artifact_path "$ML_REPO/kernels" dev.cajeta.ml.kernels 2>/dev/null)"
    fi
fi
if [[ -z "$kernels_cja" ]]; then
    K_VER="$(sed -n 's/.*"dev\.cajeta\.ml\.kernels"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$here/cajeta.json" | head -1)"
    [[ -n "$K_VER" ]] || { echo "no dev.cajeta.ml.kernels pin in cajeta.json" >&2; exit 1; }
    store_k="$OLLA_HOME/dev.cajeta.ml.kernels/$K_VER/dev.cajeta.ml.kernels-$K_VER.cja"
    cache_k="$here/build/.kernels-cache/dev.cajeta.ml.kernels-$K_VER.cja"
    if [[ -f "$store_k" ]]; then kernels_cja="$store_k"
    elif [[ -f "$cache_k" ]]; then kernels_cja="$cache_k"
    else
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.ml.kernels&version=$K_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_k")"
        curl -fsS -o "$cache_k" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256sum "$cache_k" | cut -d' ' -f1)"
        [[ "$got" == "$sha" ]] || { rm -f "$cache_k"; echo "sha256 mismatch fetching kernels" >&2; exit 1; }
        kernels_cja="$cache_k"
    fi
fi
[[ -f "$kernels_cja" ]] || { echo "could not resolve a dev.cajeta.ml.kernels archive" >&2; exit 1; }
UNIT_REPO="${UNIT_REPO:-$here/../cajeta-unit}"
unit_cja="${UNIT_CJA:-}"
if [[ -z "$unit_cja" && -d "$UNIT_REPO" ]]; then
    if ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null 2>&1 ); then
        unit_cja="$(cajeta_artifact_path "$UNIT_REPO" dev.cajeta.unit 2>/dev/null)"
    fi
fi
if [[ -z "$unit_cja" ]]; then
    UNIT_VER="$(sed -n 's/.*"dev\.cajeta\.unit"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$here/cajeta.json" | head -1)"
    [[ -n "$UNIT_VER" ]] || { echo "no dev.cajeta.unit pin in cajeta.json" >&2; exit 1; }
    store_u="$OLLA_HOME/dev.cajeta.unit/$UNIT_VER/dev.cajeta.unit-$UNIT_VER.cja"
    cache_u="$here/build/.unit-cache/dev.cajeta.unit-$UNIT_VER.cja"
    if [[ -f "$store_u" ]]; then unit_cja="$store_u"
    elif [[ -f "$cache_u" ]]; then unit_cja="$cache_u"
    else
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.unit&version=$UNIT_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_u")"
        curl -fsS -o "$cache_u" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256sum "$cache_u" | cut -d' ' -f1)"
        [[ "$got" == "$sha" ]] || { rm -f "$cache_u"; echo "sha256 mismatch fetching unit" >&2; exit 1; }
        unit_cja="$cache_u"
    fi
fi
[[ -f "$unit_cja" ]] || { echo "could not resolve a dev.cajeta.unit archive" >&2; exit 1; }

# A comma-separated --classpath for the compiler. Git Bash converts a lone
# POSIX path argument for a native Windows exe but not a comma list, so on
# MSYS each entry is converted with cygpath; elsewhere they pass unchanged.
classpath() {
    local out="" p
    for p in "$@"; do
        if command -v cygpath >/dev/null 2>&1; then p="$(cygpath -m "$p")"; fi
        out="${out:+$out,}$p"
    done
    printf '%s' "$out"
}

echo ">> compiling the tour"
mkdir -p build/tour
# Device codegen happens at EXE emission (the library's @Kernels lower here),
# so the tour binary bundles the same backends as the test suite.
XPU_BACKENDS="${XPU_BACKENDS:-nvptx,amdgpu,vulkan,cpu}"
"$CAJETA" --emit=exe --xpu-backend="$XPU_BACKENDS" --classpath="$(classpath "$art" "$ml_cja" "$kernels_cja" "$unit_cja")" \
    -o build/tour/xgboost-tour \
    dev.cajeta.xgboost.tour.Tour.main "$here/tour/src" build/tour >/dev/null

# The GPU demos want the probed SFU captures (GpuSplitFinder requires the
# RCP table); resolve or fetch them exactly as the test suite does.
source "$here/scripts/fetch-captures.sh" "$here"

echo ">> running"
exec ./build/tour/xgboost-tour
