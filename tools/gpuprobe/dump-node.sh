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

UNIT_REPO="${UNIT_REPO:-$root/../cajeta-unit}"
nid="${1:-14}"
outname="${2:-node14}"

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

unit_cja="${UNIT_CJA:-}"
if [[ -z "$unit_cja" && -d "$UNIT_REPO" ]]; then
    echo ">> building cajeta-unit from checkout ($UNIT_REPO)"
    ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null )
    unit_cja="$(cajeta_artifact_path "$UNIT_REPO" dev.cajeta.unit 2>/dev/null)"
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
