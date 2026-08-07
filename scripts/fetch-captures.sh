#!/usr/bin/env bash
# Resolve the probed SFU capture tables (RCP for __fdividef gain scoring,
# EX2 for device expf) and export RCP_TABLE / EX2_TABLE. Local gitignored
# captures win; otherwise fetch the sha256-pinned v0.3.0 release assets
# into build/.capture-cache. Sourced by run-tests.sh and run-tour.sh —
# callers keep working (FastMath falls back / GPU split finding defers to
# the CPU finder) if a fetch fails offline.
#   usage: source scripts/fetch-captures.sh <repo-root>
_cap_root="${1:?repo root}"
_cap_release="v0.3.0"
_cap_sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1;
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
_cap_fetch() {
    # $1 local path, $2 asset name, $3 pinned sha256
    if [[ -f "$1" ]]; then printf '%s' "$1"; return 0; fi
    local cache="$_cap_root/build/.capture-cache/$2"
    if [[ ! -f "$cache" ]]; then
        mkdir -p "$(dirname "$cache")"
        echo ">> fetching $2 ($_cap_release release asset)" >&2
        curl -fsSL -o "$cache.tmp" \
            "https://github.com/jklappenbach/cajeta-xgboost/releases/download/$_cap_release/$2" \
            || { rm -f "$cache.tmp"; echo ">> fetch failed — running without $2" >&2; return 1; }
        local got; got="$(_cap_sha "$cache.tmp")"
        [[ "$got" == "$3" ]] || { rm -f "$cache.tmp"; echo ">> sha256 mismatch for $2 — refusing it" >&2; return 1; }
        mv "$cache.tmp" "$cache"
    fi
    printf '%s' "$cache"
    return 0
}
_rcp="$(_cap_fetch "$_cap_root/tools/fdividef/rcp_mantissa.npy" rcp_mantissa.npy \
    234796e0057823853e8f8843556bfc23a362e76cb0871b9ae082a612568f61ae)" \
    && export RCP_TABLE="$_rcp"
_ex2="$(_cap_fetch "$_cap_root/tools/expf/ex2_table.npy" ex2_table.npy \
    fe6720ac0a3685e6925f4d739cdbfe462184df30e1b2e8e4ed5b026dae680c64)" \
    && export EX2_TABLE="$_ex2"
