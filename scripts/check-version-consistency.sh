#!/usr/bin/env bash
# Assert that every copy of the library version agrees with cajeta.json, which
# is the single source of truth. XGBoost.version() reported 0.2.0 for two cuts
# while the manifest said 0.3.0, and the suite was green the whole time because
# ScaffoldTest compares version() against a literal and can never see the drift.
#
# The manifest is JSONC, so jq cannot parse it — read details.version directly.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

want="$(sed -n 's/^[[:space:]]*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$root/cajeta.json" | head -1)"
[[ -n "$want" ]] || { echo "check-version-consistency: no details.version in cajeta.json" >&2; exit 1; }

rc=0
check() {
    # $1 label, $2 file, $3 sed extractor
    local got
    got="$(sed -n "$3" "$root/$2" | head -1)"
    if [[ -z "$got" ]]; then
        echo "check-version-consistency: $1 — no version found in $2" >&2
        rc=1
    elif [[ "$got" != "$want" ]]; then
        echo "check-version-consistency: $1 — $2 says $got, cajeta.json says $want" >&2
        rc=1
    fi
}

check "XGBoost.version()" src/main/cajeta/dev/cajeta/xgboost/XGBoost.cajeta \
    's/.*version()[[:space:]]*{[[:space:]]*return[[:space:]]*"\([^"]*\)".*/\1/p'
check "ScaffoldTest" src/test/cajeta/dev/cajeta/xgboost/selftest/ScaffoldTest.cajeta \
    's/.*Assert\.equals("\([^"]*\)",[[:space:]]*XGBoost\.version()).*/\1/p'
check "README status line" README.md \
    's/.*\*\*The [^(]*(\([0-9][^)]*\)).*/\1/p'
check "docs/Guide.md dependency peg" docs/Guide.md \
    's/.*"dev\.cajeta\.xgboost"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'

[[ $rc -eq 0 ]] && echo "check-version-consistency: 4/4 copies agree at $want"
exit $rc
