#!/usr/bin/env bash
# Build the library .cja, compile the tour against it, run it.
# The tour is self-checking: non-zero exit means a demonstrated claim failed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"

echo ">> building dev.cajeta.xgboost"
"$CAJETA" build >/dev/null
art="$(ls -t "$here"/build/archive/dev.cajeta.xgboost-*.cja | head -1)"

echo ">> compiling the tour"
mkdir -p build/tour
"$CAJETA" --emit=exe --classpath="$art" \
    -o build/tour/xgboost-tour \
    dev.cajeta.xgboost.tour.Tour.main "$here/tour/src" build/tour >/dev/null

echo ">> running"
exec ./build/tour/xgboost-tour
