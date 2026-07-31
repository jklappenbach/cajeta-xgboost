#!/usr/bin/env bash
# The tour half of CI: the self-checking tour, then the tour coverage gate.
# release.yml runs ./run-tests.sh separately (its own step, its own timeout);
# this script keeps the tour checks in one place.
set -euo pipefail
cd "$(dirname "$0")/.."

./run-tour.sh

CAJETA="$(command -v cajeta)" ./scripts/check-library-tour-coverage.sh \
    src/main/cajeta tour
