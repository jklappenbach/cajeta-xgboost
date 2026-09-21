#!/usr/bin/env bash
# The source-level half of CI: the version-consistency gate, the self-checking
# tour, then the tour coverage gate. release.yml runs ./run-tests.sh separately
# (its own step, its own timeout); this script keeps the rest in one place.
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/check-version-consistency.sh

./run-tour.sh

CAJETA="$(command -v cajeta)" ./scripts/check-library-tour-coverage.sh \
    src/main/cajeta tour
