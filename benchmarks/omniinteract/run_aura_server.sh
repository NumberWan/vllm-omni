#!/usr/bin/env bash
# Start the production AURA server used by OmniInteract.
# Full guide: benchmarks/omniinteract/SETUP_AND_RUN.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/scripts/start_aura_omni.sh" "$@"
