#!/usr/bin/env bash
# Launch Calliope. Sets LD_LIBRARY_PATH so the user-space PortAudio (installed
# without sudo) is found by sounddevice.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="$HOME/.local/palib/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec .venv/bin/python talk.py "$@"
