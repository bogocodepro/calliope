#!/usr/bin/env bash
# Launch the voice recording studio at http://localhost:7861
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec .venv/bin/python record_ui.py
