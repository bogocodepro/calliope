#!/usr/bin/env bash
# Calliope Studio — speak · record · train · use, all in one. http://localhost:7860
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec .venv/bin/python studio.py
