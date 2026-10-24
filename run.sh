#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip --disable-pip-version-check install -q -r backend/requirements.txt
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8000}"
