#!/usr/bin/env bash
# Install dependencies without requiring model weights or modifying system Python.
set -euo pipefail
TRACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACE_PYTHON="${TRACE_PYTHON:-python3}"
TRACE_VENV="${TRACE_VENV:-$TRACE_DIR/.venv}"
TRACE_TORCH_INDEX="${TRACE_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
"$TRACE_PYTHON" "$TRACE_DIR/bootstrap_env.py" --venv "$TRACE_VENV"
"$TRACE_VENV/bin/python" -m pip install torch==2.11.0 --index-url "$TRACE_TORCH_INDEX"
"$TRACE_VENV/bin/python" -m pip install -r "$TRACE_DIR/requirements.txt" -r "$TRACE_DIR/requirements-hub.txt"
"$TRACE_VENV/bin/python" -m pip check
cd "$TRACE_DIR"
"$TRACE_VENV/bin/python" -m unittest discover -s tests -v
echo 'Environment ready. Download authorized artifacts before running inference; see README.md.'
