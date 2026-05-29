#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-${SCRIPT_DIR}/wheelhouse}"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements/offline_install.txt}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"python 3.10 is required, got {sys.version.split()[0]}")
PY

mapfile -t wheels < <(find "${SCRIPT_DIR}/dist" -maxdepth 1 -name 'lmdeploy-*.whl' | sort)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one lmdeploy wheel in ${SCRIPT_DIR}/dist, found ${#wheels[@]}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install --no-index --find-links "${WHEELHOUSE_DIR}" -r "${REQ_FILE}"
"${PYTHON_BIN}" -m pip install --no-index --find-links "${WHEELHOUSE_DIR}" --no-deps "${wheels[0]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_install.py"
