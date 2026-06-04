#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-${SCRIPT_DIR}/wheelhouse}"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements/offline_install.txt}"

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import os
import sys
actual = f"{sys.version_info[0]}.{sys.version_info[1]}"
expected = os.getenv("LMDEPLOY_PYTHON_VERSION")
if expected and expected != actual:
    raise SystemExit(f"python {expected} is required, got {sys.version.split()[0]}")
print(actual)
PY
)"
export LMDEPLOY_PYTHON_VERSION="${PYTHON_VERSION}"

mapfile -t wheels < <(find "${SCRIPT_DIR}/dist" -maxdepth 1 -name 'lmdeploy-*.whl' | sort)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one lmdeploy wheel in ${SCRIPT_DIR}/dist, found ${#wheels[@]}" >&2
  exit 1
fi

pip_args=()
if compgen -G "${WHEELHOUSE_DIR}/*.whl" >/dev/null; then
  pip_args=(--no-index --find-links "${WHEELHOUSE_DIR}")
fi

"${PYTHON_BIN}" -m pip install "${pip_args[@]}" -r "${REQ_FILE}"
"${PYTHON_BIN}" -m pip install "${pip_args[@]}" --no-deps "${wheels[0]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_install.py"
