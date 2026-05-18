#!/usr/bin/env bash
set -euo pipefail

# Pure gRPC client pressure launcher. It does not start Sanic and does not import
# the model. Use INFER_TYPES="-1,0,1" to compare server-side logits modes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

CONDA_SH="${CONDA_SH:-/home/alan/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-work}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${NO_CONDA:-0}" != "1" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TARGET="${TARGET:-${TM_GRPC_TARGET:-127.0.0.1:50051}}"
REQUESTS="${REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-8}"
CHANNELS="${CHANNELS:-${CONCURRENCY}}"
INFER_TYPES="${INFER_TYPES:--1,0,1}"
REPEAT="${REPEAT:-1}"
MIN_CHARS="${MIN_CHARS:-500}"
MAX_CHARS="${MAX_CHARS:-1000}"

echo "[grpc-client] target=${TARGET}"
echo "[grpc-client] requests=${REQUESTS} concurrency=${CONCURRENCY} channels=${CHANNELS}"
echo "[grpc-client] infer_types=${INFER_TYPES} repeat=${REPEAT} chars=${MIN_CHARS}-${MAX_CHARS}"

exec "${PYTHON_BIN}" tests/test_lmdeploy/grpc_client_pressure.py \
  --target "${TARGET}" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --channels "${CHANNELS}" \
  --infer-types="${INFER_TYPES}" \
  --repeat "${REPEAT}" \
  --warmup "${WARMUP:-8}" \
  --max-tokens "${MAX_TOKENS:-1}" \
  --min-chars "${MIN_CHARS}" \
  --max-chars "${MAX_CHARS}" \
  --health \
  "$@"
