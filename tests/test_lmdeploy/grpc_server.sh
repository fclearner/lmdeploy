#!/usr/bin/env bash
set -euo pipefail

# Minimal gRPC model server launcher.
# Override any variable from the shell; this script intentionally keeps defaults
# conservative so it is easy to compare infer_type=-1/0/1 first.

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
export LMDEPLOY_LOG_LEVEL="${LMDEPLOY_LOG_LEVEL:-WARNING}"
export TM_LOG_LEVEL="${TM_LOG_LEVEL:-WARNING}"
export TM_DEBUG_LEVEL="${TM_DEBUG_LEVEL:-}"
export TM_MODEL_PATH="${TM_MODEL_PATH:-${REPO_ROOT}/.cache_models/Qwen/Qwen2.5-0.5B}"
export TM_GRPC_HOST="${TM_GRPC_HOST:-0.0.0.0}"
export TM_GRPC_PORT="${TM_GRPC_PORT:-50051}"
export TM_DTYPE="${TM_DTYPE:-auto}"
export TM_MAX_NEW_TOKENS="${TM_MAX_NEW_TOKENS:-1}"
export TM_MAX_INSTANCES="${TM_MAX_INSTANCES:-8}"
export TM_MAX_BATCH_SIZE="${TM_MAX_BATCH_SIZE:-${TM_MAX_INSTANCES}}"
export TM_ADMISSION_CONCURRENCY="${TM_ADMISSION_CONCURRENCY:-${TM_MAX_BATCH_SIZE}}"
export TM_CUDA_STREAMS="${TM_CUDA_STREAMS:-${TM_ADMISSION_CONCURRENCY}}"
export TM_MAX_QUEUE_SIZE="${TM_MAX_QUEUE_SIZE:-128}"
export TM_QUEUE_TIMEOUT_S="${TM_QUEUE_TIMEOUT_S:-5}"
export TM_GENERATION_TIMEOUT_S="${TM_GENERATION_TIMEOUT_S:-30}"
# Random pressure tests usually do not benefit from prefix caching and can show
# extra BlockTrie variance. Enable it explicitly for production prompts with a
# stable shared prefix.
export TM_ENABLE_PREFIX_CACHING="${TM_ENABLE_PREFIX_CACHING:-0}"

# Use the C++ logits processor by default. The Python builtin processor is kept
# only for A/B debugging because it requests generation logits from TurboMind and
# is much slower under pressure.
export TM_ENABLE_CPP_LOGITS_PROCESSOR="${TM_ENABLE_CPP_LOGITS_PROCESSOR:-1}"
export TM_ENABLE_BUILTIN_LOGITS_PROCESSOR="${TM_ENABLE_BUILTIN_LOGITS_PROCESSOR:-0}"
if [[ -z "${TM_LOGITS_PROCESSOR+x}" ]]; then
  if [[ "${TM_ENABLE_CPP_LOGITS_PROCESSOR}" == "1" || "${TM_ENABLE_CPP_LOGITS_PROCESSOR}" == "true" ]]; then
    export TM_LOGITS_PROCESSOR=""
  elif [[ "${TM_ENABLE_BUILTIN_LOGITS_PROCESSOR}" == "1" || "${TM_ENABLE_BUILTIN_LOGITS_PROCESSOR}" == "true" ]]; then
    export TM_LOGITS_PROCESSOR="builtin_token_decision"
  else
    export TM_LOGITS_PROCESSOR=""
  fi
fi
export TM_VALID_ID="${TM_VALID_ID:-123}"
export TM_INVALID_ID="${TM_INVALID_ID:-456}"
export TM_END_ID="${TM_END_ID:-789}"
export TM_CERTAINTY_THRESHOLD="${TM_CERTAINTY_THRESHOLD:-0}"
export TM_COMPLETION_THRESHOLD="${TM_COMPLETION_THRESHOLD:-0}"
export TM_INVALID_BIAS="${TM_INVALID_BIAS:-0}"

echo "[grpc-server] repo=${REPO_ROOT}"
echo "[grpc-server] python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[grpc-server] model=${TM_MODEL_PATH}"
echo "[grpc-server] listen=${TM_GRPC_HOST}:${TM_GRPC_PORT}"
echo "[grpc-server] dtype=${TM_DTYPE}"
echo "[grpc-server] instances=${TM_MAX_INSTANCES} batch=${TM_MAX_BATCH_SIZE} admission=${TM_ADMISSION_CONCURRENCY}"
echo "[grpc-server] prefix_caching=${TM_ENABLE_PREFIX_CACHING}"
echo "[grpc-server] cpp_logits=${TM_ENABLE_CPP_LOGITS_PROCESSOR} python_logits=${TM_ENABLE_BUILTIN_LOGITS_PROCESSOR} logits_processor=${TM_LOGITS_PROCESSOR}"
exec "${PYTHON_BIN}" grpc_turbomind_server.py
