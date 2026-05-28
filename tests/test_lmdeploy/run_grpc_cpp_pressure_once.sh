#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

CONDA_SH="${CONDA_SH:-/home/alan/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-work}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE="${LOG_FILE:-/tmp/lmdeploy_grpc_cpp_pressure_server.log}"

if [[ "${NO_CONDA:-0}" != "1" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  set +u
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
fi

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/lmdeploy/lib:${LD_LIBRARY_PATH:-}"
export TM_MODEL_PATH="${TM_MODEL_PATH:-${REPO_ROOT}/.cache_models/Qwen/Qwen2.5-0.5B}"
export TM_GRPC_PORT="${TM_GRPC_PORT:-50051}"
export TM_MAX_NEW_TOKENS="${TM_MAX_NEW_TOKENS:-1}"
export TM_MAX_INSTANCES="${TM_MAX_INSTANCES:-2}"
export TM_MAX_BATCH_SIZE="${TM_MAX_BATCH_SIZE:-${TM_MAX_INSTANCES}}"
export TM_ADMISSION_CONCURRENCY="${TM_ADMISSION_CONCURRENCY:-${TM_MAX_BATCH_SIZE}}"
export TM_CUDA_STREAMS="${TM_CUDA_STREAMS:-${TM_ADMISSION_CONCURRENCY}}"
export TM_MAX_QUEUE_SIZE="${TM_MAX_QUEUE_SIZE:-64}"
export TM_QUEUE_TIMEOUT_S="${TM_QUEUE_TIMEOUT_S:-5}"
export TM_GENERATION_TIMEOUT_S="${TM_GENERATION_TIMEOUT_S:-30}"
export TM_ENABLE_PREFIX_CACHING="${TM_ENABLE_PREFIX_CACHING:-0}"

export TM_ENABLE_CPP_LOGITS_PROCESSOR="${TM_ENABLE_CPP_LOGITS_PROCESSOR:-1}"
export TM_ENABLE_BUILTIN_LOGITS_PROCESSOR="${TM_ENABLE_BUILTIN_LOGITS_PROCESSOR:-0}"
export TM_LOGITS_PROCESSOR="${TM_LOGITS_PROCESSOR:-}"
export TM_VALID_ID="${TM_VALID_ID:-123}"
export TM_INVALID_ID="${TM_INVALID_ID:-456}"
export TM_END_ID="${TM_END_ID:-789}"
export TM_CERTAINTY_THRESHOLD="${TM_CERTAINTY_THRESHOLD:-0}"
export TM_COMPLETION_THRESHOLD="${TM_COMPLETION_THRESHOLD:-0}"
export TM_INVALID_BIAS="${TM_INVALID_BIAS:-0}"

REQUESTS="${REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-4}"
CHANNELS="${CHANNELS:-4}"
WARMUP="${WARMUP:-4}"
REPEAT="${REPEAT:-2}"
MIN_CHARS="${MIN_CHARS:-32}"
MAX_CHARS="${MAX_CHARS:-256}"
INFER_TYPES="${INFER_TYPES:--1,1}"
EXPECTED_TOKEN_ID="${EXPECTED_TOKEN_ID:-}"
TOP_SLOW="${TOP_SLOW:-0}"
RATE_QPS="${RATE_QPS:-0}"
PHASE_GAP_SEC="${PHASE_GAP_SEC:-0}"
CHANNEL_READY_TIMEOUT_SEC="${CHANNEL_READY_TIMEOUT_SEC:-10}"

rm -f "${LOG_FILE}"
"${PYTHON_BIN}" grpc_turbomind_server.py >"${LOG_FILE}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  if "${PYTHON_BIN}" tests/test_lmdeploy/grpc_client_pressure.py \
      --target "127.0.0.1:${TM_GRPC_PORT}" \
      --requests 1 \
      --concurrency 1 \
      --warmup 0 \
      --infer-type -1 \
      --max-chars 32 \
      --health >/tmp/lmdeploy_grpc_pressure_wait.out 2>/tmp/lmdeploy_grpc_pressure_wait.err; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[server] exited during startup"
    tail -160 "${LOG_FILE}" || true
    exit 1
  fi
  sleep 2
done

if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "[server] exited before pressure"
  tail -160 "${LOG_FILE}" || true
  exit 1
fi

cat /tmp/lmdeploy_grpc_pressure_wait.out
if [[ -n "${EXPECTED_TOKEN_ID}" ]]; then
  "${PYTHON_BIN}" tests/test_lmdeploy/check_grpc_generate_once.py \
    --target "127.0.0.1:${TM_GRPC_PORT}" \
    --infer-type 1 \
    --expected-token-id "${EXPECTED_TOKEN_ID}"
fi
"${PYTHON_BIN}" tests/test_lmdeploy/grpc_client_pressure.py \
  --target "127.0.0.1:${TM_GRPC_PORT}" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --channels "${CHANNELS}" \
  --warmup "${WARMUP}" \
  --infer-types="${INFER_TYPES}" \
  --repeat "${REPEAT}" \
  --rate-qps "${RATE_QPS}" \
  --min-chars "${MIN_CHARS}" \
  --max-chars "${MAX_CHARS}" \
  --top-slow "${TOP_SLOW}" \
  --phase-gap-sec "${PHASE_GAP_SEC}" \
  --channel-ready-timeout-sec "${CHANNEL_READY_TIMEOUT_SEC}" \
  --timeout-sec 120 \
  --health
