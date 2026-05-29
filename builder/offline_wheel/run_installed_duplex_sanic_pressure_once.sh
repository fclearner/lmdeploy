#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
GRPC_LOG_FILE="${GRPC_LOG_FILE:-/tmp/lmdeploy_duplex_grpc_server.log}"
SANIC_LOG_FILE="${SANIC_LOG_FILE:-/tmp/lmdeploy_duplex_sanic_server.log}"

export TM_MODEL_PATH="${TM_MODEL_PATH:?TM_MODEL_PATH is required}"
export TM_GRPC_PORT="${TM_GRPC_PORT:-50051}"
export TM_GRPC_TARGET="${TM_GRPC_TARGET:-127.0.0.1:${TM_GRPC_PORT}}"
export TM_MAX_NEW_TOKENS="${TM_MAX_NEW_TOKENS:-1}"
export TM_MAX_INSTANCES="${TM_MAX_INSTANCES:-50}"
export TM_MAX_BATCH_SIZE="${TM_MAX_BATCH_SIZE:-${TM_MAX_INSTANCES}}"
export TM_ADMISSION_CONCURRENCY="${TM_ADMISSION_CONCURRENCY:-${TM_MAX_BATCH_SIZE}}"
export TM_CUDA_STREAMS="${TM_CUDA_STREAMS:-${TM_ADMISSION_CONCURRENCY}}"
export TM_MAX_QUEUE_SIZE="${TM_MAX_QUEUE_SIZE:-256}"
export TM_QUEUE_TIMEOUT_S="${TM_QUEUE_TIMEOUT_S:-5}"
export TM_GENERATION_TIMEOUT_S="${TM_GENERATION_TIMEOUT_S:-30}"
export TM_ENABLE_PREFIX_CACHING="${TM_ENABLE_PREFIX_CACHING:-0}"

export TM_ENABLE_CPP_LOGITS_PROCESSOR="${TM_ENABLE_CPP_LOGITS_PROCESSOR:-1}"
export TM_ENABLE_BUILTIN_LOGITS_PROCESSOR="${TM_ENABLE_BUILTIN_LOGITS_PROCESSOR:-0}"
export TM_LOGITS_PROCESSOR="${TM_LOGITS_PROCESSOR:-}"
export TM_VALID_ID="${TM_VALID_ID:-123}"
export TM_INVALID_ID="${TM_INVALID_ID:-456}"
export TM_END_ID="${TM_END_ID:-789}"
export TM_CERTAINTY_THRESHOLD="${TM_CERTAINTY_THRESHOLD:-0.05}"
export TM_COMPLETION_THRESHOLD="${TM_COMPLETION_THRESHOLD:-0.1}"
export TM_INVALID_BIAS="${TM_INVALID_BIAS:-0.1}"

export DUPLEX_HOST="${DUPLEX_HOST:-127.0.0.1}"
export DUPLEX_PORT="${DUPLEX_PORT:-18080}"
export DUPLEX_GRPC_TARGET="${DUPLEX_GRPC_TARGET:-${TM_GRPC_TARGET}}"
export DUPLEX_CLIENT_POOL_SIZE="${DUPLEX_CLIENT_POOL_SIZE:-1}"
export DUPLEX_DEFAULT_GRPC_CHANNELS="${DUPLEX_DEFAULT_GRPC_CHANNELS:-50}"
export DUPLEX_GRPC_CLIENT_CHANNELS="${DUPLEX_GRPC_CLIENT_CHANNELS:-${DUPLEX_DEFAULT_GRPC_CHANNELS}}"
export DUPLEX_REQUEST_TIMEOUT="${DUPLEX_REQUEST_TIMEOUT:-10}"
export DUPLEX_MAX_NEW_TOKENS="${DUPLEX_MAX_NEW_TOKENS:-1}"
export DUPLEX_VALID_ID="${DUPLEX_VALID_ID:-${TM_VALID_ID}}"
export DUPLEX_INVALID_ID="${DUPLEX_INVALID_ID:-${TM_INVALID_ID}}"
export DUPLEX_END_ID="${DUPLEX_END_ID:-${TM_END_ID}}"
export DUPLEX_CERTAINTY_THRESHOLD="${DUPLEX_CERTAINTY_THRESHOLD:-${TM_CERTAINTY_THRESHOLD}}"
export DUPLEX_COMPLETION_THRESHOLD="${DUPLEX_COMPLETION_THRESHOLD:-${TM_COMPLETION_THRESHOLD}}"
export DUPLEX_INVALID_BIAS="${DUPLEX_INVALID_BIAS:-${TM_INVALID_BIAS}}"

REQUESTS="${REQUESTS:-1000}"
CONCURRENCY="${CONCURRENCY:-50}"
CHANNELS="${CHANNELS:-50}"
WARMUP="${WARMUP:-100}"
REPEAT="${REPEAT:-3}"
RATE_QPS="${RATE_QPS:-250}"
MIN_CHARS="${MIN_CHARS:-1}"
MAX_CHARS="${MAX_CHARS:-256}"
TOP_SLOW="${TOP_SLOW:-5}"

echo "[duplex-pressure] grpc_target=${TM_GRPC_TARGET} grpc_log=${GRPC_LOG_FILE}"
echo "[duplex-pressure] sanic_url=http://${DUPLEX_HOST}:${DUPLEX_PORT}/infer sanic_log=${SANIC_LOG_FILE}"

rm -f "${GRPC_LOG_FILE}" "${SANIC_LOG_FILE}"
"${PYTHON_BIN}" -m grpc_turbomind_server >"${GRPC_LOG_FILE}" 2>&1 &
grpc_pid=$!
sanic_pid=""

cleanup() {
  if [[ -n "${sanic_pid}" ]]; then
    kill "${sanic_pid}" 2>/dev/null || true
    wait "${sanic_pid}" 2>/dev/null || true
  fi
  kill "${grpc_pid}" 2>/dev/null || true
  wait "${grpc_pid}" 2>/dev/null || true
}
trap cleanup EXIT

grpc_ready=0
for _ in $(seq 1 120); do
  if "${PYTHON_BIN}" - <<PY >/tmp/lmdeploy_duplex_grpc_wait.out 2>/tmp/lmdeploy_duplex_grpc_wait.err
import asyncio
from turbomind_grpc_client import TurboMindGrpcClient

async def main():
    client = TurboMindGrpcClient("${TM_GRPC_TARGET}", channels=1, ready_on_start=False)
    print(await client.health_async())
    await client.close()

asyncio.run(main())
PY
  then
    grpc_ready=1
    break
  fi
  if ! kill -0 "${grpc_pid}" 2>/dev/null; then
    echo "[grpc-server] exited during startup"
    tail -160 "${GRPC_LOG_FILE}" || true
    exit 1
  fi
  sleep 2
done

if [[ "${grpc_ready}" != "1" ]]; then
  echo "[grpc-server] did not become healthy"
  tail -160 "${GRPC_LOG_FILE}" || true
  cat /tmp/lmdeploy_duplex_grpc_wait.err || true
  exit 1
fi

cat /tmp/lmdeploy_duplex_grpc_wait.out
"${PYTHON_BIN}" -m duplex.server --host "${DUPLEX_HOST}" --port "${DUPLEX_PORT}" >"${SANIC_LOG_FILE}" 2>&1 &
sanic_pid=$!

sanic_ready=0
for _ in $(seq 1 60); do
  if "${PYTHON_BIN}" - <<PY
import urllib.request
urllib.request.urlopen("http://${DUPLEX_HOST}:${DUPLEX_PORT}/health/check", timeout=1).read()
PY
  then
    sanic_ready=1
    break
  fi
  if ! kill -0 "${sanic_pid}" 2>/dev/null; then
    echo "[sanic-server] exited during startup"
    tail -160 "${SANIC_LOG_FILE}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${sanic_ready}" != "1" ]]; then
  echo "[sanic-server] did not become healthy"
  tail -160 "${SANIC_LOG_FILE}" || true
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/tools/duplex_sanic_pressure.py" \
  --url "http://${DUPLEX_HOST}:${DUPLEX_PORT}/infer" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --channels "${CHANNELS}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --rate-qps "${RATE_QPS}" \
  --min-chars "${MIN_CHARS}" \
  --max-chars "${MAX_CHARS}" \
  --top-slow "${TOP_SLOW}"
