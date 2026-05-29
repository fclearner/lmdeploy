#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CUDA_VERSION_RAW="${CUDA_VERSION:-12.4}"
CUDA_VERSION="$(printf '%s\n' "${CUDA_VERSION_RAW}" | sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/')"
CUDA_ARCHITECTURES="${LMDEPLOY_CUDA_ARCHITECTURES:-70-real;75-real}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/offline_dist}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"
ONLY_BINARY="${ONLY_BINARY:-1}"

cd "${REPO_ROOT}"

VERSION="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
ns = {}
exec(Path("lmdeploy/version.py").read_text(), ns)
print(ns["__version__"])
PY
)"
BUNDLE_NAME="${BUNDLE_NAME:-lmdeploy-${VERSION}-py310-cu124-sm70-sm75}"
BUNDLE_DIR="${OUTPUT_ROOT}/${BUNDLE_NAME}"
DIST_DIR="${BUNDLE_DIR}/dist"
WHEELHOUSE_DIR="${BUNDLE_DIR}/wheelhouse"
REQ_DIR="${BUNDLE_DIR}/requirements"
TOOLS_DIR="${BUNDLE_DIR}/tools"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"python 3.10 is required, got {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" - <<PY
import re
import subprocess
required = "${CUDA_VERSION}"
out = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\\s+(\\d+\\.\\d+)", out)
if not match:
    raise SystemExit("failed to parse nvcc version")
actual = match.group(1)
if actual != required:
    raise SystemExit(f"CUDA {required} is required, got {actual}")
print(out.strip())
PY

if [[ "${CLEAN_OUTPUT}" == "1" && -d "${BUNDLE_DIR}" ]]; then
  case "${BUNDLE_DIR}" in
    "${OUTPUT_ROOT}"/*) rm -rf "${BUNDLE_DIR}" ;;
    *) echo "Refusing to remove unexpected output dir: ${BUNDLE_DIR}" >&2; exit 1 ;;
  esac
fi
mkdir -p "${DIST_DIR}" "${WHEELHOUSE_DIR}" "${REQ_DIR}" "${TOOLS_DIR}"

cp requirements/runtime_cuda.txt "${REQ_DIR}/runtime_cuda.txt"
cp requirements/serve.txt "${REQ_DIR}/serve.txt"
cat >"${REQ_DIR}/pytorch_cu124.txt" <<'EOF'
torch==2.6.0+cu124
torchvision==0.21.0+cu124
EOF
cat >"${REQ_DIR}/offline_install.txt" <<'EOF'
-r pytorch_cu124.txt
-r runtime_cuda.txt
-r serve.txt
EOF

if [[ "${INSTALL_BUILD_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel build
  "${PYTHON_BIN}" -m pip install -r requirements/build.txt
fi

if [[ "${CLEAN_BUILD}" == "1" ]]; then
  rm -rf build
fi

export LMDEPLOY_TARGET_DEVICE=cuda
unset DISABLE_TURBOMIND
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
export CUDAARCHS="${CUDA_ARCHITECTURES}"
export LMDEPLOY_EXPECTED_CUDA_ARCHS="${LMDEPLOY_EXPECTED_CUDA_ARCHS:-${CUDA_ARCHITECTURES}}"

"${PYTHON_BIN}" -m build --wheel --no-isolation -o "${DIST_DIR}"

download_args=(download --dest "${WHEELHOUSE_DIR}" --extra-index-url "${TORCH_INDEX_URL}")
if [[ "${ONLY_BINARY}" == "1" ]]; then
  download_args+=(--only-binary=:all:)
fi
"${PYTHON_BIN}" -m pip "${download_args[@]}" -r "${REQ_DIR}/offline_install.txt"

cp "${SCRIPT_DIR}/install_offline.sh" "${BUNDLE_DIR}/install_offline.sh"
cp "${SCRIPT_DIR}/verify_install.py" "${BUNDLE_DIR}/verify_install.py"
cp "${SCRIPT_DIR}/run_installed_duplex_sanic_pressure_once.sh" \
  "${BUNDLE_DIR}/run_installed_duplex_sanic_pressure_once.sh"
cp tests/test_lmdeploy/duplex_sanic_pressure.py "${TOOLS_DIR}/duplex_sanic_pressure.py"
chmod +x "${BUNDLE_DIR}/install_offline.sh" \
  "${BUNDLE_DIR}/run_installed_duplex_sanic_pressure_once.sh" \
  "${TOOLS_DIR}/duplex_sanic_pressure.py"

{
  echo "lmdeploy_version=${VERSION}"
  echo "python_bin=${PYTHON_BIN}"
  "${PYTHON_BIN}" --version
  echo "cuda_version=${CUDA_VERSION}"
  nvcc --version
  echo "cmake_cuda_architectures=${CUDA_ARCHITECTURES}"
  echo "torch_index_url=${TORCH_INDEX_URL}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "git_status_short_begin"
  git status --short 2>/dev/null || true
  echo "git_status_short_end"
} >"${BUNDLE_DIR}/build_info.txt"

(
  cd "${BUNDLE_DIR}"
  find dist wheelhouse requirements tools -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

echo "Offline bundle is ready: ${BUNDLE_DIR}"
