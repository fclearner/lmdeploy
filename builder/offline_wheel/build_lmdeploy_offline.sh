#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-70-real;75-real}"
BUILD_ROOT="${BUILD_ROOT:-${SCRIPT_DIR}/build_src}"
DIST_DIR="${DIST_DIR:-${SCRIPT_DIR}/dist}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-${SCRIPT_DIR}/wheelhouse}"
BUILD_REQ_FILE="${BUILD_REQ_FILE:-${SCRIPT_DIR}/requirements/offline_build.txt}"
INSTALL_REQ_FILE="${INSTALL_REQ_FILE:-${SCRIPT_DIR}/requirements/offline_install.txt}"
INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-0}"

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

for path in \
  "${SCRIPT_DIR}/source/lmdeploy-source.tar.gz" \
  "${SCRIPT_DIR}/third_party/fmt" \
  "${SCRIPT_DIR}/third_party/Catch2" \
  "${SCRIPT_DIR}/third_party/repo-cutlass" \
  "${SCRIPT_DIR}/third_party/yaml-cpp" \
  "${SCRIPT_DIR}/third_party/xgrammar" \
  "${SCRIPT_DIR}/third_party/gloo" \
  "${SCRIPT_DIR}/third_party/concurrentqueue"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing offline build input: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -m pip install --no-index --find-links "${WHEELHOUSE_DIR}" -r "${BUILD_REQ_FILE}"
"${PYTHON_BIN}" -m pip install --no-index --find-links "${WHEELHOUSE_DIR}" -r "${INSTALL_REQ_FILE}"

rm -rf "${BUILD_ROOT}" "${DIST_DIR}"
mkdir -p "${BUILD_ROOT}" "${DIST_DIR}"
tar -xzf "${SCRIPT_DIR}/source/lmdeploy-source.tar.gz" -C "${BUILD_ROOT}"

cd "${BUILD_ROOT}/lmdeploy-src"

export LMDEPLOY_TARGET_DEVICE=cuda
unset DISABLE_TURBOMIND
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
export CUDAARCHS="${CUDA_ARCHITECTURES}"

offline_cmake_args=(
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  "-DFETCHCONTENT_SOURCE_DIR_FMT=${SCRIPT_DIR}/third_party/fmt"
  "-DFETCHCONTENT_SOURCE_DIR_CATCH2=${SCRIPT_DIR}/third_party/Catch2"
  "-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS=${SCRIPT_DIR}/third_party/repo-cutlass"
  "-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=${SCRIPT_DIR}/third_party/yaml-cpp"
  "-DFETCHCONTENT_SOURCE_DIR_XGRAMMAR=${SCRIPT_DIR}/third_party/xgrammar"
  "-DFETCHCONTENT_SOURCE_DIR_GLOO=${SCRIPT_DIR}/third_party/gloo"
  "-DFETCHCONTENT_SOURCE_DIR_CONCURRENTQUEUE=${SCRIPT_DIR}/third_party/concurrentqueue"
)
for arg in "${offline_cmake_args[@]}"; do
  printf -v quoted_arg '%q' "${arg}"
  export LMDEPLOY_EXTRA_CMAKE_ARGS="${LMDEPLOY_EXTRA_CMAKE_ARGS:-} ${quoted_arg}"
done

"${PYTHON_BIN}" -m build --wheel --no-isolation -o "${DIST_DIR}"

if [[ "${INSTALL_AFTER_BUILD}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" "${SCRIPT_DIR}/install_offline.sh"
fi

echo "Offline LMDeploy wheel is ready under: ${DIST_DIR}"
