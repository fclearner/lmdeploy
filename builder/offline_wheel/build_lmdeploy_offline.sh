#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CUDA_VERSION_RAW="${CUDA_VERSION:-12.2}"
CUDA_VERSION="$(printf '%s\n' "${CUDA_VERSION_RAW}" | sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/')"
CUDA_ARCHITECTURES="${LMDEPLOY_CUDA_ARCHITECTURES:-70-real;75-real}"
BUILD_ROOT="${BUILD_ROOT:-${SCRIPT_DIR}/build_src}"
DIST_DIR="${DIST_DIR:-${SCRIPT_DIR}/dist}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-${SCRIPT_DIR}/wheelhouse}"
BUILD_REQ_FILE="${BUILD_REQ_FILE:-${SCRIPT_DIR}/requirements/offline_build.txt}"
INSTALL_REQ_FILE="${INSTALL_REQ_FILE:-${SCRIPT_DIR}/requirements/offline_install.txt}"
INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-0}"
TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
MIN_BUILD_FREE_GB="${LMDEPLOY_MIN_BUILD_FREE_GB:-10}"
MIN_DIST_FREE_GB="${LMDEPLOY_MIN_DIST_FREE_GB:-1}"
MIN_TMP_FREE_GB="${LMDEPLOY_MIN_TMP_FREE_GB:-2}"
MIN_CACHE_FREE_GB="${LMDEPLOY_MIN_CACHE_FREE_GB:-1}"

mkdir -p "${TMPDIR}" "${BUILD_ROOT}" "${DIST_DIR}" "${PIP_CACHE_DIR}"
export TMPDIR XDG_CACHE_HOME PIP_CACHE_DIR

check_free_space() {
  local path="$1"
  local label="$2"
  local min_gb="$3"
  local available_kb
  available_kb="$(df -Pk "${path}" | awk 'NR == 2 {print $4}')"
  local min_kb=$((min_gb * 1024 * 1024))
  if [[ -z "${available_kb}" || "${available_kb}" -lt "${min_kb}" ]]; then
    echo "Insufficient free space for ${label}: need >= ${min_gb} GiB, current df output:" >&2
    df -h "${path}" >&2 || true
    echo "Set BUILD_ROOT and TMPDIR to a larger filesystem, for example:" >&2
    echo "  BUILD_ROOT=/data/lmdeploy_build TMPDIR=/data/tmp PYTHON_BIN=${PYTHON_BIN} bash build_lmdeploy_offline.sh" >&2
    exit 1
  fi
}

check_free_space "${BUILD_ROOT}" "BUILD_ROOT" "${MIN_BUILD_FREE_GB}"
check_free_space "${DIST_DIR}" "DIST_DIR" "${MIN_DIST_FREE_GB}"
check_free_space "${TMPDIR}" "TMPDIR" "${MIN_TMP_FREE_GB}"
check_free_space "${PIP_CACHE_DIR}" "PIP_CACHE_DIR" "${MIN_CACHE_FREE_GB}"

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

"${PYTHON_BIN}" - <<PY
import re
import subprocess
required = "${CUDA_VERSION}"
out = subprocess.check_output(["${CUDACXX:-nvcc}", "--version"], text=True)
match = re.search(r"release\\s+(\\d+\\.\\d+)", out)
if not match:
    raise SystemExit("failed to parse nvcc version")
actual = match.group(1)
if actual != required:
    raise SystemExit(f"CUDA {required} is required, got {actual}")
print(out.strip())
PY

cuda_smoke_dir="$(mktemp -d "${TMPDIR}/lmdeploy_cuda_smoke.XXXXXX")"
trap 'rm -rf "${cuda_smoke_dir}"' EXIT
cat >"${cuda_smoke_dir}/nvcc_check.cu" <<'EOF'
int main() { return 0; }
EOF
nvcc_smoke_args=(-c "${cuda_smoke_dir}/nvcc_check.cu" -o "${cuda_smoke_dir}/nvcc_check.o")
if [[ -n "${CUDAHOSTCXX:-}" ]]; then
  nvcc_smoke_args=(-ccbin "${CUDAHOSTCXX}" "${nvcc_smoke_args[@]}")
elif [[ -n "${CXX:-}" ]]; then
  nvcc_smoke_args=(-ccbin "${CXX}" "${nvcc_smoke_args[@]}")
fi
"${CUDACXX:-nvcc}" "${nvcc_smoke_args[@]}"

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

pip_args=()
if compgen -G "${WHEELHOUSE_DIR}/*.whl" >/dev/null; then
  pip_args=(--no-index --find-links "${WHEELHOUSE_DIR}")
fi

"${PYTHON_BIN}" -m pip install "${pip_args[@]}" -r "${BUILD_REQ_FILE}"
"${PYTHON_BIN}" -m pip install "${pip_args[@]}" -r "${INSTALL_REQ_FILE}"

rm -rf "${BUILD_ROOT}" "${DIST_DIR}"
mkdir -p "${BUILD_ROOT}" "${DIST_DIR}"
tar -xzf "${SCRIPT_DIR}/source/lmdeploy-source.tar.gz" -C "${BUILD_ROOT}"

cd "${BUILD_ROOT}/lmdeploy-src"

export LMDEPLOY_TARGET_DEVICE=cuda
unset DISABLE_TURBOMIND
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
export CUDAARCHS="${CUDA_ARCHITECTURES}"
export LMDEPLOY_EXPECTED_CUDA_ARCHS="${LMDEPLOY_EXPECTED_CUDA_ARCHS:-${CUDA_ARCHITECTURES}}"

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
  PYTHON_BIN="${PYTHON_BIN}" LMDEPLOY_PYTHON_VERSION="${PYTHON_VERSION}" "${SCRIPT_DIR}/install_offline.sh"
fi

echo "Offline LMDeploy wheel is ready under: ${DIST_DIR}"
