#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CUDA_VERSION_RAW="${CUDA_VERSION:-12.2}"
CUDA_VERSION="$(printf '%s\n' "${CUDA_VERSION_RAW}" | sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/')"
CUDA_ARCHITECTURES="${LMDEPLOY_CUDA_ARCHITECTURES:-70-real;75-real}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/offline_dist}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
ONLY_BINARY="${ONLY_BINARY:-1}"
INCLUDE_WHEELHOUSE="${INCLUDE_WHEELHOUSE:-0}"
MAKE_TARBALL="${MAKE_TARBALL:-1}"

cd "${REPO_ROOT}"

VERSION="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
ns = {}
exec(Path("lmdeploy/version.py").read_text(), ns)
print(ns["__version__"])
PY
)"
if [[ "${INCLUDE_WHEELHOUSE}" == "1" ]]; then
  CUDA_TAG="cu${CUDA_VERSION//./}"
  DEFAULT_BUNDLE_NAME="lmdeploy-${VERSION}-source-build-py310-${CUDA_TAG}-sm70-sm75"
else
  CUDA_TAG="cu${CUDA_VERSION//./}"
  DEFAULT_BUNDLE_NAME="lmdeploy-${VERSION}-source-build-lite-py310-${CUDA_TAG}-sm70-sm75"
fi
BUNDLE_NAME="${BUNDLE_NAME:-${DEFAULT_BUNDLE_NAME}}"
BUNDLE_DIR="${OUTPUT_ROOT}/${BUNDLE_NAME}"
WHEELHOUSE_DIR="${BUNDLE_DIR}/wheelhouse"
REQ_DIR="${BUNDLE_DIR}/requirements"
SOURCE_DIR="${BUNDLE_DIR}/source"
THIRD_PARTY_DIR="${BUNDLE_DIR}/third_party"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"python 3.10 is required, got {sys.version.split()[0]}")
PY

if [[ "${CLEAN_OUTPUT}" == "1" && -d "${BUNDLE_DIR}" ]]; then
  case "${BUNDLE_DIR}" in
    "${OUTPUT_ROOT}"/*) rm -rf "${BUNDLE_DIR}" ;;
    *) echo "Refusing to remove unexpected output dir: ${BUNDLE_DIR}" >&2; exit 1 ;;
  esac
fi

mkdir -p "${REQ_DIR}" "${SOURCE_DIR}" "${THIRD_PARTY_DIR}"
if [[ "${INCLUDE_WHEELHOUSE}" == "1" ]]; then
  mkdir -p "${WHEELHOUSE_DIR}"
fi

cp requirements/build.txt "${REQ_DIR}/build.txt"
cp requirements/runtime_cuda.txt "${REQ_DIR}/runtime_cuda.txt"
cp requirements/serve.txt "${REQ_DIR}/serve.txt"
cat >"${REQ_DIR}/offline_build.txt" <<'EOF'
pip
setuptools
wheel
build
cmake
ninja
cmake_build_extension
pybind11<=2.13.1
EOF
cat >"${REQ_DIR}/offline_install.txt" <<'EOF'
-r runtime_cuda.txt
-r serve.txt
EOF

if [[ "${INCLUDE_WHEELHOUSE}" == "1" ]]; then
  download_args=(download --dest "${WHEELHOUSE_DIR}")
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    download_args+=(--extra-index-url "${TORCH_INDEX_URL}")
  fi
  if [[ "${ONLY_BINARY}" == "1" ]]; then
    download_args+=(--only-binary=:all:)
  fi
  "${PYTHON_BIN}" -m pip "${download_args[@]}" -r "${REQ_DIR}/offline_build.txt"
  "${PYTHON_BIN}" -m pip "${download_args[@]}" -r "${REQ_DIR}/offline_install.txt"
fi

git archive --format=tar.gz --prefix=lmdeploy-src/ "${SOURCE_REF}" -o "${SOURCE_DIR}/lmdeploy-source.tar.gz"
git rev-parse "${SOURCE_REF}" >"${SOURCE_DIR}/source_ref.txt"

clone_dep() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local dest="${THIRD_PARTY_DIR}/${name}"

  rm -rf "${dest}"
  git clone --filter=blob:none "${url}" "${dest}"
  git -C "${dest}" checkout "${ref}"
  git -C "${dest}" submodule update --init --recursive --depth 1
  find "${dest}" -name .git -print0 | xargs -0 rm -rf
}

clone_dep fmt https://github.com/fmtlib/fmt.git 11.1.4
clone_dep Catch2 https://github.com/catchorg/Catch2.git v3.8.0
clone_dep repo-cutlass https://github.com/NVIDIA/cutlass.git v3.9.2
clone_dep yaml-cpp https://github.com/jbeder/yaml-cpp.git 65c1c270dbe7eec37b2df2531d7497c4eea79aee
clone_dep xgrammar https://github.com/mlc-ai/xgrammar.git v0.1.27
clone_dep gloo https://github.com/pytorch/gloo.git c7b7b022c124d9643957d9bd55f57ac59fce8fa2
clone_dep concurrentqueue https://github.com/cameron314/concurrentqueue.git v1.0.4

cat >"${BUNDLE_DIR}/third_party_manifest.txt" <<'EOF'
fmt https://github.com/fmtlib/fmt.git 11.1.4
Catch2 https://github.com/catchorg/Catch2.git v3.8.0
repo-cutlass https://github.com/NVIDIA/cutlass.git v3.9.2
yaml-cpp https://github.com/jbeder/yaml-cpp.git 65c1c270dbe7eec37b2df2531d7497c4eea79aee
xgrammar https://github.com/mlc-ai/xgrammar.git v0.1.27 with 3rdparty/dlpack submodule
gloo https://github.com/pytorch/gloo.git c7b7b022c124d9643957d9bd55f57ac59fce8fa2
concurrentqueue https://github.com/cameron314/concurrentqueue.git v1.0.4
EOF

cp "${SCRIPT_DIR}/build_lmdeploy_offline.sh" "${BUNDLE_DIR}/build_lmdeploy_offline.sh"
cp "${SCRIPT_DIR}/install_offline.sh" "${BUNDLE_DIR}/install_offline.sh"
cp "${SCRIPT_DIR}/verify_install.py" "${BUNDLE_DIR}/verify_install.py"
chmod +x "${BUNDLE_DIR}/build_lmdeploy_offline.sh" "${BUNDLE_DIR}/install_offline.sh"

{
  echo "lmdeploy_version=${VERSION}"
  echo "source_ref=${SOURCE_REF}"
  echo "source_commit=$(git rev-parse "${SOURCE_REF}" 2>/dev/null || true)"
  echo "python_bin=${PYTHON_BIN}"
  "${PYTHON_BIN}" --version
  echo "cuda_version=${CUDA_VERSION}"
  echo "cmake_cuda_architectures=${CUDA_ARCHITECTURES}"
  echo "include_wheelhouse=${INCLUDE_WHEELHOUSE}"
  echo "torch_index_url=${TORCH_INDEX_URL}"
  echo "git_status_short_begin"
  git status --short 2>/dev/null || true
  echo "git_status_short_end"
} >"${BUNDLE_DIR}/build_info.txt"

(
  cd "${BUNDLE_DIR}"
  checksum_paths=(
    build_lmdeploy_offline.sh
    install_offline.sh
    verify_install.py
    build_info.txt
    third_party_manifest.txt
    requirements
    source
    third_party
  )
  if [[ -d wheelhouse ]]; then
    checksum_paths+=(wheelhouse)
  fi
  find "${checksum_paths[@]}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

if [[ "${MAKE_TARBALL}" == "1" ]]; then
  tar -C "${OUTPUT_ROOT}" -czf "${BUNDLE_DIR}.tar.gz" "${BUNDLE_NAME}"
  sha256sum "${BUNDLE_DIR}.tar.gz" >"${BUNDLE_DIR}.tar.gz.sha256"
fi

echo "Offline source-build bundle is ready: ${BUNDLE_DIR}"
if [[ "${MAKE_TARBALL}" == "1" ]]; then
  echo "Tarball is ready: ${BUNDLE_DIR}.tar.gz"
fi
