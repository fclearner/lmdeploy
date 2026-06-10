# LMDeploy Offline Build And Wheel Bundles

This builder targets the requested production shape:

- Python 3.10 or 3.12
- CUDA 12.2 for source compilation in the target environment
- V100 + T4, compiled as `sm_70` and `sm_75`
- LMDeploy TurboMind enabled
- Sanic and gRPC serving dependencies installed through the target pip source,
  or included in the wheelhouse when `INCLUDE_WHEELHOUSE=1`

There are two bundle modes:

- `build_offline_bundle.sh`: build the LMDeploy wheel on the online machine,
  then install that prebuilt wheel offline.
- `build_offline_source_bundle.sh`: prepare a lightweight source archive and
  CMake third-party sources so the offline machine can compile LMDeploy itself.
  It does not include a pip wheelhouse by default and uses the target
  environment's configured pip source.

## Build Prebuilt Wheel Bundle On An Online CUDA 12.4 Machine

Use a Linux x86_64 host with Python 3.10 or 3.12, CUDA toolkit 12.4, `nvcc`, CMake,
Ninja and a working compiler toolchain. A GPU is not required to compile the
wheel, but a V100 or T4 is required for final runtime validation.

```bash
PYTHON_BIN=python3.10 \
CUDA_VERSION=12.4 \
LMDEPLOY_CUDA_ARCHITECTURES='70-real;75-real' \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash builder/offline_wheel/build_offline_bundle.sh
```

The bundle is written to:

```text
offline_dist/lmdeploy-<version>-py<major><minor>-cu124-sm70-sm75/
```

The important files are:

```text
dist/lmdeploy-*.whl
wheelhouse/*.whl
requirements/offline_install.txt
requirements/pytorch_cu124.txt
install_offline.sh
verify_install.py
run_installed_duplex_sanic_pressure_once.sh
build_info.txt
SHA256SUMS
```

If a dependency has no binary wheel for your Python version on your platform, the build
script fails by default. That is intentional for offline deployment. You can set
`ONLY_BINARY=0` only if you also plan to support source builds on the offline
target, which is not recommended for production rollout.

## Build Lightweight Offline Source-Compile Bundle

Use this mode when the target environment must compile LMDeploy offline rather
than install a prebuilt LMDeploy wheel. This is the recommended mode when the
offline environment already has an internal pip source:

```bash
PYTHON_BIN=python3.12 \
LMDEPLOY_PYTHON_VERSION=3.12 \
CUDA_VERSION=12.2 \
LMDEPLOY_CUDA_ARCHITECTURES='70-real;75-real' \
bash builder/offline_wheel/build_offline_source_bundle.sh
```

The bundle is written to:

```text
offline_dist/lmdeploy-<version>-source-build-lite-py312-cu122-sm70-sm75/
offline_dist/lmdeploy-<version>-source-build-lite-py312-cu122-sm70-sm75.tar.gz
```

The important files are:

```text
source/lmdeploy-source.tar.gz
third_party/fmt/
third_party/Catch2/
third_party/repo-cutlass/
third_party/yaml-cpp/
third_party/xgrammar/
third_party/gloo/
third_party/concurrentqueue/
requirements/offline_build.txt
requirements/offline_install.txt
build_lmdeploy_offline.sh
install_offline.sh
verify_install.py
build_info.txt
SHA256SUMS
```

This bundle includes the CMake `FetchContent` dependencies used by TurboMind, so
offline compilation does not need GitHub access for `fmt`, `cutlass`,
`yaml-cpp`, `xgrammar`, `gloo`, `concurrentqueue`, or `Catch2`.

On the offline machine, unpack the tarball and compile:

```bash
tar -xzf lmdeploy-<version>-source-build-lite-py312-cu122-sm70-sm75.tar.gz
cd lmdeploy-<version>-source-build-lite-py312-cu122-sm70-sm75
PYTHON_BIN=python3.12 LMDEPLOY_PYTHON_VERSION=3.12 bash build_lmdeploy_offline.sh
```

To compile and install into the active environment in one step:

```bash
PYTHON_BIN=python3.12 LMDEPLOY_PYTHON_VERSION=3.12 INSTALL_AFTER_BUILD=1 bash build_lmdeploy_offline.sh
```

`build_lmdeploy_offline.sh` installs build and runtime Python dependencies
through the active pip configuration. Set `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`
or the target environment's pip config before running it if the internal source
is not already configured.

If the environment already exports a full toolkit version such as
`CUDA_VERSION=12.2.2.009`, the scripts normalize it to `12.2` before checking
`nvcc --version` and naming the bundle.

The build scripts intentionally ignore any pre-existing
`CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS` in the environment and set them from
`LMDEPLOY_CUDA_ARCHITECTURES`, defaulting to `70-real;75-real`. This prevents a
V100-only environment from accidentally producing a wheel without `sm_75`.

If you still need a fully self-contained source-build bundle with pip wheels,
set `INCLUDE_WHEELHOUSE=1` and, when needed, provide `TORCH_INDEX_URL`.

### CUDA Host Compiler And Disk Space Checks

`build_lmdeploy_offline.sh` runs an early `nvcc` smoke test before the full
build. If it fails with:

```text
nvcc fatal: Failed to preprocess host compiler properties.
```

first check for disk pressure. A preceding error such as:

```text
No space left on device
```

means the failure is from the filesystem used by `TMPDIR` or the build
directory, not from LMDeploy C++ headers.

Use a larger filesystem for both temporary files and build output:

```bash
mkdir -p /data/lmdeploy_build /data/lmdeploy_dist /data/tmp /data/cache/pip
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
CUDAHOSTCXX=/usr/bin/g++-11 \
CUDACXX=/usr/local/cuda/bin/nvcc \
BUILD_ROOT=/data/lmdeploy_build \
DIST_DIR=/data/lmdeploy_dist \
TMPDIR=/data/tmp \
PIP_CACHE_DIR=/data/cache/pip \
PYTHON_BIN=python3.12 \
LMDEPLOY_PYTHON_VERSION=3.12 \
INSTALL_AFTER_BUILD=1 \
bash build_lmdeploy_offline.sh
```

The script requires at least 10 GiB free for `BUILD_ROOT`, 1 GiB free for
`DIST_DIR`, 2 GiB free for `TMPDIR`, and 1 GiB free for `PIP_CACHE_DIR` by
default. Override these checks only if you have measured the target environment:

```bash
LMDEPLOY_MIN_BUILD_FREE_GB=6 LMDEPLOY_MIN_TMP_FREE_GB=1 bash build_lmdeploy_offline.sh
```

## Install Offline

Copy the whole bundle directory to the target machine, then run:

```bash
PYTHON_BIN=python3.12 LMDEPLOY_PYTHON_VERSION=3.12 bash install_offline.sh
```

The install script uses `wheelhouse/` with `--no-index --find-links` when local
wheels are present. In the lightweight source-build bundle there is no
wheelhouse by default, so it uses the active pip configuration and installs the
local LMDeploy wheel from `dist/`.

## Verify Runtime

`install_offline.sh` runs `verify_install.py` automatically. You can also run it
manually:

```bash
LMDEPLOY_PYTHON_VERSION=3.12 python3.12 verify_install.py
```

The verifier checks:

- `lmdeploy`
- `duplex.server`
- `grpc_turbomind_server`
- `turbomind_grpc_client`
- `sanic`
- `grpc`
- `_turbomind`
- `_xgrammar`
- `sm_70` and `sm_75` in `_turbomind` when `cuobjdump` is available

## Optional Installed-Service Pressure Test

After setting model path and token ids:

```bash
export TM_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
export TM_VALID_ID=<valid-token-id>
export TM_INVALID_ID=<invalid-token-id>
export TM_END_ID=<im-end-token-id>

REQUESTS=1000 \
CONCURRENCY=50 \
CHANNELS=50 \
RATE_QPS=250 \
WARMUP=100 \
REPEAT=3 \
bash run_installed_duplex_sanic_pressure_once.sh
```
