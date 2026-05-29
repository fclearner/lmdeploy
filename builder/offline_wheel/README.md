# LMDeploy Offline Wheel Bundle

This builder targets the requested production shape:

- Python 3.10
- CUDA 12.4
- V100 + T4, compiled as `sm_70` and `sm_75`
- LMDeploy TurboMind enabled
- Sanic and gRPC serving dependencies included in the offline wheelhouse

## Build On An Online CUDA 12.4 Machine

Use a Linux x86_64 host with Python 3.10, CUDA toolkit 12.4, `nvcc`, CMake,
Ninja and a working compiler toolchain. A GPU is not required to compile the
wheel, but a V100 or T4 is required for final runtime validation.

```bash
PYTHON_BIN=python3.10 \
CUDA_VERSION=12.4 \
CMAKE_CUDA_ARCHITECTURES='70-real;75-real' \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash builder/offline_wheel/build_offline_bundle.sh
```

The bundle is written to:

```text
offline_dist/lmdeploy-<version>-py310-cu124-sm70-sm75/
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

If a dependency has no binary wheel for Python 3.10 on your platform, the build
script fails by default. That is intentional for offline deployment. You can set
`ONLY_BINARY=0` only if you also plan to support source builds on the offline
target, which is not recommended for production rollout.

## Install Offline

Copy the whole bundle directory to the target machine, then run:

```bash
PYTHON_BIN=python3.10 bash install_offline.sh
```

The install script uses only:

```bash
--no-index --find-links wheelhouse
```

It installs the PyTorch CUDA 12.4 wheel, LMDeploy runtime dependencies,
Sanic/gRPC serving dependencies, then the local LMDeploy wheel. The bundle pins
`torch==2.6.0+cu124` and
`torchvision==0.21.0+cu124` to avoid accidentally resolving a non-CUDA-12.4
PyTorch wheel.

## Verify Runtime

`install_offline.sh` runs `verify_install.py` automatically. You can also run it
manually:

```bash
python3.10 verify_install.py
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
