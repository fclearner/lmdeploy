#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def require_python() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"python 3.10 is required, got {sys.version.split()[0]}")


def import_required(name: str):
    module = importlib.import_module(name)
    print(f"ok import {name}: {getattr(module, '__file__', 'built-in')}")
    return module


def find_extension(lib_dir: Path, prefix: str) -> Path:
    candidates = sorted(lib_dir.glob(f"{prefix}*.so"))
    if not candidates:
        raise RuntimeError(f"cannot find {prefix}*.so under {lib_dir}")
    return candidates[0]


def verify_cuda_arches(turbomind_so: Path) -> None:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        print("skip cuda arch check: cuobjdump not found")
        return
    output = subprocess.check_output([cuobjdump, "--list-elf", str(turbomind_so)], text=True)
    missing = [arch for arch in ("sm_70", "sm_75") if arch not in output]
    if missing:
        raise RuntimeError(f"{turbomind_so} does not contain expected CUDA archs: {missing}")
    print("ok cuda archs: sm_70 sm_75")


def main() -> None:
    require_python()
    lmdeploy = import_required("lmdeploy")
    for name in (
        "aiohttp",
        "grpc",
        "google.protobuf",
        "sanic",
        "torch",
        "duplex.server",
        "turbomind_grpc_client",
        "turbomind_service_core",
        "grpc_turbomind_server",
    ):
        import_required(name)

    lib_dir = Path(lmdeploy.__file__).resolve().parent / "lib"
    sys.path.append(str(lib_dir))
    turbomind_so = find_extension(lib_dir, "_turbomind")
    find_extension(lib_dir, "_xgrammar")
    import_required("_turbomind")
    import_required("_xgrammar")
    verify_cuda_arches(turbomind_so)

    torch = importlib.import_module("torch")
    print(f"torch={torch.__version__} torch_cuda={getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    else:
        print("cuda_device_count=0")
    print("lmdeploy offline install verification passed")


if __name__ == "__main__":
    main()
