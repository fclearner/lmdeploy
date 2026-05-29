import ast
from pathlib import Path

from setuptools import find_packages


def test_duplex_package_is_included():
    assert "duplex" in find_packages()


def test_grpc_gateway_root_modules_are_packaged():
    setup_tree = ast.parse(Path("setup.py").read_text())
    setup_call = next(
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
    py_modules = {item.value for item in keywords["py_modules"].elts}
    assert {
        "grpc_turbomind_server",
        "turbomind_grpc_client",
        "turbomind_grpc_protocol",
        "turbomind_service_core",
    } <= py_modules


def test_setup_forwards_cuda_architectures_to_cmake():
    setup_text = Path("setup.py").read_text()
    assert "get_cmake_cuda_architectures_option()" in setup_text
    assert "-DCMAKE_CUDA_ARCHITECTURES=" in setup_text
