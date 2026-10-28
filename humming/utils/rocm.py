"""ROCm toolchain discovery.

Everything here is safe to import on CUDA builds: nothing touches the HIP
runtime until a function is called, and ``is_rocm_build`` only inspects the
torch build metadata.
"""

import functools
import os
import re
import shutil
from pathlib import Path

import torch


@functools.cache
def is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


@functools.cache
def get_rocm_path() -> Path:
    for variable in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        value = os.environ.get(variable)
        if value and Path(value).is_dir():
            return Path(value)
    hipcc_path = shutil.which("hipcc")
    if hipcc_path is not None:
        return Path(hipcc_path).resolve().parents[1]
    default_path = Path("/opt/rocm")
    if default_path.is_dir():
        return default_path
    raise RuntimeError("ROCm installation was not found; set ROCM_PATH")


@functools.cache
def get_rocm_version() -> str:
    rocm_path = get_rocm_path()
    version_file = rocm_path / ".info" / "version"
    if version_file.is_file():
        return version_file.read_text().strip()
    hip_version_header = rocm_path / "include" / "hip" / "hip_version.h"
    if hip_version_header.is_file():
        header_text = hip_version_header.read_text()
        version_parts = [
            re.search(rf"#define HIP_VERSION_{part}\s+(\d+)", header_text)
            for part in ("MAJOR", "MINOR", "PATCH")
        ]
        if all(version_parts):
            return ".".join(match.group(1) for match in version_parts if match is not None)
    return "unknown"


def get_rocm_include_path() -> Path:
    return get_rocm_path() / "include"


def get_rocm_library_path() -> Path:
    return get_rocm_path() / "lib"


def get_hipcc_path() -> str:
    hipcc_path = get_rocm_path() / "bin" / "hipcc"
    if hipcc_path.is_file():
        return str(hipcc_path)
    found_path = shutil.which("hipcc")
    if found_path is None:
        raise RuntimeError("hipcc was not found; set ROCM_PATH")
    return found_path


def get_amdhip_library_path() -> str:
    library_path = get_rocm_library_path() / "libamdhip64.so"
    if library_path.exists():
        return str(library_path)
    return "libamdhip64.so"


def normalize_gcn_arch(gcn_arch_name: str) -> str:
    """Strip target features: ``gfx950:sramecc+:xnack-`` -> ``gfx950``."""
    return gcn_arch_name.split(":", 1)[0]
