"""런타임 초기화 헬퍼."""

from __future__ import annotations

import os
from pathlib import Path
import site

_NVIDIA_DLL_BOOTSTRAPPED = False


def bootstrap_nvidia_dll_path() -> None:
    """Windows CUDA 런타임 DLL 경로를 PATH에 1회만 추가한다."""
    global _NVIDIA_DLL_BOOTSTRAPPED

    if _NVIDIA_DLL_BOOTSTRAPPED:
        return

    for sp in site.getsitepackages():
        nvidia_base = Path(sp) / "nvidia"
        if not nvidia_base.exists():
            continue
        for sub in nvidia_base.iterdir():
            bin_dir = sub / "bin"
            if bin_dir.exists():
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    _NVIDIA_DLL_BOOTSTRAPPED = True


def detect_device() -> tuple[str, str]:
    """GPU 사용 가능 여부 감지, (device, compute_type) 반환."""
    try:
        bootstrap_nvidia_dll_path()
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"
