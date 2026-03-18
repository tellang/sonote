"""런타임 초기화 헬퍼."""

from __future__ import annotations

import os
from pathlib import Path
import site

from .runtime.context import get_project_root, is_frozen

_NVIDIA_DLL_BOOTSTRAPPED = False


def bootstrap_nvidia_dll_path() -> None:
    """Windows CUDA 런타임 DLL 경로를 PATH에 1회만 추가한다.

    PyInstaller frozen 환경에서는 exe 디렉토리(_internal/)에
    DLL이 번들되므로 해당 경로를 우선 추가한다.
    """
    global _NVIDIA_DLL_BOOTSTRAPPED

    if _NVIDIA_DLL_BOOTSTRAPPED:
        return

    # frozen exe: DLL이 exe와 같은 디렉토리에 번들됨
    if is_frozen():
        exe_root = get_project_root()
        exe_dir = str(exe_root)
        internal_dir = str(exe_root / "_internal")
        for d in [exe_dir, internal_dir]:
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    else:
        # 개발 환경: site-packages/nvidia/*/bin 경로 추가
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


def get_available_vram() -> int:
    """사용 가능한 VRAM(bytes) 반환. GPU 미감지 시 0.

    torch 없이도 동작하도록 nvidia-smi → pynvml → torch 순으로 시도.
    """
    # 1차: nvidia-smi (가장 가볍고 확실)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            free_mib = int(result.stdout.strip().split("\n")[0])
            return free_mib * 1024 * 1024
    except Exception:
        pass

    # 2차: torch (설치되어 있으면)
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            return int(max(free_bytes, 0))
    except Exception:
        pass

    return 0


def calculate_bon_workers(
    model_size_mb: int = 800,
    reserve_mb: int = 1500,
    max_workers: int = 4,
) -> int:
    """VRAM 기반 최적 BoN worker 수 계산. 최소 1."""
    safe_model_size_mb = max(int(model_size_mb), 1)
    safe_reserve_mb = max(int(reserve_mb), 0)
    safe_max_workers = max(int(max_workers), 1)

    available_vram_mb = get_available_vram() // (1024 * 1024)
    usable_vram_mb = max(available_vram_mb - safe_reserve_mb, 0)
    workers_by_vram = usable_vram_mb // safe_model_size_mb

    return max(1, min(int(workers_by_vram), safe_max_workers))
