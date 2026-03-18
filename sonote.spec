# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os

project_root = Path(r"C:\Users\SSAFY\Desktop\Projects\tools\sonote")
build_mode = os.environ.get("SONOTE_BUILD_MODE", "onedir")
is_onefile = build_mode == "onefile"

hiddenimports = [
    "pystray._win32",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "watchdog.observers.winapi"
]

import sounddevice
_sd_dir = Path(sounddevice.__file__).parent / "_sounddevice_data"

import faster_whisper as _fw
_fw_assets = Path(_fw.__file__).parent / "assets"

# CUDA DLL 경로 (nvidia-cublas, nvidia-cudnn)
_nvidia_base = Path(sounddevice.__file__).parent / "nvidia"
_cublas_bin = _nvidia_base / "cublas" / "bin"
_cudnn_bin = _nvidia_base / "cudnn" / "bin"

datas = [
    (str(project_root / "static"), "static"),
    (str(_sd_dir), "_sounddevice_data"),
    (str(_fw_assets), "faster_whisper/assets"),
]

# CUDA DLL을 binaries로 번들
binaries = []
for cuda_dir in [_cublas_bin, _cudnn_bin]:
    if cuda_dir.exists():
        for dll in cuda_dir.glob("*.dll"):
            binaries.append((str(dll), "."))

a = Analysis(
    [str(project_root / "src" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "cefpython3", "webview", "pywebview"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if is_onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="sonote",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="sonote",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="sonote",
    )
