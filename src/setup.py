"""sonote setup — 대화형/자동 환경 설정 + 의존성 자동 설치.

사용법:
    sonote setup              # 대화형 가이드
    sonote setup --all        # 모든 선택적 의존성 자동 설치
    sonote setup --gpu        # CUDA 환경 설정
    sonote setup --diarize    # 화자 분리 (pyannote + torch)
    sonote setup --fix        # doctor 기반 누락 항목 자동 수정
    sonote setup --json       # JSON 결과 출력
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["run_setup"]

# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# 설치 액션 정의
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"


def _pip_install(*packages: str, timeout: int = 300) -> tuple[bool, str]:
    """pip install 실행."""
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "타임아웃"
    except Exception as e:
        return False, str(e)


def _run_cmd(cmd: list[str], timeout: int = 120) -> tuple[bool, str]:
    """외부 명령 실행."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            shell=_IS_WINDOWS,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip() or result.stdout.strip()
    except FileNotFoundError:
        return False, f"명령어를 찾을 수 없음: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, "타임아웃"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# 개별 설치/수정 함수
# ---------------------------------------------------------------------------

def _install_ffmpeg() -> tuple[bool, str]:
    """ffmpeg 자동 설치."""
    if shutil.which("ffmpeg"):
        return True, "이미 설치됨"

    if _IS_WINDOWS:
        # winget 시도
        if shutil.which("winget"):
            ok, msg = _run_cmd(
                ["winget", "install", "Gyan.FFmpeg", "-e", "--accept-source-agreements", "--accept-package-agreements"],
                timeout=300,
            )
            if ok:
                return True, "winget으로 설치 완료"
            # choco 시도
        if shutil.which("choco"):
            ok, msg = _run_cmd(["choco", "install", "ffmpeg", "-y"], timeout=300)
            if ok:
                return True, "choco로 설치 완료"
        return False, "winget/choco 모두 실패 — https://ffmpeg.org/download.html 에서 수동 설치"
    else:
        # macOS
        if shutil.which("brew"):
            ok, msg = _run_cmd(["brew", "install", "ffmpeg"], timeout=300)
            if ok:
                return True, "brew로 설치 완료"
        # Linux
        if shutil.which("apt-get"):
            ok, msg = _run_cmd(["sudo", "apt-get", "install", "-y", "ffmpeg"], timeout=300)
            if ok:
                return True, "apt-get으로 설치 완료"
        return False, "패키지 매니저 미감지 — 수동 설치 필요"


def _setup_cuda_dll_path() -> tuple[bool, str]:
    """CUDA DLL 경로 패치 (Windows)."""
    if not _IS_WINDOWS:
        return True, "Windows 전용 — 건너뜀"

    try:
        from .runtime_env import bootstrap_nvidia_dll_path, detect_device
        bootstrap_nvidia_dll_path()
        device, compute_type = detect_device()
        if device == "cuda":
            return True, f"CUDA 정상 (device={device}, compute={compute_type})"
        return True, f"CPU 모드 (CUDA 미감지, device={device})"
    except Exception as e:
        return False, f"CUDA 감지 실패: {e}"


def _detect_torch_index_url() -> str | None:
    """CUDA가 사용 가능하면 PyTorch CUDA 인덱스 URL 반환, 아니면 None(CPU)."""
    # 이미 설치된 torch에서 CUDA 빌드 감지
    try:
        import torch
        if torch.cuda.is_available():
            # 설치된 torch의 CUDA 버전에서 인덱스 추론
            cuda_ver = torch.version.cuda  # e.g. "12.6"
            if cuda_ver:
                tag = "cu" + cuda_ver.replace(".", "")  # "cu126"
                return f"https://download.pytorch.org/whl/{tag}"
        return None
    except ImportError:
        pass

    # torch 미설치 시 nvidia-smi로 CUDA 존재 여부 확인
    if shutil.which("nvidia-smi"):
        return "https://download.pytorch.org/whl/cu126"
    return None


def _check_torch_consistency() -> tuple[bool, str]:
    """torch/torchaudio/torchvision 빌드 버전(CUDA suffix)이 일치하는지 확인."""
    try:
        import torch
        torch_ver = torch.__version__  # e.g. "2.8.0+cu126"
    except ImportError:
        return False, "torch 미설치"

    suffix = ""
    if "+" in torch_ver:
        suffix = torch_ver.split("+")[1]  # "cu126" or "cpu"

    mismatches: list[str] = []
    for pkg_name in ("torchaudio", "torchvision"):
        try:
            mod = __import__(pkg_name)
            pkg_ver = mod.__version__
            pkg_suffix = pkg_ver.split("+")[1] if "+" in pkg_ver else ""
            if pkg_suffix != suffix:
                mismatches.append(f"{pkg_name} {pkg_ver} (expected +{suffix})")
        except ImportError:
            mismatches.append(f"{pkg_name} 미설치")

    if mismatches:
        return False, f"torch {torch_ver}와 불일치: {', '.join(mismatches)}"
    return True, f"torch 스택 일관됨 ({torch_ver})"


def _install_diarize() -> tuple[bool, str]:
    """화자 분리 의존성 설치 (torch + torchaudio + pyannote-audio)."""
    index_url = _detect_torch_index_url()
    torch_packages = ["torch", "torchaudio", "torchvision"]

    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        torch_ok = False

    if not torch_ok:
        # 신규 설치: CUDA 인덱스 URL 포함
        args = torch_packages.copy()
        if index_url:
            args = ["--index-url", index_url] + args
        ok, msg = _pip_install(*args, timeout=600)
        if not ok:
            return False, f"torch 설치 실패: {msg}"
    else:
        # 이미 설치됨 → 빌드 일관성 체크
        consistent, detail = _check_torch_consistency()
        if not consistent:
            # 불일치 시 재설치
            if not index_url:
                index_url = _detect_torch_index_url()
            args = torch_packages.copy()
            if index_url:
                args = ["--index-url", index_url] + args
            ok, msg = _pip_install(*args, timeout=600)
            if not ok:
                return False, f"torch 스택 재설치 실패: {msg} (원인: {detail})"

    # pyannote-audio
    try:
        import pyannote.audio  # noqa: F401
        return True, "이미 설치됨"
    except ImportError:
        pass

    ok, msg = _pip_install("pyannote-audio", timeout=600)
    if ok:
        return True, "pyannote-audio 설치 완료"
    return False, f"pyannote-audio 설치 실패: {msg}"


def _setup_hf_token(token: str | None = None) -> tuple[bool, str]:
    """HuggingFace 토큰 설정."""
    existing = os.environ.get("HF_TOKEN", "")
    if existing:
        masked = existing[:4] + "..." + existing[-4:] if len(existing) > 8 else "***"
        return True, f"이미 설정됨 ({masked})"

    if token:
        os.environ["HF_TOKEN"] = token
        # 영속화: .env 파일에 저장
        env_path = Path.home() / ".sonote" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)

        # 기존 .env 내용 읽기
        env_lines: list[str] = []
        if env_path.exists():
            env_lines = env_path.read_text(encoding="utf-8").splitlines()

        # HF_TOKEN 줄 교체 또는 추가
        found = False
        for i, line in enumerate(env_lines):
            if line.startswith("HF_TOKEN="):
                env_lines[i] = f"HF_TOKEN={token}"
                found = True
                break
        if not found:
            env_lines.append(f"HF_TOKEN={token}")

        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        return True, f"설정 완료 → {env_path}"

    return False, "HF_TOKEN 미설정 — sonote setup --hf-token hf_xxx 로 설정"


def _install_desktop_extras() -> tuple[bool, str]:
    """데스크톱 extras 설치 (pystray + Pillow)."""
    ok, msg = _pip_install("pystray", "Pillow")
    if ok:
        return True, "pystray + Pillow 설치 완료"
    return False, f"설치 실패: {msg}"


def _install_media_extras() -> tuple[bool, str]:
    """미디어 제어 extras 설치 (pycaw — Windows 전용)."""
    if not _IS_WINDOWS:
        return True, "Windows 전용 — 건너뜀"
    ok, msg = _pip_install("pycaw")
    if ok:
        return True, "pycaw 설치 완료"
    return False, f"설치 실패: {msg}"


def _verify_model() -> tuple[bool, str]:
    """기본 Whisper 모델 접근 가능 여부 확인."""
    model_id = "tellang/whisper-medium-ko-ct2"
    try:
        from faster_whisper.utils import download_model

        cache_dir = download_model(model_id, cache_dir=None)
        return True, f"모델 캐시: {cache_dir}"
    except Exception:
        # faster_whisper 내부 다운로드 실패 시 직접 체크
        try:
            import httpx
            resp = httpx.head(
                f"https://huggingface.co/{model_id}/resolve/main/config.json",
                follow_redirects=True,
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "모델 접근 가능 (첫 실행 시 자동 다운로드)"
            return False, f"모델 접근 불가 (HTTP {resp.status_code}) — HF 토큰 또는 모델 공개 설정 확인"
        except Exception as e:
            return False, f"모델 확인 실패: {e}"


# ---------------------------------------------------------------------------
# 설치 항목 레지스트리
# ---------------------------------------------------------------------------

_SETUP_ITEMS: list[dict[str, Any]] = [
    {
        "key": "ffmpeg",
        "label": "FFmpeg (오디오/비디오 처리)",
        "installer": _install_ffmpeg,
        "category": "core",
        "required": True,
    },
    {
        "key": "cuda",
        "label": "CUDA DLL 경로 설정",
        "installer": _setup_cuda_dll_path,
        "category": "gpu",
        "required": False,
    },
    {
        "key": "model",
        "label": "Whisper 모델 (tellang/whisper-medium-ko-ct2)",
        "installer": _verify_model,
        "category": "core",
        "required": True,
    },
    {
        "key": "diarize",
        "label": "화자 분리 (torch + pyannote-audio)",
        "installer": _install_diarize,
        "category": "diarize",
        "required": False,
    },
    {
        "key": "hf_token",
        "label": "HuggingFace 토큰",
        "installer": lambda: _setup_hf_token(),
        "category": "diarize",
        "required": False,
    },
    {
        "key": "desktop",
        "label": "데스크톱 트레이 (pystray + Pillow)",
        "installer": _install_desktop_extras,
        "category": "desktop",
        "required": False,
    },
    {
        "key": "media",
        "label": "미디어 제어 (pycaw)",
        "installer": _install_media_extras,
        "category": "desktop",
        "required": False,
    },
]


# ---------------------------------------------------------------------------
# 메인 setup 함수
# ---------------------------------------------------------------------------

def run_setup(
    *,
    all_extras: bool = False,
    gpu: bool = False,
    diarize: bool = False,
    desktop: bool = False,
    fix: bool = False,
    hf_token: str | None = None,
    use_json: bool = False,
) -> dict[str, Any]:
    """sonote setup 실행.

    Args:
        all_extras: 모든 선택적 의존성 설치
        gpu: CUDA 환경 설정
        diarize: 화자 분리 의존성 설치
        desktop: 데스크톱 extras 설치
        fix: doctor 결과 기반 누락 항목만 수정
        hf_token: HuggingFace 토큰 직접 전달
        use_json: JSON 출력

    Returns:
        {"items": [...], "summary": {"ok": N, "failed": N, "skipped": N}}
    """
    # HF 토큰이 전달된 경우 먼저 설정
    if hf_token:
        _setup_hf_token(hf_token)

    # .env 파일에서 HF_TOKEN 로드 (있으면)
    _load_env_file()

    # 실행할 카테고리 결정
    categories: set[str] = {"core"}  # core는 항상 실행
    if all_extras:
        categories = {"core", "gpu", "diarize", "desktop"}
    if gpu:
        categories.add("gpu")
    if diarize:
        categories.add("diarize")
    if desktop:
        categories.add("desktop")

    # fix 모드: doctor 진단 후 missing 항목만 설치
    if fix:
        from .doctor import run_diagnosis
        diag = run_diagnosis()
        missing_keys = {
            k for k, v in diag["items"].items()
            if v["status"] == "missing"
        }
        # doctor key → setup key 매핑
        _doctor_to_setup = {
            "ffmpeg": "ffmpeg",
            "cuda": "cuda",
            "pyannote_audio": "diarize",
            "hf_token": "hf_token",
            "pystray_pillow": "desktop",
        }
        fix_keys = {_doctor_to_setup.get(k, k) for k in missing_keys}
        # fix 모드에서는 모든 카테고리 허용
        categories = {"core", "gpu", "diarize", "desktop"}
    else:
        fix_keys = None

    results: list[dict[str, Any]] = []

    if not use_json:
        print()
        print(f"{_BOLD}{_CYAN}{'=' * 60}{_RESET}")
        print(f"{_BOLD}{_CYAN}  sonote setup{_RESET}")
        print(f"{_BOLD}{_CYAN}{'=' * 60}{_RESET}")
        print()

    for item in _SETUP_ITEMS:
        key = item["key"]
        label = item["label"]
        category = item["category"]
        installer = item["installer"]

        # 카테고리 필터
        if category not in categories:
            results.append({"key": key, "label": label, "status": "skipped", "detail": "카테고리 미선택"})
            continue

        # fix 모드: 누락 항목만
        if fix_keys is not None and key not in fix_keys:
            results.append({"key": key, "label": label, "status": "skipped", "detail": "정상 — 수정 불필요"})
            continue

        if not use_json:
            print(f"  {_CYAN}▶{_RESET} {label}...", end=" ", flush=True)

        try:
            ok, detail = installer()
            status = "ok" if ok else "failed"
        except Exception as e:
            status = "failed"
            detail = str(e)

        results.append({"key": key, "label": label, "status": status, "detail": detail})

        if not use_json:
            if status == "ok":
                print(f"{_GREEN}✔{_RESET} {_DIM}{detail}{_RESET}")
            else:
                print(f"{_RED}✘{_RESET} {detail}")

    # 요약
    summary = {
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
    }

    output = {"items": results, "summary": summary}

    if use_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print()
        ok_count = summary["ok"]
        fail_count = summary["failed"]
        total = ok_count + fail_count
        print(
            f"  {_BOLD}결과:{_RESET} "
            f"{_GREEN}{ok_count}{_RESET}/{total} 성공"
        )
        if fail_count:
            print(f"         {_RED}{fail_count}{_RESET}개 실패")
        print()

        if fail_count:
            print(f"  {_YELLOW}실패 항목을 수동으로 설치하거나 'sonote doctor'로 상태를 확인하세요.{_RESET}")
            print()

    return output


def _load_env_file() -> None:
    """~/.sonote/.env 파일에서 환경변수 로드."""
    env_path = Path.home() / ".sonote" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
