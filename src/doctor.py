"""환경 진단 모듈 — sonote 실행에 필요한 도구/라이브러리/환경변수 점검.

사용법:
    from src.doctor import run_diagnosis, print_diagnosis

    result = run_diagnosis()
    print_diagnosis(result)            # 컬러 터미널 출력
    print_diagnosis(result, use_json=True)  # JSON 출력
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


__all__ = [
    "run_diagnosis",
    "print_diagnosis",
]

# ---------------------------------------------------------------------------
# ANSI 컬러 코드 (직접 정의 — 외부 의존성 없음)
# ---------------------------------------------------------------------------
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# 진단 항목 정의
# ---------------------------------------------------------------------------

# 각 항목: (key, 표시명, 체크 함수, 설치 힌트)
# 체크 함수는 (status, detail) 튜플 반환
# status: "ok" | "missing" | "warning"


def _check_ffmpeg() -> tuple[str, str]:
    """ffmpeg 설치 여부 확인."""
    path = shutil.which("ffmpeg")
    if path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # 첫 줄에서 버전 추출
            first_line = result.stdout.split("\n")[0] if result.stdout else ""
            return "ok", first_line.strip()
        except Exception:
            return "ok", path
    return "missing", ""


def _check_cuda() -> tuple[str, str]:
    """CUDA GPU 가용 여부 확인."""
    try:
        from .runtime_env import detect_device

        device, compute_type = detect_device()
        if device == "cuda":
            return "ok", f"device={device}, compute_type={compute_type}"
        return "warning", f"CPU 모드 (device={device}, compute_type={compute_type})"
    except Exception as e:
        return "warning", f"감지 실패: {e}"


def _check_pyannote() -> tuple[str, str]:
    """pyannote-audio (화자 분리) 설치 여부 확인."""
    try:
        from .diarize import SpeakerDiarizer

        if SpeakerDiarizer.is_available():
            return "ok", "pyannote-audio 사용 가능"
        return "missing", "pyannote-audio 미설치 또는 torch 누락"
    except ImportError:
        return "missing", "pyannote-audio / torch 미설치"
    except Exception as e:
        return "warning", str(e)


def _check_yt_dlp() -> tuple[str, str]:
    """yt-dlp 설치 여부 확인."""
    path = shutil.which("yt-dlp")
    if path:
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = result.stdout.strip()
            return "ok", f"v{version}" if version else path
        except Exception:
            return "ok", path
    return "missing", ""


def _check_ollama() -> tuple[str, str]:
    """Ollama 서버 + 모델 가용 여부 확인."""
    path = shutil.which("ollama")
    if not path:
        return "missing", ""
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if models:
                return "ok", f"서버 실행 중, 모델: {', '.join(models[:5])}"
            return "warning", "서버 실행 중이나 설치된 모델 없음"
        return "warning", f"서버 응답 이상 (HTTP {resp.status_code})"
    except ImportError:
        return "warning", "httpx 미설치 — 서버 연결 확인 불가"
    except Exception:
        return "warning", "CLI 설치됨, 서버 미실행"


def _check_codex_cli() -> tuple[str, str]:
    """Codex CLI 설치 여부 확인."""
    path = shutil.which("codex")
    if path:
        return "ok", path
    return "missing", ""


def _check_gemini_cli() -> tuple[str, str]:
    """Gemini CLI 설치 여부 확인."""
    path = shutil.which("gemini")
    if path:
        return "ok", path
    return "missing", ""


def _check_tray() -> tuple[str, str]:
    """pystray + Pillow (시스템 트레이) 설치 여부 확인."""
    try:
        from .tray import is_available as _tray_ok

        if _tray_ok():
            return "ok", "pystray + Pillow 사용 가능"
        return "missing", "pystray 또는 Pillow 미설치"
    except ImportError:
        return "missing", "pystray 또는 Pillow 미설치"
    except Exception as e:
        return "warning", str(e)


def _check_hf_token() -> tuple[str, str]:
    """HF_TOKEN 환경변수 설정 여부 확인."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        # 토큰 일부만 표시 (보안)
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        return "ok", f"설정됨 ({masked})"
    return "warning", "미설정 — 화자 분리 시 필요"


def _check_meeting_api_key() -> tuple[str, str]:
    """MEETING_API_KEY 환경변수 설정 여부 확인."""
    key = os.environ.get("MEETING_API_KEY", "")
    if key:
        return "ok", "설정됨"
    return "warning", "미설정 — 웹 API 인증 비활성"


# 진단 항목 레지스트리
_CHECKS: list[tuple[str, str, Any, str]] = [
    (
        "ffmpeg",
        "FFmpeg",
        _check_ffmpeg,
        "pip: 불가 / https://ffmpeg.org/download.html 또는 choco install ffmpeg",
    ),
    (
        "cuda",
        "CUDA (GPU 가속)",
        _check_cuda,
        "https://developer.nvidia.com/cuda-downloads (CPU 모드에서도 동작)",
    ),
    (
        "pyannote_audio",
        "pyannote-audio (화자 분리)",
        _check_pyannote,
        "pip install pyannote-audio torch",
    ),
    (
        "yt_dlp",
        "yt-dlp (스트림 다운로드)",
        _check_yt_dlp,
        "pip install yt-dlp",
    ),
    (
        "ollama",
        "Ollama (로컬 LLM)",
        _check_ollama,
        "https://ollama.com/download → ollama pull gemma3:27b",
    ),
    (
        "codex_cli",
        "Codex CLI (STT 교정)",
        _check_codex_cli,
        "npm install -g @openai/codex",
    ),
    (
        "gemini_cli",
        "Gemini CLI (요약 생성)",
        _check_gemini_cli,
        "npm install -g @anthropic-ai/gemini (또는 해당 CLI)",
    ),
    (
        "pystray_pillow",
        "pystray + Pillow (시스템 트레이)",
        _check_tray,
        "pip install pystray Pillow",
    ),
    (
        "hf_token",
        "HF_TOKEN (HuggingFace 토큰)",
        _check_hf_token,
        "export HF_TOKEN=hf_xxx  # PowerShell: $env:HF_TOKEN='hf_xxx'",
    ),
    (
        "meeting_api_key",
        "MEETING_API_KEY (웹 API 인증)",
        _check_meeting_api_key,
        "export MEETING_API_KEY=secret  # PowerShell: $env:MEETING_API_KEY='secret'",
    ),
]


# ---------------------------------------------------------------------------
# 폴백 체인 진단
# ---------------------------------------------------------------------------

def _build_fallback_chains(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """STT 교정/요약 폴백 체인 구성."""
    chains: list[dict[str, Any]] = []

    # STT 교정 폴백: Codex → Gemini → Ollama
    stt_chain: list[dict[str, str]] = [
        {
            "tool": "Codex",
            "status": items.get("codex_cli", {}).get("status", "missing"),
        },
        {
            "tool": "Gemini",
            "status": items.get("gemini_cli", {}).get("status", "missing"),
        },
        {
            "tool": "Ollama",
            "status": items.get("ollama", {}).get("status", "missing"),
        },
    ]
    chains.append({
        "name": "STT 교정",
        "chain": stt_chain,
    })

    # 요약 폴백: Gemini → Ollama
    summary_chain: list[dict[str, str]] = [
        {
            "tool": "Gemini",
            "status": items.get("gemini_cli", {}).get("status", "missing"),
        },
        {
            "tool": "Ollama",
            "status": items.get("ollama", {}).get("status", "missing"),
        },
    ]
    chains.append({
        "name": "요약 생성",
        "chain": summary_chain,
    })

    return chains


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def run_diagnosis() -> dict[str, Any]:
    """전체 환경 스캔을 수행하고 결과를 반환한다.

    Returns:
        {
            "items": {
                "ffmpeg": {"status": "ok", "detail": "...", "install_hint": "..."},
                ...
            },
            "fallback_chains": [...],
            "summary": {"ok": N, "warning": N, "missing": N},
        }
    """
    items: dict[str, dict[str, Any]] = {}

    for key, label, check_fn, hint in _CHECKS:
        try:
            status, detail = check_fn()
        except Exception as e:
            status, detail = "warning", f"점검 중 오류: {e}"

        items[key] = {
            "label": label,
            "status": status,
            "detail": detail,
            "install_hint": hint,
        }

    # 요약 카운터
    summary: dict[str, int] = {"ok": 0, "warning": 0, "missing": 0}
    for item in items.values():
        s = item["status"]
        if s in summary:
            summary[s] += 1

    # 폴백 체인
    fallback_chains = _build_fallback_chains(items)

    return {
        "items": items,
        "fallback_chains": fallback_chains,
        "summary": summary,
    }


def print_diagnosis(result: dict[str, Any], use_json: bool = False) -> None:
    """진단 결과를 터미널에 출력한다.

    Args:
        result: run_diagnosis() 반환값
        use_json: True면 JSON으로 출력
    """
    if use_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    items: dict[str, dict[str, Any]] = result.get("items", {})
    chains: list[dict[str, Any]] = result.get("fallback_chains", [])
    summary: dict[str, int] = result.get("summary", {})

    # 상태 아이콘
    _STATUS_ICON: dict[str, str] = {
        "ok": f"{_GREEN}\u2714{_RESET}",       # ✔
        "warning": f"{_YELLOW}\u26A0{_RESET}",  # ⚠
        "missing": f"{_RED}\u2718{_RESET}",     # ✘
    }

    print()
    print(f"{_BOLD}{_CYAN}{'=' * 60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  sonote 환경 진단{_RESET}")
    print(f"{_BOLD}{_CYAN}{'=' * 60}{_RESET}")
    print()

    # 항목별 출력
    for key, info in items.items():
        status = info["status"]
        icon = _STATUS_ICON.get(status, "?")
        label = info["label"]
        detail = info["detail"]
        hint = info["install_hint"]

        detail_str = f"  {_DIM}{detail}{_RESET}" if detail else ""
        print(f"  {icon} {label}{detail_str}")

        if status == "missing" and hint:
            print(f"    {_DIM}설치: {hint}{_RESET}")

    # 폴백 체인 표시
    print()
    print(f"{_BOLD}  폴백 체인:{_RESET}")
    for chain_info in chains:
        name = chain_info["name"]
        chain = chain_info["chain"]

        parts: list[str] = []
        for step in chain:
            tool = step["tool"]
            s = step["status"]
            if s == "ok":
                parts.append(f"{_GREEN}{tool} \u2714{_RESET}")
            elif s == "warning":
                parts.append(f"{_YELLOW}{tool} \u26A0{_RESET}")
            else:
                parts.append(f"{_RED}{tool} \u2718{_RESET}")

        chain_str = f" {_DIM}\u2192{_RESET} ".join(parts)
        print(f"    {name}: {chain_str}")

    # 요약
    print()
    ok_count = summary.get("ok", 0)
    warn_count = summary.get("warning", 0)
    miss_count = summary.get("missing", 0)
    total = ok_count + warn_count + miss_count

    print(
        f"  {_BOLD}결과:{_RESET} "
        f"{_GREEN}{ok_count}{_RESET}/{total} 정상"
    )
    if warn_count:
        print(f"         {_YELLOW}{warn_count}{_RESET}개 경고")
    if miss_count:
        print(f"         {_RED}{miss_count}{_RESET}개 누락")
    print()
