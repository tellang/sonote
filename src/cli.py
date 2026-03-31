"""
media-transcriber CLI 진입점
YouTube 라이브 스트림/회의 음성 전사 도구
"""
import argparse
import json as _json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from src.db import save_profile, get_profile, list_profiles, update_captured, delete_profile
from src.domain_keywords import DEFAULT_DOMAIN_HINT
from .validate import sanitize_input, ValidationError
from .cookies import (
    check_cookies_file,
    export_chrome_cookies_to_netscape,
    resolve_cookies_path,
)
from src.paths import (
    transcripts_dir, audio_dir,
    speakers_json_path,
)
from .download import download_live_audio

# --- Structured exit codes ---
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_ARG_ERROR = 2
EXIT_PREFLIGHT_FAIL = 5
EXIT_MODEL_ERROR = 3
EXIT_NOT_FOUND = 4

# --- EXIT_* → reason enum 매핑 (AI 에이전트용 기계 판독 가능 에러 분류) ---
_EXIT_REASON_MAP: dict[int, str] = {
    EXIT_ERROR: "runtimeError",
    EXIT_ARG_ERROR: "argError",
    EXIT_MODEL_ERROR: "modelError",
    EXIT_NOT_FOUND: "notFound",
    EXIT_PREFLIGHT_FAIL: "preflightFail",
}

# --- code 문자열 → reason enum 매핑 (json_output의 code 파라미터용) ---
_CODE_REASON_MAP: dict[str, str] = {
    "ERROR": "runtimeError",
    "ARG_ERROR": "argError",
    "MODEL_ERROR": "modelError",
    "NOT_FOUND": "notFound",
    "PREFLIGHT_FAIL": "preflightFail",
}

_BETA_ENV_KEY = "SONOTE_BETA"
_BETA_MODEL_ID = "tellang/whisper-medium-ko-ct2"
# v2.2.0: seastar105/whisper-medium-ko-zeroth → CT2 변환 → tellang/whisper-medium-ko-ct2
_DEFAULT_MODEL_ID = "tellang/whisper-medium-ko-ct2"


def _is_beta_mode_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "beta", False) or os.getenv(_BETA_ENV_KEY) == "1")


def _has_explicit_model_arg(argv: list[str]) -> bool:
    for token in argv:
        if token in ("-m", "--model"):
            return True
        if token.startswith("--model="):
            return True
    return False


def _apply_beta_mode(args: argparse.Namespace, argv: list[str]) -> bool:
    beta_mode = _is_beta_mode_enabled(args)
    if beta_mode:
        os.environ[_BETA_ENV_KEY] = "1"

    try:
        from .postprocess import set_beta_mode

        set_beta_mode(beta_mode)
    except Exception:
        pass

    if beta_mode and hasattr(args, "model") and not _has_explicit_model_arg(argv):
        setattr(args, "model", _BETA_MODEL_ID)

    return beta_mode


# ---------------------------------------------------------------------------
# 1) Dynamic Command Surface — 런타임 환경에 따라 기능 가용성 표시
# ---------------------------------------------------------------------------
def check_capabilities() -> dict[str, Any]:
    """런타임 환경을 점검하고 사용 가능한 기능 사전을 반환한다."""
    caps: dict[str, Any] = {
        "cuda_available": False,
        "ffmpeg_available": False,
        "diarize_available": False,
        "yt_dlp_available": False,
        "ollama_available": False,
        "tray_available": False,
    }

    # CUDA
    try:
        from .runtime_env import detect_device
        device, _ = detect_device()
        caps["cuda_available"] = device == "cuda"
    except Exception:
        pass

    # ffmpeg
    caps["ffmpeg_available"] = shutil.which("ffmpeg") is not None

    # diarize (pyannote-audio)
    try:
        from .diarize import SpeakerDiarizer
        caps["diarize_available"] = SpeakerDiarizer.is_available()
    except Exception:
        pass

    # yt-dlp
    caps["yt_dlp_available"] = shutil.which("yt-dlp") is not None

    # ollama
    caps["ollama_available"] = shutil.which("ollama") is not None

    # 시스템 트레이
    try:
        from .tray import is_available as _tray_ok
        caps["tray_available"] = _tray_ok()
    except Exception:
        pass

    return caps


# ---------------------------------------------------------------------------
# 2) Explicit Handoff [MANUAL] — 자동화 불가능한 단계 명시
# ---------------------------------------------------------------------------
def _collect_manual_steps(args: argparse.Namespace) -> list[dict[str, str]]:
    """현재 환경에서 사용자가 수동 수행해야 할 단계를 반환한다."""
    steps: list[dict[str, str]] = []

    command = getattr(args, "command", "")

    # HF_TOKEN 미설정 + 화자 분리 필요 커맨드
    if command in ("meeting", "enroll") and not os.environ.get("HF_TOKEN"):
        if command == "meeting" and not getattr(args, "no_diarize", False):
            steps.append({
                "tag": "MANUAL",
                "message": "HF_TOKEN을 환경변수에 설정하세요 (화자 분리에 필요)",
                "hint": "export HF_TOKEN=hf_xxx  # 또는 $env:HF_TOKEN='hf_xxx' (PowerShell)",
            })
        elif command == "enroll":
            steps.append({
                "tag": "MANUAL",
                "message": "HF_TOKEN을 환경변수에 설정하세요 (화자 임베딩 추출에 필요)",
                "hint": "export HF_TOKEN=hf_xxx",
            })

    # MEETING_API_KEY 미설정 시 보안 경고
    if command == "meeting" and not os.environ.get("MEETING_API_KEY"):
        steps.append({
            "tag": "MANUAL",
            "message": "MEETING_API_KEY를 설정하면 보호 엔드포인트 인증이 활성화됩니다",
            "hint": "export MEETING_API_KEY=secret",
        })

    # ffmpeg 미설치 + ffmpeg 필요 커맨드
    if command in ("live", "download", "smart", "probe", "scan", "auto", "detect", "map") and not shutil.which("ffmpeg"):
        steps.append({
            "tag": "MANUAL",
            "message": "ffmpeg를 설치하세요 (오디오 다운로드/변환에 필수)",
            "hint": "https://ffmpeg.org/download.html",
        })

    return steps


# ---------------------------------------------------------------------------
# 3) Preflight Diagnostics — 커맨드 실행 전 필수 조건 점검
# ---------------------------------------------------------------------------
def preflight_check(args: argparse.Namespace) -> dict[str, Any]:
    """커맨드 실행 전 필수 조건을 점검하고 결과를 반환한다.

    반환값:
        {"passed": bool, "checks": [...], "manual_steps": [...]}
    """
    checks: list[dict[str, Any]] = []
    command = getattr(args, "command", "")
    all_passed = True

    # ffmpeg 점검 (라이브/다운로드/분석 커맨드)
    if command in ("live", "download", "smart", "probe", "scan", "auto", "detect", "map", "meeting"):
        has_ffmpeg = shutil.which("ffmpeg") is not None
        checks.append({
            "name": "ffmpeg",
            "passed": has_ffmpeg,
            "required": command != "meeting",  # meeting에서는 선택
            "message": "ffmpeg 설치됨" if has_ffmpeg else "ffmpeg 미설치",
        })
        if not has_ffmpeg and command != "meeting":
            all_passed = False

    # CUDA/모델 점검 (전사 관련)
    if command in ("transcribe", "live", "smart", "auto", "meeting"):
        try:
            from .runtime_env import detect_device
            device, compute_type = detect_device()
            checks.append({
                "name": "gpu",
                "passed": True,
                "required": False,
                "message": f"가속기: {device}/{compute_type}",
            })
        except Exception as e:
            checks.append({
                "name": "gpu",
                "passed": True,  # CPU 폴백 가능
                "required": False,
                "message": f"GPU 감지 실패 (CPU 폴백): {e}",
            })

    # 오디오 파일 존재 점검 (경고만, 파이프라인에서 동적 생성 가능)
    if command == "transcribe":
        audio_path = Path(getattr(args, "audio", ""))
        exists = audio_path.exists()
        checks.append({
            "name": "audio_file",
            "passed": exists,
            "required": False,
            "message": f"오디오 파일 {'존재' if exists else '없음'}: {audio_path}",
        })

    # HF_TOKEN 점검 (화자 분리) — meeting에서는 경고만, enroll에서만 필수
    if command in ("meeting", "enroll"):
        has_token = bool(os.environ.get("HF_TOKEN"))
        needed = command == "enroll"  # enroll에서만 필수
        checks.append({
            "name": "hf_token",
            "passed": has_token,
            "required": needed,
            "message": "HF_TOKEN 설정됨" if has_token else "HF_TOKEN 미설정",
        })
        if not has_token and needed:
            all_passed = False

    # yt-dlp 점검 (YouTube 관련)
    if command in ("live", "download", "smart", "probe", "scan", "auto", "detect", "map"):
        has_ytdlp = shutil.which("yt-dlp") is not None
        checks.append({
            "name": "yt_dlp",
            "passed": has_ytdlp,
            "required": True,
            "message": "yt-dlp 설치됨" if has_ytdlp else "yt-dlp 미설치",
        })
        if not has_ytdlp:
            all_passed = False

    manual_steps = _collect_manual_steps(args)

    return {
        "passed": all_passed,
        "checks": checks,
        "manual_steps": manual_steps,
    }


# ---------------------------------------------------------------------------
# 4) Stream-friendly Output (NDJSON) 헬퍼
# ---------------------------------------------------------------------------
def _ndjson_line(event: str, **data: Any) -> str:
    """NDJSON 한 줄 생성 (각 줄이 독립 JSON 객체)."""
    payload: dict[str, Any] = {"event": event, **data}
    return _json.dumps(payload, ensure_ascii=False)


def json_output(
    status: str,
    command: str,
    *,
    data: dict[str, Any] | None = None,
    error: str | None = None,
    code: str | None = None,
) -> str:
    """--json 모드용 구조화 출력 생성.

    에러 시 ``reason`` enum을 자동 매핑하여 AI 에이전트가
    기계적으로 에러 유형을 분기할 수 있게 한다.
    """
    payload: dict[str, Any] = {"status": status, "command": command}
    if status == "success":
        payload["data"] = data or {}
    else:
        payload["error"] = error or "unknown error"
        if code:
            payload["code"] = code
            payload["reason"] = _CODE_REASON_MAP.get(code, "runtimeError")
        else:
            payload["reason"] = "runtimeError"
        if data:
            payload["data"] = data
    return _json.dumps(payload, ensure_ascii=False, indent=2)


def _filter_fields(data: dict[str, Any], fields: str | None) -> dict[str, Any]:
    """--fields 옵션으로 지정된 필드만 추출한다. None이면 전체 반환."""
    if not fields:
        return data
    keys = {k.strip() for k in fields.split(",") if k.strip()}
    if not keys:
        return data
    return {k: v for k, v in data.items() if k in keys}


def _validate_cli_inputs(args: argparse.Namespace) -> None:
    """CLI 입력값에 대해 제어문자/위험 유니코드/경로 탈출을 검증한다."""
    # URL 인수 검증
    url = getattr(args, "url", None)
    if url and isinstance(url, str):
        sanitize_input(url, field="url")

    # 오디오 파일 경로 검증
    audio = getattr(args, "audio", None)
    if audio and isinstance(audio, str):
        sanitize_input(audio, field="audio")

    # 출력 경로 검증
    output = getattr(args, "output", None)
    if output and isinstance(output, str):
        sanitize_input(output, field="output")

    # 화자 이름 검증 (enroll)
    name = getattr(args, "name", None)
    if name and isinstance(name, str):
        sanitize_input(name, field="name")

    # 프롬프트 검증 (meeting)
    prompt = getattr(args, "prompt", None)
    if prompt and isinstance(prompt, str):
        sanitize_input(prompt, field="prompt")


# --- 서브커맨드별 필수 외부 의존성 (schema 출력에 포함) ---
_COMMAND_DEPENDENCIES: dict[str, list[dict[str, Any]]] = {
    "desktop": [
        {"name": "pywebview", "required": True, "description": "네이티브 데스크톱 웹뷰"},
        {"name": "pystray", "required": True, "description": "시스템 트레이 통합"},
    ],
    "transcribe": [
        {"name": "GPU/CUDA", "required": False, "description": "CUDA GPU 가속 (CPU 폴백 가능)"},
    ],
    "live": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
        {"name": "GPU/CUDA", "required": False, "description": "CUDA GPU 가속 (CPU 폴백 가능)"},
    ],
    "download": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
    ],
    "probe": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
    ],
    "scan": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
    ],
    "smart": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
        {"name": "GPU/CUDA", "required": False, "description": "CUDA GPU 가속 (CPU 폴백 가능)"},
    ],
    "detect": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
    ],
    "map": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
    ],
    "auto": [
        {"name": "ffmpeg", "required": True, "description": "오디오 다운로드/변환"},
        {"name": "yt-dlp", "required": True, "description": "YouTube 라이브 오디오 추출"},
        {"name": "GPU/CUDA", "required": False, "description": "CUDA GPU 가속 (CPU 폴백 가능)"},
    ],
    "meeting": [
        {"name": "GPU/CUDA", "required": False, "description": "CUDA GPU 가속 (CPU 폴백 가능)"},
        {"name": "ffmpeg", "required": False, "description": "오디오 변환 (선택)"},
    ],
    "enroll": [
        {"name": "HF_TOKEN", "required": True, "description": "Hugging Face 토큰 (화자 임베딩 추출)"},
        {"name": "GPU/CUDA", "required": False, "description": "CUDA GPU 가속 (CPU 폴백 가능)"},
    ],
    "cookies": [
        {"name": "Windows", "required": True, "description": "Chrome DPAPI 쿠키 추출은 Windows 전용"},
        {"name": "Chrome", "required": True, "description": "기본 프로필 쿠키 DB 접근"},
    ],
}


def _extract_parser_schema(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """argparse 파서에서 서브커맨드 스키마를 자동 추출.

    각 서브커맨드의 필수 외부 의존성(GPU, CUDA, ffmpeg 등)도 포함한다.
    """
    schema: dict[str, Any] = {
        "name": parser.prog,
        "description": parser.description or "",
    }
    params = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue
        param: dict[str, Any] = {
            "name": action.dest,
            "flags": action.option_strings or [],
            "required": action.required if hasattr(action, "required") else False,
            "help": action.help or "",
        }
        if action.default is not argparse.SUPPRESS and action.default is not None:
            param["default"] = action.default
        if action.type:
            param["type"] = getattr(action.type, "__name__", str(action.type))
        if hasattr(action, "choices") and action.choices:
            param["choices"] = list(action.choices)
        params.append(param)
    schema["parameters"] = params

    # 서브커맨드 이름 추출 (prog에서 마지막 토큰)
    cmd_name = parser.prog.rsplit(None, 1)[-1] if parser.prog else ""
    deps = _COMMAND_DEPENDENCIES.get(cmd_name, [])
    if deps:
        schema["dependencies"] = deps

    return schema


def extract_video_id(url: str) -> str | None:
    """YouTube URL에서 video_id를 추출."""
    m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None


def _run_async_polish_process(
    md_path_str: str,
    segment_count: int,
    use_ollama_flag: bool,
    ollama_model_name: str | None,
    status_file_str: str | None = None,
) -> None:
    import json as _json
    from pathlib import Path

    from .polish import polish_meeting

    status_path = Path(status_file_str) if status_file_str else None

    def _write_status(phase: str, progress: float) -> None:
        if status_path is None:
            return
        try:
            status_path.write_text(
                _json.dumps({"phase": phase, "progress": round(progress, 1)}),
                encoding="utf-8",
            )
        except OSError:
            pass

    _write_status("stt", 0)

    try:
        polish_meeting(
            Path(md_path_str),
            segment_count=segment_count,
            use_ollama=use_ollama_flag,
            ollama_model=ollama_model_name,
            progress_callback=lambda phase, pct: _write_status(phase, pct),
        )
    finally:
        _write_status("done", 100)


def main():
    parser = argparse.ArgumentParser(
        description="sonote — AI 에이전트를 위한 소리 노트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
서브커맨드 그룹:
  전사:  transcribe, live, download
  분석:  detect, map, auto
  회의:  meeting, approve
  관리:  profile, schema, autostart

사용 예시:
  # 로컬 오디오 파일 변환
  media-transcriber transcribe audio.wav

  # YouTube 라이브에서 50분 전부터 녹음 후 변환
  media-transcriber live https://youtube.com/watch?v=XXX --back 50
        """,
    )

    # 전역 플래그
    parser.add_argument(
        "--json", action="store_true", dest="json_mode",
        help="모든 출력을 JSON 구조화 형식으로 출력",
    )
    parser.add_argument(
        "--ndjson", action="store_true", dest="ndjson_mode",
        help="줄 단위 JSON 스트리밍 출력 (transcribe/meeting용)",
    )
    parser.add_argument(
        "--fields", default=None,
        help="출력 필드 선택 (쉼표 구분, --json과 함께 사용). 예: --fields text,start,end",
    )
    parser.add_argument(
        "--beta", action="store_true",
        help="베타 모드 활성화 (또는 SONOTE_BETA=1)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # schema: 서브커맨드 스키마 자동 출력
    p_schema = subparsers.add_parser("schema", help="서브커맨드 스키마 JSON 출력 (AI 친화적)")
    p_schema.add_argument("target", nargs="?", default=None, help="특정 서브커맨드 이름 (생략 시 전체)")

    # transcribe: 로컬 오디오 파일 변환
    p_trans = subparsers.add_parser("transcribe", help="로컬 오디오 파일 변환")
    p_trans.add_argument("audio", help="오디오 파일 경로 (WAV, MP3 등)")
    p_trans.add_argument("-o", "--output", help="출력 파일 경로 (기본: 입력파일_transcript.txt)")
    p_trans.add_argument("-m", "--model", default=_DEFAULT_MODEL_ID, help="Whisper 모델 (기본: seastar105/medium-ko)")
    p_trans.add_argument("-l", "--language", default="ko", help="언어 코드 (기본: ko)")
    p_trans.add_argument("--cpu", action="store_true", help="CPU 강제 사용")
    p_trans.add_argument("--fmt", choices=["txt", "srt"], default="txt", help="출력 형식 (기본: txt)")
    p_trans.add_argument("--beam", type=int, default=5, help="빔 서치 크기 (기본: 5)")
    p_trans.add_argument(
        "--chunk-minutes", type=int, default=0,
        help="청크 단위(분) 분할 변환 — 긴 파일용 (기본: 0=단일 처리)",
    )
    p_trans.add_argument(
        "--dry-run", action="store_true",
        help="실제 전사 없이 설정값만 출력 (--json과 함께 사용 권장)",
    )

    # live: YouTube 라이브 다운로드 + 변환
    p_live = subparsers.add_parser("live", help="YouTube 라이브 녹음 + 변환")
    p_live.add_argument("url", help="YouTube URL")
    p_live.add_argument("-o", "--output", help="출력 디렉토리 (기본: output/transcripts/YYYY-MM-DD/)")
    p_live.add_argument("-b", "--back", type=int, default=0, help="N분 전부터 시작 (기본: 0=현재)")
    p_live.add_argument("-d", "--duration", type=int, default=50, help="녹음 시간(분) (기본: 50)")
    p_live.add_argument("-m", "--model", default=_DEFAULT_MODEL_ID, help="Whisper 모델")
    p_live.add_argument("-l", "--language", default="ko", help="언어 코드")
    p_live.add_argument("--cpu", action="store_true", help="CPU 강제 사용")
    p_live.add_argument(
        "--cookies",
        default=None,
        help="cookies.txt 경로 (미지정 시 자동 탐색: SONOTE_COOKIES → ./cookies.txt → output/data/cookies.txt)",
    )
    p_live.add_argument("--fmt", choices=["txt", "srt"], default="txt", help="출력 형식")
    p_live.add_argument(
        "--resume", metavar="FILE",
        help="기존 스크립트 파일과 이어붙이기 (오버랩 자동 제거)",
    )
    p_live.add_argument(
        "--continuous", action="store_true",
        help="연속 변환 모드 (Ctrl+C로 종료)",
    )
    p_live.add_argument(
        "--chunk-size", type=int, default=120,
        help="연속 모드 청크 크기(초) (기본: 120)",
    )
    p_live.add_argument(
        "--auto-start", action="store_true",
        help="BGM 구간을 건너뛰고 강의 시작 지점 자동 탐색",
    )
    p_live.add_argument(
        "--no-polish", action="store_true",
        help="LLM 후처리 비활성화 (STT 교정 + 요약 생성 건너뜀)",
    )
    p_live.add_argument(
        "--ollama", action="store_true",
        help="Ollama 로컬 LLM으로 STT 교정/키워드 추출 (Codex/Gemini 대신)",
    )
    p_live.add_argument(
        "--ollama-model", default="qwen3.5:9b",
        help="Ollama 모델명 (기본: qwen3.5:9b)",
    )

    # detect (구: probe): BGM↔강의 경계 탐색
    p_probe = subparsers.add_parser(
        "detect", help="라이브 스트림에서 강의 시작 지점 탐색 (구: probe)",
    )
    p_probe.add_argument("url", help="YouTube URL")
    p_probe.add_argument(
        "--max-back", type=int, default=180,
        help="최대 탐색 범위(분) (기본: 180)",
    )

    # map (구: scan): 전체 구간 맵핑
    p_scan = subparsers.add_parser(
        "map", help="라이브 스트림 전체 구간 맵핑 (구: scan)",
    )
    p_scan.add_argument("url", help="YouTube URL")
    p_scan.add_argument(
        "--max-back", type=int, default=180,
        help="최대 탐색 범위(분) (기본: 180)",
    )
    p_scan.add_argument(
        "--step", type=int, default=5,
        help="프로브 간격(분) (기본: 5)",
    )
    p_scan.add_argument(
        "--force-scan", action="store_true",
        help="DB 캐시 무시, 강제 재스캔",
    )

    # auto (구: smart): 스캔 → 병렬 다운로드 → 변환 → 병합 (올인원)
    p_smart = subparsers.add_parser(
        "auto", help="스캔+병렬다운로드+변환 올인원 (구: smart)",
    )
    p_smart.add_argument("url", help="YouTube URL")
    p_smart.add_argument("-o", "--output", default=None, help="출력 디렉토리 (기본: output/transcripts/YYYY-MM-DD/)")
    p_smart.add_argument("-m", "--model", default=_DEFAULT_MODEL_ID, help="Whisper 모델")
    p_smart.add_argument("-l", "--language", default="ko", help="언어 코드")
    p_smart.add_argument("--cpu", action="store_true", help="CPU 강제 사용")
    p_smart.add_argument(
        "--cookies",
        default=None,
        help="cookies.txt 경로 (미지정 시 자동 탐색: SONOTE_COOKIES → ./cookies.txt → output/data/cookies.txt)",
    )
    p_smart.add_argument("--fmt", choices=["txt", "srt"], default="txt", help="출력 형식")
    p_smart.add_argument(
        "--max-back", type=int, default=180,
        help="최대 탐색 범위(분) (기본: 180)",
    )
    p_smart.add_argument(
        "--step", type=int, default=5,
        help="스캔 프로브 간격(분) (기본: 5)",
    )
    p_smart.add_argument(
        "--resume", metavar="FILE",
        help="기존 스크립트와 병합",
    )
    p_smart.add_argument(
        "--force-scan", action="store_true",
        help="DB 캐시 무시, 강제 재스캔",
    )
    p_smart.add_argument(
        "--watch", type=int, default=0, metavar="MINUTES",
        help="주기적 증분 캡처 간격(분). 0이면 1회만 실행 (기본: 0)",
    )

    # profile: 비디오 프로필 DB 관리
    sub_profile = subparsers.add_parser("profile", help="비디오 프로필 DB 관리")
    profile_sub = sub_profile.add_subparsers(dest="profile_cmd", required=True)
    profile_sub.add_parser("list", help="프로필 목록")
    show_parser = profile_sub.add_parser("show", help="프로필 상세")
    show_parser.add_argument("video_id", help="YouTube video ID")
    del_parser = profile_sub.add_parser("delete", help="프로필 삭제")
    del_parser.add_argument("video_id", help="YouTube video ID")

    # cookies: Chrome 쿠키 export/check/path
    p_cookies = subparsers.add_parser("cookies", help="Chrome 쿠키 관리")
    cookies_sub = p_cookies.add_subparsers(dest="cookies_cmd", required=True)

    p_cookies_export = cookies_sub.add_parser(
        "export",
        help="Chrome 쿠키를 Netscape cookies.txt로 추출 (output/data/cookies.txt)",
    )
    p_cookies_export.add_argument(
        "-o",
        "--output",
        default=None,
        help="저장 경로 (기본: output/data/cookies.txt)",
    )
    cookies_sub.add_parser("check", help="현재 cookies.txt 유효성(만료) 점검")
    cookies_sub.add_parser("path", help="현재 사용 중인 cookies.txt 경로 출력")

    # download: 오디오만 다운로드
    p_dl = subparsers.add_parser("download", help="YouTube 라이브 오디오만 다운로드")
    p_dl.add_argument("url", help="YouTube URL")
    p_dl.add_argument("-o", "--output", default="live_audio.wav", help="출력 파일")
    p_dl.add_argument("-b", "--back", type=int, default=0, help="N분 전부터 시작")
    p_dl.add_argument("-d", "--duration", type=int, default=50, help="녹음 시간(분)")

    # enroll: 화자 목소리 사전 등록
    p_enroll = subparsers.add_parser(
        "enroll",
        help="화자 목소리 사전 등록 — 회의 전 참석자 등록",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 화자 등록 (10~30초 말하기)
  media-transcriber enroll "민수"
  media-transcriber enroll "지영" --duration 20

  # 등록된 화자 목록
  media-transcriber enroll --list

  # 화자 삭제
  media-transcriber enroll --delete "민수"

  # 회의에서 사용
  media-transcriber meeting --profiles output/data/speakers.json
        """,
    )
    p_enroll.add_argument("name", nargs="?", help="등록할 화자 이름")
    p_enroll.add_argument("--duration", type=int, default=15, help="녹음 시간(초) (기본: 15, 최소: 5)")
    p_enroll.add_argument("--profiles", default=str(speakers_json_path()), help="프로필 파일 경로 (기본: output/data/speakers.json)")
    p_enroll.add_argument("--device", type=int, default=None, help="오디오 입력 장치 인덱스")
    p_enroll.add_argument("--list", action="store_true", dest="list_profiles", help="등록된 화자 목록 출력")
    p_enroll.add_argument("--delete", metavar="NAME", help="화자 삭제")
    p_enroll.add_argument("--cpu", action="store_true", help="CPU 강제 사용")

    # meeting: 회의 실시간 전사
    p_meeting = subparsers.add_parser(
        "meeting",
        help="회의 실시간 전사 — 마이크 → 화자 분리 + SSE 자막 + 파일 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (마이크 → SSE + 파일)
  media-transcriber meeting

  # 화자 분리 비활성화
  media-transcriber meeting --no-diarize

  # 포트/출력 지정
  media-transcriber meeting --port 8080 -o standup.txt

  # 마이크 장치 선택
  media-transcriber meeting --list-devices
  media-transcriber meeting --device 1

  # 청크 크기 조정 (실시간성 vs 정확도)
  media-transcriber meeting --chunk 10

  # 사전 등록된 화자 프로필 사용
  media-transcriber meeting --profiles output/data/speakers.json

  # 프로필 승인 대기 파일 생성 (원본 profiles는 덮어쓰지 않음)
  media-transcriber meeting --profiles output/data/speakers.json --auto-update
        """,
    )
    p_meeting.add_argument(
        "--file", metavar="PATH", default=None,
        help="오프라인 모드: 오디오/비디오 파일 → 회의록 .md 변환 (마이크 대신 파일 입력)",
    )
    p_meeting.add_argument("--no-diarize", action="store_true", help="화자 분리 비활성화")
    p_meeting.add_argument("--port", type=int, default=8000, help="SSE 서버 포트 (기본: 8000)")
    p_meeting.add_argument("-o", "--output", help="출력 파일 경로 (기본: output/meetings/YYYY-MM-DD/meeting_HHMMSS.md)")
    p_meeting.add_argument("--chunk", type=float, default=15.0, help="오디오 청크 크기(초) (기본: 15.0)")
    p_meeting.add_argument("--device", type=int, default=None, help="오디오 입력 장치 인덱스")
    p_meeting.add_argument("--list-devices", action="store_true", help="사용 가능한 마이크 목록 출력")
    p_meeting.add_argument("--profiles", help="화자 프로필 JSON (enroll로 생성, 화자 분리 정확도 향상)")
    p_meeting.add_argument("--auto-update", action="store_true", help="회의 종료 시 승인 대기 프로필 스냅샷 생성")
    p_meeting.add_argument("-m", "--model", default=_DEFAULT_MODEL_ID, help="Whisper 모델 (기본: large-v3-turbo)")
    p_meeting.add_argument("-l", "--language", default="ko", help="언어 코드 (기본: ko)")
    p_meeting.add_argument("--cpu", action="store_true", help="CPU 강제 사용")
    p_meeting.add_argument(
        "--prompt", default="회의 내용을 한국어로 정확히 받아적습니다.",
        help="Whisper initial_prompt (한국어 힌트, 기본: 회의 내용 한국어 프롬프트)",
    )
    p_meeting.add_argument(
        "--no-polish", action="store_true",
        help="LLM 후처리 비활성화 (STT 교정 + 요약 생성 건너뜀)",
    )
    p_meeting.add_argument(
        "--ollama", action="store_true",
        help="Ollama 로컬 LLM으로 STT 교정/키워드 추출 (Codex/Gemini 대신)",
    )
    p_meeting.add_argument(
        "--ollama-model", default="qwen3.5:9b",
        help="Ollama 모델명 (기본: qwen3.5:9b)",
    )
    p_meeting.add_argument(
        "--fixed-chunking", action="store_true",
        help="휴리스틱 청킹 비활성화 (기본은 침묵 기반 가변 청킹)",
    )
    p_meeting.add_argument(
        "--dry-run", action="store_true",
        help="실제 전사 없이 설정값만 출력 (--json과 함께 사용 권장)",
    )
    p_meeting.add_argument(
        "--auto-mute", action="store_true",
        help="녹음 시작 시 다른 미디어 앱 자동 음소거 (종료 시 복원)",
    )

    p_desktop = subparsers.add_parser(
        "desktop",
        help="pywebview 기반 네이티브 데스크톱 앱 실행",
    )
    p_desktop.add_argument(
        "--host",
        default="127.0.0.1",
        help="내장 FastAPI 서버 호스트 (기본: 127.0.0.1)",
    )
    p_desktop.add_argument(
        "--port",
        type=int,
        default=0,
        help="내장 FastAPI 서버 포트 (기본: 0=자동 선택)",
    )

    # status: 환경 진단 독립 실행
    subparsers.add_parser(
        "status",
        help="GPU/모델/디스크/외부 도구 상태 진단 (preflight_check 독립 실행)",
    )

    # autostart: Windows 자동 시작 등록/해제/상태 확인
    p_autostart = subparsers.add_parser(
        "autostart",
        help="Windows 자동 시작 등록/해제/상태 확인",
    )
    autostart_group = p_autostart.add_mutually_exclusive_group(required=True)
    autostart_group.add_argument(
        "--enable",
        action="store_true",
        help="현재 실행 경로를 Windows 자동 시작에 등록",
    )
    autostart_group.add_argument(
        "--disable",
        action="store_true",
        help="Windows 자동 시작 등록 해제",
    )
    autostart_group.add_argument(
        "--status",
        action="store_true",
        help="현재 실행 경로의 자동 시작 등록 상태 확인",
    )

    # setup: 대화형/자동 환경 설정 + 의존성 설치
    p_setup = subparsers.add_parser(
        "setup",
        help="환경 설정 — 의존성 자동 설치 + HF 토큰 설정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  sonote setup                # 핵심 의존성 설치 (ffmpeg, 모델 확인)
  sonote setup --all          # 모든 선택적 의존성 설치
  sonote setup --diarize      # 화자 분리 (torch + pyannote)
  sonote setup --gpu          # CUDA 환경 설정
  sonote setup --fix          # doctor 진단 후 누락 항목 자동 수정
  sonote setup --hf-token hf_xxx  # HuggingFace 토큰 설정
        """,
    )
    p_setup.add_argument("--all", action="store_true", dest="setup_all", help="모든 선택적 의존성 설치")
    p_setup.add_argument("--gpu", action="store_true", help="CUDA 환경 설정")
    p_setup.add_argument("--diarize", action="store_true", help="화자 분리 의존성 설치")
    p_setup.add_argument("--desktop", action="store_true", help="데스크톱 extras 설치")
    p_setup.add_argument("--fix", action="store_true", help="doctor 진단 후 누락 항목 자동 수정")
    p_setup.add_argument("--hf-token", default=None, help="HuggingFace 토큰 설정")
    p_setup.add_argument("--json", action="store_true", dest="setup_json", help="JSON 형식 출력")

    # doctor: 종합 환경 진단 (설치 안내 포함)
    p_doctor = subparsers.add_parser(
        "doctor",
        help="종합 환경 진단 — 의존성 점검 + 설치 안내 + 폴백 체인 표시",
    )
    p_doctor.add_argument(
        "--json", action="store_true", dest="doctor_json",
        help="JSON 형식으로 출력 (AI 에이전트용)",
    )

    p_approve = subparsers.add_parser("approve", help="회의 후 후보 프로필 적용 (구: approve-profiles)")
    p_approve.add_argument("review", help="meeting *.profile-review.json 경로")
    p_approve.add_argument("--profiles", help="적용 대상 프로필 JSON (기본: review 파일의 base_profiles_path)")

    # update: GitHub Releases 기반 자동 업데이트
    p_update = subparsers.add_parser(
        "update",
        help="GitHub Releases에서 최신 버전 확인 및 설치",
    )
    update_action = p_update.add_mutually_exclusive_group(required=True)
    update_action.add_argument(
        "--check",
        action="store_true",
        help="업데이트 가능 여부만 확인 (다운로드 없음)",
    )
    update_action.add_argument(
        "--install",
        action="store_true",
        help="최신 버전 다운로드 후 설치 (사용자 확인 필요)",
    )
    p_update.add_argument(
        "--json", action="store_true", dest="update_json",
        help="JSON 형식으로 출력",
    )

    # --- deprecated aliases (하위 호환) ---
    # probe → detect
    _p_probe_old = subparsers.add_parser("probe", help="[deprecated] detect를 사용하세요")
    _p_probe_old.add_argument("url", help="YouTube URL")
    _p_probe_old.add_argument("--max-back", type=int, default=180, help="최대 탐색 범위(분)")
    # scan → map
    _p_scan_old = subparsers.add_parser("scan", help="[deprecated] map을 사용하세요")
    _p_scan_old.add_argument("url", help="YouTube URL")
    _p_scan_old.add_argument("--max-back", type=int, default=180, help="최대 탐색 범위(분)")
    _p_scan_old.add_argument("--step", type=int, default=5, help="프로브 간격(분)")
    _p_scan_old.add_argument("--force-scan", action="store_true", help="DB 캐시 무시")
    # smart → auto
    _p_smart_old = subparsers.add_parser("smart", help="[deprecated] auto를 사용하세요")
    _p_smart_old.add_argument("url", help="YouTube URL")
    _p_smart_old.add_argument("-o", "--output", default=None, help="출력 디렉토리")
    _p_smart_old.add_argument("-m", "--model", default=_DEFAULT_MODEL_ID, help="Whisper 모델")
    _p_smart_old.add_argument("-l", "--language", default="ko", help="언어 코드")
    _p_smart_old.add_argument("--cpu", action="store_true", help="CPU 강제 사용")
    _p_smart_old.add_argument(
        "--cookies",
        default=None,
        help="cookies.txt 경로 (미지정 시 자동 탐색)",
    )
    _p_smart_old.add_argument("--fmt", choices=["txt", "srt"], default="txt", help="출력 형식")
    _p_smart_old.add_argument("--max-back", type=int, default=180, help="최대 탐색 범위(분)")
    _p_smart_old.add_argument("--step", type=int, default=5, help="스캔 프로브 간격(분)")
    _p_smart_old.add_argument("--resume", metavar="FILE", help="기존 스크립트와 병합")
    _p_smart_old.add_argument("--force-scan", action="store_true", help="DB 캐시 무시")
    _p_smart_old.add_argument("--watch", type=int, default=0, metavar="MINUTES", help="주기적 증분 캡처 간격(분)")
    # approve-profiles → approve
    _p_approve_old = subparsers.add_parser("approve-profiles", help="[deprecated] approve를 사용하세요")
    _p_approve_old.add_argument("review", help="meeting *.profile-review.json 경로")
    _p_approve_old.add_argument("--profiles", help="적용 대상 프로필 JSON")

    _ALIASES = {
        "probe": "detect",
        "scan": "map",
        "smart": "auto",
        "approve-profiles": "approve",
    }

    args = parser.parse_args()
    beta_mode = _apply_beta_mode(args, sys.argv[1:])
    setattr(args, "_beta_mode", beta_mode)

    # deprecated alias 감지 시 stderr 경고
    if args.command in _ALIASES:
        new_name = _ALIASES[args.command]
        print(
            f"[경고] '{args.command}'는 deprecated입니다. '{new_name}'을 사용하세요.",
            file=sys.stderr,
        )

    is_json = getattr(args, "json_mode", False)
    is_ndjson = getattr(args, "ndjson_mode", False)

    # 입력 검증 — 제어문자, 위험 유니코드, 경로 탈출 차단
    try:
        _validate_cli_inputs(args)
    except ValidationError as ve:
        if is_json:
            print(json_output(
                "error", args.command,
                error=str(ve),
                code="ARG_ERROR",
                data={"field": ve.field, "validation_code": ve.code},
            ))
        else:
            print(f"[입력 검증 실패] {ve}", file=sys.stderr)
        sys.exit(EXIT_ARG_ERROR)

    # schema 커맨드: 서브커맨드 스키마 자동 출력 + capabilities
    if args.command == "schema":
        _cmd_schema(subparsers, args)
        return

    # status 커맨드: 환경 진단 독립 실행
    if args.command == "status":
        _cmd_status(args)
        return

    # autostart 커맨드: Windows 자동 시작 제어
    if args.command == "autostart":
        _cmd_autostart(args)
        return

    # setup 커맨드: 환경 설정 + 의존성 설치
    if args.command == "setup":
        from .setup import run_setup
        run_setup(
            all_extras=getattr(args, "setup_all", False),
            gpu=getattr(args, "gpu", False),
            diarize=getattr(args, "diarize", False),
            desktop=getattr(args, "desktop", False),
            fix=getattr(args, "fix", False),
            hf_token=getattr(args, "hf_token", None),
            use_json=getattr(args, "setup_json", False),
        )
        return

    # doctor 커맨드: 종합 환경 진단
    if args.command == "doctor":
        from .doctor import run_diagnosis, print_diagnosis
        result = run_diagnosis()
        print_diagnosis(result, use_json=getattr(args, "doctor_json", False))
        return

    # --dry-run 처리 (transcribe, meeting) — capabilities/diagnostics 포함
    if getattr(args, "dry_run", False):
        _cmd_dry_run(args)
        return

    # Preflight Diagnostics — 실행 전 필수 조건 점검
    diagnostics = preflight_check(args)
    manual_steps = diagnostics["manual_steps"]

    if not diagnostics["passed"]:
        failed = [c for c in diagnostics["checks"] if not c["passed"] and c["required"]]
        if is_json or is_ndjson:
            output = json_output(
                "error", args.command,
                error="사전 점검 실패",
                code="PREFLIGHT_FAIL",
                data={
                    "diagnostics": diagnostics["checks"],
                    "manual_steps": manual_steps,
                },
            ) if is_json else _ndjson_line(
                "preflight_fail",
                command=args.command,
                diagnostics=diagnostics["checks"],
                manual_steps=manual_steps,
            )
            print(output)
        else:
            print("[사전 점검 실패]", file=sys.stderr)
            for c in failed:
                print(f"  ✗ {c['name']}: {c['message']}", file=sys.stderr)
            for step in manual_steps:
                print(f"  [{step['tag']}] {step['message']}", file=sys.stderr)
                if step.get("hint"):
                    print(f"    → {step['hint']}", file=sys.stderr)
        sys.exit(EXIT_PREFLIGHT_FAIL)

    # [MANUAL] 단계 경고 출력 (점검 통과했지만 수동 단계가 남은 경우)
    if manual_steps:
        if is_json:
            pass  # JSON 모드에서는 응답 data에 포함
        elif is_ndjson:
            for step in manual_steps:
                print(_ndjson_line("manual_step", **step))
        else:
            for step in manual_steps:
                print(f"[{step['tag']}] {step['message']}", file=sys.stderr)
                if step.get("hint"):
                    print(f"  → {step['hint']}", file=sys.stderr)

    # 커맨드 디스패치 (--json 모드 에러 핸들링 포함)
    _dispatch = {
        "desktop": _cmd_desktop,
        "transcribe": _cmd_transcribe,
        "live": _cmd_live,
        "download": _cmd_download,
        "detect": _cmd_probe,
        "map": _cmd_scan,
        "auto": _cmd_smart,
        "profile": _cmd_profile,
        "cookies": _cmd_cookies,
        "update": _cmd_update,
        "enroll": _cmd_enroll,
        "approve": _cmd_approve_profiles,
        "meeting": _cmd_meeting,
        # deprecated aliases
        "probe": _cmd_probe,
        "scan": _cmd_scan,
        "smart": _cmd_smart,
        "approve-profiles": _cmd_approve_profiles,
    }

    handler = _dispatch.get(args.command)
    if not handler:
        if is_json:
            print(json_output("error", args.command, error=f"알 수 없는 커맨드: {args.command}", code="ARG_ERROR"))
        sys.exit(EXIT_ARG_ERROR)

    if is_json:
        try:
            handler(args)
        except FileNotFoundError as e:
            print(json_output("error", args.command, error=str(e), code="NOT_FOUND"))
            sys.exit(EXIT_NOT_FOUND)
        except (ValueError, argparse.ArgumentError) as e:
            print(json_output("error", args.command, error=str(e), code="ARG_ERROR"))
            sys.exit(EXIT_ARG_ERROR)
        except Exception as e:
            print(json_output("error", args.command, error=str(e), code="ERROR"))
            sys.exit(EXIT_ERROR)
    else:
        handler(args)


def _cmd_schema(subparsers: argparse._SubParsersAction, args: argparse.Namespace) -> None:
    """서브커맨드 스키마 JSON 출력 (capabilities 포함)."""
    choices = subparsers.choices or {}
    caps = check_capabilities()

    if args.target:
        if args.target not in choices:
            print(json_output(
                "error", "schema",
                error=f"알 수 없는 서브커맨드: {args.target}",
                code="NOT_FOUND",
            ))
            sys.exit(EXIT_NOT_FOUND)
        schema = _extract_parser_schema(choices[args.target])
        schema["capabilities"] = caps
        print(json_output("success", "schema", data=schema))
    else:
        all_schemas = {}
        for name, sub_parser in choices.items():
            if name == "schema":
                continue
            all_schemas[name] = _extract_parser_schema(sub_parser)
        print(json_output("success", "schema", data={
            "commands": all_schemas,
            "capabilities": caps,
        }))


def _cmd_status(args: argparse.Namespace) -> None:
    """환경 진단 독립 실행 — preflight_check + capabilities를 JSON으로 출력."""
    caps = check_capabilities()
    diagnostics = preflight_check(args)

    status_data: dict[str, Any] = {
        "capabilities": caps,
        "diagnostics": diagnostics["checks"],
        "manual_steps": diagnostics["manual_steps"],
        "passed": diagnostics["passed"],
    }

    is_json = getattr(args, "json_mode", False)
    if is_json:
        print(json_output("success", "status", data=status_data))
    else:
        print(_json.dumps(status_data, ensure_ascii=False, indent=2))


def _cmd_autostart(args: argparse.Namespace) -> None:
    """Windows 자동 시작 등록/해제/상태 확인을 수행한다."""
    from .autostart import (
        AutostartError,
        get_exe_path,
        is_registered,
        register,
        unregister,
    )

    is_json = getattr(args, "json_mode", False)

    try:
        if args.enable:
            register()
            enabled = True
            action = "enable"
            message = "Windows 자동 시작을 등록했습니다."
        elif args.disable:
            unregister()
            enabled = False
            action = "disable"
            message = "Windows 자동 시작을 해제했습니다."
        else:
            enabled = is_registered()
            action = "status"
            message = "Windows 자동 시작이 등록되어 있습니다." if enabled else "Windows 자동 시작이 비활성화되어 있습니다."
    except AutostartError as exc:
        if is_json:
            print(json_output("error", "autostart", error=str(exc), code="ERROR"))
        else:
            print(f"[자동 시작] 오류: {exc}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    data: dict[str, Any] = {
        "action": action,
        "enabled": enabled,
        "exe_path": get_exe_path(),
    }

    if is_json:
        print(json_output("success", "autostart", data=data))
        return

    print(message)
    print(f"등록 상태: {'사용' if enabled else '사용 안 함'}")
    print(f"실행 경로: {data['exe_path']}")


def _cmd_desktop(args: argparse.Namespace) -> None:
    """pywebview 기반 데스크톱 모드를 실행한다."""
    from .desktop import run_desktop

    run_desktop(
        host=args.host,
        port=args.port,
        beta_mode=getattr(args, "_beta_mode", False),
    )


def _cmd_dry_run(args: argparse.Namespace) -> None:
    """--dry-run: 실제 전사 없이 설정값 + capabilities + diagnostics JSON 출력."""
    from .runtime_env import detect_device

    if args.cpu:
        device, compute_type = "cpu", "int8"
    else:
        device, compute_type = detect_device()

    config: dict[str, Any] = {
        "command": args.command,
        "model": getattr(args, "model", _DEFAULT_MODEL_ID),
        "language": getattr(args, "language", "ko"),
        "device": device,
        "compute_type": compute_type,
    }

    if args.command == "transcribe":
        audio_path = Path(args.audio)
        output_path = args.output or str(audio_path.stem) + "_transcript." + getattr(args, "fmt", "txt")
        config.update({
            "audio": str(audio_path.resolve()),
            "output": str(Path(output_path).resolve()),
            "format": getattr(args, "fmt", "txt"),
            "beam_size": getattr(args, "beam", 5),
            "chunk_minutes": getattr(args, "chunk_minutes", 0),
        })
    elif args.command == "meeting":
        config.update({
            "port": getattr(args, "port", 8000),
            "output": args.output or "output/meetings/YYYY-MM-DD/meeting_HHMMSS.md",
            "chunk_seconds": getattr(args, "chunk", 15.0),
            "diarize": not getattr(args, "no_diarize", False),
            "profiles": getattr(args, "profiles", None),
            "polish": not getattr(args, "no_polish", False),
            "ollama": getattr(args, "ollama", False),
            "ollama_model": getattr(args, "ollama_model", "qwen3.5:9b"),
            "prompt": getattr(args, "prompt", ""),
        })

    caps = check_capabilities()
    diagnostics = preflight_check(args)

    dry_data: dict[str, Any] = {
        "dry_run": True,
        "config": config,
        "capabilities": caps,
        "diagnostics": diagnostics["checks"],
        "manual_steps": diagnostics["manual_steps"],
    }

    if getattr(args, "json_mode", False):
        print(json_output("success", args.command, data=dry_data))
    else:
        print(_json.dumps(dry_data, ensure_ascii=False, indent=2))


def _cmd_transcribe(args):
    """로컬 오디오 파일 변환."""
    from .transcribe import save_transcript, transcribe_audio, transcribe_chunks

    audio_path = Path(args.audio)
    output_path = args.output or str(audio_path.stem) + "_transcript." + args.fmt
    device = "cpu" if args.cpu else None
    compute_type = "int8" if args.cpu else None
    is_ndjson = getattr(args, "ndjson_mode", False)

    if is_ndjson:
        print(_ndjson_line("start", command="transcribe", audio=str(audio_path)))

    if getattr(args, "chunk", None) and args.chunk > 0:
        segments = transcribe_chunks(
            str(audio_path),
            model_id=args.model,
            language=args.language,
            chunk_minutes=args.chunk,
            device=device,
            compute_type=compute_type,
        )
    else:
        segments = transcribe_audio(
            str(audio_path),
            model_id=args.model,
            language=args.language,
            device=device,
            compute_type=compute_type,
        )

    save_transcript(segments, output_path, fmt=args.fmt)

    # NDJSON: 세그먼트별 줄 단위 출력 (--fields 적용)
    if is_ndjson:
        _fields_set = None
        _raw_fields = getattr(args, "fields", None)
        if _raw_fields:
            _fields_set = {k.strip() for k in _raw_fields.split(",") if k.strip()}
        for seg in segments:
            _seg_data = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": (seg.get("text") or "").strip(),
            }
            if _fields_set:
                _seg_data = {k: v for k, v in _seg_data.items() if k in _fields_set}
            print(_ndjson_line("segment", **_seg_data))

    if is_ndjson:
        print(_ndjson_line(
            "complete",
            command="transcribe",
            audio=str(audio_path),
            output=str(output_path),
            segments=len(segments),
        ))
    elif getattr(args, "json_mode", False):
        _tdata = {
            "audio": str(audio_path),
            "output": str(output_path),
            "segments": len(segments),
            "format": args.fmt,
        }
        print(json_output("success", "transcribe", data=_filter_fields(
            _tdata, getattr(args, "fields", None),
        )))


def _cmd_probe(args):
    """라이브 스트림에서 강의 시작 지점 탐색."""
    from .probe import find_content_start
    result = find_content_start(args.url, max_back_minutes=args.max_back)

    if result["status"] == "found":
        print("\n사용 예시:")
        print(f"  python -m src.cli live \"{args.url}\" --back {result['content_back']} -d 60")
    elif result["status"] == "all_speech":
        print("\n사용 예시:")
        print(f"  python -m src.cli live \"{args.url}\" --back {result['content_back']} -d 120")


def _cmd_scan(args: argparse.Namespace) -> None:
    """라이브 스트림 전체 구간 맵핑."""
    from .probe import scan_stream
    scan_result = scan_stream(
        args.url, max_back_minutes=args.max_back, step_minutes=args.step,
    )

    video_id = extract_video_id(args.url)
    if video_id:
        save_profile(
            video_id,
            args.url,
            scan_result=scan_result,
            total_speech_min=scan_result.get("total_speech_min"),
        )
    else:
        print("[DB] video_id 추출 실패: 스캔 결과 저장 생략", file=sys.stderr)


def _watch_sleep(minutes: int) -> None:
    """watch 모드 대기. KeyboardInterrupt 시 프로그램 종료."""
    print(f"\n[watch] {minutes}분 후 재스캔... (Ctrl+C로 종료)", file=sys.stderr)
    try:
        time.sleep(minutes * 60)
    except KeyboardInterrupt:
        print("\n[watch] 사용자 중단으로 종료", file=sys.stderr)
        raise SystemExit(0)


def _cmd_smart(args: argparse.Namespace) -> None:
    """스캔 → 병렬 다운로드 → 변환 → 병합 올인원. --watch 시 주기적 증분 캡처."""
    from .probe import scan_stream
    from .download import download_speech_blocks
    from .transcribe import save_transcript, transcribe_chunks

    output_dir = Path(args.output) if args.output else transcripts_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_output = audio_dir()

    device = "cpu" if args.cpu else None
    compute_type = "int8" if args.cpu else None

    video_id = extract_video_id(args.url)
    watch_interval = getattr(args, "watch", 0)
    iteration = 0

    while True:
        iteration += 1
        force_scan = args.force_scan or (watch_interval > 0 and iteration > 1)

        if watch_interval and iteration > 1:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"[watch #{iteration}] 증분 캡처 시작", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)

        profile = get_profile(video_id) if video_id else None

        # Phase 1: 스캔
        print("=" * 60, file=sys.stderr)
        print("[Phase 1/3] 스트림 구간 스캔", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        if profile and profile.get("scan_result") and not force_scan:
            print(f"[DB] 기존 스캔 결과 사용 ({video_id})", file=sys.stderr)
            scan_result = profile["scan_result"]
        else:
            scan_result = scan_stream(
                args.url,
                max_back_minutes=args.max_back,
                step_minutes=args.step,
                cookies_path=args.cookies,
            )
            if video_id:
                save_profile(
                    video_id,
                    args.url,
                    scan_result=scan_result,
                    total_speech_min=scan_result.get("total_speech_min"),
                )

        speech_ranges = scan_result.get("speech_ranges") or []
        if not speech_ranges:
            print("[중단] 음성 구간을 찾지 못했습니다", file=sys.stderr)
            if watch_interval > 0:
                _watch_sleep(watch_interval)
                continue
            return

        # Phase 2: 음성 블록만 병렬 다운로드
        print(file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("[Phase 2/3] 음성 블록 병렬 다운로드", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        blocks = scan_result.get("blocks") or []
        audio_files = download_speech_blocks(
            args.url,
            blocks,
            audio_output,
            cookies_path=args.cookies,
        )

        if not audio_files:
            print("[중단] 다운로드된 파일 없음", file=sys.stderr)
            if watch_interval > 0:
                _watch_sleep(watch_interval)
                continue
            return

        # Phase 3: 각 블록 변환 + 병합
        print(file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"[Phase 3/3] 변환 ({len(audio_files)}개 블록)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        all_segments = []
        cumulative_offset = 0.0

        for i, audio_path in enumerate(audio_files):
            print(f"\n--- 블록 {i}/{len(audio_files) - 1}: {audio_path.name} ---", file=sys.stderr)
            segments = transcribe_chunks(
                audio_path,
                chunk_minutes=10,
                model_id=args.model,
                language=args.language,
                device=device,
                compute_type=compute_type,
            )

            for seg in segments:
                seg["start"] += cumulative_offset
                seg["end"] += cumulative_offset
            all_segments.extend(segments)

            if segments:
                cumulative_offset = segments[-1]["end"]

        # 기존 스크립트와 병합
        if args.resume:
            from .merge import merge_transcripts, load_transcript
            existing = load_transcript(args.resume)
            if existing:
                all_segments = merge_transcripts(existing, all_segments)

        # 저장
        transcript_path = output_dir / f"transcript.{args.fmt}"
        save_transcript(all_segments, transcript_path, fmt=args.fmt)

        if args.resume:
            save_transcript(all_segments, args.resume, fmt=args.fmt)

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[완료] {len(all_segments)}개 세그먼트 → {transcript_path}", file=sys.stderr)

        total_mb = sum(f.stat().st_size for f in audio_files) / 1024 / 1024
        print(f"[다운로드] 총 {total_mb:.0f}MB ({len(audio_files)}개 블록)", file=sys.stderr)
        print(f"[정리] 오디오 파일은 {audio_output}에 보존됨", file=sys.stderr)

        if video_id:
            captured_files = [str(path) for path in audio_files]
            update_captured(video_id, speech_ranges, captured_files)
            print(f"[DB] 캡처 정보 갱신 완료 ({video_id})", file=sys.stderr)

        # watch 모드: 다음 반복을 위해 resume 설정 후 대기
        if watch_interval > 0:
            if not args.resume:
                args.resume = str(transcript_path)
            _watch_sleep(watch_interval)
        else:
            break


def _cmd_profile(args: argparse.Namespace) -> None:
    """비디오 프로필 DB 조회/삭제."""
    if args.profile_cmd == "list":
        profiles = list_profiles()
        if not profiles:
            print("저장된 프로필 없음")
            return
        for p in profiles:
            status = p.get("stream_status", "?")
            speech = p.get("total_speech_min", 0) or 0
            segs = p.get("transcript_segments", 0) or 0
            title = p.get("title", "")
            print(f"  {p['video_id']}  [{status}]  강의 {speech}분  {segs}세그먼트  ({title})")
    elif args.profile_cmd == "show":
        p = get_profile(args.video_id)  # type: ignore[assignment]
        if not p:
            print(f"프로필 없음: {args.video_id}")
            return
        import json
        for k, v in p.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False, indent=2)
            print(f"  {k}: {v}")
    elif args.profile_cmd == "delete":
        if delete_profile(args.video_id):
            print(f"삭제 완료: {args.video_id}")
        else:
            print(f"프로필 없음: {args.video_id}")


def _cmd_cookies(args: argparse.Namespace) -> None:
    """Chrome cookies.txt export/check/path."""
    is_json = getattr(args, "json_mode", False)

    if args.cookies_cmd == "export":
        result = export_chrome_cookies_to_netscape(output_path=args.output)
        if is_json:
            print(json_output("success", "cookies", data={
                "action": "export",
                **result,
            }))
            return

        print(f"[cookies] 저장: {result['output_path']}")
        print(
            "[cookies] 내보냄 "
            f"{result['exported_count']}/{result['total_count']} "
            f"(v20 스킵: {result['skipped_v20']}, 오류 스킵: {result['skipped_error']})"
        )
        if result["skipped_v20"] > 0:
            print(
                "[안내] v20 App-Bound 쿠키는 외부 복호화가 불가합니다. "
                "Chrome 확장으로 export한 cookies.txt 사용이 필요합니다."
            )
        return

    if args.cookies_cmd == "check":
        result = check_cookies_file()
        if not result["exists"]:
            message = "사용 가능한 cookies.txt를 찾지 못했습니다."
            if is_json:
                print(json_output("error", "cookies", error=message, code="NOT_FOUND"))
            else:
                print(f"[cookies] {message}")
            return

        status = "valid" if result["valid"] > 0 else "expired"
        if is_json:
            print(json_output("success", "cookies", data={
                "action": "check",
                "status": status,
                **result,
            }))
            return

        print(f"[cookies] 파일: {result['path']}")
        print(
            f"[cookies] 전체 {result['total']} / 유효 {result['valid']} / "
            f"만료 {result['expired']} / 세션 {result['session']} / 손상 {result['malformed']}"
        )
        if status == "expired":
            print("[안내] 유효한 쿠키가 없어 재-export가 필요합니다.")
        return

    if args.cookies_cmd == "path":
        path = resolve_cookies_path()
        if is_json:
            if path:
                print(json_output("success", "cookies", data={
                    "action": "path",
                    "path": str(path),
                }))
            else:
                print(json_output("error", "cookies", error="사용 가능한 cookies.txt 없음", code="NOT_FOUND"))
            return

        if path:
            print(path)
        else:
            print("사용 가능한 cookies.txt 없음")
        return

    raise ValueError(f"알 수 없는 cookies 서브커맨드: {args.cookies_cmd}")


def _cmd_live(args):
    """YouTube 라이브 녹음 + 변환."""
    from .transcribe import save_transcript, transcribe_audio

    output_dir = Path(args.output) if args.output else transcripts_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else None
    compute_type = "int8" if args.cpu else None

    # 자동 시작 지점 탐색: BGM 건너뛰기
    if args.auto_start and args.back == 0:
        from .probe import find_content_start
        print("[자동 탐색] BGM 구간 건너뛰기...", file=sys.stderr)
        result = find_content_start(args.url, cookies_path=args.cookies)
        if result["status"] in ("found", "all_speech"):
            args.back = result["content_back"]
            print(f"[자동 탐색] --back {args.back} 으로 설정", file=sys.stderr)
        else:
            print("[자동 탐색] 강의 시작 지점을 찾지 못함 — 현재 시점부터 녹음", file=sys.stderr)

    # 연속 모드: 청크 단위 실시간 변환
    if args.continuous:
        from .continuous import continuous_live
        transcript_path = output_dir / f"transcript.{args.fmt}"
        continuous_live(
            args.url,
            transcript_path,
            chunk_seconds=args.chunk_size,
            model_id=args.model,
            language=args.language,
            device=device,
            compute_type=compute_type,
            fmt=args.fmt,
            cookies_path=args.cookies,
            use_polish=not getattr(args, "no_polish", False),
            use_ollama=getattr(args, "ollama", False),
            ollama_model=getattr(args, "ollama_model", None),
        )
        return

    audio_path = output_dir / "live_audio.wav"
    transcript_path = output_dir / f"transcript.{args.fmt}"

    print("=" * 60, file=sys.stderr)
    print(f"[1/2] 오디오 다운로드 ({args.duration}분, {args.back}분 전부터)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    download_live_audio(
        args.url,
        audio_path,
        minutes_back=args.back,
        duration_minutes=args.duration,
        cookies_path=args.cookies,
    )

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("[2/2] 음성 인식", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    segments = transcribe_audio(
        audio_path,
        model_id=args.model,
        language=args.language,
        device=device,
        compute_type=compute_type,
    )

    # 이어붙이기 모드: 기존 스크립트와 병합
    if args.resume:
        from .merge import merge_transcripts, load_transcript
        existing = load_transcript(args.resume)
        if existing:
            segments = merge_transcripts(existing, segments)
            print(f"[이어붙이기] {args.resume} 기반 병합 완료", file=sys.stderr)
        # 병합 결과를 기존 파일에도 덮어쓰기
        save_transcript(segments, args.resume, fmt=args.fmt)
        print(f"[저장] {args.resume} (병합본)", file=sys.stderr)

    save_transcript(segments, transcript_path, fmt=args.fmt)


def _cmd_download(args):
    """YouTube 라이브 오디오만 다운로드."""
    print(f"[다운로드] {args.url} -> {args.output}", file=sys.stderr)
    download_live_audio(
        args.url,
        args.output,
        minutes_back=args.back,
        duration_minutes=args.duration,
    )
    print(f"[완료] {args.output}", file=sys.stderr)


def _cmd_enroll(args):
    """화자 목소리 사전 등록."""
    from .diarize import SpeakerDiarizer
    from .audio_capture import capture_audio, find_builtin_mic, SAMPLE_RATE

    # --device 미지정 시 내장 마이크 자동 탐색
    if args.device is None:
        builtin = find_builtin_mic()
        if builtin is not None:
            print(f"[자동 감지] 내장 마이크 사용: [{builtin}]", file=sys.stderr)
            args.device = builtin

    profiles_path = Path(args.profiles)

    # --list: 등록된 화자 목록
    if args.list_profiles:
        entries = SpeakerDiarizer.list_profiles(profiles_path)
        if not entries:
            print(f"등록된 화자 없음 ({profiles_path})")
            return
        print(f"등록된 화자 ({profiles_path}):")
        for e in entries:
            print(f"  - {e['name']}  (등록: {e['enrolled_at']})")
        return

    # --delete: 화자 삭제
    if args.delete:
        if SpeakerDiarizer.delete_from_profiles(profiles_path, args.delete):
            print(f"[삭제] '{args.delete}' 삭제 완료", file=sys.stderr)
        else:
            print(f"[오류] '{args.delete}' 찾을 수 없음", file=sys.stderr)
        return

    # 이름 필수 확인
    if not args.name:
        print("[오류] 등록할 화자 이름을 지정하세요.", file=sys.stderr)
        print("  사용법: python -m src.cli enroll \"이름\"", file=sys.stderr)
        print("  목록:   python -m src.cli enroll --list", file=sys.stderr)
        sys.exit(EXIT_ARG_ERROR)

    duration = max(args.duration, 5)  # 최소 5초

    # 기존 화자 중복 확인
    existing = SpeakerDiarizer.list_profiles(profiles_path)
    existing_names = {e["name"] for e in existing}
    if args.name in existing_names:
        answer = input(f"'{args.name}' 이미 등록됨. 덮어쓸까요? (y/N): ").strip().lower()
        if answer != "y":
            print("[취소]", file=sys.stderr)
            return

    # pyannote 모델 로드
    if not SpeakerDiarizer.is_available():
        print("[오류] pyannote-audio 미설치. pip install pyannote-audio", file=sys.stderr)
        return

    if args.cpu:
        device = "cpu"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[모델] pyannote 임베딩 모델 로드 중 ({device})...", file=sys.stderr)
    diarizer = SpeakerDiarizer(hf_token=os.getenv("HF_TOKEN"), device=device)
    print("[모델] 로드 완료", file=sys.stderr)

    # 녹음
    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"  '{args.name}' 목소리 등록 — {duration}초 녹음", file=sys.stderr)
    print("  녹음 시작 후 자연스럽게 말해주세요.", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    input("Enter를 눌러 녹음 시작...")

    print(f"\n[녹음 중] {duration}초...", file=sys.stderr)
    chunks = []
    recorded = 0.0
    for chunk in capture_audio(chunk_seconds=1.0, device=args.device):
        chunks.append(chunk)
        recorded += 1.0
        remaining = duration - recorded
        if remaining > 0:
            print(f"  {remaining:.0f}초 남음...", end="\r", file=sys.stderr)
        if recorded >= duration:
            break

    audio = np.concatenate(chunks)
    print(f"\n[녹음 완료] {recorded:.0f}초, {len(audio)} 샘플", file=sys.stderr)

    # 임베딩 추출 + 등록
    print("[등록] 임베딩 추출 중...", file=sys.stderr)
    diarizer.enroll(args.name, audio, SAMPLE_RATE)

    # 저장
    diarizer.save_profiles(profiles_path)
    print(f"[저장] {profiles_path}", file=sys.stderr)
    print(f"[완료] '{args.name}' 등록 성공!", file=sys.stderr)
    print("\n회의에서 사용:", file=sys.stderr)
    print(f"  python -m src.cli meeting --profiles {profiles_path}", file=sys.stderr)


class StreamingAGC:
    """스트리밍 볼륨 자동 보정 (Automatic Gain Control)"""

    def __init__(self, target_rms: float = 0.05, attack: float = 0.1, release: float = 0.01):
        self.target_rms = target_rms
        self.attack = attack
        self.release = release
        self.gain = 1.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        audio = chunk.astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            desired = self.target_rms / rms
            alpha = self.attack if desired < self.gain else self.release
            self.gain = alpha * desired + (1 - alpha) * self.gain
            self.gain = np.clip(self.gain, 0.1, 10.0)
        return np.clip(audio * self.gain, -0.95, 0.95)


def _match_speaker_segment(
    seg_start: float, seg_end: float, speaker_segments: list[dict],
) -> str:
    """STT 세그먼트 시간과 화자 구간을 매칭하여 화자 라벨 반환.

    겹치는 구간이 가장 긴 화자를 선택한다.
    매칭 실패 시 "화자" 반환.
    """
    best_speaker = "화자"
    best_overlap = 0.0

    for spk_seg in speaker_segments:
        overlap_start = max(seg_start, spk_seg["start"])
        overlap_end = min(seg_end, spk_seg["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = spk_seg["speaker"]

    return best_speaker


def _build_meeting_prompt(domain_hint: str, rolling: str, max_chars: int = 800) -> str:
    """도메인 힌트 + rolling context 조합"""
    remaining = max_chars - len(domain_hint)
    return domain_hint + rolling[-remaining:] if remaining > 0 else domain_hint[:max_chars]


def _wait_for_server_ready(port: int, timeout_seconds: float = 1.5) -> None:
    """로컬 서버 readiness를 짧게 poll해 고정 sleep을 줄인다."""
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    status_url = f"http://127.0.0.1:{port}/status"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=0.25):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)


def _segment_retry_score(seg: dict, active_keywords: set[str]) -> float:
    """저신뢰 세그먼트 BoN 후보 선택 점수."""
    text = (seg.get("text") or "").strip()
    score = float(seg.get("avg_logprob", -10.0))
    score -= float(seg.get("compression_ratio", 0.0)) * 0.35
    score -= float(seg.get("no_speech_prob", 0.0)) * 0.6
    if active_keywords and text:
        lowered = text.lower()
        score += 0.2 * sum(1 for kw in active_keywords if kw.lower() in lowered)
    return score


def _cmd_approve_profiles(args):
    """회의 검토 후 후보 프로필을 승인 적용한다."""
    import json
    import shutil

    review_path = Path(args.review)
    if not review_path.exists():
        raise FileNotFoundError(f"리뷰 파일 없음: {review_path}")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidate_path = Path(
        review.get("candidate_profiles_path")
        or review.get("pending_profiles_path")
        or ""
    )
    if not candidate_path.exists():
        raise FileNotFoundError(f"후보 프로필 파일 없음: {candidate_path}")

    target_path = Path(
        args.profiles
        or review.get("base_profiles_path")
        or review.get("profiles_source")
        or ""
    )
    if not str(target_path):
        raise ValueError("적용 대상 profiles 경로를 찾을 수 없습니다.")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    if target_path.exists():
        shutil.copyfile(target_path, backup_path)

    shutil.copyfile(candidate_path, target_path)
    review["approved"] = True
    review["approved_at"] = __import__("datetime").datetime.now().isoformat()
    review["applied_profiles_path"] = str(target_path)
    review["backup_path"] = str(backup_path) if backup_path.exists() else ""
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[프로필 승인] {target_path}", file=sys.stderr)
    if backup_path.exists():
        print(f"[백업] {backup_path}", file=sys.stderr)


def _mean_word_probability(seg: dict) -> float:
    words = seg.get("words") or []
    if not words:
        return 0.0
    probs = [float(word.get("probability", 0.0)) for word in words]
    return sum(probs) / len(probs)


def _score_transcription_candidate(segments: list[dict]) -> float:
    if not segments:
        return float("-inf")

    total_score = 0.0
    total_duration = 0.0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        duration = max(float(seg.get("end", 0)) - float(seg.get("start", 0)), 0.1)
        if not text:
            total_score -= 2.0 * duration
            total_duration += duration
            continue
        avg_lp = float(seg.get("avg_logprob", -2.0))
        no_speech = float(seg.get("no_speech_prob", 0.0))
        compression = float(seg.get("compression_ratio", 0.0))
        wp = _mean_word_probability(seg)

        seg_score = avg_lp
        seg_score -= no_speech * 1.5
        seg_score -= max(0.0, compression - 1.8) * 0.8
        seg_score += wp
        # 환각 복합 페널티
        if compression > 2.0 and avg_lp < -0.7:
            seg_score -= 1.5
        # 반복 bigram 페널티
        words = text.split()
        if len(words) >= 4:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if len(set(bigrams)) / len(bigrams) < 0.5:
                seg_score -= 2.0

        total_score += seg_score * duration
        total_duration += duration
    return total_score / max(total_duration, 0.1)


def _needs_bon_retry(segments: list[dict], bad_ratio_threshold: float = 0.35) -> bool:
    if not segments:
        return False
    bad_count = 0
    evaluated_count = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        duration = max(float(seg.get("end", 0)) - float(seg.get("start", 0)), 0.0)
        # 2초 미만 짧은 추임새는 평가 제외
        if duration < 2.0 and len(text) < 10:
            continue
        evaluated_count += 1
        is_bad = (
            float(seg.get("avg_logprob", 0.0)) < -0.9
            or float(seg.get("no_speech_prob", 0.0)) > 0.5
            or float(seg.get("compression_ratio", 0.0)) > 2.4
            or (seg.get("words") and _mean_word_probability(seg) < 0.45)
        )
        if is_bad:
            bad_count += 1
    return evaluated_count > 0 and (bad_count / evaluated_count) >= bad_ratio_threshold


def _run_selective_bon(
    worker,
    chunk,
    args,
    rolling_context: str,
    domain_hint: str,
    baseline_segments: list[dict],
) -> list[dict]:
    # Pool 워커 >= 2면 이미 Pool에서 BoN 완료 → skip
    if hasattr(worker, 'ready_workers') and worker.ready_workers >= 2:
        return baseline_segments
    if not _needs_bon_retry(baseline_segments):
        return baseline_segments
    # no_speech > 0.6 && logprob < -1.0 세그먼트만 있으면 skip (무음)
    real_segments = [s for s in baseline_segments if not (
        float(s.get("no_speech_prob", 0)) > 0.6 and float(s.get("avg_logprob", 0)) < -1.0
    )]
    if not real_segments:
        return baseline_segments

    # 단일 retry: baseline이 매우 나쁘면 beam_size=8, 아니면 beam_size=5
    avg_lp = sum(float(s.get("avg_logprob", -2)) for s in baseline_segments) / max(len(baseline_segments), 1)
    retry_config = {
        "beam_size": 8 if avg_lp < -1.2 else 5,
        "temperature": 0.2,
        "condition_on_previous_text": False,
        "initial_prompt": _build_meeting_prompt(domain_hint, rolling_context),
    }
    try:
        retry_segments = worker.transcribe(
            chunk, language=args.language, vad_filter=True,
            vad_parameters={"threshold": 0.45, "min_speech_duration_ms": 250,
                           "min_silence_duration_ms": 500, "speech_pad_ms": 400},
            hallucination_silence_threshold=2.0, compression_ratio_threshold=2.4,
            no_speech_threshold=0.45, log_prob_threshold=-1.0, word_timestamps=True,
            **retry_config,
        )
    except Exception:
        return baseline_segments
    return max([baseline_segments, retry_segments], key=_score_transcription_candidate)


def _collect_live_corrections(
    polish_pool,
    correction_futures: list[tuple],
    wait_budget_seconds: float = 8.0,
) -> dict[str, str]:
    corrected_map: dict[str, str] = {}
    if not polish_pool:
        return corrected_map

    deadline = time.monotonic() + max(wait_budget_seconds, 0.0)
    try:
        for future, original_batch in correction_futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("[후처리] 실시간 교정 정리 시간 초과, 남은 배치는 건너뜁니다.", file=sys.stderr)
                break
            try:
                _, ok, corrected = future.result(timeout=remaining)
            except TimeoutError:
                print("[후처리] 실시간 교정 정리 시간 초과, 남은 배치는 건너뜁니다.", file=sys.stderr)
                break
            except Exception:
                continue
            if not ok:
                continue
            for original_line, corrected_line in zip(original_batch, corrected):
                if original_line != corrected_line:
                    corrected_map[original_line] = corrected_line
    finally:
        polish_pool.shutdown(wait=False, cancel_futures=True)

    return corrected_map

def _cmd_meeting_offline(args):
    """오프라인 회의록 변환 — 파일 → Whisper 전사 → MeetingWriter(.md) → 후처리."""
    from .transcribe import transcribe_audio, transcribe_chunks, get_audio_duration
    from .meeting_writer import MeetingWriter

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {audio_path}", file=sys.stderr)
        sys.exit(EXIT_NOT_FOUND)

    is_json = getattr(args, "json_mode", False)
    getattr(args, "ndjson_mode", False)

    print(f"[오프라인 회의록] {audio_path}", file=sys.stderr)
    print("[모드] 파일 → 회의록 .md 변환", file=sys.stderr)

    # 디바이스 설정
    device_name = "cpu" if args.cpu else None
    compute_type = "int8" if args.cpu else None

    # 오디오 길이 확인 → 청크 분할 결정
    try:
        duration_seconds = get_audio_duration(str(audio_path))
        duration_minutes = duration_seconds / 60
        print(f"[오디오] {duration_minutes:.1f}분", file=sys.stderr)
    except Exception:
        duration_seconds = 0.0
        duration_minutes = 0.0

    # 10분 이상이면 자동 청크 분할
    chunk_minutes = 10 if duration_minutes > 10 else 0

    # 전사 실행
    print(f"[전사] 모델: {args.model} | 청크: {chunk_minutes}분", file=sys.stderr)
    if chunk_minutes > 0:
        segments = transcribe_chunks(
            audio_path,
            chunk_minutes=chunk_minutes,
            model_id=args.model,
            language=args.language,
            device=device_name,
            compute_type=compute_type,
            beam_size=getattr(args, "beam", 5),
        )
    else:
        segments = transcribe_audio(
            audio_path,
            model_id=args.model,
            language=args.language,
            device=device_name,
            compute_type=compute_type,
            beam_size=getattr(args, "beam", 5),
        )

    if not segments:
        print("[오류] 전사 결과가 비어 있습니다.", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    # MeetingWriter로 회의록 양식 생성
    writer = MeetingWriter(output_path=args.output)

    for seg in segments:
        start_sec = seg.get("start", 0.0)
        end_sec = seg.get("end", 0.0)
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # 타임스탬프 포맷
        start_ts = time.strftime("%H:%M:%S", time.gmtime(start_sec))
        end_ts = time.strftime("%H:%M:%S", time.gmtime(end_sec))
        timestamp = f"{start_ts} ~ {end_ts}"

        writer.append_segment(
            speaker="Unknown",
            text=text,
            timestamp=timestamp,
            metadata={"start": start_sec, "end": end_sec},
        )

    # 종료 처리
    segment_count = len([s for s in segments if (s.get("text") or "").strip()])
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))
    speaker_count = len(writer._speakers)

    writer.write_footer(duration_str, segment_count, speaker_count)
    writer.close()

    print(f"\n[완료] {writer.output_path}", file=sys.stderr)
    print(f"[통계] {segment_count}개 세그먼트 | {duration_str} | {speaker_count}명", file=sys.stderr)

    # LLM 후처리 (meeting과 동일)
    _use_ollama = getattr(args, "ollama", False)
    _ollama_model = getattr(args, "ollama_model", "qwen3.5:9b")

    if not getattr(args, "no_polish", False) and segment_count > 0:
        from .polish import is_codex_available, is_gemini_available, is_ollama_available

        if is_codex_available() or is_gemini_available() or is_ollama_available(_ollama_model):
            import multiprocessing

            print(file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print("[후처리] LLM 후처리 (STT 교정 + 요약)", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            try:
                _status_file = str(writer.output_path.with_suffix(".postprocess-status.json"))
                proc = multiprocessing.Process(
                    target=_run_async_polish_process,
                    args=(
                        str(writer.output_path),
                        segment_count,
                        _use_ollama,
                        _ollama_model,
                        _status_file,
                    ),
                    daemon=False,
                )
                proc.start()
                print(f"[후처리] 백그라운드 실행 중 (PID: {proc.pid})", file=sys.stderr)
                print("[후처리] 터미널을 닫아도 후처리는 계속됩니다.", file=sys.stderr)
            except Exception as e:
                print(f"[후처리] 실패: {e}", file=sys.stderr)
        else:
            print("[후처리] Codex/Gemini/Ollama 미가용 — 후처리 건너뜀", file=sys.stderr)

    if is_json:
        print(json_output("success", "meeting", data={
            "mode": "offline",
            "audio": str(audio_path),
            "output": str(writer.output_path),
            "segments": segment_count,
            "duration": duration_str,
        }))


def _cmd_meeting(args):
    """회의 전사 — 마이크(실시간) 또는 파일(오프라인) → 회의록 .md."""

    # --file: 오프라인 모드 — 파일 → 회의록 .md 변환
    if getattr(args, "file", None):
        _cmd_meeting_offline(args)
        return

    from .audio_capture import list_audio_devices, capture_audio, find_builtin_mic

    # --list-devices: 장치 목록만 출력
    if args.list_devices:
        devices = list_audio_devices()
        if not devices:
            print("[오류] 사용 가능한 오디오 입력 장치가 없습니다.", file=sys.stderr)
            return
        print("사용 가능한 오디오 입력 장치:")
        for dev in devices:
            print(
                f"  [{dev['index']}] {dev['name']} "
                f"(ch={dev['channels']}, rate={dev['sample_rate']:.0f})"
            )
        return

    # --device 미지정 시 내장 마이크 자동 탐색
    if args.device is None:
        builtin = find_builtin_mic()
        if builtin is not None:
            print(f"[자동 감지] 내장 마이크 사용: [{builtin}]", file=sys.stderr)
            args.device = builtin

    import threading
    import webbrowser
    from .runtime_env import detect_device
    from .server import (
        run_server,
        push_transcript_sync,
        push_correction_sync,
        is_paused,
        is_shutdown_requested,
        get_keywords,
        set_postprocess_status,
        set_diarizer,
        add_extracted_keywords,
        consume_audio_device_switch,
        get_audio_device_switch_event,
        set_current_audio_device,
        set_startup_status,
        consume_session_rotate,
        set_session_rotate_callback,
        get_keywords_snapshot,
    )
    from .meeting import PipelineAdapter, PipelineContext, run_capture_loop
    from .meeting_writer import MeetingWriter
    from .tray import MeetingTray, is_available as tray_available

    # SSE 서버 시작 (데몬 스레드)
    set_current_audio_device(args.device)
    set_startup_status("booting", "로컬 UI 시작 중...")
    print(f"[서버] http://localhost:{args.port} 시작 중...", file=sys.stderr)
    server_thread = threading.Thread(
        target=run_server,
        args=("127.0.0.1", args.port),
        kwargs={"beta_mode": getattr(args, "_beta_mode", False)},
        daemon=True,
    )
    server_thread.start()
    _wait_for_server_ready(args.port, timeout_seconds=1.5)
    threading.Thread(
        target=webbrowser.open,
        args=(f"http://localhost:{args.port}",),
        daemon=True,
    ).start()

    # 파일 저장
    writer = MeetingWriter(output_path=args.output)
    writer.write_header()
    print(f"[저장] {writer.output_path}", file=sys.stderr)

    pending_profiles_path = writer.output_path.with_suffix(".profiles.pending.json")
    audio_offset_seconds = 0.0

    # 디바이스 설정
    set_startup_status("device", "가속기 감지 중...")
    if args.cpu:
        device, compute_type = "cpu", "int8"
    else:
        device, compute_type = detect_device()

    # Whisper 모델 동기 로드 (메인 프로세스 — Windows CUDA 서브프로세스 hang 회피)
    set_startup_status("loading_asr", f"STT 모델 로드 중 ({device}/{compute_type})...")
    print(f"[모델] {args.model} 로드 중 ({device}/{compute_type})...", file=sys.stderr)

    from .runtime_env import bootstrap_nvidia_dll_path
    bootstrap_nvidia_dll_path()
    from faster_whisper import WhisperModel as _WM
    _whisper_model = _WM(args.model, device=device, compute_type=compute_type)
    if device == "cuda":
        from .transcribe import _register_cuda_exit_guard
        _register_cuda_exit_guard()
    print("[모델] 로드 완료", file=sys.stderr)

    # WhisperWorkerPool 호환 래퍼 (is_ready / transcribe 인터페이스)
    class _InlineWhisperWorker:
        @property
        def is_ready(self):
            return True

        def transcribe(self, audio, **kwargs):
            segs, _info = _whisper_model.transcribe(audio, **kwargs)
            result = []
            for s in segs:
                d = {
                    "start": s.start, "end": s.end, "text": s.text,
                    "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                    "compression_ratio": s.compression_ratio,
                }
                if hasattr(s, "words") and s.words:
                    d["words"] = [
                        {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                        for w in s.words
                    ]
                result.append(d)
            return result

        def stop(self):
            pass

    worker = _InlineWhisperWorker()
    # AGC (볼륨 자동 보정) — 마이크 거리 차이 보정
    agc = StreamingAGC()

    # 화자 분리기 (선택)
    diarizer = None
    if not args.no_diarize:
        from .diarize import SpeakerDiarizer
        if SpeakerDiarizer.is_available():
            set_startup_status("loading_diarizer", "화자 분리 모델 로드 중...")
            print("[화자 분리] pyannote 모델 로드 중...", file=sys.stderr)
            try:
                profiles_path = getattr(args, "profiles", None)
                diarizer = SpeakerDiarizer(
                    hf_token=os.getenv("HF_TOKEN"),
                    device=device,
                    profiles_path=profiles_path,
                )
                # pyannote CUDA 모델이 메인 프로세스에 로드됨
                # → 종료 시 torch CUDA 정리 segfault (exit 127) 방지
                if device == "cuda":
                    from .transcribe import _register_cuda_exit_guard
                    _register_cuda_exit_guard()
                print("[화자 분리] 로드 완료", file=sys.stderr)
                if profiles_path:
                    names = list(diarizer._enrolled_names)
                    print(f"[화자 분리] 프로필 모드: {', '.join(names)} ({len(names)}명)", file=sys.stderr)
            except Exception as e:
                print(f"[화자 분리] 로드 실패: {e}", file=sys.stderr)
                print("[화자 분리] 화자 분리 없이 진행합니다.", file=sys.stderr)
        else:
            print("[화자 분리] pyannote-audio 미설치 — 화자 분리 비활성화", file=sys.stderr)
            print("[화자 분리] 설치: pip install 'media-transcriber[diarize]'", file=sys.stderr)

    # diarizer를 서버에 연결 (웹 UI 화자 등록용)
    profiles_path = getattr(args, "profiles", None)
    set_diarizer(diarizer, profiles_path)
    # 모델 로드 완료 시에만 ready 전환 (비동기 로드 중이면 캡처 루프에서 전환)
    if worker.is_ready:
        set_startup_status("ready", "녹음 준비 완료", ready=True)

    # 트레이 아이콘 (Windows)
    tray: MeetingTray | None = None
    if tray_available():
        import src.server as _srv

        def _toggle_pause_from_tray() -> None:
            _srv.toggle_pause_state()

        def _shutdown_from_tray() -> None:
            _srv.request_shutdown()

        tray = MeetingTray(
            port=args.port,
            on_toggle_pause=_toggle_pause_from_tray,
            on_shutdown=_shutdown_from_tray,
        )
        tray.start()
        print("[트레이] 시스템 트레이 아이콘 활성화", file=sys.stderr)

    # 미디어 자동 음소거 (--auto-mute 또는 config)
    _media_muted = False
    if getattr(args, "auto_mute", False):
        try:
            from .media_control import is_available as _media_ok, mute_media_apps
            if _media_ok():
                muted = mute_media_apps(mute=True)
                if muted:
                    _media_muted = True
                    print(f"[미디어] 음소거 적용: {', '.join(muted.keys())}", file=sys.stderr)
        except Exception as e:
            print(f"[미디어] 음소거 실패 (무시): {e}", file=sys.stderr)

    def _build_dynamic_domain_hint() -> str:
        """기본 도메인 힌트 + 웹 UI에서 추가한 동적 키워드 결합."""
        dynamic_keywords = sorted(get_keywords())
        if not dynamic_keywords:
            return DEFAULT_DOMAIN_HINT
        return DEFAULT_DOMAIN_HINT + ", ".join(dynamic_keywords) + ". "

    def _format_input_device(device_index: int | None) -> str:
        return "기본 장치" if device_index is None else f"[{device_index}]"

    # 세션 회전 콜백 등록 (녹음 루프 진입 전)
    def _save_current_session():
        """세션 회전 시 현재 세션 저장."""
        nonlocal writer, segment_count, start_time
        duration = time.time() - start_time
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        speaker_count = diarizer.get_speaker_count() if diarizer else 1
        writer.set_keywords(get_keywords_snapshot())
        write_session_json = getattr(writer, "write_session_json", None)
        if callable(write_session_json):
            write_session_json(duration, segment_count, speaker_count)
        writer.write_footer(duration_str, segment_count, speaker_count)

    set_session_rotate_callback(_save_current_session)

    # 실시간 처리 루프
    _model_announced = False
    if worker.is_ready:
        print("[모델] 로드 완료", file=sys.stderr)
        _model_announced = True
    _audio_prefetch_buffer: list[np.ndarray] = []  # 모델 로드 중 오디오 선캡처 버퍼

    print(f"[캡처] 마이크 캡처 시작 (청크={args.chunk}초)", file=sys.stderr)
    print("[캡처] Ctrl+C로 종료", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    start_time = time.time()
    segment_count = 0
    # 이전 전사 결과를 rolling context로 유지 (최대 800자)
    rolling_context = _build_meeting_prompt(_build_dynamic_domain_hint(), args.prompt)

    # 실시간 후처리 초기화 (녹음 중 백그라운드 교정 + 키워드 추출)
    # 폴백 체인: gemini-2.5-flash → gemini-3-flash-preview → codex(codex_low) → ollama
    _polish_pool = None
    _correction_futures: list[tuple] = []
    _use_ollama = getattr(args, "ollama", False)
    _ollama_model = getattr(args, "ollama_model", None) or "qwen3.5:9b"
    _correct_fn = None
    _extract_kw_fn = None
    _polish_engine = "none"
    if not args.no_polish:
        from .polish import (
            is_codex_available as _codex_ok,
            is_gemini_available as _gemini_ok,
            is_ollama_available as _ollama_ok,
            _GEMINI_MODELS,
        )
        if _use_ollama and _ollama_ok(_ollama_model):
            _polish_engine = "ollama"
        elif _gemini_ok():
            _polish_engine = "gemini"
        elif _codex_ok():
            _polish_engine = "codex"
        elif _ollama_ok(_ollama_model):
            _polish_engine = "ollama"
            print(f"[후처리] Gemini/Codex 미설치 → Ollama ({_ollama_model}) 폴백", file=sys.stderr)

        if _polish_engine != "none":
            from concurrent.futures import ThreadPoolExecutor as _TPE
            _polish_pool = _TPE(max_workers=2, thread_name_prefix="live-polish")

            if _polish_engine == "gemini":
                from .polish import _correct_batch_gemini, extract_keywords_with_gemini
                _gemini_model_idx = 0
                _gemini_lock = threading.Lock()

                def _correct_fn(batch, idx, t=120):
                    nonlocal _gemini_model_idx
                    with _gemini_lock:
                        model = _GEMINI_MODELS[_gemini_model_idx]
                    result = _correct_batch_gemini(batch, idx, model, t)
                    if not result[1] and len(_GEMINI_MODELS) > 1:
                        with _gemini_lock:
                            if _gemini_model_idx == 0:
                                _gemini_model_idx = 1
                                fallback = _GEMINI_MODELS[1]
                                print(f"[후처리] {model} 실패 → {fallback} 폴백", file=sys.stderr)
                            else:
                                fallback = _GEMINI_MODELS[_gemini_model_idx]
                        result = _correct_batch_gemini(batch, idx, fallback, t)
                    return result

                def _extract_kw_fn(text):
                    with _gemini_lock:
                        model = _GEMINI_MODELS[_gemini_model_idx]
                    return extract_keywords_with_gemini(text, model=model)

                print(f"[후처리] 엔진: Gemini ({_GEMINI_MODELS[0]})", file=sys.stderr)

            elif _polish_engine == "codex":
                from .polish import _correct_batch, extract_keywords_with_codex
                _work_dir = writer.output_path.parent

                def _correct_fn(batch, idx, t=120):
                    return _correct_batch(batch, idx, _work_dir, t)

                def _extract_kw_fn(text):
                    return extract_keywords_with_codex(text, _work_dir)

                print("[후처리] 엔진: Codex (codex_low)", file=sys.stderr)

            else:  # ollama
                from .polish import _correct_batch_ollama, extract_keywords_with_ollama

                def _correct_fn(batch, idx, t=120):
                    return _correct_batch_ollama(batch, idx, _ollama_model, t)

                def _extract_kw_fn(text):
                    return extract_keywords_with_ollama(text, model=_ollama_model)

                print(f"[후처리] 엔진: Ollama ({_ollama_model})", file=sys.stderr)

    _capture_switch_event = get_audio_device_switch_event()
    _active_input_device = args.device
    _fallback_input_device = args.device
    _shutdown_logged = False
    chunk_offset_seconds = 0.0
    heuristic_min = max(4.0, min(args.chunk * 0.55, args.chunk - 1.0))

    def _rotate_session() -> None:
        nonlocal writer, segment_count, start_time, rolling_context
        nonlocal pending_profiles_path, audio_offset_seconds, chunk_offset_seconds

        _save_current_session()
        writer.close()

        writer = MeetingWriter(output_path=args.output)
        writer.write_header()
        print(f"\n[새 세션] {writer.output_path}", file=sys.stderr)

        segment_count = 0
        start_time = time.time()
        rolling_context = _build_meeting_prompt(_build_dynamic_domain_hint(), args.prompt)
        audio_offset_seconds = 0.0
        chunk_offset_seconds = 0.0
        _audio_prefetch_buffer.clear()

        set_session_rotate_callback(_save_current_session)
        pending_profiles_path = writer.output_path.with_suffix(".profiles.pending.json")
        print("[새 세션] 녹음 계속 중 (모델 유지)", file=sys.stderr)

    def _is_shutdown_requested() -> bool:
        nonlocal _shutdown_logged
        requested = is_shutdown_requested()
        if requested and not _shutdown_logged:
            print("\n[종료 요청] 웹 UI에서 저장 & 종료 요청", file=sys.stderr)
            _shutdown_logged = True
        return requested

    def _preprocess_chunk(chunk: np.ndarray) -> np.ndarray | None:
        nonlocal audio_offset_seconds, chunk_offset_seconds, _model_announced

        chunk = agc.process(chunk)
        chunk_duration_seconds = len(chunk) / 16000.0
        chunk_offset_seconds = audio_offset_seconds
        audio_offset_seconds += chunk_duration_seconds
        writer.append_audio(chunk)

        try:
            _worker_ready = worker.is_ready
        except RuntimeError as exc:
            print(f"[오류] {exc}", file=sys.stderr)
            set_startup_status("error", str(exc))
            raise
        if not _worker_ready:
            _audio_prefetch_buffer.append(chunk)
            set_startup_status(
                "loading_asr",
                f"모델 로드 중… (오디오 {len(_audio_prefetch_buffer)}청크 버퍼링)",
            )
            return None

        if not _model_announced:
            print("[모델] 로드 완료", file=sys.stderr)
            _model_announced = True
            set_startup_status("ready", "녹음 준비 완료", ready=True)

        if _audio_prefetch_buffer:
            buffered_count = len(_audio_prefetch_buffer)
            print(
                f"[선캡처] 버퍼된 {buffered_count}청크를 현재 청크에 병합",
                file=sys.stderr,
            )
            _audio_prefetch_buffer.append(chunk)
            chunk = np.concatenate(_audio_prefetch_buffer)
            _audio_prefetch_buffer.clear()

        return chunk

    def _timestamp_provider() -> str:
        elapsed = time.time() - start_time
        return time.strftime("%H:%M:%S", time.gmtime(elapsed))

    def _transcribe_kwargs_factory(_chunk: np.ndarray) -> dict[str, Any]:
        nonlocal rolling_context
        rolling_context = _build_meeting_prompt(_build_dynamic_domain_hint(), rolling_context)
        return {
            "temperature": 0.0,
            "initial_prompt": rolling_context,
        }

    def _transcribe_runner(
        stt_worker: Any,
        chunk: np.ndarray,
        kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        stt_segments = stt_worker.transcribe(chunk, **kwargs)
        return _run_selective_bon(
            stt_worker,
            chunk,
            args,
            rolling_context,
            _build_dynamic_domain_hint(),
            stt_segments,
        )

    def _segment_time_mapper(seg: dict[str, Any]) -> tuple[float, float]:
        return (
            chunk_offset_seconds + float(seg.get("start", 0.0)),
            chunk_offset_seconds + float(seg.get("end", 0.0)),
        )

    def _on_raw_segment(payload: dict[str, Any]) -> None:
        alignment_payload = dict(payload)
        alignment_payload["chunk_offset"] = chunk_offset_seconds
        writer.append_alignment(alignment_payload)

    def _on_transcript(payload: dict[str, Any]) -> None:
        push_transcript_sync(payload["speaker"], payload["text"], payload["timestamp"], confidence=payload.get("confidence"))
        writer.append_segment(
            payload["speaker"],
            payload["text"],
            payload["timestamp"],
            metadata={
                "start": payload["start"],
                "end": payload["end"],
                "feedback_text": payload["feedback_text"],
                "confidence": payload.get("confidence"),
            },
        )
        print(f"[{payload['timestamp']}] [{payload['speaker']}] {payload['text']}")

    def _on_segment_emitted(
        _segment: Any,
        timestamp: str,
        _stripped: str,
        emitted_count: int,
    ) -> None:
        nonlocal segment_count
        segment_count = emitted_count
        if tray:
            speaker_count = diarizer.get_speaker_count() if diarizer else 1
            tray.update_status(
                elapsed=timestamp,
                segments=segment_count,
                speakers=speaker_count,
                paused=is_paused(),
            )

    def _on_chunk_processed(_segments: list[Any], chunk_text: str, _count: int) -> None:
        nonlocal rolling_context
        rolling_context = _build_meeting_prompt(
            _build_dynamic_domain_hint(),
            rolling_context + " " + chunk_text,
        )

    def _submit_correction(batch: list[str], idx: int) -> Any:
        if _polish_pool is None or _correct_fn is None:
            return None
        return _polish_pool.submit(_correct_fn, batch, idx, 120)

    def _submit_keyword(text: str) -> None:
        if _polish_pool is None or _extract_kw_fn is None:
            return

        def _do_kw_extract(_text: str = text, _fn: Any = _extract_kw_fn) -> None:
            try:
                kws = _fn(_text)
                payload = add_extracted_keywords(list(kws or []))
                promoted = payload.get("promoted") or []
                extracted = payload.get("extracted") or []
                if kws:
                    print(
                        f"[키워드 추출] promoted={len(promoted)} extracted={len(extracted)}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"[키워드 추출] 실패: {exc}", file=sys.stderr)

        _polish_pool.submit(_do_kw_extract)

    def _on_correction_future(future: Any, batch: list[str]) -> None:
        _correction_futures.append((future, batch))

    def _on_capture_error(
        exc: Exception,
        active_device: int | None,
        _error_count: int,
    ) -> tuple[int | None, bool]:
        nonlocal _active_input_device

        if active_device != _fallback_input_device:
            print(
                "[마이크] 장치 전환 실패:"
                f" {_format_input_device(active_device)} ({exc})",
                file=sys.stderr,
            )
            print(
                "[마이크] 이전 장치로 복귀:"
                f" {_format_input_device(_fallback_input_device)}",
                file=sys.stderr,
            )
            _active_input_device = _fallback_input_device
            set_current_audio_device(
                _fallback_input_device,
                error=f"전환 실패: {exc}",
            )
            return _fallback_input_device, True
        raise exc

    def _on_device_switched(previous: int | None, current: int | None) -> None:
        nonlocal _active_input_device, _fallback_input_device

        print(
            "[마이크] 입력 장치 전환:"
            f" {_format_input_device(previous)} -> {_format_input_device(current)}",
            file=sys.stderr,
        )
        _fallback_input_device = previous
        _active_input_device = current

    context = PipelineContext(
        worker=worker,
        diarizer=diarizer,
        language=args.language,
        chunk_seconds=args.chunk,
        on_transcript=_on_transcript,
        on_correction=push_correction_sync,
        stop_event=threading.Event(),
    )
    adapter = PipelineAdapter(
        capture_audio=capture_audio,
        is_paused=is_paused,
        is_shutdown_requested=_is_shutdown_requested,
        consume_audio_device_switch=consume_audio_device_switch,
        set_current_audio_device=set_current_audio_device,
        capture_stop_event=_capture_switch_event,
        capture_kwargs={
            "heuristic_split": not getattr(args, "fixed_chunking", False),
            "min_chunk_seconds": heuristic_min,
        },
        consume_session_rotate=consume_session_rotate,
        on_session_rotate=_rotate_session,
        skip_chunk_on_rotate=True,
        reset_runtime_on_rotate=True,
        preprocess_chunk=_preprocess_chunk,
        transcribe_kwargs_factory=_transcribe_kwargs_factory,
        transcribe_runner=_transcribe_runner,
        timestamp_provider=_timestamp_provider,
        segment_time_mapper=_segment_time_mapper,
        on_raw_segment=_on_raw_segment,
        on_segment_emitted=_on_segment_emitted,
        on_chunk_processed=_on_chunk_processed,
        dedupe_recent_texts=True,
        recent_text_limit=10,
        submit_correction_batch=_submit_correction,
        on_correction_future=_on_correction_future,
        submit_keyword_job=_submit_keyword,
        keyword_every_segments=10,
        keyword_window=10,
        on_capture_error=_on_capture_error,
        on_device_switched=_on_device_switched,
        diarize_error_label="meeting",
    )
    try:
        run_capture_loop(
            context,
            adapter,
            initial_device=_active_input_device,
        )
    except KeyboardInterrupt:
        pass

    # 트레이 아이콘 종료
    if tray:
        tray.stop()
    worker.stop()

    # 종료 처리
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("[종료] 저장 중...", file=sys.stderr)
    set_postprocess_status("saving")

    duration = time.time() - start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
    speaker_count = diarizer.get_speaker_count() if diarizer else 1
    profiles_path = getattr(args, "profiles", None)
    auto_update = getattr(args, "auto_update", False)

    writer.set_artifact("alignment", writer.alignment_path)
    if profiles_path:
        writer.set_artifact("profiles_source", profiles_path)

    if diarizer and profiles_path:
        try:
            diarizer.save_profiles(pending_profiles_path)
            writer.set_artifact("pending_profiles", pending_profiles_path)
            writer.write_profile_review(
                {
                    "status": "pending_review",
                    "profiles_source": str(profiles_path),
                    "base_profiles_path": str(profiles_path),
                    "pending_profiles_path": str(pending_profiles_path),
                    "candidate_profiles_path": str(pending_profiles_path),
                    "auto_update_requested": auto_update,
                    "speaker_count": speaker_count,
                    "note": "이번 회의 라벨 검토 후 승인된 경우에만 profiles_source로 반영하세요.",
                }
            )
            print(f"[프로필 검토] 승인 대기 파일 생성: {pending_profiles_path}", file=sys.stderr)
        except Exception as e:
            print(f"[프로필 검토] 승인 대기 파일 생성 실패: {e}", file=sys.stderr)

    writer.write_footer(duration_str, segment_count, speaker_count)
    try:
        _corrected_map = _collect_live_corrections(_polish_pool, _correction_futures)
        if _corrected_map:
            corrected_count = writer.apply_segment_corrections(_corrected_map)
            if corrected_count:
                writer.write_footer(duration_str, segment_count, speaker_count)
                print(f"[실시간 교정] {corrected_count}줄 사전 교정 적용", file=sys.stderr)
    finally:
        writer.close()

    # Whisper는 별도 프로세스에서 종료되어 메인 프로세스 CUDA 정리 불필요
    if diarizer:
        del diarizer
        diarizer = None

    launched_postprocess = False

    # LLM 후처리 (Codex/Ollama: STT 교정, Gemini/Ollama: 요약 생성)
    if not args.no_polish and segment_count > 0:
        from .polish import is_codex_available, is_gemini_available, is_ollama_available

        if is_codex_available() or is_gemini_available() or is_ollama_available(_ollama_model):
            import multiprocessing

            print(file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print("[후처리] LLM 후처리", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            try:
                _status_file = str(writer.output_path.with_suffix(".postprocess-status.json"))
                proc = multiprocessing.Process(
                    target=_run_async_polish_process,
                    args=(
                        str(writer.output_path),
                        segment_count,
                        _use_ollama,
                        _ollama_model,
                        _status_file,
                    ),
                    daemon=False,
                )
                proc.start()
                from .server import set_postprocess_status_file
                set_postprocess_status_file(_status_file)
                print(f"[후처리] 백그라운드 실행 중 (PID: {proc.pid})", file=sys.stderr)
                print("[후처리] 터미널을 닫아도 후처리는 계속됩니다.", file=sys.stderr)
            except KeyboardInterrupt:
                print("[후처리] 사용자 중단 — 현재까지 저장된 회의록은 유지합니다.", file=sys.stderr)
            except Exception as e:
                print(f"[후처리] 실패: {e}", file=sys.stderr)
            else:
                launched_postprocess = True
                set_postprocess_status("queued", 0)

    if not launched_postprocess:
        set_postprocess_status("done", 100)
    print(f"[저장] {writer.output_path}", file=sys.stderr)
    print(f"[완료] {segment_count}개 세그먼트 | {duration_str} | 화자 {speaker_count}명", file=sys.stderr)

    # 미디어 상태 복원
    if _media_muted:
        try:
            from .media_control import restore_media_state
            restore_media_state()
            print("[미디어] 음소거 해제 — 이전 상태 복원", file=sys.stderr)
        except Exception as e:
            print(f"[미디어] 복원 실패: {e}", file=sys.stderr)

    # Discord Webhook 회의 요약 전송
    try:
        from .discord_notify import is_configured, send_meeting_summary
        if is_configured():
            from .discord_notify import _get_webhook_url
            _wh_url = _get_webhook_url()
            if _wh_url:
                _summary = f"{segment_count}개 세그먼트 | {duration_str} | 화자 {speaker_count}명"
                _session_id = writer.output_path.parent.name if hasattr(writer, "output_path") else "unknown"
                send_meeting_summary(_wh_url, _summary, speaker_count, duration_str, _session_id)
                print("[디스코드] 회의 요약 전송 완료", file=sys.stderr)
    except Exception as e:
        print(f"[디스코드] 전송 실패 (무시): {e}", file=sys.stderr)

    # 완료 후 출력 폴더 열기 (Windows)
    if sys.platform == "win32":
        os.startfile(str(writer.output_path.parent))


def _cmd_update(args: argparse.Namespace) -> None:
    """GitHub Releases 기반 업데이트 확인 및 설치."""
    from .updater import (
        UpdateError,
        check_for_update,
        download_update,
        get_current_version,
        verify_checksum,
    )

    use_json = getattr(args, "update_json", False)
    current_version = get_current_version()

    # --check: 업데이트 가능 여부만 확인
    if args.check:
        try:
            info = check_for_update()
        except UpdateError as exc:
            if use_json:
                print(json_output("error", "update", error=str(exc), code="ERROR"))
            else:
                print(f"[업데이트] 확인 실패: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

        if info is None:
            if use_json:
                print(json_output("success", "update", data={
                    "current_version": current_version,
                    "update_available": False,
                }))
            else:
                print(f"[업데이트] 최신 버전입니다. (현재: {current_version})")
            return

        if use_json:
            print(json_output("success", "update", data={
                "current_version": current_version,
                "update_available": True,
                "latest_version": info.version,
                "download_url": info.download_url,
                "published_at": info.published_at,
                "release_notes": info.release_notes,
            }))
        else:
            print(f"[업데이트] 새 버전 발견: {info.version} (현재: {current_version})")
            print(f"  게시일: {info.published_at}")
            print("  설치: sonote update --install")
            if info.release_notes:
                # 릴리스 노트 앞 3줄만 미리보기
                preview_lines = info.release_notes.strip().splitlines()[:3]
                print("  릴리스 노트:")
                for line in preview_lines:
                    print(f"    {line}")
        return

    # --install: 다운로드 + 설치
    if args.install:
        # PyInstaller EXE 환경 여부 확인
        import sys as _sys
        if not getattr(_sys, "frozen", False):
            msg = "EXE 환경에서만 자동 설치가 가능합니다. (개발 환경에서는 --check만 사용 가능)"
            if use_json:
                print(json_output("error", "update", error=msg, code="ERROR"))
            else:
                print(f"[업데이트] {msg}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

        try:
            info = check_for_update()
        except UpdateError as exc:
            if use_json:
                print(json_output("error", "update", error=str(exc), code="ERROR"))
            else:
                print(f"[업데이트] 확인 실패: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

        if info is None:
            if use_json:
                print(json_output("success", "update", data={
                    "current_version": current_version,
                    "update_available": False,
                }))
            else:
                print(f"[업데이트] 이미 최신 버전입니다. (현재: {current_version})")
            return

        # 사용자 확인 (JSON 모드는 자동 진행)
        if not use_json:
            print(f"[업데이트] {current_version} → {info.version}")
            print(f"  다운로드 URL: {info.download_url}")
            answer = input("  설치하시겠습니까? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("[업데이트] 설치를 취소했습니다.")
                return

        # 다운로드
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        dest = tmp_dir / f"sonote_{info.version}.exe"

        if not use_json:
            print(f"[업데이트] 다운로드 중... {info.download_url}")

        last_pct = [-1]

        def _progress(downloaded: int, total: int) -> None:
            if total <= 0 or use_json:
                return
            pct = int(downloaded * 100 / total)
            if pct != last_pct[0] and pct % 10 == 0:
                last_pct[0] = pct
                print(f"  [{pct:3d}%] {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB")

        try:
            from .updater import download_update
            download_update(info.download_url, dest, progress_callback=_progress)
        except UpdateError as exc:
            if use_json:
                print(json_output("error", "update", error=str(exc), code="ERROR"))
            else:
                print(f"[업데이트] 다운로드 실패: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)

        # SHA256 체크섬 검증
        if info.checksum_sha256:
            if not use_json:
                print("[업데이트] SHA256 체크섬 검증 중...")
            if not verify_checksum(dest, info.checksum_sha256):
                dest.unlink(missing_ok=True)
                msg = "체크섬 검증 실패 — 파일이 손상되었거나 변조되었습니다."
                if use_json:
                    print(json_output("error", "update", error=msg, code="ERROR"))
                else:
                    print(f"[업데이트] {msg}", file=sys.stderr)
                sys.exit(EXIT_ERROR)
            if not use_json:
                print("[업데이트] 체크섬 검증 통과")
        else:
            if not use_json:
                print("[업데이트] 체크섬 정보 없음 — 검증 건너뜀")

        if use_json:
            print(json_output("success", "update", data={
                "current_version": current_version,
                "new_version": info.version,
                "status": "applying",
            }))
        else:
            print("[업데이트] 설치 중... (앱이 재시작됩니다)")

        # 업데이트 적용 (재시작 포함)
        try:
            from .updater import apply_update
            apply_update(dest)
        except UpdateError as exc:
            if use_json:
                print(json_output("error", "update", error=str(exc), code="ERROR"))
            else:
                print(f"[업데이트] 설치 실패: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
