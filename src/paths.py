"""출력 디렉토리 경로 관리 — 유형+날짜 기반 구조.

구조:
    output/
    ├── data/                     # 공유 데이터 (날짜 무관)
    │   ├── profiles.db
    │   └── speakers.json
    ├── meetings/                 # 회의록
    │   └── YYYY-MM-DD/
    ├── transcripts/              # 강의/라이브 전사
    │   └── YYYY-MM-DD/
    └── audio/                    # WAV 오디오 (대용량)
        └── YYYY-MM-DD/
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """프로젝트(또는 exe 번들) 루트 디렉토리를 반환한다."""
    if getattr(sys, "frozen", False):
        # PyInstaller onedir: exe 옆이 루트
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def static_dir() -> Path:
    """static 리소스 디렉토리를 반환한다."""
    if getattr(sys, "frozen", False):
        # PyInstaller: _MEIPASS 내부에 datas로 포함됨
        return Path(getattr(sys, "_MEIPASS", "")) / "static"
    return Path(__file__).resolve().parent.parent / "static"


# 프로젝트 루트 기준 output 디렉토리
OUTPUT_ROOT: Path = project_root() / "output"


def _ensure_dir(path: Path) -> Path:
    """디렉토리 생성 후 반환."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """공유 데이터 디렉토리 (profiles.db, speakers.json 등)."""
    return _ensure_dir(OUTPUT_ROOT / "data")


def meetings_dir(date: datetime | None = None) -> Path:
    """회의록 날짜별 디렉토리.

    Args:
        date: 기준 날짜. None이면 오늘.
    """
    date_str = (date or datetime.now()).strftime("%Y-%m-%d")
    return _ensure_dir(OUTPUT_ROOT / "meetings" / date_str)


def transcripts_dir(date: datetime | None = None) -> Path:
    """전사 결과 날짜별 디렉토리.

    Args:
        date: 기준 날짜. None이면 오늘.
    """
    date_str = (date or datetime.now()).strftime("%Y-%m-%d")
    return _ensure_dir(OUTPUT_ROOT / "transcripts" / date_str)


def audio_dir(date: datetime | None = None) -> Path:
    """오디오 파일 날짜별 디렉토리.

    Args:
        date: 기준 날짜. None이면 오늘.
    """
    date_str = (date or datetime.now()).strftime("%Y-%m-%d")
    return _ensure_dir(OUTPUT_ROOT / "audio" / date_str)


def profiles_db_path() -> Path:
    """프로필 DB 경로."""
    return data_dir() / "profiles.db"


def speakers_json_path() -> Path:
    """화자 프로필 JSON 경로."""
    return data_dir() / "speakers.json"
