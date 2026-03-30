"""src.export.export_session 단위 테스트."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from src.export import export_session


def _create_session_dir(
    tmp_path: Path,
    *,
    session_meta: dict | None = None,
    transcript_text: str | None = None,
    segments_payload: list[dict] | dict | None = None,
) -> Path:
    session_dir = tmp_path / "meetings" / "2026-03-13" / "101500"
    session_dir.mkdir(parents=True, exist_ok=True)

    if session_meta is not None:
        (session_dir / "session.json").write_text(
            json.dumps(session_meta, ensure_ascii=False),
            encoding="utf-8",
        )

    if transcript_text is not None:
        (session_dir / "transcript_001.txt").write_text(transcript_text, encoding="utf-8")

    if segments_payload is not None:
        (session_dir / "segments.json").write_text(
            json.dumps(segments_payload, ensure_ascii=False),
            encoding="utf-8",
        )

    return session_dir


def test_export_session_txt_parses_segments_and_renders_output(tmp_path: Path) -> None:
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"], "duration": "300"},
        transcript_text="\n".join(
            [
                "# comment",
                "[00:00:05] [Alice] 다섯 초 발화",
                "[00:00:03] Bob: 세 초 발화",
                "---",
                "[00:00:07 ~ 00:00:09] 범위 기반 발화",
            ]
        ),
    )

    exported = export_session(session_dir, "txt")

    assert exported == "\n".join(
        [
            "[00:00:03] Bob: 세 초 발화",
            "[00:00:05] Alice: 다섯 초 발화",
            "[00:00:07] Alice: 범위 기반 발화",
        ]
    )


def test_export_session_md_renders_title_attendees_and_transcript(tmp_path: Path) -> None:
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={
            "started_at": "2026-03-13T09:00:00",
            "duration": 90,
            "speakers": ["Bob", "Alice"],
        },
        transcript_text="[00:00:01] [Alice] 회의를 시작합니다.\n[00:00:05] [Bob] 진행 상황을 공유합니다.",
    )

    exported = export_session(session_dir, "md")

    assert exported.startswith("# 회의록 - 2026-03-13")
    assert "- 소요 시간: 00:01:30" in exported
    assert "## 참석자" in exported
    assert "- Alice" in exported
    assert "- Bob" in exported
    assert "## 전사 내용" in exported
    assert "- [00:00:01] Alice: 회의를 시작합니다." in exported
    assert "- [00:00:05] Bob: 진행 상황을 공유합니다." in exported


def test_export_session_docx_returns_bytes_when_python_docx_available(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"], "started_at": "2026-03-13T09:00:00"},
        transcript_text="[00:00:01] [Alice] DOCX 테스트",
    )

    exported = export_session(session_dir, "docx")

    assert isinstance(exported, bytes)
    assert exported.startswith(b"PK")


def test_export_session_docx_raises_import_error_when_python_docx_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"]},
        transcript_text="[00:00:01] [Alice] DOCX 테스트",
    )
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "docx" or name.startswith("docx."):
            raise ImportError("No module named 'docx'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="python-docx"):
        export_session(session_dir, "docx")


def test_export_session_raises_file_not_found_when_session_json_is_missing(
    tmp_path: Path,
) -> None:
    session_dir = _create_session_dir(
        tmp_path,
        transcript_text="[00:00:01] [Alice] session.json 없음",
    )

    with pytest.raises(FileNotFoundError, match="session.json"):
        export_session(session_dir, "txt")


def test_export_session_pdf_returns_bytes_when_fpdf2_available(tmp_path: Path) -> None:
    pytest.importorskip("fpdf")
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={
            "speakers": ["Alice", "Bob"],
            "started_at": "2026-03-13T09:00:00",
            "duration": 180,
        },
        transcript_text=(
            "[00:00:01] [Alice] PDF 테스트 첫 번째 발화\n"
            "[00:00:05] [Bob] PDF 테스트 두 번째 발화"
        ),
    )

    exported = export_session(session_dir, "pdf")

    assert isinstance(exported, bytes)
    # PDF 파일은 %PDF- 매직 바이트로 시작
    assert exported[:5] == b"%PDF-"


def test_export_session_pdf_raises_import_error_when_fpdf2_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"]},
        transcript_text="[00:00:01] [Alice] PDF 테스트",
    )
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fpdf" or name.startswith("fpdf."):
            raise ImportError("No module named 'fpdf'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="fpdf2"):
        export_session(session_dir, "pdf")


def test_export_session_pdf_handles_empty_segments(tmp_path: Path) -> None:
    """빈 세그먼트 배열만 있으면 전사 파일이 없는 것과 동일 → FileNotFoundError."""
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"]},
        segments_payload={"segments": []},
    )

    with pytest.raises(FileNotFoundError, match="전사 파일"):
        export_session(session_dir, "pdf")


def test_export_session_rejects_unsupported_format(tmp_path: Path) -> None:
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"]},
        transcript_text="[00:00:01] [Alice] 테스트",
    )

    with pytest.raises(ValueError, match="지원하지 않는 포맷"):
        export_session(session_dir, "csv")


def test_export_session_handles_empty_segments(tmp_path: Path) -> None:
    """빈 세그먼트 배열만 있으면 전사 파일이 없는 것과 동일 → FileNotFoundError."""
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"]},
        segments_payload={"segments": []},
    )

    with pytest.raises(FileNotFoundError, match="전사 파일"):
        export_session(session_dir, "txt")


# ===================================================================
# 엣지케이스: _find_korean_font — 폰트 미존재 시 None 반환
# ===================================================================


def test_find_korean_font_returns_none_when_no_font(monkeypatch: pytest.MonkeyPatch) -> None:
    """시스템에 한국어 폰트가 없으면 None을 반환한다."""
    from src.export import _find_korean_font

    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.warns(UserWarning, match="한국어 폰트를 찾을 수 없습니다"):
        result = _find_korean_font()
    assert result is None


# ===================================================================
# 엣지케이스: _render_pdf — 다수 화자(>6) 색상 순환
# ===================================================================


def test_export_pdf_many_speakers_color_cycle(tmp_path: Path) -> None:
    """화자가 6명을 초과해도 색상이 순환되며 PDF가 정상 생성된다."""
    pytest.importorskip("fpdf")

    # 8명의 화자 생성 (색상 팔레트는 6개)
    speakers = [f"Speaker{i}" for i in range(8)]
    lines = []
    for i, speaker in enumerate(speakers):
        lines.append(f"[00:00:{i:02d}] [{speaker}] 화자 {i} 발화")

    session_dir = _create_session_dir(
        tmp_path,
        session_meta={
            "speakers": speakers,
            "started_at": "2026-03-13T09:00:00",
            "duration": 60,
        },
        transcript_text="\n".join(lines),
    )

    exported = export_session(session_dir, "pdf")
    assert isinstance(exported, bytes)
    assert exported[:5] == b"%PDF-"


# ===================================================================
# 엣지케이스: export — 존재하지 않는 세션 디렉토리
# ===================================================================


def test_export_session_nonexistent_directory(tmp_path: Path) -> None:
    """존재하지 않는 디렉토리 경로에 대해 FileNotFoundError를 발생시킨다."""
    fake_dir = tmp_path / "meetings" / "nonexistent"
    with pytest.raises(FileNotFoundError):
        export_session(fake_dir, "txt")


# ===================================================================
# 엣지케이스: export PDF — 단일 화자 + 긴 텍스트 (multi_cell 경계)
# ===================================================================


def test_export_pdf_single_speaker_long_text(tmp_path: Path) -> None:
    """단일 화자의 긴 텍스트가 multi_cell로 올바르게 래핑된다."""
    pytest.importorskip("fpdf")

    long_text = "가나다라마바사 " * 50  # 반복 텍스트
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={
            "speakers": ["Alice"],
            "started_at": "2026-03-13T09:00:00",
        },
        transcript_text=f"[00:00:01] [Alice] {long_text}",
    )

    exported = export_session(session_dir, "pdf")
    assert isinstance(exported, bytes)
    assert exported[:5] == b"%PDF-"


# ===================================================================
# 엣지케이스: export — 화자 없는 세그먼트 (speakers=[])
# ===================================================================


def test_export_md_no_speakers(tmp_path: Path) -> None:
    """화자 목록이 비어 있어도 MD 내보내기가 정상 동작한다."""
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={
            "speakers": [],
            "started_at": "2026-03-13T09:00:00",
            "duration": 30,
        },
        transcript_text="[00:00:01] Unknown: 발화 내용",
    )

    exported = export_session(session_dir, "md")
    assert "## 전사 내용" in exported
    # 참석자 섹션은 있되 목록은 비어 있거나 Unknown 포함
    assert "## 참석자" in exported


# ===================================================================
# 엣지케이스: export — format 대소문자 무관
# ===================================================================


def test_export_session_format_case_insensitive(tmp_path: Path) -> None:
    """포맷 문자열의 대소문자 및 공백을 정규화한다."""
    session_dir = _create_session_dir(
        tmp_path,
        session_meta={"speakers": ["Alice"]},
        transcript_text="[00:00:01] [Alice] 대소문자 테스트",
    )

    # 대문자 + 앞뒤 공백
    exported = export_session(session_dir, " TXT ")
    assert "[00:00:01] Alice: 대소문자 테스트" in exported
