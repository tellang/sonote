from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.paths import profiles_db_path

# 프로젝트 루트 기준 기본 DB 경로
DEFAULT_DB_PATH: Path = profiles_db_path()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS video_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    stream_status TEXT DEFAULT "live",
    total_duration_sec REAL,
    scan_result TEXT,
    speech_blocks TEXT,
    total_speech_min INTEGER,
    captured_ranges TEXT,
    transcript_files TEXT,
    transcript_segments INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""

_JSON_FIELDS: set[str] = {
    "scan_result",
    "speech_blocks",
    "captured_ranges",
    "transcript_files",
}

_UPSERT_ALLOWED_FIELDS: set[str] = {
    "title",
    "stream_status",
    "total_duration_sec",
    "speech_blocks",
    "total_speech_min",
    "captured_ranges",
    "transcript_files",
    "transcript_segments",
}


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    """DB 경로를 Path로 정규화."""
    return DEFAULT_DB_PATH if db_path is None else Path(db_path)


def _json_dumps(value: Any) -> str | None:
    """JSON 컬럼 저장용 문자열 변환."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Any) -> Any:
    """JSON 컬럼 조회 시 파싱."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """sqlite3.Row를 dict로 변환하고 JSON 컬럼을 파싱."""
    data = dict(row)
    for field in _JSON_FIELDS:
        data[field] = _json_loads(data.get(field))
    return data


def init_db(db_path: str | Path | None = None) -> Path:
    """DB 파일/디렉토리를 준비하고 스키마를 생성."""
    resolved_path = _resolve_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path)
    try:
        conn.execute(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

    return resolved_path


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """sqlite3.Row 기반 연결 객체를 반환."""
    resolved_path = init_db(db_path)
    conn = sqlite3.connect(resolved_path)
    conn.row_factory = sqlite3.Row
    return conn


def save_profile(
    video_id: str,
    url: str,
    scan_result: Any = None,
    **kwargs: Any,
) -> None:
    """video_id 기준으로 프로필을 저장/갱신(UPSERT)."""
    fields: dict[str, Any] = {
        "video_id": video_id,
        "url": url,
    }
    if scan_result is not None:
        fields["scan_result"] = scan_result

    for key, value in kwargs.items():
        if key in _UPSERT_ALLOWED_FIELDS:
            fields[key] = value

    for json_field in _JSON_FIELDS:
        if json_field in fields:
            fields[json_field] = _json_dumps(fields[json_field])

    columns = list(fields.keys())
    placeholders = ", ".join("?" for _ in columns)
    insert_cols = ", ".join(columns)

    update_cols = [col for col in columns if col != "video_id"]
    update_clause_parts = [f"{col}=excluded.{col}" for col in update_cols]
    update_clause_parts.append("updated_at=datetime('now')")
    update_clause = ", ".join(update_clause_parts)

    sql = f"""
    INSERT INTO video_profiles ({insert_cols})
    VALUES ({placeholders})
    ON CONFLICT(video_id) DO UPDATE SET
    {update_clause};
    """

    values = [fields[col] for col in columns]
    conn = get_db()
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def get_profile(video_id: str) -> dict[str, Any] | None:
    """video_id로 단일 프로필 조회. JSON 컬럼은 파싱해 반환."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM video_profiles WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def list_profiles() -> list[dict[str, Any]]:
    """생성 최신순으로 전체 프로필 목록 반환."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM video_profiles ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def update_captured(
    video_id: str,
    ranges: Any,
    files: Any,
) -> None:
    """캡처 구간/파일 정보를 갱신하고 updated_at을 갱신."""
    captured_ranges = _json_dumps(ranges)
    transcript_files = _json_dumps(files)
    transcript_segments = len(files) if isinstance(files, list) else None

    conn = get_db()
    try:
        conn.execute(
            """
            UPDATE video_profiles
            SET captured_ranges = ?,
                transcript_files = ?,
                transcript_segments = ?,
                updated_at = datetime('now')
            WHERE video_id = ?
            """,
            (captured_ranges, transcript_files, transcript_segments, video_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_profile(video_id: str) -> bool:
    """video_id 프로필 삭제 성공 여부 반환."""
    conn = get_db()
    try:
        cur = conn.execute(
            "DELETE FROM video_profiles WHERE video_id = ?",
            (video_id,),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
