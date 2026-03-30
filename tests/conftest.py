"""공통 테스트 fixture 정의."""

from pathlib import Path

import pytest

from src import server
from src.postprocess import Segment

# 스냅샷 디렉토리 경로
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


@pytest.fixture(autouse=True)
def _reset_server_state():
    """각 테스트 전에 서버 글로벌 상태를 초기화한다."""
    server._client_queues.clear()
    server._transcript_history.clear()
    server._manual_keywords.clear()
    server._extracted_keywords.clear()
    server._promoted_keywords.clear()
    server._blocked_keywords.clear()
    server._keyword_seen_counts.clear()
    server._segment_count = 0
    server._speakers.clear()
    server._session_rotate_event.clear()
    server._session_rotate_callback = None
    yield


@pytest.fixture
def transcript_history():
    """테스트용 전사 내역 3건을 생성하여 서버에 주입한다."""
    items = [
        {"speaker": "A", "text": "안녕하세요.", "ts": "00:00:01"},
        {"speaker": "B", "text": "네, 반갑습니다.", "ts": "00:00:05"},
        {"speaker": "A", "text": "오늘 회의를 시작하겠습니다.", "ts": "00:00:10"},
    ]
    server._transcript_history.extend(items)
    return items


@pytest.fixture
def sample_segments():
    """후처리 테스트용 세그먼트 목록을 생성한다."""
    return [
        Segment(speaker="A", text="어 안녕하세요", start=0.0, end=1.0),
        Segment(speaker="A", text="파이선 프로젝트 시작합니다", start=1.5, end=3.0),
        Segment(speaker="B", text="네 알겠습니다", start=4.0, end=5.0),
    ]


@pytest.fixture
def snapshot_cli_commands():
    """CLI 명령 surface 스냅샷 텍스트를 반환한다."""
    path = SNAPSHOTS_DIR / "cli_commands.txt"
    assert path.exists(), f"CLI 스냅샷 파일이 없습니다: {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def snapshot_rest_routes():
    """REST 라우트 스냅샷 라인 목록을 반환한다."""
    path = SNAPSHOTS_DIR / "rest_routes.txt"
    assert path.exists(), f"REST 라우트 스냅샷 파일이 없습니다: {path}"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return lines


@pytest.fixture
def snapshot_ws_messages():
    """WebSocket 메시지 타입 스냅샷 텍스트를 반환한다."""
    path = SNAPSHOTS_DIR / "ws_messages.txt"
    assert path.exists(), f"WS 메시지 스냅샷 파일이 없습니다: {path}"
    return path.read_text(encoding="utf-8")
