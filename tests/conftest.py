"""공통 테스트 fixture 정의."""

import pytest

from src import server
from src.postprocess import Segment


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
