"""WebSocket /ws/transcribe 엔드포인트 테스트.

연결/인증, 초기 전사 내역 수신, 양방향 메시지(edit_speaker, edit_segment,
search, ping/pong), 에러 처리, 동시 연결을 검증한다.
"""

import json

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from src import server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ws_state():
    """각 테스트 전에 WebSocket/인증 관련 상태를 초기화한다."""
    original_api_key = server._api_key
    server._api_key = ""
    server._connected_websockets.clear()
    server._startup_ready = True
    yield
    server._api_key = original_api_key
    server._connected_websockets.clear()


@pytest.fixture
def client() -> TestClient:
    """Starlette TestClient를 FastAPI 앱에 연결한다."""
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# 1. 연결 성공 + 초기 전사 내역 수신
# ---------------------------------------------------------------------------


class TestWSConnectionAndInitialHistory:
    """WS 연결 성공 후 기존 전사 내역이 순서대로 전달되는지 검증한다."""

    def test_connect_success_no_auth(self, client: TestClient):
        """API 키 미설정 시 인증 없이 연결이 성공한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            # 연결 성공 — 전사 내역이 없으므로 send로 ping 확인
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_initial_history_sent_on_connect(self, client: TestClient, transcript_history):
        """연결 시 기존 전사 내역이 transcript 타입으로 전송된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for expected in transcript_history:
                msg = ws.receive_json()
                assert msg["type"] == "transcript"
                assert msg["speaker"] == expected["speaker"]
                assert msg["text"] == expected["text"]
                assert msg["ts"] == expected["ts"]

    def test_empty_history_no_initial_messages(self, client: TestClient):
        """전사 내역이 비어 있으면 초기 메시지가 전송되지 않는다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            # 내역 없이 바로 ping/pong 가능
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ---------------------------------------------------------------------------
# 2. API 키 인증
# ---------------------------------------------------------------------------


class TestWSAuthentication:
    """WebSocket 연결 시 API 키 인증 로직을 검증한다."""

    def test_valid_key_via_query_param(self, client: TestClient):
        """쿼리 파라미터로 유효한 API 키를 전달하면 연결이 성공한다."""
        server._api_key = "test-secret"
        with client.websocket_connect("/ws/transcribe?api_key=test-secret") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_valid_key_via_header(self, client: TestClient):
        """x-api-key 헤더로 유효한 API 키를 전달하면 연결이 성공한다."""
        server._api_key = "test-secret"
        with client.websocket_connect(
            "/ws/transcribe",
            headers={"x-api-key": "test-secret"},
        ) as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_invalid_key_rejected(self, client: TestClient):
        """잘못된 API 키로 연결하면 4003 코드로 거부된다."""
        server._api_key = "test-secret"
        with pytest.raises(Exception):
            # TestClient는 WebSocket 거부 시 예외를 발생시킴
            with client.websocket_connect("/ws/transcribe?api_key=wrong-key") as ws:
                ws.send_json({"type": "ping"})

    def test_missing_key_rejected_when_required(self, client: TestClient):
        """API 키가 설정되었는데 키 없이 연결하면 거부된다."""
        server._api_key = "test-secret"
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.send_json({"type": "ping"})

    def test_no_auth_when_api_key_not_set(self, client: TestClient):
        """API 키가 빈 문자열이면 인증 없이 연결이 허용된다."""
        server._api_key = ""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ---------------------------------------------------------------------------
# 3. edit_speaker 메시지
# ---------------------------------------------------------------------------


class TestWSEditSpeaker:
    """edit_speaker 메시지로 화자 이름을 수정하는 로직을 검증한다."""

    def test_edit_speaker_updates_history(self, client: TestClient, transcript_history):
        """유효한 edit_speaker 메시지로 화자 이름이 변경된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            # 초기 내역 소비
            for _ in transcript_history:
                ws.receive_json()

            ws.send_json({"type": "edit_speaker", "index": 0, "speaker": "홍길동"})
            # edit_speaker는 SSE 브로드캐스트만 하고 WS 직접 응답 없음
            # 상태 변경 확인
        assert server._transcript_history[0]["speaker"] == "홍길동"
        assert "홍길동" in server._speakers

    def test_edit_speaker_invalid_index_ignored(self, client: TestClient, transcript_history):
        """범위 밖 index의 edit_speaker는 무시된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_speaker", "index": 999, "speaker": "무시됨"})
        # 원본 유지
        assert server._transcript_history[0]["speaker"] == "A"

    def test_edit_speaker_empty_name_ignored(self, client: TestClient, transcript_history):
        """빈 화자 이름의 edit_speaker는 무시된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_speaker", "index": 0, "speaker": ""})
        assert server._transcript_history[0]["speaker"] == "A"

    def test_edit_speaker_negative_index_ignored(self, client: TestClient, transcript_history):
        """음수 index의 edit_speaker는 무시된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_speaker", "index": -1, "speaker": "음수"})
        assert server._transcript_history[0]["speaker"] == "A"


# ---------------------------------------------------------------------------
# 4. edit_segment 메시지
# ---------------------------------------------------------------------------


class TestWSEditSegment:
    """edit_segment 메시지로 세그먼트 텍스트를 수정하는 로직을 검증한다."""

    def test_edit_segment_updates_text(self, client: TestClient, transcript_history):
        """유효한 edit_segment 메시지로 텍스트가 변경된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_segment", "index": 1, "text": "수정된 텍스트"})
        assert server._transcript_history[1]["text"] == "수정된 텍스트"

    def test_edit_segment_invalid_index_ignored(self, client: TestClient, transcript_history):
        """범위 밖 index의 edit_segment는 무시된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_segment", "index": 999, "text": "무시됨"})
        assert server._transcript_history[1]["text"] == "네, 반갑습니다."

    def test_edit_segment_empty_text_ignored(self, client: TestClient, transcript_history):
        """빈 텍스트의 edit_segment는 무시된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_segment", "index": 0, "text": ""})
        assert server._transcript_history[0]["text"] == "안녕하세요."


# ---------------------------------------------------------------------------
# 5. search 메시지
# ---------------------------------------------------------------------------


class TestWSSearch:
    """search 메시지로 전사 내역을 검색하는 로직을 검증한다."""

    def test_search_returns_matching_results(self, client: TestClient, transcript_history):
        """검색어에 일치하는 결과를 search_result 타입으로 반환한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "search", "query": "안녕"})
            resp = ws.receive_json()
            assert resp["type"] == "search_result"
            assert resp["query"] == "안녕"
            assert resp["count"] == 1
            assert resp["results"][0]["index"] == 0

    def test_search_case_insensitive(self, client: TestClient, transcript_history):
        """검색은 대소문자를 구분하지 않는다."""
        # 영문이 포함된 전사 내역 추가
        server._transcript_history.append(
            {"speaker": "C", "text": "Hello World", "ts": "00:00:15"}
        )
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in range(len(server._transcript_history)):
                ws.receive_json()
            ws.send_json({"type": "search", "query": "hello"})
            resp = ws.receive_json()
            assert resp["count"] == 1
            assert resp["results"][0]["text"] == "Hello World"

    def test_search_by_speaker_name(self, client: TestClient, transcript_history):
        """화자 이름으로도 검색할 수 있다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "search", "query": "b"})
            resp = ws.receive_json()
            assert resp["count"] == 1
            assert resp["results"][0]["speaker"] == "B"

    def test_search_no_results(self, client: TestClient, transcript_history):
        """일치하는 결과가 없으면 빈 리스트를 반환한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "search", "query": "존재하지않는검색어"})
            resp = ws.receive_json()
            assert resp["type"] == "search_result"
            assert resp["count"] == 0
            assert resp["results"] == []

    def test_search_empty_query_ignored(self, client: TestClient, transcript_history):
        """빈 검색어는 무시된다 (응답 없음)."""
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "search", "query": ""})
            # 빈 검색어는 응답이 없으므로 ping으로 확인
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ---------------------------------------------------------------------------
# 6. ping/pong
# ---------------------------------------------------------------------------


class TestWSPingPong:
    """ping 메시지 전송 시 pong 응답을 검증한다."""

    def test_ping_returns_pong(self, client: TestClient):
        """ping 전송 시 pong과 타임스탬프를 응답한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"
            assert "ts" in resp

    def test_pong_message_is_silently_ignored(self, client: TestClient):
        """클라이언트가 보낸 pong 메시지는 무시된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "pong"})
            # pong은 무시되므로 후속 ping이 정상 동작해야 함
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ---------------------------------------------------------------------------
# 7. 연결 해제 시 _connected_websockets에서 제거
# ---------------------------------------------------------------------------


class TestWSDisconnectCleanup:
    """WebSocket 연결 해제 시 리소스가 정리되는지 검증한다."""

    def test_disconnect_removes_from_connected_set(self, client: TestClient):
        """연결 종료 후 _connected_websockets에서 제거된다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "ping"})
            ws.receive_json()
            # 연결 중 set에 존재 확인
            assert len(server._connected_websockets) == 1
        # context manager 종료 후 (연결 해제) — 정리 확인
        assert len(server._connected_websockets) == 0


# ---------------------------------------------------------------------------
# 8. 동시 2개 WebSocket 연결
# ---------------------------------------------------------------------------


class TestWSConcurrentConnections:
    """동시에 여러 WebSocket 연결이 관리되는지 검증한다."""

    def test_two_concurrent_connections(self, client: TestClient):
        """두 개의 WebSocket이 동시에 연결되고 독립적으로 동작한다."""
        with client.websocket_connect("/ws/transcribe") as ws1:
            with client.websocket_connect("/ws/transcribe") as ws2:
                assert len(server._connected_websockets) == 2

                # 각각 독립적으로 ping/pong
                ws1.send_json({"type": "ping"})
                resp1 = ws1.receive_json()
                assert resp1["type"] == "pong"

                ws2.send_json({"type": "ping"})
                resp2 = ws2.receive_json()
                assert resp2["type"] == "pong"

            # ws2 종료 후
            assert len(server._connected_websockets) == 1
        # ws1도 종료 후
        assert len(server._connected_websockets) == 0

    def test_both_receive_initial_history(self, client: TestClient, transcript_history):
        """두 연결 모두 초기 전사 내역을 수신한다."""
        with client.websocket_connect("/ws/transcribe") as ws1:
            # ws1이 내역 수신
            for expected in transcript_history:
                msg = ws1.receive_json()
                assert msg["type"] == "transcript"
                assert msg["text"] == expected["text"]

            with client.websocket_connect("/ws/transcribe") as ws2:
                # ws2도 내역 수신
                for expected in transcript_history:
                    msg = ws2.receive_json()
                    assert msg["type"] == "transcript"
                    assert msg["text"] == expected["text"]


# ---------------------------------------------------------------------------
# 9. 잘못된 JSON 전송 → 에러 응답
# ---------------------------------------------------------------------------


class TestWSInvalidJSON:
    """잘못된 JSON 메시지 전송 시 에러 응답을 검증한다."""

    def test_invalid_json_returns_error(self, client: TestClient):
        """파싱 불가능한 문자열 전송 시 에러 메시지를 반환한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text("이것은 JSON이 아닙니다{{{")
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "잘못된 JSON" in resp["message"]

    def test_unknown_message_type_returns_error(self, client: TestClient):
        """알 수 없는 메시지 타입 전송 시 에러 메시지를 반환한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "unknown_action"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "알 수 없는 메시지 타입" in resp["message"]

    def test_connection_continues_after_error(self, client: TestClient):
        """에러 발생 후에도 연결이 유지되어 후속 메시지를 처리한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            # 잘못된 JSON 전송
            ws.send_text("not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"

            # 연결이 여전히 활성 — 정상 메시지 처리
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_empty_type_returns_error(self, client: TestClient):
        """type 필드가 없는 JSON 전송 시 에러를 반환한다."""
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"data": "no type field"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "알 수 없는 메시지 타입" in resp["message"]


# ---------------------------------------------------------------------------
# 10. 대용량 메시지 전송
# ---------------------------------------------------------------------------


class TestWSLargeMessage:
    """과대 메시지 전송 시 적절히 처리되는지 검증한다."""

    def test_large_json_message_processed(self, client: TestClient, transcript_history):
        """큰 텍스트가 포함된 메시지도 정상 처리된다."""
        large_text = "가" * 10000
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "edit_segment", "index": 0, "text": large_text})
        assert server._transcript_history[0]["text"] == large_text

    def test_large_search_query(self, client: TestClient, transcript_history):
        """긴 검색어도 에러 없이 처리된다 (결과 0건)."""
        long_query = "테스트" * 1000
        with client.websocket_connect("/ws/transcribe") as ws:
            for _ in transcript_history:
                ws.receive_json()
            ws.send_json({"type": "search", "query": long_query})
            resp = ws.receive_json()
            assert resp["type"] == "search_result"
            assert resp["count"] == 0

    def test_large_invalid_text_returns_error(self, client: TestClient):
        """큰 비-JSON 문자열도 에러 메시지로 응답한다."""
        large_text = "x" * 50000
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text(large_text)
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "잘못된 JSON" in resp["message"]
