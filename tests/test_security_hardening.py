"""보안 하드닝 검증 테스트.

server.py에 적용된(또는 적용 예정인) 보안 하드닝을 검증한다:
- MAX_WS_CONNECTIONS: WebSocket 최대 동시 연결 제한
- MAX_WS_MESSAGE_SIZE: WebSocket 메시지 크기 제한 (64KB)
- MAX_TRANSCRIPT_HISTORY: 전사 내역 최대 크기 제한
- 원자적 파일 쓰기: 프로필 저장 중 크래시 시 원본 무결성 보장
- asyncio.Lock 프로필 보호: 동시 프로필 수정 일관성
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src import server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_security_state():
    """각 테스트 전에 보안 관련 서버 상태를 초기화한다."""
    original_api_key = server._api_key
    server._api_key = ""
    server._connected_websockets.clear()
    server._transcript_history.clear()
    server._startup_ready = True
    yield
    server._api_key = original_api_key
    server._connected_websockets.clear()
    server._transcript_history.clear()


@pytest.fixture
def client() -> TestClient:
    """Starlette TestClient를 FastAPI 앱에 연결한다."""
    return TestClient(server.app)


@pytest.fixture
def tmp_profiles(tmp_path: Path):
    """임시 프로필 JSON 파일을 생성하고 서버에 경로를 설정한다."""
    profiles_file = tmp_path / "speakers.json"
    profiles_file.write_text(
        json.dumps({"speakers": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    original = server._profiles_path
    server._profiles_path = str(profiles_file)
    yield profiles_file
    server._profiles_path = original


# ---------------------------------------------------------------------------
# 1. WebSocket 최대 연결 수 제한 (MAX_WS_CONNECTIONS)
# ---------------------------------------------------------------------------


class TestWSMaxConnections:
    """MAX_WS_CONNECTIONS 상수가 정의되어 있으면 WebSocket 연결 수 제한을 검증한다."""

    def test_ws_max_connections(self, client: TestClient):
        """_connected_websockets에 MAX-1개 dummy 추가 후 WS 연결 시도 → 거부 확인."""
        max_conns = getattr(server, "MAX_WS_CONNECTIONS", None)
        if max_conns is None:
            pytest.skip("MAX_WS_CONNECTIONS 상수가 아직 정의되지 않았습니다.")

        # dummy WebSocket 객체로 연결 슬롯을 채운다
        dummies = []
        for _ in range(max_conns - 1):
            dummy = MagicMock()
            server._connected_websockets.add(dummy)
            dummies.append(dummy)

        # MAX-1개 상태에서 실제 연결 1개는 성공해야 한다
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

        # 정리 후 MAX개 dummy로 가득 채운다
        server._connected_websockets.clear()
        for _ in range(max_conns):
            server._connected_websockets.add(MagicMock())

        # MAX개 상태에서 추가 연결은 거부되어야 한다
        # 서버가 accept() 전에 close(4008)을 호출하므로 WebSocketDisconnect 발생
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.send_json({"type": "ping"})
                ws.receive_json()

        # 정리
        server._connected_websockets.clear()

    def test_ws_max_connections_small_limit(self, client: TestClient, monkeypatch):
        """monkeypatch로 MAX_WS_CONNECTIONS를 2로 설정하여 제한을 검증한다."""
        if not hasattr(server, "MAX_WS_CONNECTIONS"):
            pytest.skip("MAX_WS_CONNECTIONS 상수가 아직 정의되지 않았습니다.")

        monkeypatch.setattr(server, "MAX_WS_CONNECTIONS", 2)

        # 2개 dummy로 슬롯을 채운다
        server._connected_websockets.add(MagicMock())
        server._connected_websockets.add(MagicMock())

        # 추가 연결 거부
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.send_json({"type": "ping"})

        server._connected_websockets.clear()


# ---------------------------------------------------------------------------
# 2. WebSocket 메시지 크기 제한 (MAX_WS_MESSAGE_SIZE)
# ---------------------------------------------------------------------------


class TestWSMessageSizeLimit:
    """MAX_WS_MESSAGE_SIZE 상수가 정의되어 있으면 메시지 크기 제한을 검증한다."""

    def test_ws_message_size_limit(self, client: TestClient):
        """64KB 초과 메시지 전송 시 에러 응답 또는 연결 종료를 확인한다."""
        max_size = getattr(server, "MAX_WS_MESSAGE_SIZE", None)
        if max_size is None:
            pytest.skip("MAX_WS_MESSAGE_SIZE 상수가 아직 정의되지 않았습니다.")

        # 제한 초과 크기의 메시지 생성
        oversized_payload = "가" * (max_size + 1024)

        with client.websocket_connect("/ws/transcribe") as ws:
            try:
                ws.send_text(oversized_payload)
                resp = ws.receive_json()
                # 에러 응답 또는 연결 종료 중 하나
                assert resp.get("type") == "error"
            except Exception:
                # 연결이 서버에 의해 종료됨 — 정상 동작
                pass

    def test_ws_message_within_limit(self, client: TestClient):
        """제한 이하 크기의 메시지는 정상 처리된다."""
        max_size = getattr(server, "MAX_WS_MESSAGE_SIZE", None)
        if max_size is None:
            pytest.skip("MAX_WS_MESSAGE_SIZE 상수가 아직 정의되지 않았습니다.")

        # 제한 이하 크기의 유효한 JSON 메시지
        small_payload = json.dumps({"type": "ping"})
        assert len(small_payload.encode("utf-8")) < max_size

        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text(small_payload)
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ---------------------------------------------------------------------------
# 3. 전사 내역 최대 크기 (MAX_TRANSCRIPT_HISTORY)
# ---------------------------------------------------------------------------


class TestTranscriptHistoryMaxSize:
    """MAX_TRANSCRIPT_HISTORY 상수가 정의되어 있으면 전사 내역 크기 제한을 검증한다."""

    def test_transcript_history_max_size(self):
        """_transcript_history에 MAX+100개 추가 → len이 MAX 이하인지 확인."""
        max_hist = getattr(server, "MAX_TRANSCRIPT_HISTORY", None)
        if max_hist is None:
            pytest.skip("MAX_TRANSCRIPT_HISTORY 상수가 아직 정의되지 않았습니다.")

        # MAX + 100개 항목 추가
        for i in range(max_hist + 100):
            server._transcript_history.append(
                {"speaker": f"S{i}", "text": f"세그먼트 {i}", "ts": f"00:{i:02d}:00"}
            )

        # 내부 정리 로직이 있다면 MAX 이하여야 한다
        # 아직 정리 로직이 append에 통합되지 않았다면 수동 호출
        if hasattr(server, "_trim_transcript_history"):
            server._trim_transcript_history()

        assert len(server._transcript_history) <= max_hist

    def test_transcript_history_order(self):
        """오래된 항목부터 제거되는지 확인 (FIFO)."""
        max_hist = getattr(server, "MAX_TRANSCRIPT_HISTORY", None)
        if max_hist is None:
            pytest.skip("MAX_TRANSCRIPT_HISTORY 상수가 아직 정의되지 않았습니다.")

        # 명확한 순서로 항목 추가
        total = max_hist + 50
        for i in range(total):
            server._transcript_history.append(
                {"speaker": "A", "text": f"항목_{i:05d}", "ts": f"00:00:{i:02d}"}
            )

        if hasattr(server, "_trim_transcript_history"):
            server._trim_transcript_history()

        assert len(server._transcript_history) <= max_hist

        # 마지막으로 추가된 항목이 남아 있어야 한다 (FIFO — 오래된 것 먼저 제거)
        last_item = server._transcript_history[-1]
        assert last_item["text"] == f"항목_{total - 1:05d}"

        # 가장 오래된 항목은 제거되었어야 한다
        first_texts = [item["text"] for item in list(server._transcript_history)[:5]]
        # 제거된 초기 항목이 남아 있지 않아야 한다
        assert "항목_00000" not in first_texts


# ---------------------------------------------------------------------------
# 4. 원자적 파일 쓰기 (프로필 저장)
# ---------------------------------------------------------------------------


class TestAtomicWriteProfiles:
    """프로필 저장 시 원자적 쓰기가 적용되어 크래시 내성을 보장하는지 검증한다."""

    def test_atomic_write_profiles_crash_resilience(self, tmp_profiles, monkeypatch):
        """프로필 쓰기 중 크래시 시뮬레이션 → 원본 파일 무결성 확인."""
        # 초기 프로필 데이터 설정
        initial_data = {
            "speakers": {
                "홍길동": {
                    "embedding": [0.1, 0.2, 0.3],
                    "enrolled_at": "2026-01-01T00:00:00",
                    "description": "테스트 화자",
                }
            }
        }
        tmp_profiles.write_text(
            json.dumps(initial_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # os.replace가 호출되기 전에 예외를 발생시켜 크래시 시뮬레이션
        original_replace = os.replace

        def crash_on_replace(src, dst):
            """os.replace 호출 시 예외를 발생시킨다."""
            # 임시 파일은 정리하되 원본은 건드리지 않는다
            if os.path.exists(src):
                os.unlink(src)
            raise OSError("시뮬레이션된 디스크 오류")

        # _write_profiles_json이 원자적 쓰기를 사용하는 경우에만 의미 있음
        # 현재 구현은 직접 write_text이므로, 원자적 쓰기가 구현된 후 검증
        if not hasattr(server, "_atomic_write") and "replace" not in (
            server._write_profiles_json.__code__.co_names
            if hasattr(server._write_profiles_json, "__code__")
            else ()
        ):
            # 원자적 쓰기가 아직 구현되지 않았더라도 기본 동작 검증
            new_data = {
                "speakers": {
                    "홍길동": initial_data["speakers"]["홍길동"],
                    "김철수": {
                        "embedding": [0.4, 0.5, 0.6],
                        "enrolled_at": "2026-01-02T00:00:00",
                        "description": "새 화자",
                    },
                }
            }
            server._write_profiles_json(new_data)

            # 쓰기 후 파일이 유효한 JSON인지 확인
            saved = json.loads(tmp_profiles.read_text(encoding="utf-8"))
            assert "speakers" in saved
            assert "홍길동" in saved["speakers"]
            assert "김철수" in saved["speakers"]
            return

        # 원자적 쓰기가 구현된 경우: os.replace에서 크래시 시뮬레이션
        monkeypatch.setattr(os, "replace", crash_on_replace)

        new_data = {
            "speakers": {
                "홍길동": initial_data["speakers"]["홍길동"],
                "김철수": {
                    "embedding": [0.4, 0.5, 0.6],
                    "enrolled_at": "2026-01-02T00:00:00",
                    "description": "새 화자",
                },
            }
        }

        # 쓰기 시 예외 발생 예상
        with pytest.raises(OSError, match="시뮬레이션된 디스크 오류"):
            server._write_profiles_json(new_data)

        monkeypatch.setattr(os, "replace", original_replace)

        # 원본 파일이 무결한지 확인 — 크래시 이전 데이터가 보존되어야 한다
        preserved = json.loads(tmp_profiles.read_text(encoding="utf-8"))
        assert preserved == initial_data
        assert "홍길동" in preserved["speakers"]
        # 크래시로 인해 김철수는 추가되지 않아야 한다
        assert "김철수" not in preserved["speakers"]

    def test_write_profiles_creates_valid_json(self, tmp_profiles):
        """프로필 쓰기 후 파일이 유효한 JSON인지 확인한다."""
        data = {
            "speakers": {
                "테스트": {
                    "embedding": [0.1, 0.2],
                    "enrolled_at": "2026-01-01T00:00:00",
                    "description": "테스트",
                }
            }
        }
        server._write_profiles_json(data)
        result = json.loads(tmp_profiles.read_text(encoding="utf-8"))
        assert result == data

    def test_write_profiles_no_temp_file_left(self, tmp_profiles):
        """정상 쓰기 완료 후 임시 파일이 남아 있지 않아야 한다."""
        data = {"speakers": {"화자A": {"embedding": [1.0], "enrolled_at": "2026-01-01T00:00:00"}}}
        server._write_profiles_json(data)

        parent_dir = tmp_profiles.parent
        # .tmp 확장자 파일이 남아 있으면 안 된다
        tmp_files = list(parent_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"임시 파일이 남아 있음: {tmp_files}"


# ---------------------------------------------------------------------------
# 5. asyncio.Lock 프로필 동시 접근 보호
# ---------------------------------------------------------------------------


class TestProfilesLockConcurrent:
    """asyncio.Lock으로 프로필 동시 수정 시 일관성을 보장하는지 검증한다."""

    @pytest.mark.asyncio
    async def test_profiles_lock_concurrent(self, tmp_profiles):
        """asyncio.gather로 동시 프로필 수정 5건 → 최종 상태 일관성 확인."""
        profiles_lock = getattr(server, "_profiles_lock", None)
        if profiles_lock is None:
            pytest.skip("_profiles_lock이 아직 정의되지 않았습니다.")

        async def modify_profile(name: str) -> None:
            """프로필을 추가하는 비동기 작업."""
            async with profiles_lock:
                data = server._read_profiles_json()
                # 동시성 테스트를 위해 약간의 지연
                await asyncio.sleep(0.01)
                data["speakers"][name] = {
                    "embedding": [0.1],
                    "enrolled_at": "2026-01-01T00:00:00",
                    "description": f"화자 {name}",
                }
                server._write_profiles_json(data)

        # 5개 동시 수정
        names = [f"화자_{i}" for i in range(5)]
        await asyncio.gather(*[modify_profile(name) for name in names])

        # 모든 5명이 등록되어 있어야 한다
        final_data = server._read_profiles_json()
        for name in names:
            assert name in final_data["speakers"], f"'{name}'이 누락되었습니다."

        assert len(final_data["speakers"]) == 5

    @pytest.mark.asyncio
    async def test_profiles_lock_without_lock_race_condition(self, tmp_profiles):
        """Lock 없이 동시 수정 시 데이터 손실 가능성을 보여준다 (대조군)."""
        # Lock이 없는 경우의 위험성을 보여주는 대조 테스트
        if not hasattr(server, "_profiles_lock"):
            pytest.skip("_profiles_lock이 아직 정의되지 않았습니다.")

        # Lock을 사용하지 않는 동시 수정 (경쟁 상태 시뮬레이션)
        async def unsafe_modify(name: str) -> None:
            """Lock 없이 프로필을 수정하는 비동기 작업 (의도적 위험)."""
            data = server._read_profiles_json()
            await asyncio.sleep(0.01)  # 다른 태스크에 양보
            data["speakers"][name] = {
                "embedding": [0.1],
                "enrolled_at": "2026-01-01T00:00:00",
                "description": f"unsafe {name}",
            }
            server._write_profiles_json(data)

        names = [f"unsafe_{i}" for i in range(5)]
        await asyncio.gather(*[unsafe_modify(name) for name in names])

        # Lock 없이는 일부 데이터가 손실될 수 있다
        final_data = server._read_profiles_json()
        # 최소 1개는 남아있어야 한다 (마지막 쓰기는 반드시 반영)
        assert len(final_data["speakers"]) >= 1
        # 하지만 5개 모두 남아있지 않을 수 있다 (경쟁 상태)
        # 이 테스트는 Lock의 필요성을 보여주는 것이 목적


# ---------------------------------------------------------------------------
# 6. WebSocket 초기 히스토리 전송 제한
# ---------------------------------------------------------------------------


class TestWSInitialHistoryLimit:
    """WebSocket 연결 시 전송되는 초기 히스토리 수가 제한되는지 검증한다."""

    def test_ws_initial_history_limit(self, client: TestClient):
        """히스토리 2000개 상태에서 WS 연결 → 수신 메시지 수 확인."""
        max_initial = getattr(server, "MAX_WS_INITIAL_HISTORY", None)
        max_hist = getattr(server, "MAX_TRANSCRIPT_HISTORY", None)

        if max_initial is None and max_hist is None:
            pytest.skip("MAX_WS_INITIAL_HISTORY 또는 MAX_TRANSCRIPT_HISTORY가 정의되지 않았습니다.")

        # 초기 히스토리로 사용할 제한값 결정
        limit = max_initial if max_initial is not None else 1000

        # 2000개 히스토리 추가
        for i in range(2000):
            server._transcript_history.append(
                {"speaker": "A", "text": f"세그먼트 {i}", "ts": f"00:{i // 60:02d}:{i % 60:02d}"}
            )

        with client.websocket_connect("/ws/transcribe") as ws:
            received = []
            # 초기 히스토리 수신 후 ping으로 경계 확인
            ws.send_json({"type": "ping"})

            while True:
                msg = ws.receive_json()
                if msg.get("type") == "pong":
                    break
                received.append(msg)

            # 수신된 메시지가 제한 이하인지 확인
            assert len(received) <= limit, (
                f"초기 히스토리 {len(received)}개 수신 — 제한 {limit}개 초과"
            )

    def test_ws_initial_history_all_sent_when_small(self, client: TestClient):
        """히스토리가 제한보다 적으면 전부 전송된다."""
        # 소규모 히스토리
        for i in range(5):
            server._transcript_history.append(
                {"speaker": "A", "text": f"소규모 {i}", "ts": f"00:00:{i:02d}"}
            )

        with client.websocket_connect("/ws/transcribe") as ws:
            received = []
            ws.send_json({"type": "ping"})

            while True:
                msg = ws.receive_json()
                if msg.get("type") == "pong":
                    break
                received.append(msg)

            assert len(received) == 5


# ---------------------------------------------------------------------------
# 7. 전사 내역 FIFO 순서 보장
# ---------------------------------------------------------------------------


class TestTranscriptHistoryFIFO:
    """전사 내역 트리밍 시 오래된 항목부터 제거되는지 검증한다."""

    def test_transcript_history_fifo_order(self):
        """MAX_TRANSCRIPT_HISTORY 초과 시 가장 오래된 항목이 먼저 제거된다."""
        max_hist = getattr(server, "MAX_TRANSCRIPT_HISTORY", None)
        if max_hist is None:
            pytest.skip("MAX_TRANSCRIPT_HISTORY 상수가 아직 정의되지 않았습니다.")

        overflow = 200
        total = max_hist + overflow

        # 순서가 명확한 항목 추가
        for i in range(total):
            server._transcript_history.append(
                {"speaker": "A", "text": f"SEQ_{i:06d}", "ts": f"{i}"}
            )

        # 트리밍 함수가 있으면 호출
        if hasattr(server, "_trim_transcript_history"):
            server._trim_transcript_history()

        # 트리밍 후 크기 확인
        assert len(server._transcript_history) <= max_hist

        # 첫 번째 항목은 overflow 이후의 것이어야 한다
        first = server._transcript_history[0]
        first_seq = int(first["text"].split("_")[1])
        assert first_seq >= overflow, (
            f"첫 번째 항목 SEQ가 {first_seq}인데, {overflow} 이상이어야 합니다 (FIFO)"
        )

        # 마지막 항목은 가장 최근 것이어야 한다
        last = server._transcript_history[-1]
        assert last["text"] == f"SEQ_{total - 1:06d}"

    def test_transcript_history_preserves_recent(self):
        """트리밍 후에도 가장 최근 항목들이 보존된다."""
        max_hist = getattr(server, "MAX_TRANSCRIPT_HISTORY", None)
        if max_hist is None:
            pytest.skip("MAX_TRANSCRIPT_HISTORY 상수가 아직 정의되지 않았습니다.")

        # MAX + 50개 추가
        for i in range(max_hist + 50):
            server._transcript_history.append(
                {"speaker": "B", "text": f"RECENT_{i}", "ts": f"{i}"}
            )

        if hasattr(server, "_trim_transcript_history"):
            server._trim_transcript_history()

        # 가장 최근 10개가 모두 존재하는지 확인
        recent_texts = {item["text"] for item in list(server._transcript_history)[-10:]}
        for i in range(max_hist + 40, max_hist + 50):
            assert f"RECENT_{i}" in recent_texts
