"""server.py Critical 단위 테스트.

push_correction 엣지 케이스, add_extracted_keywords 승격 로직,
_normalize_keyword 필터링을 검증한다.
"""

import asyncio

import pytest

from src import server


class TestPushCorrection:
    """push_correction의 정상 동작 및 경계값 처리를 검증한다."""

    def test_valid_index_updates_history_text(self, transcript_history):
        """유효한 index로 교정하면 _transcript_history의 text가 변경된다."""
        corrections = [
            {"index": 0, "original": "안녕하세요.", "corrected": "안녕하세요!"}
        ]
        asyncio.run(server.push_correction(corrections))
        assert server._transcript_history[0]["text"] == "안녕하세요!"

    def test_negative_index_no_crash(self, transcript_history):
        """음수 index는 무시되며 예외를 발생시키지 않는다."""
        corrections = [
            {"index": -1, "original": "X", "corrected": "Y"}
        ]
        asyncio.run(server.push_correction(corrections))
        # 원본이 변경되지 않음
        for i, item in enumerate(transcript_history):
            assert server._transcript_history[i]["text"] == item["text"]

    def test_out_of_range_index_no_crash(self, transcript_history):
        """범위 밖 index는 무시되며 예외를 발생시키지 않는다."""
        corrections = [
            {"index": 999, "original": "X", "corrected": "Y"}
        ]
        asyncio.run(server.push_correction(corrections))
        for i, item in enumerate(transcript_history):
            assert server._transcript_history[i]["text"] == item["text"]

    def test_updates_specific_index_text(self, transcript_history):
        """중간 index(1)의 text가 정확히 교정된다."""
        corrections = [
            {"index": 1, "original": "네, 반갑습니다.", "corrected": "네, 만나서 반갑습니다."}
        ]
        asyncio.run(server.push_correction(corrections))
        assert server._transcript_history[1]["text"] == "네, 만나서 반갑습니다."
        # 다른 항목은 변경되지 않음
        assert server._transcript_history[0]["text"] == "안녕하세요."
        assert server._transcript_history[2]["text"] == "오늘 회의를 시작하겠습니다."

    def test_multiple_corrections_at_once(self, transcript_history):
        """여러 교정을 동시에 적용할 수 있다."""
        corrections = [
            {"index": 0, "original": "안녕하세요.", "corrected": "안녕하세요!"},
            {"index": 2, "original": "오늘 회의를 시작하겠습니다.", "corrected": "오늘 미팅을 시작합니다."},
        ]
        asyncio.run(server.push_correction(corrections))
        assert server._transcript_history[0]["text"] == "안녕하세요!"
        assert server._transcript_history[2]["text"] == "오늘 미팅을 시작합니다."

    def test_empty_corrected_is_ignored(self, transcript_history):
        """corrected가 빈 문자열이면 교정이 무시된다."""
        corrections = [
            {"index": 0, "original": "안녕하세요.", "corrected": ""}
        ]
        asyncio.run(server.push_correction(corrections))
        assert server._transcript_history[0]["text"] == "안녕하세요."

    def test_non_dict_correction_is_skipped(self, transcript_history):
        """correction이 dict가 아니면 건너뛴다."""
        corrections = ["invalid", 123, None]
        asyncio.run(server.push_correction(corrections))
        for i, item in enumerate(transcript_history):
            assert server._transcript_history[i]["text"] == item["text"]

    def test_non_int_index_is_skipped(self, transcript_history):
        """index가 정수가 아니면 건너뛴다."""
        corrections = [
            {"index": "abc", "original": "X", "corrected": "Y"}
        ]
        asyncio.run(server.push_correction(corrections))
        for i, item in enumerate(transcript_history):
            assert server._transcript_history[i]["text"] == item["text"]

    def test_broadcasts_correction_event(self, transcript_history):
        """교정 성공 시 SSE correction 이벤트가 브로드캐스트된다."""
        queue: asyncio.Queue = asyncio.Queue()
        server._client_queues.add(queue)
        corrections = [
            {"index": 0, "original": "안녕하세요.", "corrected": "안녕하세요!"}
        ]
        asyncio.run(server.push_correction(corrections))
        item = queue.get_nowait()
        assert item["_type"] == "correction"
        assert len(item["_payload"]["corrections"]) == 1
        assert item["_payload"]["corrections"][0]["corrected"] == "안녕하세요!"

    def test_no_broadcast_when_all_invalid(self, transcript_history):
        """모든 교정이 무효하면 브로드캐스트하지 않는다."""
        queue: asyncio.Queue = asyncio.Queue()
        server._client_queues.add(queue)
        corrections = [
            {"index": -1, "original": "X", "corrected": "Y"},
            {"index": 999, "original": "X", "corrected": "Y"},
        ]
        asyncio.run(server.push_correction(corrections))
        assert queue.empty()


class TestAddExtractedKeywords:
    """add_extracted_keywords의 승격 로직을 검증한다."""

    def test_promote_threshold_2_promotes_after_two_sightings(self):
        """promote_threshold=2일 때 2번 이상 출현한 키워드가 승격된다."""
        # 첫 번째: extracted에만 추가
        payload = server.add_extracted_keywords(["RAG"], promote_threshold=2)
        assert "RAG" in payload["extracted"]
        assert "RAG" not in payload["promoted"]

        # 두 번째: promoted로 승격
        payload = server.add_extracted_keywords(["RAG"], promote_threshold=2)
        assert "RAG" not in payload["extracted"]
        assert "RAG" in payload["promoted"]

    def test_manual_keyword_not_added_to_extracted(self):
        """수동 등록 키워드는 extracted에 추가되지 않는다."""
        server._manual_keywords.add("LLM")
        payload = server.add_extracted_keywords(["LLM"])
        assert "LLM" not in payload["extracted"]

    def test_blocked_keyword_not_added(self):
        """차단된 키워드는 extracted에 추가되지 않는다."""
        server._blocked_keywords.add("테스트")
        payload = server.add_extracted_keywords(["테스트"])
        assert "테스트" not in payload["extracted"]
        assert "테스트" not in payload["promoted"]

    def test_custom_promote_threshold(self):
        """promote_threshold=3이면 3번 출현 후 승격된다."""
        for _ in range(2):
            payload = server.add_extracted_keywords(["FastAPI"], promote_threshold=3)
        assert "FastAPI" in payload["extracted"]
        assert "FastAPI" not in payload["promoted"]

        payload = server.add_extracted_keywords(["FastAPI"], promote_threshold=3)
        assert "FastAPI" not in payload["extracted"]
        assert "FastAPI" in payload["promoted"]

    def test_already_promoted_stays_promoted(self):
        """이미 승격된 키워드는 재추가해도 promoted에 유지된다."""
        server._promoted_keywords.add("Whisper")
        payload = server.add_extracted_keywords(["Whisper"])
        assert "Whisper" in payload["promoted"]
        assert "Whisper" not in payload["extracted"]

    def test_short_keyword_filtered_by_normalize(self):
        """2자 미만 키워드는 _normalize_keyword에 의해 필터된다."""
        payload = server.add_extracted_keywords(["A"])
        assert "A" not in payload["extracted"]
        assert "A" not in payload["promoted"]


class TestNormalizeKeyword:
    """_normalize_keyword의 필터링 로직을 검증한다."""

    def test_filters_single_char(self):
        """1자 문자열은 빈 문자열로 변환된다."""
        assert server._normalize_keyword("A") == ""

    def test_filters_empty_string(self):
        """빈 문자열은 빈 문자열로 변환된다."""
        assert server._normalize_keyword("") == ""

    def test_passes_two_char_keyword(self):
        """2자 이상 문자열은 그대로 반환된다."""
        assert server._normalize_keyword("AI") == "AI"

    def test_strips_surrounding_punctuation(self):
        """앞뒤 구두점과 공백이 제거된다."""
        assert server._normalize_keyword(",LLM;") == "LLM"

    def test_collapses_internal_whitespace(self):
        """내부 연속 공백이 단일 공백으로 축소된다."""
        assert server._normalize_keyword("Large  Language   Model") == "Large Language Model"

    def test_none_returns_empty(self):
        """None 입력은 빈 문자열로 변환된다."""
        assert server._normalize_keyword(None) == ""

    def test_korean_two_chars_passes(self):
        """한글 2자 이상은 정상 통과한다."""
        assert server._normalize_keyword("한글") == "한글"

    def test_korean_single_char_filtered(self):
        """한글 1자는 빈 문자열로 필터된다."""
        assert server._normalize_keyword("가") == ""

    def test_mixed_language_passes(self):
        """영한 혼합 키워드가 정상 통과한다."""
        assert server._normalize_keyword("AI모델") == "AI모델"

    def test_numbers_only_two_digits_passes(self):
        """숫자만 2자 이상이면 통과한다."""
        assert server._normalize_keyword("42") == "42"

    def test_numbers_only_single_digit_filtered(self):
        """숫자 1자는 빈 문자열로 필터된다."""
        assert server._normalize_keyword("7") == ""

    def test_multiple_punctuation_stripped(self):
        """여러 종류의 구두점이 모두 제거된다."""
        assert server._normalize_keyword(",,;;:://키워드,,;;::") == "키워드"

    def test_only_punctuation_returns_empty(self):
        """구두점만 있으면 빈 문자열로 변환된다."""
        assert server._normalize_keyword(",;:/") == ""
