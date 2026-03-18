import unittest

from src.postprocess import (
    Segment,
    is_looping,
    normalize_feedback_text,
    postprocess,
    remove_overlap,
)


class PostprocessFeedbackTests(unittest.TestCase):
    def test_normalize_feedback_text_reduces_ellipsis_noise(self) -> None:
        text = "이거는 ... 테스트 ... 입니다..."
        self.assertEqual(normalize_feedback_text(text), "이거는 테스트 입니다.")


class TestPostprocessPipeline(unittest.TestCase):
    """postprocess 파이프라인의 전체 흐름을 검증한다."""

    def test_filler_removal_correction_merge_punctuation_dedup(self) -> None:
        """필러 제거 → 도메인 교정 → 병합 → 구두점 → 중복제거 순서로 동작한다."""
        segments = [
            Segment(speaker="A", text="어", start=0.0, end=0.5),       # 필러 → 제거
            Segment(speaker="A", text="파이선 시작", start=1.0, end=2.0),  # 교정 → 파이썬
            Segment(speaker="A", text="합니다", start=2.5, end=3.0),     # 병합 대상
            Segment(speaker="B", text="네 알겠습니다", start=8.0, end=9.0),  # gap > 3초 → 마침표
        ]
        result = postprocess(segments)

        # 필러 "어"는 제거됨
        all_text = " ".join(seg.text for seg in result)
        self.assertNotIn("어", all_text.split()[0:1])

        # "파이선" → "파이썬" 교정
        self.assertIn("파이썬", all_text)

        # A의 세그먼트가 병합되어 1개 그룹이 됨
        a_segments = [seg for seg in result if seg.speaker == "A"]
        self.assertEqual(len(a_segments), 1)
        self.assertIn("파이썬", a_segments[0].text)
        self.assertIn("합니다", a_segments[0].text)

    def test_hallucination_segments_removed(self) -> None:
        """환각 텍스트는 파이프라인에서 제거된다."""
        segments = [
            Segment(speaker="A", text="감사합니다", start=0.0, end=1.0),
            Segment(speaker="A", text="유효한 내용입니다", start=2.0, end=3.0),
        ]
        result = postprocess(segments)
        all_text = " ".join(seg.text for seg in result)
        self.assertNotIn("감사합니다", all_text)
        self.assertIn("유효한 내용", all_text)

    def test_looping_segments_removed(self) -> None:
        """루핑(반복) 텍스트는 파이프라인에서 제거된다."""
        looping_text = " ".join(["반복 패턴 입니다"] * 5)
        segments = [
            Segment(speaker="A", text=looping_text, start=0.0, end=5.0),
            Segment(speaker="A", text="정상 텍스트입니다", start=6.0, end=7.0),
        ]
        result = postprocess(segments)
        all_text = " ".join(seg.text for seg in result)
        self.assertIn("정상 텍스트", all_text)

    def test_duplicate_segments_deduplicated(self) -> None:
        """연속 동일 텍스트의 다른 화자 세그먼트가 dedup으로 하나만 남는다."""
        # 다른 화자이므로 merge 안 됨 → 독립 세그먼트 2개 → dedup이 연속 동일 텍스트 제거
        segments = [
            Segment(speaker="A", text="동일한 텍스트입니다", start=0.0, end=1.0),
            Segment(speaker="B", text="동일한 텍스트입니다", start=1.5, end=2.5),
        ]
        result = postprocess(segments)
        self.assertEqual(len(result), 1)

    def test_empty_input_returns_empty(self) -> None:
        """빈 입력 리스트는 빈 결과를 반환한다."""
        result = postprocess([])
        self.assertEqual(result, [])

    def test_punctuation_added_on_long_gap(self) -> None:
        """세그먼트 간 gap이 3초 이상이면 마침표가 추가된다."""
        segments = [
            Segment(speaker="A", text="첫 문장", start=0.0, end=1.0),
            Segment(speaker="B", text="두번째 문장", start=5.0, end=6.0),
        ]
        result = postprocess(segments)
        # 첫 세그먼트 끝에 마침표가 추가되어야 함
        self.assertTrue(result[0].text.endswith("."))


class TestIsLooping(unittest.TestCase):
    """is_looping의 반복 패턴 감지를 검증한다."""

    def test_detects_repeated_text_pattern(self) -> None:
        """동일 구절이 threshold 이상 반복되면 True를 반환한다."""
        text = " ".join(["이것은 반복"] * 5)
        self.assertTrue(is_looping(text, phrase_len=2, threshold=3))

    def test_no_repetition_returns_false(self) -> None:
        """반복이 없으면 False를 반환한다."""
        text = "이것은 전혀 반복되지 않는 문장입니다"
        self.assertFalse(is_looping(text))

    def test_short_text_returns_false(self) -> None:
        """텍스트가 phrase_len * threshold 미만이면 False를 반환한다."""
        text = "짧은 문장"
        self.assertFalse(is_looping(text, phrase_len=3, threshold=3))

    def test_exact_threshold_triggers(self) -> None:
        """정확히 threshold 횟수만큼 반복하면 True를 반환한다."""
        # phrase_len=2, threshold=3 → 첫 2단어가 3번 반복
        text = "테스트 구절 테스트 구절 테스트 구절"
        self.assertTrue(is_looping(text, phrase_len=2, threshold=3))

    def test_below_threshold_returns_false(self) -> None:
        """threshold 미만 반복은 False를 반환한다."""
        text = "테스트 구절 테스트 구절"
        self.assertFalse(is_looping(text, phrase_len=2, threshold=3))

    def test_exact_boundary_ratio_equals_threshold(self) -> None:
        """반복 횟수가 정확히 threshold와 같으면 True를 반환한다 (경계값)."""
        # phrase_len=3, threshold=2 → 첫 3단어가 정확히 2번 반복
        text = "가 나 다 가 나 다"
        self.assertTrue(is_looping(text, phrase_len=3, threshold=2))

    def test_mixed_repeated_non_repeated_text(self) -> None:
        """반복 구절 뒤에 비반복 텍스트가 있으면 반복 체크가 중단된다."""
        # 첫 구절 2번 반복 후 다른 텍스트 → threshold=3 미달
        text = "반복 구절 반복 구절 다른 텍스트"
        self.assertFalse(is_looping(text, phrase_len=2, threshold=3))


class TestRemoveOverlap(unittest.TestCase):
    """remove_overlap의 중복 제거를 검증한다."""

    def test_removes_duplicate_words_at_boundary(self) -> None:
        """이전 청크 끝과 현재 청크 앞의 중복 단어를 제거한다."""
        prev = "첫번째 문장 그리고 겹치는 부분"
        curr = "겹치는 부분 두번째 문장"
        result = remove_overlap(prev, curr)
        self.assertEqual(result, "두번째 문장")

    def test_no_overlap_returns_original(self) -> None:
        """중복이 없으면 현재 텍스트를 그대로 반환한다."""
        prev = "완전히 다른 문장"
        curr = "새로운 내용입니다"
        result = remove_overlap(prev, curr)
        self.assertEqual(result, "새로운 내용입니다")

    def test_full_overlap_returns_empty(self) -> None:
        """현재 텍스트가 이전 텍스트 끝과 완전히 겹치면 빈 결과를 반환한다."""
        prev = "안녕 하세요"
        curr = "안녕 하세요"
        result = remove_overlap(prev, curr, max_words=10)
        self.assertEqual(result, "")

    def test_single_word_overlap(self) -> None:
        """단일 단어 중복도 정상적으로 제거된다."""
        prev = "문장 끝"
        curr = "끝 새로운 시작"
        result = remove_overlap(prev, curr)
        self.assertEqual(result, "새로운 시작")

    def test_max_words_limits_search(self) -> None:
        """max_words가 탐색 범위를 제한한다."""
        prev = "A B C D E"
        curr = "A B C D E F"
        # max_words=2이면 끝 2단어("D E")만 비교
        result = remove_overlap(prev, curr, max_words=2)
        # "D E"는 curr 앞쪽에 없으므로 원본 반환
        self.assertEqual(result, "A B C D E F")

    def test_overlap_at_different_position(self) -> None:
        """이전 청크 끝 3단어가 현재 청크 앞 3단어와 겹칠 때 제거된다."""
        prev = "시작 문장 중간 부분 겹치는 세 단어"
        curr = "겹치는 세 단어 이후 새로운 내용"
        result = remove_overlap(prev, curr)
        self.assertEqual(result, "이후 새로운 내용")

    def test_unicode_text_overlap(self) -> None:
        """유니코드(이모지/특수문자 포함) 텍스트의 중복도 정상 처리된다."""
        prev = "회의 📝 결과 정리"
        curr = "결과 정리 다음 단계"
        result = remove_overlap(prev, curr)
        self.assertEqual(result, "다음 단계")


if __name__ == "__main__":
    unittest.main()
