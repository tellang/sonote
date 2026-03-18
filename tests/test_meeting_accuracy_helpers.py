import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import server
from src.audio_capture import _count_trailing_silence, _should_emit_chunk
from src.cli import _collect_live_corrections
from src.meeting_writer import MeetingWriter
from src.postprocess import clean_ellipsis, normalize_feedback_text


class PostprocessHelperTests(unittest.TestCase):
    def test_clean_ellipsis_reduces_mid_sentence_and_trailing_dots(self) -> None:
        self.assertEqual(clean_ellipsis("이거... 진짜 중요...."), "이거 진짜 중요.")
        self.assertEqual(clean_ellipsis(". . . 다음 안건"), "다음 안건")

    def test_normalize_feedback_text_strips_prompt_noise(self) -> None:
        self.assertEqual(
            normalize_feedback_text("그 그거... 그거... 맞습니다...."),
            "그거 맞습니다.",
        )


class KeywordStateTests(unittest.TestCase):
    def setUp(self) -> None:
        server._manual_keywords.clear()
        server._extracted_keywords.clear()
        server._promoted_keywords.clear()
        server._blocked_keywords.clear()
        server._keyword_seen_counts.clear()

    def test_repeated_extracted_keywords_are_promoted(self) -> None:
        first = server.add_extracted_keywords(["RAG", "Vector DB"])
        second = server.add_extracted_keywords(["RAG"])

        self.assertIn("RAG", second["promoted"])
        self.assertIn("Vector DB", second["extracted"])
        self.assertNotIn("RAG", second["extracted"])
        self.assertNotIn("Vector DB", first["prompt_keywords"])
        self.assertIn("RAG", second["prompt_keywords"])


class AudioChunkHelperTests(unittest.TestCase):
    def test_trailing_silence_is_counted(self) -> None:
        samples = np.array([0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(_count_trailing_silence(samples, 0.001), 3)

    def test_should_emit_chunk_for_heuristic_silence(self) -> None:
        self.assertTrue(
            _should_emit_chunk(
                buffered_samples=120,
                min_chunk_samples=100,
                max_chunk_samples=200,
                trailing_silence_samples=20,
                silence_trigger_samples=20,
                heuristic_split=True,
            )
        )
        self.assertFalse(
            _should_emit_chunk(
                buffered_samples=120,
                min_chunk_samples=100,
                max_chunk_samples=200,
                trailing_silence_samples=5,
                silence_trigger_samples=20,
                heuristic_split=True,
            )
        )


class LiveCorrectionHelperTests(unittest.TestCase):
    class _FakeFuture:
        def __init__(self, value=None, exc: Exception | None = None) -> None:
            self._value = value
            self._exc = exc

        def result(self, timeout=None):
            _ = timeout
            if self._exc is not None:
                raise self._exc
            return self._value

    class _FakePool:
        def __init__(self) -> None:
            self.calls: list[tuple[bool, bool]] = []

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            self.calls.append((wait, cancel_futures))

    def test_collect_live_corrections_is_best_effort(self) -> None:
        pool = self._FakePool()
        corrections = _collect_live_corrections(
            pool,
            [
                (
                    self._FakeFuture(
                        (0, True, ["- [00:00:01] [A] 안녕하세요."])
                    ),
                    ["- [00:00:01] [A] 안녕하세여."],
                ),
                (
                    self._FakeFuture(exc=TimeoutError()),
                    ["- [00:00:02] [A] 테스트입니다."],
                ),
            ],
            wait_budget_seconds=1.0,
        )

        self.assertEqual(
            corrections["- [00:00:01] [A] 안녕하세여."],
            "- [00:00:01] [A] 안녕하세요.",
        )
        self.assertEqual(pool.calls, [(False, True)])


class MeetingWriterArtifactTests(unittest.TestCase):
    def test_writer_persists_audio_and_session_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_audio(np.ones(1600, dtype=np.float32) * 0.05)
            writer.append_alignment({"kind": "stt_segment", "text": "테스트"})
            writer.append_segment("A", "테스트입니다.", "00:00:01", metadata={"start": 0.0, "end": 1.0})
            writer.write_profile_review({"status": "pending_review"})
            writer.write_footer("00:00:01", 1, 1)
            writer.close()

            session = json.loads(writer.session_path.read_text(encoding="utf-8"))
            self.assertTrue(writer.audio_path.exists())
            self.assertGreater(writer.audio_path.stat().st_size, 44)
            self.assertEqual(session["audio_path"], str(writer.audio_path))
            self.assertIn("profile_review", session["artifacts"])


if __name__ == "__main__":
    unittest.main()
