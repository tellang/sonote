import json
from datetime import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.meeting_writer import MeetingWriter


class MeetingWriterArtifactTests(unittest.TestCase):
    def test_default_filename_uses_session_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_now = datetime(2026, 3, 11, 14, 1, 17)
            with patch("src.meeting_writer.datetime") as mock_datetime, patch(
                "src.meeting_writer.meetings_dir",
                return_value=Path(tmpdir),
            ):
                mock_datetime.now.return_value = fixed_now
                writer = MeetingWriter()
                try:
                    # 세션 서브디렉토리: HHMMSS/meeting.md
                    self.assertEqual(writer.output_path.name, "meeting.md")
                    self.assertEqual(writer.output_path.parent.name, "140117")
                finally:
                    writer.close()

    def test_writer_persists_alignment_audio_and_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_audio(np.zeros(1600, dtype=np.float32))
            writer.append_alignment({"kind": "stt_segment", "text": "테스트"})
            writer.append_segment("A", "안녕하세요", "00:00:01", {"start": 0.0, "end": 1.0})
            writer.set_keywords(
                {
                    "manual": ["안건"],
                    "extracted": [],
                    "promoted": [],
                    "blocked": [],
                }
            )
            review_path = writer.write_profile_review(
                {
                    "profiles_source": "base.json",
                    "candidate_profiles_path": "candidate.json",
                }
            )
            writer.write_footer("00:00:01", 1, 1)
            writer.close()

            self.assertTrue(output_path.exists())
            self.assertTrue(writer.alignment_path.exists())
            self.assertTrue(review_path.exists())

            alignment_entries = [
                json.loads(line)
                for line in writer.alignment_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(alignment_entries[0]["entry_kind"], "raw_stt")
            self.assertEqual(alignment_entries[1]["entry_kind"], "display_segment")

            session_path = output_path.parent / "session.json"
            session = json.loads(session_path.read_text(encoding="utf-8"))
            self.assertIn("session_audio", session["artifacts"])
            self.assertEqual(session["segment_count"], 1)
            self.assertEqual(session["keywords"]["manual"], ["안건"])

    def test_writer_renders_grouped_paragraph_section_and_preserves_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_segment("A", "안녕하세요.", "00:00:01", {"start": 0.0, "end": 0.8})
            writer.append_segment("A", "오늘 안건 공유드릴게요.", "00:00:02", {"start": 1.0, "end": 2.0})
            writer.append_segment("B", "좋습니다.", "00:00:05", {"start": 5.0, "end": 5.7})
            writer.write_footer("00:00:05", 3, 2)
            writer.close()

            content = output_path.read_text(encoding="utf-8")
            self.assertIn("# 대화 정리", content)
            self.assertIn("### A · 00:00:01 ~ 00:00:02", content)
            self.assertIn("안녕하세요. 오늘 안건 공유드릴게요.", content)
            self.assertIn("# Raw Data", content)
            self.assertIn("- [00:00:01] [A] 안녕하세요.", content)
            self.assertIn("- [00:00:02] [A] 오늘 안건 공유드릴게요.", content)
            self.assertIn("- [00:00:05] [B] 좋습니다.", content)

    def test_apply_segment_corrections_updates_grouped_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_segment("A", "안녕하세여.", "00:00:01", {"start": 0.0, "end": 0.8})
            writer.write_footer("00:00:01", 1, 1)

            applied = writer.apply_segment_corrections(
                {
                    "- [00:00:01] [A] 안녕하세여.": "- [00:00:01] [A] 안녕하세요."
                }
            )
            writer.write_footer("00:00:01", 1, 1)
            writer.close()

            content = output_path.read_text(encoding="utf-8")
            self.assertEqual(applied, 1)
            self.assertIn("안녕하세요.", content)
            self.assertNotIn("안녕하세여.", content)


class MeetingWriterApplyCorrectionsTests(unittest.TestCase):
    """apply_segment_corrections의 경계값 및 정상 동작을 검증한다."""

    def test_corrections_dict_properly_applied(self) -> None:
        """corrections dict에 일치하는 원본 라인이 있으면 text가 교정된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_segment("A", "파이선 시작합니다.", "00:00:01", {"start": 0.0, "end": 1.0})
            writer.append_segment("B", "네 알겠습니다.", "00:00:05", {"start": 5.0, "end": 6.0})

            applied = writer.apply_segment_corrections(
                {
                    "- [00:00:01] [A] 파이선 시작합니다.": "- [00:00:01] [A] 파이썬 시작합니다.",
                    "- [00:00:05] [B] 네 알겠습니다.": "- [00:00:05] [B] 네, 알겠습니다.",
                }
            )
            writer.close()

            self.assertEqual(applied, 2)
            self.assertEqual(writer._segments[0]["text"], "파이썬 시작합니다.")
            self.assertEqual(writer._segments[1]["text"], "네, 알겠습니다.")

    def test_missing_segment_index_gracefully_skipped(self) -> None:
        """corrections에 일치하지 않는 raw_line은 건너뛴다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_segment("A", "테스트.", "00:00:01", {"start": 0.0, "end": 1.0})

            applied = writer.apply_segment_corrections(
                {
                    "- [00:00:99] [Z] 존재하지 않음": "- [00:00:99] [Z] 교정됨",
                }
            )
            writer.close()

            self.assertEqual(applied, 0)
            self.assertEqual(writer._segments[0]["text"], "테스트.")

    def test_same_text_correction_is_skipped(self) -> None:
        """교정 후 텍스트가 원본과 동일하면 적용되지 않는다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_segment("A", "동일.", "00:00:01", {"start": 0.0, "end": 1.0})

            applied = writer.apply_segment_corrections(
                {
                    "- [00:00:01] [A] 동일.": "- [00:00:01] [A] 동일.",
                }
            )
            writer.close()

            self.assertEqual(applied, 0)

    def test_wrong_prefix_correction_is_skipped(self) -> None:
        """교정 라인이 올바른 prefix로 시작하지 않으면 무시된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            writer = MeetingWriter(output_path=output_path)
            writer.write_header()
            writer.append_segment("A", "원본.", "00:00:01", {"start": 0.0, "end": 1.0})

            applied = writer.apply_segment_corrections(
                {
                    "- [00:00:01] [A] 원본.": "잘못된 형식의 교정",
                }
            )
            writer.close()

            self.assertEqual(applied, 0)
            self.assertEqual(writer._segments[0]["text"], "원본.")


class MeetingWriterGroupSegmentsTests(unittest.TestCase):
    """_group_segments_for_display의 병합/분리 로직을 검증한다."""

    def _make_writer(self, tmpdir: str) -> MeetingWriter:
        output_path = Path(tmpdir) / "meeting.md"
        writer = MeetingWriter(output_path=output_path)
        writer.write_header()
        return writer

    def test_segments_within_gap_merged(self) -> None:
        """gap이 _DISPLAY_PARAGRAPH_GAP_SECONDS 이내이면 같은 그룹으로 병합된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer(tmpdir)
            writer.append_segment("A", "첫 문장.", "00:00:01", {"start": 0.0, "end": 1.0})
            writer.append_segment("A", "두번째.", "00:00:02", {"start": 1.2, "end": 2.0})

            groups = writer._group_segments_for_display()
            writer.close()

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["texts"], ["첫 문장.", "두번째."])
            self.assertEqual(groups[0]["segment_count"], 2)

    def test_segments_beyond_gap_split(self) -> None:
        """gap이 _DISPLAY_PARAGRAPH_GAP_SECONDS 초과이면 별도 그룹으로 분리된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer(tmpdir)
            writer.append_segment("A", "첫 문장.", "00:00:01", {"start": 0.0, "end": 1.0})
            # gap = 10.0 - 1.0 = 9.0 > 1.5
            writer.append_segment("A", "두번째.", "00:00:11", {"start": 10.0, "end": 11.0})

            groups = writer._group_segments_for_display()
            writer.close()

            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]["texts"], ["첫 문장."])
            self.assertEqual(groups[1]["texts"], ["두번째."])

    def test_max_chars_limit_causes_split(self) -> None:
        """char_count 합이 _DISPLAY_PARAGRAPH_MAX_CHARS(280)를 초과하면 분리된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer(tmpdir)
            # 첫 세그먼트 275자 + 구분자 1 + 두번째 6자 = 282 > 280 → 분리
            long_text = "가" * 275
            writer.append_segment("A", long_text, "00:00:01", {"start": 0.0, "end": 1.0})
            writer.append_segment("A", "추가텍스트입니다.", "00:00:02", {"start": 1.2, "end": 2.0})

            groups = writer._group_segments_for_display()
            writer.close()

            self.assertEqual(len(groups), 2)

    def test_different_speakers_always_split(self) -> None:
        """다른 화자의 세그먼트는 gap이 짧아도 항상 분리된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer(tmpdir)
            writer.append_segment("A", "화자 A.", "00:00:01", {"start": 0.0, "end": 1.0})
            writer.append_segment("B", "화자 B.", "00:00:02", {"start": 1.2, "end": 2.0})

            groups = writer._group_segments_for_display()
            writer.close()

            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]["speaker"], "A")
            self.assertEqual(groups[1]["speaker"], "B")

    def test_max_segments_limit_causes_split(self) -> None:
        """세그먼트 수가 _DISPLAY_PARAGRAPH_MAX_SEGMENTS(4)를 초과하면 분리된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer(tmpdir)
            for i in range(5):
                start = float(i)
                end = start + 0.5
                writer.append_segment("A", f"문장{i}.", f"00:00:{i:02d}", {"start": start, "end": end})

            groups = writer._group_segments_for_display()
            writer.close()

            # 4개 병합 + 1개 별도 = 2그룹
            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]["segment_count"], 4)
            self.assertEqual(groups[1]["segment_count"], 1)

    def test_empty_text_segments_are_skipped(self) -> None:
        """빈 텍스트 세그먼트는 그룹에 포함되지 않는다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer(tmpdir)
            writer.append_segment("A", "", "00:00:01", {"start": 0.0, "end": 1.0})
            writer.append_segment("A", "유효한 문장.", "00:00:02", {"start": 1.2, "end": 2.0})

            groups = writer._group_segments_for_display()
            writer.close()

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["texts"], ["유효한 문장."])


if __name__ == "__main__":
    unittest.main()
