import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.polish import calc_timeout, _cleanup_orphaned_temp_files, polish_meeting


class PolishMeetingTests(unittest.TestCase):
    def test_polish_meeting_merges_parallel_summary_and_correction_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "meeting.md"
            md_path.write_text(
                "\n".join(
                    [
                        "# 회의록",
                        "",
                        "# 회의 내용 요약",
                        "",
                        "- 일시: 2026-03-13 09:00:00",
                        "- 경과 시간: 00:00:01",
                        "- 참석자: A (1명)",
                        "- 총 세그먼트: 1개",
                        "- (요약은 회의 후 별도 작성)",
                        "",
                        "---",
                        "",
                        "# To-do",
                        "",
                        "- [ ] (회의 후 별도 작성)",
                        "",
                        "---",
                        "",
                        "# 대화 정리",
                        "",
                        "### A · 00:00:01",
                        "",
                        "안녕하세여.",
                        "",
                        "---",
                        "",
                        "# Raw Data",
                        "",
                        "- [00:00:01] [A] 안녕하세여.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            def _fake_correct(
                target_path: Path,
                timeout: int = 0,
                model: str = "",
                progress_callback=None,
            ) -> bool:
                _ = timeout, model
                content = target_path.read_text(encoding="utf-8")
                target_path.write_text(
                    content.replace("안녕하세여.", "안녕하세요."),
                    encoding="utf-8",
                )
                if progress_callback:
                    progress_callback(100)
                return True

            def _fake_summary(target_path: Path, timeout: int = 0) -> bool:
                _ = timeout
                content = target_path.read_text(encoding="utf-8")
                content = content.replace(
                    "- (요약은 회의 후 별도 작성)",
                    "## 주제\n병렬 후처리 테스트",
                )
                content = content.replace(
                    "- [ ] (회의 후 별도 작성)",
                    "- [ ] 결과 확인",
                )
                target_path.write_text(content, encoding="utf-8")
                return True

            with (
                mock.patch("src.polish.is_codex_available", return_value=False),
                mock.patch("src.polish.is_gemini_available", return_value=True),
                mock.patch("src.polish.is_ollama_available", return_value=True),
                mock.patch("src.polish.correct_with_ollama_parallel", side_effect=_fake_correct),
                mock.patch("src.polish.summarize_with_gemini", side_effect=_fake_summary),
            ):
                result = polish_meeting(
                    md_path,
                    segment_count=1,
                    use_ollama=True,
                    ollama_model="gemma3:27b",
                )

            content = md_path.read_text(encoding="utf-8")
            self.assertEqual(result, {"corrected": True, "summarized": True})
            self.assertIn("## 주제\n병렬 후처리 테스트", content)
            self.assertIn("- [ ] 결과 확인", content)
            self.assertIn("- [00:00:01] [A] 안녕하세요.", content)
            self.assertFalse(md_path.with_suffix(".summary.tmp.md").exists())


class TestCalcTimeout(unittest.TestCase):
    """calc_timeout의 세그먼트 수 비례 타임아웃 계산을 검증한다."""

    def test_zero_segments_returns_minimum(self) -> None:
        """0 세그먼트일 때 최소값(300초)을 반환한다."""
        result = calc_timeout(0)
        self.assertEqual(result, 300)

    def test_100_segments_proportional(self) -> None:
        """100 세그먼트일 때 base + per_segment * 100 = 180 + 300 = 480초."""
        result = calc_timeout(100)
        self.assertEqual(result, 480)

    def test_1000_segments_no_cap(self) -> None:
        """1000 세그먼트일 때 상한 없이 계산된다 (180 + 3000 = 3180초)."""
        result = calc_timeout(1000)
        self.assertEqual(result, 3180)

    def test_small_segments_below_minimum_clamped(self) -> None:
        """base + per_segment * count < 300 이면 최소 300초로 클램프된다."""
        # base=180, per_segment=3, count=30 → 180+90=270 < 300 → 300
        result = calc_timeout(30)
        self.assertEqual(result, 300)

    def test_custom_base_and_per_segment(self) -> None:
        """사용자 정의 base, per_segment 인자가 반영된다."""
        result = calc_timeout(50, base=100, per_segment=5)
        # 100 + 250 = 350 > 300
        self.assertEqual(result, 350)


class TestCleanupOrphanedTempFiles(unittest.TestCase):
    """_cleanup_orphaned_temp_files의 임시 파일 정리를 검증한다."""

    def test_matching_patterns_deleted(self) -> None:
        """매칭되는 패턴의 임시 파일이 삭제된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            # 삭제 대상 파일 생성
            targets = [
                work_dir / ".polish-batch-001.txt",
                work_dir / "meeting.pre-polish.md",
                work_dir / "meeting.summary.tmp.md",
                work_dir / ".kw-input.txt",
                work_dir / ".kw-output.txt",
                work_dir / "session.postprocess-status.json",
            ]
            for f in targets:
                f.write_text("임시 내용")

            _cleanup_orphaned_temp_files(work_dir)

            for f in targets:
                self.assertFalse(f.exists(), f"{f.name}이 삭제되지 않았다")

    def test_non_matching_files_preserved(self) -> None:
        """매칭되지 않는 파일은 보존된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            # 보존 대상 파일 생성
            preserved = [
                work_dir / "meeting.md",
                work_dir / "output.json",
                work_dir / "notes.txt",
                work_dir / "polish-result.md",
            ]
            for f in preserved:
                f.write_text("보존할 내용")

            _cleanup_orphaned_temp_files(work_dir)

            for f in preserved:
                self.assertTrue(f.exists(), f"{f.name}이 잘못 삭제되었다")

    def test_empty_directory_no_error(self) -> None:
        """빈 디렉토리에서 실행해도 에러가 발생하지 않는다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _cleanup_orphaned_temp_files(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
