import sys
import unittest
from io import StringIO
from unittest.mock import patch

from src import cli


class CliDispatchTests(unittest.TestCase):
    def test_transcribe_command_dispatch(self) -> None:
        with patch.object(cli, "_cmd_transcribe") as handler:
            with patch.object(
                sys,
                "argv",
                ["media-transcriber", "transcribe", "sample.wav"],
            ):
                cli.main()

        handler.assert_called_once()
        parsed_args = handler.call_args.args[0]
        self.assertEqual(parsed_args.command, "transcribe")
        self.assertEqual(parsed_args.audio, "sample.wav")

    def test_meeting_command_dispatch(self) -> None:
        with patch.object(cli, "_cmd_meeting") as handler:
            with patch.object(
                sys,
                "argv",
                ["media-transcriber", "meeting", "--no-diarize"],
            ):
                cli.main()

        handler.assert_called_once()
        parsed_args = handler.call_args.args[0]
        self.assertEqual(parsed_args.command, "meeting")
        self.assertTrue(parsed_args.no_diarize)

    def test_approve_profiles_command_dispatch(self) -> None:
        with patch.object(cli, "_cmd_approve_profiles") as handler:
            with patch.object(
                sys,
                "argv",
                ["media-transcriber", "approve-profiles", "review.json"],
            ):
                cli.main()

        handler.assert_called_once()
        parsed_args = handler.call_args.args[0]
        self.assertEqual(parsed_args.command, "approve-profiles")
        self.assertEqual(parsed_args.review, "review.json")

    def test_auto_command_dispatch(self) -> None:
        with patch.object(cli, "_cmd_smart") as handler:
            with patch.object(
                sys,
                "argv",
                ["media-transcriber", "auto", "https://youtube.com/watch?v=test"],
            ):
                cli.main()

        handler.assert_called_once()
        parsed_args = handler.call_args.args[0]
        self.assertEqual(parsed_args.url, "https://youtube.com/watch?v=test")

    def test_detect_command_dispatch(self) -> None:
        with patch.object(cli, "_cmd_probe") as handler:
            with patch.object(
                sys,
                "argv",
                ["media-transcriber", "detect", "https://youtube.com/watch?v=test"],
            ):
                cli.main()

        handler.assert_called_once()
        parsed_args = handler.call_args.args[0]
        self.assertEqual(parsed_args.url, "https://youtube.com/watch?v=test")

    def test_map_command_dispatch(self) -> None:
        with patch.object(cli, "_cmd_scan") as handler:
            with patch.object(
                sys,
                "argv",
                ["media-transcriber", "map", "https://youtube.com/watch?v=test"],
            ):
                cli.main()

        handler.assert_called_once()
        parsed_args = handler.call_args.args[0]
        self.assertEqual(parsed_args.url, "https://youtube.com/watch?v=test")

    def test_ssafy_command_is_not_available(self) -> None:
        with patch.object(sys, "argv", ["media-transcriber", "ssafy"]):
            with patch("sys.stderr", new=StringIO()) as fake_stderr:
                with self.assertRaises(SystemExit) as exc:
                    cli.main()
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("invalid choice: 'ssafy'", fake_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
