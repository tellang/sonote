import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src import cli


class _FakeThread:
    events: list[str] = []

    def __init__(self, target=None, args=(), daemon=None, kwargs=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.kwargs = kwargs or {}

    def start(self) -> None:
        label = getattr(self.target, "__name__", repr(self.target))
        self.events.append(f"thread_start:{label}")


class _FakeMeetingWriter:
    def __init__(self, output_path=None):
        base = Path(output_path) if output_path is not None else Path("meeting.md")
        self.output_path = base
        self.alignment_path = base.with_suffix(".stt.jsonl")

    def write_header(self) -> None:
        return None

    def set_artifact(self, *_args, **_kwargs) -> None:
        return None

    def write_footer(self, *_args, **_kwargs) -> None:
        return None

    def write_profile_review(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeWhisperWorker:
    instances: list["_FakeWhisperWorker"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    @property
    def is_ready(self) -> bool:
        return self.started

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class MeetingStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeThread.events = []
        _FakeWhisperWorker.instances = []

    def _build_args(self, **overrides) -> argparse.Namespace:
        values = {
            "list_devices": False,
            "device": 7,
            "port": 8765,
            "output": None,
            "cpu": True,
            "model": "large-v3-turbo",
            "language": "ko",
            "no_diarize": True,
            "profiles": None,
            "chunk": 5.0,
            "prompt": "",
            "fixed_chunking": False,
            "no_polish": True,
            "ollama": False,
            "ollama_model": None,
            "auto_update": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_meeting_startup_reports_expected_status_sequence_before_capture(self) -> None:
        startup_calls: list[tuple[str, str, bool]] = []

        def fake_set_startup_status(phase: str, message: str = "", ready: bool = False) -> None:
            startup_calls.append((phase, message, ready))

        def fake_browser_open(*_args, **_kwargs) -> None:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"

            with patch("threading.Thread", new=_FakeThread), patch(
                "src.audio_capture.capture_audio",
                new=lambda **_kwargs: iter(()),
            ), patch("src.audio_capture.find_builtin_mic", return_value=None), patch(
                "src.server.set_current_audio_device",
            ) as set_current_audio_device, patch(
                "src.server.set_startup_status",
                side_effect=fake_set_startup_status,
            ), patch("src.server.run_server"), patch(
                "src.server.set_diarizer",
            ) as set_diarizer, patch(
                "src.server.is_shutdown_requested",
                return_value=True,
            ), patch(
                "src.server.get_audio_device_switch_event",
                return_value=object(),
            ), patch(
                "src.server.consume_audio_device_switch",
                return_value=(False, None),
            ), patch(
                "src.server.set_postprocess_status",
            ), patch(
                "src.runtime_env.detect_device",
                return_value=("cuda", "float16"),
            ), patch(
                "src.meeting_writer.MeetingWriter",
                side_effect=lambda output_path=None: _FakeMeetingWriter(output_path),
            ), patch(
                "src.whisper_worker.WhisperWorker",
                new=_FakeWhisperWorker,
            ), patch(
                "src.tray.is_available",
                return_value=False,
            ), patch(
                "src.cli._wait_for_server_ready",
            ), patch(
                "webbrowser.open",
                new=fake_browser_open,
            ), patch(
                "os.startfile",
                create=True,
            ):
                cli._cmd_meeting(self._build_args(output=output_path))

        self.assertEqual(
            startup_calls,
            [
                ("booting", "로컬 UI 시작 중...", False),
                ("device", "가속기 감지 중...", False),
                ("loading_asr", "STT 모델 로드 중 (cpu/int8)...", False),
                ("ready", "녹음 준비 완료", True),
            ],
        )
        set_current_audio_device.assert_called_once_with(7)
        set_diarizer.assert_called_once_with(None, None)
        self.assertEqual(len(_FakeWhisperWorker.instances), 1)
        self.assertTrue(_FakeWhisperWorker.instances[0].started)
        self.assertTrue(_FakeWhisperWorker.instances[0].stopped)

    def test_meeting_startup_waits_for_server_before_opening_browser_and_marks_diarizer_loading(self) -> None:
        startup_calls: list[tuple[str, str, bool]] = []

        def fake_set_startup_status(phase: str, message: str = "", ready: bool = False) -> None:
            startup_calls.append((phase, message, ready))

        def fake_run_server(*_args, **_kwargs) -> None:
            return None

        def fake_wait_for_server_ready(*_args, **_kwargs) -> None:
            _FakeThread.events.append("wait_for_server_ready")

        def fake_browser_open(*_args, **_kwargs) -> None:
            return None

        class FakeSpeakerDiarizer:
            instances: list["FakeSpeakerDiarizer"] = []

            @staticmethod
            def is_available() -> bool:
                return True

            def __init__(self, hf_token=None, device=None, profiles_path=None):
                self.hf_token = hf_token
                self.device = device
                self.profiles_path = profiles_path
                self._enrolled_names = []
                self.__class__.instances.append(self)

            def get_speaker_count(self) -> int:
                return 1

            def save_profiles(self, _path: Path) -> None:
                return None

        fake_diarize_module = types.SimpleNamespace(SpeakerDiarizer=FakeSpeakerDiarizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.md"
            profiles_path = Path(tmpdir) / "profiles.json"

            with patch("threading.Thread", new=_FakeThread), patch.dict(
                sys.modules,
                {"src.diarize": fake_diarize_module},
            ), patch(
                "src.audio_capture.capture_audio",
                new=lambda **_kwargs: iter(()),
            ), patch(
                "src.server.run_server",
                new=fake_run_server,
            ), patch(
                "src.server.set_startup_status",
                side_effect=fake_set_startup_status,
            ), patch(
                "src.server.set_diarizer",
            ) as set_diarizer, patch(
                "src.server.set_current_audio_device",
            ), patch(
                "src.server.is_shutdown_requested",
                return_value=True,
            ), patch(
                "src.server.get_audio_device_switch_event",
                return_value=object(),
            ), patch(
                "src.server.consume_audio_device_switch",
                return_value=(False, None),
            ), patch(
                "src.server.set_postprocess_status",
            ), patch(
                "src.meeting_writer.MeetingWriter",
                side_effect=lambda output_path=None: _FakeMeetingWriter(output_path),
            ), patch(
                "src.whisper_worker.WhisperWorker",
                new=_FakeWhisperWorker,
            ), patch(
                "src.tray.is_available",
                return_value=False,
            ), patch(
                "src.cli._wait_for_server_ready",
                side_effect=fake_wait_for_server_ready,
            ), patch(
                "webbrowser.open",
                new=fake_browser_open,
            ), patch(
                "os.startfile",
                create=True,
            ):
                cli._cmd_meeting(
                    self._build_args(
                        output=output_path,
                        no_diarize=False,
                        profiles=profiles_path,
                    )
                )

        self.assertIn("thread_start:fake_run_server", _FakeThread.events)
        self.assertIn("wait_for_server_ready", _FakeThread.events)
        self.assertIn("thread_start:fake_browser_open", _FakeThread.events)
        self.assertLess(
            _FakeThread.events.index("thread_start:fake_run_server"),
            _FakeThread.events.index("wait_for_server_ready"),
        )
        self.assertLess(
            _FakeThread.events.index("wait_for_server_ready"),
            _FakeThread.events.index("thread_start:fake_browser_open"),
        )
        self.assertIn(
            ("loading_diarizer", "화자 분리 모델 로드 중...", False),
            startup_calls,
        )
        diarizer = FakeSpeakerDiarizer.instances[0]
        set_diarizer.assert_called_once_with(diarizer, profiles_path)
        self.assertEqual(
            [call[0] for call in startup_calls[-2:]],
            ["loading_diarizer", "ready"],
        )


if __name__ == "__main__":
    unittest.main()
