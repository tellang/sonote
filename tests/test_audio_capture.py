import unittest
import threading
from unittest import mock

import numpy as np

from src import audio_capture


class AudioCaptureHeuristicTests(unittest.TestCase):
    def test_count_trailing_silence_counts_only_tail(self) -> None:
        samples = np.array([0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(audio_capture._count_trailing_silence(samples, 0.01), 3)

    def test_should_emit_chunk_for_fixed_max(self) -> None:
        self.assertTrue(
            audio_capture._should_emit_chunk(
                buffered_samples=16000,
                min_chunk_samples=8000,
                max_chunk_samples=16000,
                trailing_silence_samples=0,
                silence_trigger_samples=4000,
                heuristic_split=False,
            )
        )

    def test_should_emit_chunk_for_trailing_silence_after_min(self) -> None:
        self.assertTrue(
            audio_capture._should_emit_chunk(
                buffered_samples=12000,
                min_chunk_samples=8000,
                max_chunk_samples=16000,
                trailing_silence_samples=5000,
                silence_trigger_samples=4000,
                heuristic_split=True,
            )
        )
        self.assertFalse(
            audio_capture._should_emit_chunk(
                buffered_samples=6000,
                min_chunk_samples=8000,
                max_chunk_samples=16000,
                trailing_silence_samples=5000,
                silence_trigger_samples=4000,
                heuristic_split=True,
            )
        )

    def test_capture_audio_stops_when_stop_event_is_set(self) -> None:
        class DummyInputStream:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        stop_event = threading.Event()
        stop_event.set()
        started = []

        with mock.patch.object(audio_capture.sd, "InputStream", DummyInputStream):
            generator = audio_capture.capture_audio(
                chunk_seconds=1.0,
                stop_event=stop_event,
                on_stream_started=lambda: started.append(True),
            )
            with self.assertRaises(StopIteration):
                next(generator)

        self.assertEqual(started, [True])


if __name__ == "__main__":
    unittest.main()
