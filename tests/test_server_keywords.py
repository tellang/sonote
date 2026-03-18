import asyncio
import unittest

from src import server


class ServerKeywordStateTests(unittest.TestCase):
    def setUp(self) -> None:
        server._client_queues.clear()
        server._transcript_history.clear()
        server._manual_keywords.clear()
        server._extracted_keywords.clear()
        server._promoted_keywords.clear()
        server._blocked_keywords.clear()
        server._keyword_seen_counts.clear()
        server.set_current_audio_device(None)
        server.set_startup_status("", "", ready=False)

    def test_extracted_keywords_promote_after_repeat(self) -> None:
        payload = server.add_extracted_keywords(["LangGraph"])
        self.assertEqual(payload["extracted"], ["LangGraph"])
        self.assertEqual(payload["promoted"], [])
        self.assertEqual(server.get_keywords(), set())

        payload = server.add_extracted_keywords(["LangGraph"])
        self.assertEqual(payload["extracted"], [])
        self.assertEqual(payload["promoted"], ["LangGraph"])
        self.assertEqual(server.get_keywords(), {"LangGraph"})

    def test_blocked_keyword_is_not_reintroduced(self) -> None:
        server._blocked_keywords.add("회의록")
        payload = server.add_extracted_keywords(["회의록", "LangSmith"])
        self.assertEqual(payload["extracted"], ["LangSmith"])
        self.assertNotIn("회의록", payload["extracted"])

    def test_audio_device_switch_state_round_trip(self) -> None:
        server.set_current_audio_device(1)
        self.assertEqual(server.consume_audio_device_switch(), (False, None))

        changed = server.request_audio_device_switch(3)
        self.assertTrue(changed)
        self.assertEqual(server.consume_audio_device_switch(), (True, 3))

        payload = server._audio_device_payload([
            {"index": 1, "name": "Mic A"},
            {"index": 3, "name": "Mic B"},
        ])
        self.assertTrue(payload["switching"])
        self.assertEqual(payload["current_device"], 1)
        self.assertEqual(payload["requested_device"], 3)

        server.set_current_audio_device(3)
        self.assertEqual(server.consume_audio_device_switch(), (False, None))

    def test_requesting_same_audio_device_is_noop(self) -> None:
        server.set_current_audio_device(2)
        changed = server.request_audio_device_switch(2)
        self.assertFalse(changed)
        self.assertEqual(server.consume_audio_device_switch(), (False, None))

    def test_status_exposes_startup_state(self) -> None:
        server.set_startup_status("loading_asr", "STT 모델 로드 중...", ready=False)
        payload = asyncio.run(server.status())
        self.assertIn("startup", payload)
        self.assertEqual(payload["startup"]["phase"], "loading_asr")
        self.assertEqual(payload["startup"]["message"], "STT 모델 로드 중...")
        self.assertFalse(payload["startup"]["ready"])

    def test_push_correction_updates_history_and_broadcasts_named_event(self) -> None:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        server._client_queues.add(queue)
        server._transcript_history.append(
            {"speaker": "A", "text": "안녕하세여.", "ts": "00:00:01"}
        )

        asyncio.run(
            server.push_correction(
                [
                    {
                        "index": 0,
                        "original": "- [00:00:01] [A] 안녕하세여.",
                        "corrected": "안녕하세요.",
                    }
                ]
            )
        )

        self.assertEqual(server._transcript_history[0]["text"], "안녕하세요.")
        item = queue.get_nowait()
        self.assertEqual(item["_type"], "correction")
        self.assertEqual(
            item["_payload"],
            {
                "corrections": [
                    {
                        "index": 0,
                        "original": "- [00:00:01] [A] 안녕하세여.",
                        "corrected": "안녕하세요.",
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
