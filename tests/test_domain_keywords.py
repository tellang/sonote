import unittest

from src.domain_keywords import (
    DEFAULT_DOMAIN_HINT,
    DEFAULT_MEETING_DOMAIN_KEYWORDS,
    DEFAULT_MEETING_PROMPT_KEYWORDS,
    build_default_domain_hint,
)


class DomainKeywordTests(unittest.TestCase):
    def test_default_keyword_inventory_is_deduplicated(self) -> None:
        self.assertEqual(
            len(DEFAULT_MEETING_DOMAIN_KEYWORDS),
            len(set(DEFAULT_MEETING_DOMAIN_KEYWORDS)),
        )

    def test_prompt_keywords_include_dev_and_exam_terms(self) -> None:
        expected = {"R&R", "API", "도커", "쿠버네티스", "정규화", "요구사항", "LLM"}
        self.assertTrue(expected.issubset(set(DEFAULT_MEETING_PROMPT_KEYWORDS)))

    def test_default_domain_hint_is_bounded(self) -> None:
        hint = build_default_domain_hint(320)
        self.assertLessEqual(len(hint), 320)
        self.assertTrue(hint.endswith(". "))
        self.assertIn("API", hint)
        self.assertIn("쿠버네티스", hint)
        self.assertEqual(DEFAULT_DOMAIN_HINT, hint)


if __name__ == "__main__":
    unittest.main()
