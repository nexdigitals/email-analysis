import json
import unittest
from unittest import mock

from analyze import (
    _parse_gemini_payload,
    _strip_code_fences,
    fetch_screenshot_and_text,
    validate_url,
)


class HelperTests(unittest.TestCase):
    def test_strip_code_fences_removes_triple_backticks(self):
        payload = "```json\n{\"a\":1}\n```"
        self.assertEqual(_strip_code_fences(payload), '{"a":1}')

    def test_parse_gemini_payload_handles_code_fence(self):
        payload = "```json\n{\"a\":1}\n```"
        self.assertEqual(_parse_gemini_payload(payload), {"a": 1})

    def test_parse_gemini_payload_invalid_returns_none(self):
        self.assertIsNone(_parse_gemini_payload("not-json"))

    def test_validate_url_accepts_http_and_https(self):
        for url in ("http://example.com", "https://example.com"):
            self.assertEqual(validate_url(url), url)

    def test_validate_url_rejects_missing_scheme(self):
        with self.assertRaises(ValueError):
            validate_url("example.com")


class FetchTests(unittest.TestCase):
    @mock.patch("analyze.requests.get")
    def test_fetch_screenshot_and_text_http_mode(self, mock_get):
        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                return None

            @property
            def text(self):
                return "<html><body><h1>Hi</h1></body></html>"

        mock_get.return_value = FakeResp()
        screenshot, text, html, error = fetch_screenshot_and_text("https://example.com", render_js=False, timeout_sec=5)
        self.assertIsNone(screenshot)
        self.assertIn("Hi", text)
        self.assertIn("<h1>", html)
        self.assertIsNone(error)

    @mock.patch("analyze.requests.get")
    def test_fetch_screenshot_and_text_http_mode_error(self, mock_get):
        mock_get.side_effect = Exception("boom")
        screenshot, text, html, error = fetch_screenshot_and_text("https://example.com", render_js=False, timeout_sec=5)
        self.assertIsNone(screenshot)
        self.assertIsNone(text)
        self.assertIsNone(html)
        self.assertIn("boom", error)


if __name__ == "__main__":
    unittest.main()
