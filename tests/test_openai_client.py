import os
import unittest
from unittest.mock import patch

from openai_client import create_openai_client


class OpenAIClientTests(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_runtime_base_url_is_passed_without_being_stored(self, client_class):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
        }, clear=False):
            create_openai_client()
        client_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )

    @patch("openai.OpenAI")
    def test_default_endpoint_is_left_to_openai_sdk(self, client_class):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            create_openai_client()
        client_class.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
