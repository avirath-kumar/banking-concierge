"""Check provider routing without sending requests or using real credentials."""

import os
import unittest
from unittest.mock import patch

from pydantic import SecretStr

from concierge.retrieval import _make_embeddings


class EmbeddingsAuthenticationTests(unittest.TestCase):
    def test_direct_openai_works_without_langsmith_credentials(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder-openai"}, clear=True):
            embeddings = _make_embeddings()

        self.assertEqual(embeddings.openai_api_key, SecretStr("placeholder-openai"))
        self.assertIsNone(embeddings.openai_api_base)
        self.assertEqual(str(embeddings.client._client.base_url), "https://api.openai.com/v1/")

    def test_gateway_uses_langsmith_credentials(self) -> None:
        environment = {
            "OPENAI_API_KEY": "placeholder-openai",
            "LANGSMITH_API_KEY": "placeholder-langsmith",
            "BASE_URL": "https://gateway.smith.langchain.com/openai/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            embeddings = _make_embeddings()

        self.assertEqual(embeddings.openai_api_key, SecretStr("placeholder-langsmith"))
        self.assertEqual(embeddings.openai_api_base, environment["BASE_URL"])
        self.assertEqual(str(embeddings.client._client.base_url), environment["BASE_URL"] + "/")

    def test_gateway_does_not_fall_back_to_openai_credentials(self) -> None:
        environment = {
            "OPENAI_API_KEY": "placeholder-openai",
            "BASE_URL": "https://gateway.smith.langchain.com/openai/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(KeyError):
                _make_embeddings()

    def test_cloud_gateway_override_works_without_reserved_key(self) -> None:
        environment = {
            "CONCIERGE_GATEWAY_API_KEY": "placeholder-runtime",
            "BASE_URL": "https://gateway.smith.langchain.com/openai/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            embeddings = _make_embeddings()

        self.assertEqual(embeddings.openai_api_key, SecretStr("placeholder-runtime"))


if __name__ == "__main__":
    unittest.main()
