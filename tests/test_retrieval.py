"""Check provider routing without sending requests or using real credentials."""

import os
import unittest
from unittest.mock import patch

from pydantic import SecretStr

from concierge.retrieval import _make_embeddings, retrieve


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


class KeywordRetrievalTests(unittest.TestCase):
    def test_faq_retrieval_needs_no_embedding_credentials(self) -> None:
        with patch.dict(os.environ, {"CONCIERGE_RETRIEVAL": "keyword"}, clear=True):
            with patch("concierge.retrieval._make_embeddings", side_effect=AssertionError("No provider calls")):
                documents = retrieve("Everyday Checking monthly service fee waived", k=2)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["source"], "checking_accounts.md")
        self.assertIn("$10", documents[0].page_content)
        self.assertIn("$500", documents[0].page_content)

    def test_empty_and_unmatched_queries_return_no_documents(self) -> None:
        with patch.dict(os.environ, {"CONCIERGE_RETRIEVAL": "keyword"}, clear=True):
            self.assertEqual(retrieve(""), [])
            self.assertEqual(retrieve("nonexistentxyzterm"), [])
            self.assertEqual(retrieve("banking", k=0), [])

    def test_unknown_retrieval_mode_fails_explicitly(self) -> None:
        with patch.dict(os.environ, {"CONCIERGE_RETRIEVAL": "invalid"}, clear=True):
            with self.assertRaises(ValueError):
                retrieve("checking fees")


if __name__ == "__main__":
    unittest.main()
