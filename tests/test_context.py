"""Keep the seed prompt available before LangSmith credentials are configured."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langsmith.utils import LangSmithNotFoundError

from concierge.context import get_prompt
from concierge.prompts import SYSTEM_PROMPT


class PromptFallbackTests(unittest.TestCase):
    def test_missing_credentials_skip_context_hub(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("concierge.context.Client") as client:
                prompt = get_prompt()

        self.assertEqual(prompt, SYSTEM_PROMPT)
        client.assert_not_called()

    def test_configured_credentials_load_the_hub_prompt(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "placeholder"}, clear=True):
            with patch("concierge.context.Client") as client:
                client.return_value.pull_agent.return_value = SimpleNamespace(
                    files={"AGENTS.md": SimpleNamespace(content="Hub instructions")}
                )
                prompt = get_prompt()

        self.assertEqual(prompt, "Hub instructions")

    def test_unseeded_hub_falls_back_without_logging_exception_details(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "placeholder"}, clear=True):
            with patch("concierge.context.Client") as client:
                client.return_value.pull_agent.side_effect = LangSmithNotFoundError(
                    "Private response details"
                )
                with self.assertLogs("concierge.context", level="WARNING") as logs:
                    prompt = get_prompt()

        self.assertEqual(prompt, SYSTEM_PROMPT)
        self.assertNotIn("Private response details", "\n".join(logs.output))

    def test_missing_agents_file_falls_back_to_seed(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "placeholder"}, clear=True):
            with patch("concierge.context.Client") as client:
                client.return_value.pull_agent.return_value = SimpleNamespace(files={})
                with self.assertLogs("concierge.context", level="WARNING"):
                    prompt = get_prompt()

        self.assertEqual(prompt, SYSTEM_PROMPT)

    def test_cloud_gateway_override_loads_context_hub(self) -> None:
        with patch.dict(os.environ, {"CONCIERGE_GATEWAY_API_KEY": "placeholder"}, clear=True):
            with patch("concierge.context.Client") as client:
                client.return_value.pull_agent.return_value = SimpleNamespace(
                    files={"AGENTS.md": SimpleNamespace(content="Hub instructions")}
                )
                prompt = get_prompt()

        self.assertEqual(prompt, "Hub instructions")
        client.assert_called_once_with(api_key="placeholder")


if __name__ == "__main__":
    unittest.main()
