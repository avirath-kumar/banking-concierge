"""Exercise find_branch directory lookups."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

from concierge.graph import graph
from concierge.mock_data import BRANCHES
from concierge.tools import find_branch


class FindBranchTests(unittest.TestCase):
    def test_unmatched_zip_reports_no_match_without_nearest_branch(self) -> None:
        result = find_branch.invoke({"zip_code": "80211"})

        self.assertEqual(result["match"], False)
        self.assertIn("branch locator", result["message"])
        self.assertNotIn("nearest_known", result)
        self.assertNotIn("San Francisco", str(result))

    def test_matched_zip_returns_branch(self) -> None:
        result = find_branch.invoke({"zip_code": "94103"})

        self.assertEqual(result, {"match": True, **BRANCHES[0]})

    def test_invalid_zip_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "5-digit U.S. ZIP code"):
            find_branch.invoke({"zip_code": "80211-3476"})

    def test_unmatched_zip_tool_message_does_not_name_a_branch(self) -> None:
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_branch",
                        "args": {"zip_code": "80211"},
                        "id": "branch-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="No branch in our directory matches 80211."),
        ]

        with patch("concierge.graph._make_model", return_value=model):
            result = graph.invoke(
                {"messages": [{"role": "user", "content": "Nearest branch to 80211?"}]}
            )

        tool_message = result["messages"][2]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertIn('"match": false', tool_message.content)
        self.assertNotIn("San Francisco", tool_message.content)
        self.assertNotIn("nearest_known", tool_message.content)


if __name__ == "__main__":
    unittest.main()
