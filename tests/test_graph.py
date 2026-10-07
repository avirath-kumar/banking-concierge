"""Exercise the compiled graph's tool loop without provider calls."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from concierge.graph import graph


class GraphToolLoopTests(unittest.TestCase):
    def test_retrieval_result_returns_to_agent_and_increments_counter(self) -> None:
        expected_answer = "The retrieved document describes a monthly fee."
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_banking_docs",
                        "args": {"query": "Everyday Checking fee"},
                        "id": "retrieval-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=expected_answer),
        ]
        documents = [
            Document(
                page_content="Everyday Checking has a monthly fee.",
                metadata={"source": "checking_accounts.md"},
            )
        ]

        with patch("concierge.graph._make_model", return_value=model):
            with patch("concierge.tools.retrieve", return_value=documents) as retrieve:
                result = graph.invoke(
                    {"messages": [{"role": "user", "content": "What is the fee?"}]}
                )

        retrieve.assert_called_once_with("Everyday Checking fee", k=4)
        self.assertEqual(result["retrieval_calls"], 1)
        self.assertIsInstance(result["messages"][2], ToolMessage)
        self.assertIn("[source: checking_accounts.md]", result["messages"][2].content)
        second_prompt = model.invoke.call_args_list[1].args[0]
        self.assertIsInstance(second_prompt[-1], ToolMessage)
        self.assertEqual(result["messages"][-1].content, expected_answer)

    def test_model_receives_redacted_human_pii(self) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(content="Please share the customer ID.")

        with patch("concierge.graph._make_model", return_value=model):
            graph.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Customer's SSN is 552-77-1230, look them up?",
                        }
                    ]
                }
            )

        prompt = model.invoke.call_args.args[0]
        rendered = " ".join(str(message.content) for message in prompt[1:])
        self.assertNotIn("552-77-1230", rendered)
        self.assertIn("***-**-1230", rendered)


if __name__ == "__main__":
    unittest.main()
