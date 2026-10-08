"""Exercise the two-phase transfer flow through the compiled graph without provider calls."""

import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from concierge.graph import graph
from concierge.prompts import SYSTEM_PROMPT


def _call(name: str, call_id: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _run(messages: list, *calls: dict) -> list:
    model = Mock()
    model.invoke.side_effect = [
        AIMessage(content="", tool_calls=list(calls)),
        AIMessage(content="done"),
    ]
    with patch("concierge.graph._make_model", return_value=model):
        return graph.invoke({"messages": messages})["messages"]


def _tool_results(messages: list) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _submitted(messages: list) -> list[dict]:
    return [
        json.loads(m.content)
        for m in _tool_results(messages)
        if m.name == "transfer_funds" and m.status != "error"
    ]


PREPARE = _call("prepare_transfer", "prep-1", from_account="1234", to_account="5678", amount=50)


class TransferFlowTests(unittest.TestCase):
    def _prepared_thread(self) -> list:
        return _run([HumanMessage("Move $50 from 1234 to 5678.")], PREPARE)

    def test_prepare_returns_pending_preview_without_submitting(self) -> None:
        messages = self._prepared_thread()
        [result] = _tool_results(messages)
        preview = json.loads(result.content)
        self.assertEqual(preview["status"], "pending_confirmation")
        self.assertFalse(preview["submitted"])
        self.assertEqual(preview["amount"], 50.0)
        self.assertEqual(preview["confirmation_token"], "prep-1")
        self.assertEqual(_submitted(messages), [])

    def test_prepare_rejects_invalid_amounts_and_accounts(self) -> None:
        cases = [
            {"from_account": "1234", "to_account": "5678", "amount": 0},
            {"from_account": "1234", "to_account": "5678", "amount": -5},
            {"from_account": "1234", "to_account": "5678", "amount": 10.005},
            {"from_account": "source_account_id", "to_account": "5678", "amount": 275},
            {"from_account": "", "to_account": "5678", "amount": 275},
            {"from_account": "MNB-438271", "to_account": "MNB-7359042", "amount": 18.47},
        ]
        for args in cases:
            with self.subTest(args=args):
                messages = _run(
                    [HumanMessage("transfer")], _call("prepare_transfer", "p", **args)
                )
                [result] = _tool_results(messages)
                self.assertEqual(result.status, "error")
                self.assertEqual(_submitted(messages), [])

    def test_transfer_rejected_without_valid_confirmation(self) -> None:
        prepared = self._prepared_thread()
        confirmed = [*prepared, HumanMessage("Yes, submit it.")]
        cases = {
            "no prepare": ([HumanMessage("Move $50 from 1234 to 5678.")], "prep-1", 50),
            "invalid token": (confirmed, "bogus", 50),
            "altered amount": (confirmed, "prep-1", 500),
            "not yet confirmed": (prepared, "prep-1", 50),
        }
        for label, (messages, token, amount) in cases.items():
            with self.subTest(label):
                result = _run(
                    list(messages),
                    _call(
                        "transfer_funds",
                        "xfer",
                        from_account="1234",
                        to_account="5678",
                        amount=amount,
                        confirmation_token=token,
                    ),
                )
                self.assertEqual(_tool_results(result)[-1].status, "error")
                self.assertEqual(_submitted(result), [])

    def test_missing_token_is_rejected(self) -> None:
        result = _run(
            [HumanMessage("Move $50 from 1234 to 5678.")],
            _call("transfer_funds", "xfer", from_account="1234", to_account="5678", amount=50),
        )
        self.assertEqual(_tool_results(result)[-1].status, "error")
        self.assertEqual(_submitted(result), [])

    def test_same_turn_prepare_and_submit_is_rejected(self) -> None:
        result = _run(
            [HumanMessage("Move $50 from 1234 to 5678.")],
            PREPARE,
            _call(
                "transfer_funds",
                "xfer",
                from_account="1234",
                to_account="5678",
                amount=50,
                confirmation_token="prep-1",
            ),
        )
        self.assertEqual(_submitted(result), [])

    def test_confirmed_transfer_submits_exactly_once(self) -> None:
        confirmed = [*self._prepared_thread(), HumanMessage("Yes, submit it.")]
        submit = {
            "from_account": "1234",
            "to_account": "5678",
            "amount": 50.0,
            "confirmation_token": "prep-1",
        }
        result = _run(
            confirmed,
            _call("transfer_funds", "xfer-1", **submit),
            _call("transfer_funds", "xfer-2", **submit),
        )
        submitted = _submitted(result)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["status"], "submitted")
        self.assertEqual(
            (submitted[0]["from_account"], submitted[0]["to_account"], submitted[0]["amount"]),
            ("1234", "5678", 50.0),
        )

        replay = _run(
            [*result, HumanMessage("Do it again.")],
            _call("transfer_funds", "xfer-3", **submit),
        )
        self.assertEqual(len(_submitted(replay)), 1)

    def test_prompt_requires_confirmation_and_accurate_status(self) -> None:
        for phrase in ("prepare_transfer", "explicitly confirms", "draft", "never describe a submitted transfer"):
            self.assertIn(phrase, SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
