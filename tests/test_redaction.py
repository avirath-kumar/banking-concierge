"""Check rule-based PII masking applied to representative messages."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from concierge.redaction import redact_human_messages, redact_text


class RedactTextTests(unittest.TestCase):
    def test_masks_ssn_keeping_last_four(self) -> None:
        self.assertEqual(
            redact_text("Customer's SSN is 552-77-1230."),
            "Customer's SSN is ***-**-1230.",
        )

    def test_masks_card_numbers_keeping_last_four(self) -> None:
        self.assertEqual(redact_text("Card 4242 4242 4242 4242"), "Card **** 4242")
        self.assertEqual(redact_text("Card 5555-5555-5555-4444"), "Card **** 4444")
        self.assertEqual(redact_text("Card 4012888888881881"), "Card **** 1881")

    def test_masks_cvv(self) -> None:
        self.assertEqual(redact_text("CVV 314"), "CVV ***")
        self.assertEqual(redact_text("security code: 0991"), "security code: ***")
        self.assertEqual(redact_text("cvc 208"), "cvc ***")

    def test_masks_email(self) -> None:
        self.assertEqual(
            redact_text("email priya.shah@example.com please"),
            "email p***@example.com please",
        )

    def test_masks_phone_numbers(self) -> None:
        self.assertEqual(redact_text("phone (212) 555-0193"), "phone ***-***-0193")
        self.assertEqual(redact_text("phone 212-555-0193"), "phone ***-***-0193")

    def test_leaves_non_pii_unchanged(self) -> None:
        text = (
            "CUST-0001 paid a $35 overdraft fee on 2024-05-01 near ZIP 94103; "
            "transfer $250.00 from account 1234."
        )
        self.assertEqual(redact_text(text), text)


class RedactHumanMessagesTests(unittest.TestCase):
    def test_only_human_messages_are_redacted(self) -> None:
        ai = AIMessage(content="SSN on file is 552-77-1230")
        tool = ToolMessage(content="ssn: 552-77-1230", tool_call_id="t1")
        human = HumanMessage(content="SSN 552-77-1230", id="h1")

        redacted = redact_human_messages([human, ai, tool])

        self.assertEqual(redacted[0].content, "SSN ***-**-1230")
        self.assertEqual(redacted[0].id, "h1")
        self.assertIs(redacted[1], ai)
        self.assertIs(redacted[2], tool)
        self.assertEqual(human.content, "SSN 552-77-1230")

    def test_redacts_list_content_text_blocks(self) -> None:
        human = HumanMessage(
            content=[{"type": "text", "text": "email a.b@example.com"}, "SSN 552-77-1230"]
        )

        redacted = redact_human_messages([human])

        self.assertEqual(
            redacted[0].content,
            [{"type": "text", "text": "email a***@example.com"}, "SSN ***-**-1230"],
        )


if __name__ == "__main__":
    unittest.main()
