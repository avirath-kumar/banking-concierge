"""Keep raw customer identifiers out of tool results and the system prompt contract."""

import json
import unittest
from unittest.mock import patch

from concierge.prompts import SYSTEM_PROMPT
from concierge.tools import account_lookup

CUSTOMER = {
    "customer_id": "CUST-9999",
    "name": "Test Holder",
    "ssn": "552-11-2233",
    "phone": "(415) 555-0199",
    "email": "test.holder@example.com",
    "credit_cards": [
        {"brand": "Visa", "number": "4111 1111 1111 1234", "cvv": "987", "exp": "09/28"},
        {"brand": "Mastercard", "number": "5500 0000 0000 5678", "cvv": "654", "exp": "01/30"},
    ],
    "accounts": [{"account_id": "1111", "type": "Everyday Checking", "balance": 10.5}],
}


class AccountLookupMaskingTests(unittest.TestCase):
    def lookup(self) -> dict:
        with patch.dict("concierge.tools.CUSTOMERS", {"CUST-9999": CUSTOMER}):
            return account_lookup.invoke({"customer_id": "CUST-9999"})

    def test_returns_allow_listed_and_masked_fields_only(self) -> None:
        result = self.lookup()

        self.assertEqual(
            set(result),
            {
                "customer_id",
                "name",
                "ssn_last4",
                "phone_masked",
                "email_masked",
                "credit_cards",
                "accounts",
            },
        )
        self.assertEqual(result["customer_id"], "CUST-9999")
        self.assertEqual(result["name"], "Test Holder")
        self.assertEqual(result["accounts"], CUSTOMER["accounts"])
        self.assertEqual(result["ssn_last4"], "2233")
        self.assertEqual(result["phone_masked"], "***-***-0199")
        self.assertEqual(result["email_masked"], "t***@example.com")

        serialized = json.dumps(result)
        for raw in ("552-11-2233", "(415) 555-0199", "test.holder@example.com"):
            self.assertNotIn(raw, serialized)

    def test_cards_expose_brand_last4_and_exp_only(self) -> None:
        result = self.lookup()

        self.assertEqual(
            result["credit_cards"],
            [
                {"brand": "Visa", "last4": "1234", "exp": "09/28"},
                {"brand": "Mastercard", "last4": "5678", "exp": "01/30"},
            ],
        )
        serialized = json.dumps(result)
        for card in CUSTOMER["credit_cards"]:
            self.assertNotIn(card["number"], serialized)
            self.assertNotIn(card["cvv"], serialized)
        for key in ("ssn", "cvv", "number"):
            self.assertNotIn(f'"{key}"', serialized)


class SystemPromptSensitiveDataTests(unittest.TestCase):
    def test_prompt_prohibits_reading_back_sensitive_identifiers(self) -> None:
        self.assertIn("never read back a full SSN, full card number, or CVV", SYSTEM_PROMPT)
        self.assertIn("have the caller state the identifier", SYSTEM_PROMPT)
        self.assertNotIn("verbatim", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
