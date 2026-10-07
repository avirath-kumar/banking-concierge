"""Check that tool schemas expose the input constraints enforced at runtime."""

import unittest

from langchain_core.utils.function_calling import convert_to_openai_tool

from concierge.tools import account_lookup, recent_transactions


def _description(tool) -> str:
    return convert_to_openai_tool(tool)["function"]["description"]


class ToolSchemaConstraintTests(unittest.TestCase):
    def test_recent_transactions_documents_limit_range_and_cap(self) -> None:
        description = _description(recent_transactions)
        self.assertIn("from 1 to 50", description)
        self.assertIn("maximum 50", description)
        self.assertIn("request 50 and tell them", description)

    def test_recent_transactions_documents_customer_id_format(self) -> None:
        self.assertIn(
            "customer_id: The customer ID in the format CUST-####",
            _description(recent_transactions),
        )

    def test_account_lookup_documents_customer_id_format(self) -> None:
        description = _description(account_lookup)
        self.assertIn("customer_id: The customer ID in the format CUST-####", description)
        self.assertIn("instead of calling this tool", description)

    def test_runtime_validation_still_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum of 50"):
            recent_transactions.invoke({"customer_id": "CUST-0001", "limit": 200})
        with self.assertRaisesRegex(ValueError, "CUST-####"):
            recent_transactions.invoke({"customer_id": "12345"})
        with self.assertRaisesRegex(ValueError, "CUST-####"):
            account_lookup.invoke({"customer_id": "12345"})


if __name__ == "__main__":
    unittest.main()
