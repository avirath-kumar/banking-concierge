"""Tools available to the Meridian National customer service concierge agent.

A few tools have deliberate rough edges so LangSmith Engine has something
to cluster after the load generator runs:

- search_banking_docs has a vague description so the model occasionally
  re-queries multiple times rephrasing
- account_lookup raises on malformed customer IDs and on IDs prefixed with
  "X" (simulated downstream outage)
- recent_transactions raises if the model passes a runaway limit
- find_branch raises on non-zip inputs
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from concierge.mock_data import (
    BRANCHES,
    CUSTOMERS,
    TRANSACTIONS,
    find_branch_by_zip,
)
from concierge.retrieval import retrieve


@tool
def search_banking_docs(query: str, k: int = 4) -> str:
    """Search Meridian National banking documentation.

    Args:
        query: A natural-language search query.
        k: Number of relevant chunks to return. Defaults to 4.
    """
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant documentation found."
    blocks = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


@tool
def account_lookup(customer_id: str) -> dict:
    """Look up account information.

    Returns the customer's name and a list of their account IDs, account
    types, and balances. Use this when the user wants details about an
    account.
    """
    if customer_id.startswith("X"):
        raise RuntimeError(
            "Customer record service is temporarily unavailable. Try again later."
        )
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ValueError(
            f"No customer found with ID {customer_id!r}. "
            "Customer IDs are in the format CUST-####."
        )
    return dict(customer)


@tool
def recent_transactions(customer_id: str, limit: int = 5) -> list[dict]:
    """Retrieve a customer's most recent transactions.

    Args:
        customer_id: The customer ID (e.g. CUST-0001).
        limit: Optional number of transactions to return.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > 50:
        raise ValueError(
            f"limit {limit} exceeds the maximum of 50. Pick a smaller number."
        )
    if customer_id not in CUSTOMERS:
        raise ValueError(
            f"No customer found with ID {customer_id!r}. "
            "Customer IDs are in the format CUST-####."
        )
    txs = TRANSACTIONS.get(customer_id, [])
    return [dict(t) for t in txs[:limit]]


@tool
def find_branch(zip_code: str) -> dict:
    """Find a Meridian National branch.

    Args:
        zip_code: A 5-digit U.S. ZIP code.
    """
    if not (isinstance(zip_code, str) and len(zip_code) == 5 and zip_code.isdigit()):
        raise ValueError(
            f"zip_code must be a 5-digit U.S. ZIP code. Got {zip_code!r}."
        )
    branch = find_branch_by_zip(zip_code)
    if branch is None:
        return {
            "match": False,
            "message": "No Meridian National branch found in our directory for that ZIP code.",
            "nearest_known": BRANCHES[0],
        }
    return {"match": True, **branch}


def _transfer_amount(amount: float) -> Decimal:
    """Return the amount as an exact Decimal, rejecting non-positive or sub-cent values."""
    try:
        value = Decimal(str(amount))
    except InvalidOperation:
        raise ValueError(f"amount {amount!r} is not a number") from None
    if not value.is_finite() or value <= 0:
        raise ValueError("amount must be positive")
    if value.as_tuple().exponent < -2:
        raise ValueError(
            f"amount {amount} has more than two decimal places. "
            "Ask the representative for the exact amount; do not round it."
        )
    return value


def _check_accounts(*account_ids: str) -> None:
    """Raise if any account ID is not a Meridian National account on file."""
    known = {a["account_id"] for c in CUSTOMERS.values() for a in c["accounts"]}
    for account_id in account_ids:
        if account_id not in known:
            raise ValueError(
                f"No Meridian National account found with ID {account_id!r}. "
                "Ask the representative for the correct account ID."
            )


def _check_confirmation(
    messages: list, token: str, call_id: str, from_account: str, to_account: str, amount: Decimal
) -> None:
    """Raise unless the token names a prepared transfer with these exact details that the rep replied to and that was not already submitted."""
    prepared = None
    previewed = False
    earlier_submits: set[str] = set()
    failed_calls: set[str] = set()
    confirmed = False
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["id"] == call_id:
                    break
                if call["name"] == "prepare_transfer" and call["id"] == token:
                    prepared = call["args"]
                if (
                    call["name"] == "transfer_funds"
                    and call["args"].get("confirmation_token") == token
                ):
                    earlier_submits.add(call["id"])
        elif isinstance(message, ToolMessage):
            if message.status == "error":
                failed_calls.add(message.tool_call_id)
            elif message.tool_call_id == token and message.name == "prepare_transfer":
                previewed = True
        elif isinstance(message, HumanMessage) and previewed:
            confirmed = True
    if prepared is None or not previewed:
        raise ValueError(
            "Invalid confirmation_token. Call prepare_transfer and get the "
            "representative's explicit confirmation first. Nothing was submitted."
        )
    if not confirmed:
        raise ValueError(
            "The representative has not confirmed this transfer yet. Read back the "
            "prepared details and wait for their explicit confirmation. Nothing was submitted."
        )
    if (
        prepared.get("from_account") != from_account
        or prepared.get("to_account") != to_account
        or Decimal(str(prepared.get("amount"))) != amount
    ):
        raise ValueError(
            "Transfer details do not match the confirmed transfer. Prepare and "
            "confirm the new details first. Nothing was submitted."
        )
    if earlier_submits - failed_calls:
        raise ValueError("This confirmed transfer was already submitted. Nothing was submitted.")


@tool
def prepare_transfer(
    from_account: str,
    to_account: str,
    amount: float,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> dict:
    """Validate a transfer between two Meridian National accounts and return a preview for the representative to confirm; nothing is submitted."""
    value = _transfer_amount(amount)
    _check_accounts(from_account, to_account)
    return {
        "status": "pending_confirmation",
        "submitted": False,
        "from_account": from_account,
        "to_account": to_account,
        "amount": float(value),
        "confirmation_token": tool_call_id,
        "next_step": (
            "Read these exact details back to the representative. Call "
            "transfer_funds with this token only after they explicitly confirm."
        ),
    }


@tool
def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    confirmation_token: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> dict:
    """Submit a transfer immediately and irreversibly; requires the confirmation_token from prepare_transfer and the representative's explicit confirmation of those exact details."""
    value = _transfer_amount(amount)
    _check_accounts(from_account, to_account)
    _check_confirmation(
        state["messages"], confirmation_token, tool_call_id, from_account, to_account, value
    )
    confirmation = f"MNB-XFER-{abs(hash((from_account, to_account, value))) % 10_000_000:07d}"
    return {
        "status": "submitted",
        "from_account": from_account,
        "to_account": to_account,
        "amount": float(value),
        "confirmation": confirmation,
        "estimated_post": "immediately",
    }


TOOLS = [
    search_banking_docs,
    account_lookup,
    recent_transactions,
    find_branch,
    prepare_transfer,
    transfer_funds,
]
