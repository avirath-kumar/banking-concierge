"""Rule-based masking of PII in representative messages before they reach the model."""

from __future__ import annotations

import re

from langchain_core.messages import AnyMessage, HumanMessage

# Kept in sync with evals/evaluators.py `_detect_pii` so anything the
# pii_leak_rate evaluator flags is masked before the model can echo it.
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_EMAIL_PATTERN = re.compile(r"\b([\w.+-])[\w.+-]*@([\w-]+\.[\w.-]+)\b")
_PHONE_PATTERN = re.compile(r"(?:\(\d{3}\)|\b\d{3})[\s.-]?\d{3}[\s.-]?(\d{4})\b")
_CVV_LABEL_PATTERN = re.compile(
    r"(\b(?:cvv|cv2|cvc|security\s*code)\b[:\s]*)\d{3,4}\b",
    re.IGNORECASE,
)


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if not 13 <= len(digits) <= 19:
        return match.group(0)
    return f"**** {digits[-4:]}"


def redact_text(text: str) -> str:
    """Mask SSNs, card numbers, CVVs, phone numbers, and emails in text."""
    text = _SSN_PATTERN.sub(lambda m: f"***-**-{m.group(0)[-4:]}", text)
    text = _CARD_PATTERN.sub(_mask_card, text)
    text = _CVV_LABEL_PATTERN.sub(lambda m: f"{m.group(1)}***", text)
    text = _PHONE_PATTERN.sub(lambda m: f"***-***-{m.group(1)}", text)
    text = _EMAIL_PATTERN.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)
    return text


def _redact_content(content: str | list) -> str | list:
    if isinstance(content, str):
        return redact_text(content)
    redacted: list = []
    for block in content:
        if isinstance(block, str):
            redacted.append(redact_text(block))
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            redacted.append({**block, "text": redact_text(block["text"])})
        else:
            redacted.append(block)
    return redacted


def redact_human_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Return messages with human message content redacted; others are unchanged."""
    return [
        message.model_copy(update={"content": _redact_content(message.content)})
        if isinstance(message, HumanMessage)
        else message
        for message in messages
    ]
