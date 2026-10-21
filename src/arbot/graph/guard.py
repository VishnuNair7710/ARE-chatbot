"""Input and output guards. They are a second line of defence: the first is architectural
(qualification outcomes come from code, never from model text), so even a fully jailbroken
model cannot qualify a lead or invent contact details that reach the visitor."""
from __future__ import annotations

import re
from typing import Iterable

from arbot.graph.prompts import NOT_IN_DATA, OFFICE_PHONE

_INJECTION = re.compile(
    r"(ignore|disregard|forget|override|bypass)\s+(?:(?:all|any|your|the|previous|prior|above|earlier|these|those)\s+)*"
    r"(instructions?|rules?|prompts?|guidelines?|constraints?|restrictions?)"
    r"|system\s+prompt|developer\s+mode|jailbreak|you\s+are\s+now\b|pretend\s+(to\s+be|you\s+are)"
    r"|act\s+as\s+(an?\s+)?(admin|developer|dan|root)\b|reveal\s+(your|the)\s+(prompt|instructions?)",
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?<![\w])\+?\d[\d\s().-]{7,}\d(?![\w])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)[^\s)]+", re.IGNORECASE)
_LEAK_MARKERS = ("only from what they return", "visitor_question", "not_in_data", "system prompt",
                 "my instructions", "language-understanding component")


_ODD_SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " ", "‑": "-"})


def normalise_spaces(text: str) -> str:
    """Models like to emit narrow no-break spaces and non-breaking hyphens; they render oddly
    and break exact matching, so replace them with plain characters."""
    return text.translate(_ODD_SPACES)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def mask_pii(text: str) -> str:
    """For logs: never write visitors' phone numbers or emails to disk."""
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text or ""))


class OutputGuard:
    """Replies may only contain phone numbers, emails and URLs that already appear in the
    company's own data (the FAQ) or are the office number."""

    def __init__(self, allowed_texts: Iterable[str]):
        corpus = " ".join(allowed_texts) + " " + OFFICE_PHONE
        self.phones = {_digits(m) for m in _PHONE.findall(corpus)}
        self.emails = {m.lower() for m in _EMAIL.findall(corpus)}
        self.urls = {m.lower().rstrip(".,") for m in _URL.findall(corpus)}

    def check(self, reply: str) -> str:
        low = reply.lower()
        if any(marker in low for marker in _LEAK_MARKERS):
            return NOT_IN_DATA
        if any(_digits(m) not in self.phones for m in _PHONE.findall(reply)):
            return NOT_IN_DATA
        if any(m.lower() not in self.emails for m in _EMAIL.findall(reply)):
            return NOT_IN_DATA
        if any(m.lower().rstrip(".,") not in self.urls for m in _URL.findall(reply)):
            return NOT_IN_DATA
        return reply
