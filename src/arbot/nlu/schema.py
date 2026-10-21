"""What the language-understanding layer returns. Closed vocabularies only: whatever the
visitor types (or tries to inject), the NLU can only answer with these values.

Numbers, postcodes, phones and emails are NOT extracted here: deterministic parsers in
`arbot.domain` own those, so a model can never invent a budget.
"""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

SideKind = Literal["none", "faq", "property", "off_topic"]


class Understanding(BaseModel):
    side: SideKind = "none"                           # side question to answer, then resume
    yes_no: bool | None = None
    intent: Literal["buy", "sell"] | None = None
    category: Literal["new_build", "resale"] | None = None
    name: str | None = None
    switch_intent: Literal["buy", "sell"] | None = None   # visitor changed their mind


class NLU(Protocol):
    def understand(self, text: str, step: str, state: dict) -> Understanding: ...
