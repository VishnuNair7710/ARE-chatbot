"""Conversation state: everything the bot remembers for one visitor (one thread)."""
from __future__ import annotations

from typing import Any, TypedDict

from arbot.graph.steps import Step


class BotState(TypedDict, total=False):
    step: str                       # Step value; the only thing that decides what happens next
    name: str | None
    intent: str | None              # "buy" | "sell"
    category: str | None            # "new_build" | "resale"
    budget_gbp: int | None
    postcode: str | None            # normalised, e.g. "GU5 7AQ"
    matches: list[dict[str, Any]]   # matches shown to the visitor
    property_id: str | None         # lane C: the specific property
    property_candidates: list[dict[str, Any]]  # lane C: several matches, visitor must pick
    turns: int
    thread_id: str
    phone: str | None
    email: str | None
    invalid: dict[str, int]         # consecutive invalid attempts per field
    outcome: str | None             # "M1" | "M2" | "M3" | "QUALIFIED"
    qualified: bool
    ended: bool
    # per-turn scratch
    user_text: str
    understanding: dict[str, Any]
    replies: list[str]
    history: list[dict[str, str]]   # trimmed window of the conversation


def initial_state() -> BotState:
    return BotState(
        step=Step.HOME.value, name=None, intent=None, category=None, budget_gbp=None,
        postcode=None, matches=[], property_id=None, property_candidates=[], turns=0, thread_id="",
        phone=None, email=None, invalid={}, outcome=None,
        qualified=False, ended=False, user_text="", understanding={}, replies=[], history=[],
    )
