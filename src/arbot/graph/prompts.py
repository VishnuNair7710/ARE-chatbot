"""Fixed wording from flowchart.png / the brief. Flow text is never LLM-generated,
so every question and closing message can be asserted exactly in tests."""
from __future__ import annotations

from arbot.domain.budget import format_gbp
from arbot.graph.state import BotState
from arbot.graph.steps import Step

OFFICE_PHONE = "020 7946 0100"
NOT_IN_DATA = f"I'm sorry, I don't have that information. Please call our office on {OFFICE_PHONE}."

GREETING = "How may I help you?"
ASK_NAME = "Can I get your name?"
BUY_OR_SELL = "Are you looking to buy or sell a property?"
SPECIFIC_PROPERTY = "Do you have a specific property in mind?"
NEW_OR_RESALE = "Are you looking for a new-build or a resale home?"
BUDGET = "What is your budget?"
POSTCODE_INTEREST = "Postcode of your location of interest?"
POSTCODE_SELLER = "What is your postcode?"
WHICH_PROPERTY = "Which property is it?"
PROPERTY_RETRY = "Sorry, I couldn't find that property. Could you give me its name or address once more?"
SIMILAR_HOME = "Sorry, that one's gone. I can help you find a similar home. What's your budget?"
ASK_PHONE = "Can I get your phone number?"
ASK_EMAIL = "Can I get your email address?"
CALL_BACK = ("Someone will get in touch within 24 hours via phone or email. "
             "Is there anything else I can help you with?")
REASSIST = "How can I help you?"
GOODBYE = "Thank you for chatting with us. Goodbye."
CHAT_ENDED = f"This chat has ended. For anything else, please call our office on {OFFICE_PHONE}."

ENDINGS = {
    "M1": "Sorry, we don't cater to new-build properties under £1 million.",
    "M2": "Sorry, we don't cater to the postcode you provided.",
    "M3": "Sorry, we have no available homes matching your requirements right now.",
}


def ending(kind: str) -> str:
    """M1/M2/M3 all give the office number, then say goodbye."""
    return f"{ENDINGS[kind]} You can reach our office on {OFFICE_PHONE}. Goodbye."


def format_matches(matches: list[dict]) -> str:
    lines = [f"{i}. {m['name']} – {m['bedrooms']} bed – {format_gbp(m['price'])}"
             for i, m in enumerate(matches, 1)]
    return "Here are some homes that match what you're looking for:\n" + "\n".join(lines)


_STATIC = {
    Step.HOME: GREETING,
    Step.ASK_NAME: ASK_NAME,
    Step.BUY_OR_SELL: BUY_OR_SELL,
    Step.SPECIFIC_PROPERTY: SPECIFIC_PROPERTY,
    Step.NEW_OR_RESALE: NEW_OR_RESALE,
    Step.A1_BUDGET: BUDGET,
    Step.B1_BUDGET: BUDGET,
    Step.A3_POSTCODE: POSTCODE_INTEREST,
    Step.B2_POSTCODE: POSTCODE_INTEREST,
    Step.D1_POSTCODE: POSTCODE_SELLER,
    Step.C1_PROPERTY: WHICH_PROPERTY,
    Step.C1_RETRY: PROPERTY_RETRY,
    Step.C5_BUDGET: SIMILAR_HOME,
    Step.E1_PHONE: ASK_PHONE,
    Step.E2_EMAIL: ASK_EMAIL,
    Step.E3_ANYTHING_ELSE: "Is there anything else I can help you with?",
    Step.ENDED: CHAT_ENDED,
}


def prompt_for(step: Step | str, state: BotState | None = None) -> str:
    """The question for a step. Used both to advance the flow and to resume it after a
    side question, so the wording can never drift between the two."""
    step = Step(step)
    if step == Step.C_PICK:
        options = "\n".join(f"{i}. {c['name']} – {c['address']}"
                            for i, c in enumerate((state or {}).get("property_candidates", []), 1))
        return f"I found a few matches:\n\n{options}\n\nWhich one do you mean?"   # blank lines: markdown list
    return _STATIC[step]
