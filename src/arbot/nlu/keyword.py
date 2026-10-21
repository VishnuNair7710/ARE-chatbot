"""Rule-based NLU. Used (a) as the offline fallback when the LLM is down and (b) as the
deterministic NLU for flow tests. The LLM-backed NLU (P3) returns the same schema."""
from __future__ import annotations

import re

from arbot.domain.budget import parse_budget
from arbot.domain.postcode import extract_postcode
from arbot.domain.validators import validate_email_address, validate_phone
from arbot.graph.steps import Step
from arbot.nlu.schema import Understanding

_I = re.IGNORECASE
_NO = re.compile(r"\b(no|nope|nah|nothing|not really|that'?s all|all good|i'?m good|no thanks)\b", _I)
_YES = re.compile(r"\b(yes|yeah|yep|yup|sure|please|ok(ay)?|correct|of course|definitely|i do)\b", _I)
_SELL = re.compile(r"\b(sell|selling|valuation|value my|list my)\b", _I)
_BUY = re.compile(r"\b(buy|buying|purchase|purchasing|looking for|looking to|interested in)\b", _I)
_NEW = re.compile(r"new[- ]?builds?|brand[- ]new|off[- ]plan", _I)
_RESALE = re.compile(r"resale|re-sale|second[- ]hand|pre-owned|existing|resell", _I)
_CUE = re.compile(r"\b(actually|instead|rather|changed my mind|change my mind|switch|on second thought)\b", _I)
_QUESTION = re.compile(
    r"\?|^\s*(what|where|when|who|why|how|which|do you|does|can you|could you|are you|is there|is it|will)\b", _I)
_REQUEST = re.compile(
    r"^\s*(please\s+)?(tell|give|show|explain|list|let me know|send)\b|\bi('d| would)? (like|want) to know\b"
    r"|\bi wonder\b|\bcan you tell\b", _I)
_PROPERTY_Q = re.compile(
    r"\b(bedrooms?|bathrooms?|price of|how much is|cost of|still available|listing|garden|parking|"
    r"apartment|penthouse|townhouse|cottage|house|flat|property|properties)\b", _I)
_OFF_TOPIC = re.compile(
    r"\b(weather|joke|poem|recipe|football|capital of|stock|crypto|bitcoin|movie|write (me )?(a|some)|"
    r"ignore (all |your |previous )|system prompt)\b", _I)
_NAME_LEAD = re.compile(
    r"(?:my name is|my name's|i'?m|i am|it'?s|it is|this is|call me|name'?s)\s+"
    r"([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)", _I)
_NOT_NAMES = {
    "looking", "want", "to", "buy", "sell", "a", "the", "interested", "here", "just", "trying", "not",
    "so", "yes", "no", "hi", "hello", "hey", "please", "thanks", "thank", "you", "new", "build",
    "resale", "property", "house", "home", "flat", "selling", "buying", "need", "would", "like",
}
EXCLUDED_FROM_SWITCH = {Step.HOME, Step.ASK_NAME, Step.BUY_OR_SELL, Step.E3_ANYTHING_ELSE, Step.ENDED}


def _extract_name(text: str) -> str | None:
    m = _NAME_LEAD.search(text)
    candidate = m.group(1) if m else text.strip().strip(".!")
    words = candidate.split()
    if not 1 <= len(words) <= 3:
        return None
    if any(w.lower() in _NOT_NAMES or not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w) for w in words):
        return None
    return " ".join(w.capitalize() for w in words)


def looks_like_question(text: str) -> bool:
    """A question or a request for information ('tell me about your fees'). Used so that a
    model cannot turn a plain answer ('Castle Anthrax', 'not-an-email') into a side question."""
    return bool(_QUESTION.search(text) or _REQUEST.search(text))


def answers_current_step(text: str, step: Step, u: Understanding) -> bool:
    if step in (Step.A1_BUDGET, Step.B1_BUDGET, Step.C5_BUDGET):
        b = parse_budget(text)
        return b.value is not None or b.ambiguous
    if step in (Step.A3_POSTCODE, Step.B2_POSTCODE, Step.D1_POSTCODE):
        return extract_postcode(text) is not None
    if step == Step.E1_PHONE:
        return validate_phone(text) is not None
    if step == Step.E2_EMAIL:
        return validate_email_address(text) is not None
    if step == Step.ASK_NAME:
        return u.name is not None
    if step == Step.BUY_OR_SELL:
        return u.intent is not None
    if step == Step.NEW_OR_RESALE:
        return u.category is not None
    if step in (Step.SPECIFIC_PROPERTY, Step.E3_ANYTHING_ELSE):
        return u.yes_no is not None or u.intent is not None
    return False


class KeywordNLU:
    def understand(self, text: str, step: str, state: dict) -> Understanding:
        step_ = Step(step)
        u = Understanding()
        if _NO.search(text):
            u.yes_no = False
        elif _YES.search(text):
            u.yes_no = True
        sell, buy = bool(_SELL.search(text)), bool(_BUY.search(text))
        if sell != buy:
            u.intent = "sell" if sell else "buy"
        new, resale = bool(_NEW.search(text)), bool(_RESALE.search(text))
        if new != resale:
            u.category = "new_build" if new else "resale"
        if step_ == Step.ASK_NAME:
            u.name = _extract_name(text)

        answered = answers_current_step(text, step_, u)
        if not answered and looks_like_question(text):      # includes 'tell me about...' requests
            if _OFF_TOPIC.search(text):
                u.side = "off_topic"
            elif _PROPERTY_Q.search(text):
                u.side = "property"
            else:
                u.side = "faq"
        if (u.side == "none" and u.intent and step_ not in EXCLUDED_FROM_SWITCH
                and u.intent != state.get("intent") and (_CUE.search(text) or not answered)):
            u.switch_intent = u.intent
        return u
