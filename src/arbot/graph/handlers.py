"""One handler per flowchart step. Each consumes the visitor's answer to the question
last asked, applies a deterministic rule, then moves `step` and says what comes next.

Handlers mutate a plain-dict copy of the state (see build.py) and append to
`state["replies"]`. They never call an LLM: language understanding is already in `u`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from arbot.agents.side_question import SideAnswerer
from arbot.data.property_index import PropertyIndex
from arbot.domain.budget import format_gbp, parse_budget
from arbot.domain.postcode import extract_postcode
from arbot.domain.qualify import (MIN_NEW_BUILD_BUDGET, Outcome, qualify_new_build, qualify_resale,
                                  qualify_seller)
from arbot.domain.validators import validate_email_address, validate_phone
from arbot.graph import prompts as P
from arbot.graph.guard import normalise_spaces
from arbot.graph.state import BotState
from arbot.graph.steps import Step
from arbot.nlu.schema import NLU, Understanding

MAX_INVALID_BEFORE_OFFICE_HINT = 3
_BUDGET_WORDS = re.compile(r"£|\d\s*(k|m|mn|mil|million|thousand|grand)\b", re.IGNORECASE)


@dataclass
class Ctx:
    listings: pd.DataFrame
    coverage: frozenset[str]
    nlu: NLU
    side: SideAnswerer
    props: PropertyIndex | None = None
    lane_c: bool = True                             # bonus lane: buyer with a specific property
    strict_flow: bool = False                       # always ask every question verbatim
    on_lead: Callable[[BotState], None] = lambda s: None


# ---- helpers ---------------------------------------------------------------
def say(s: BotState, text: str) -> None:
    s["replies"].append(normalise_spaces(text))


def goto(s: BotState, step: Step, *, ask: bool = True, prefix: str = "") -> None:
    s["step"] = step.value
    if ask:
        say(s, prefix + P.prompt_for(step, s))


def invalid(s: BotState, field: str, message: str, step: Step) -> None:
    """Re-ask politely. After repeated failures also offer the office number."""
    count = s["invalid"][field] = s["invalid"].get(field, 0) + 1
    text = f"{message} {P.prompt_for(step, s)}"
    if count >= MAX_INVALID_BEFORE_OFFICE_HINT:
        text += f" (If you'd prefer, you can call our office on {P.OFFICE_PHONE}.)"
    say(s, text)


def reset_journey(s: BotState) -> None:
    """New journey: forget lane-specific answers, keep who the visitor is and how to reach them."""
    s.update(category=None, budget_gbp=None, postcode=None, matches=[], outcome=None,
             qualified=False, invalid={}, property_id=None, property_candidates=[])


def end_chat(s: BotState, kind: str) -> None:
    s.update(outcome=kind, ended=True, step=Step.ENDED.value)
    say(s, P.ending(kind))


def absorb_volunteered(s: BotState, u: Understanding, c: Ctx) -> None:
    """Visitors often answer ahead ('new-build in SW3, £2m'). Keep what they offered so we
    don't re-ask, but every flowchart check still runs in order. Off in strict mode."""
    if c.strict_flow:
        return
    text = s["user_text"]
    if u.intent and not s.get("intent"):
        s["intent"] = u.intent
    if u.category and not s.get("category"):
        s["category"] = u.category
        s["intent"] = s.get("intent") or "buy"   # only buyers choose new-build/resale
    if _BUDGET_WORDS.search(text) and s.get("budget_gbp") is None:
        s["budget_gbp"] = parse_budget(text).value
    if s.get("postcode") is None:
        s["postcode"] = extract_postcode(text)
    if c.lane_c and c.props is not None and s.get("intent") != "sell" and not s.get("property_candidates"):
        named = c.props.mentioned(text)       # 'Is Apartment 3, Aldermoor Place still available?'
        if named:
            s["intent"] = "buy"
            s["property_candidates"] = [{"id": h.property_id, "name": h.name, "address": h.address}
                                        for h in named]


# ---- routing between journeys ---------------------------------------------
def after_identification(s: BotState, c: Ctx, *, prefix: str = "") -> None:
    """We know the visitor's name (and maybe more): pick the next flowchart question."""
    if s.get("intent") == "sell":
        return enter_d(s, c, prefix)
    if s.get("intent") == "buy":
        if s.get("property_candidates"):          # visitor already named a property: lane C
            return resolve_candidates(s, c)
        if s.get("category"):
            return enter_lane(s, c, prefix)
        return goto(s, Step.SPECIFIC_PROPERTY, prefix=prefix)
    goto(s, Step.BUY_OR_SELL, prefix=prefix)


# ---- lane C: buyer with a specific property (bonus) ----------------------------
def resolve_candidates(s: BotState, c: Ctx) -> None:
    cands = s["property_candidates"]
    if len(cands) == 1:
        return found_property(s, c, cands[0]["id"])
    goto(s, Step.C_PICK)


def _with_article(noun: str) -> str:
    return f"{'an' if noun[:1] in 'aeiou' else 'a'} {noun}"


def found_property(s: BotState, c: Ctx, property_id: str) -> None:
    p = c.props.get(property_id)
    s.update(property_id=property_id, property_candidates=[], category=p["listing_category"], invalid={})
    if p["status"] == "available":                      # C3 yes -> C4 share details -> E
        s.update(outcome="QUALIFIED", qualified=True,
                 matches=[{"property_id": property_id, "name": p["property_name"],
                           "bedrooms": p["bedrooms"], "price": p["price_gbp"]}])
        say(s, f"{p['property_name']} is {_with_article(p['property_type'].lower())} with {p['bedrooms']} bedrooms, "
               f"priced at {format_gbp(p['price_gbp'])}.")
        return enter_e(s, c)
    goto(s, Step.C5_BUDGET)                              # sold / under offer -> C5


def _lookup(s: BotState, c: Ctx) -> bool:
    """Try to resolve the visitor's text to a property. True if the step was handled."""
    hits = c.props.find(s["user_text"])
    if len(hits) == 1:
        found_property(s, c, hits[0].property_id)
        return True
    if len(hits) > 1:
        s["property_candidates"] = [{"id": h.property_id, "name": h.name, "address": h.address} for h in hits]
        goto(s, Step.C_PICK)
        return True
    return False


def h_c1_property(s: BotState, u: Understanding, c: Ctx) -> None:
    if not _lookup(s, c):
        goto(s, Step.C1_RETRY)                           # not found: ask once more


def h_c1_retry(s: BotState, u: Understanding, c: Ctx) -> None:
    if not _lookup(s, c):                                # still not found: general buyer
        goto(s, Step.NEW_OR_RESALE, prefix="No problem, I'll help you find a home another way. ")


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _ordinal_choice(text: str, count: int) -> int | None:
    """'2', '2nd one', 'the second', 'number 3', 'last' -> zero-based index into the list shown.
    Only for short replies, so a sentence that merely contains a number is not misread."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words or len(words) > 4:
        return None
    if "last" in words:
        return count - 1
    for w in words:
        m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", w)
        n = int(m.group(1)) if m else _ORDINALS.get(w)
        if n is not None and 1 <= n <= count:
            return n - 1
    return None


def _pick_by_distinguishing_words(text: str, cands: list[dict]) -> str | None:
    """'the one on Sheen Road': choose the candidate whose address holds the most query words
    that the other candidates lack. None if that is not a unique best."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    tokens = {x["id"]: set(re.findall(r"[a-z0-9]+", f"{x['name']} {x['address']}".lower())) for x in cands}
    common = set.intersection(*tokens.values())
    scores = {pid: len(words & (toks - common)) for pid, toks in tokens.items()}
    best = max(scores.values())
    leaders = [pid for pid, sc in scores.items() if sc == best]
    return leaders[0] if best > 0 and len(leaders) == 1 else None


def h_c_pick(s: BotState, u: Understanding, c: Ctx) -> None:
    cands = s["property_candidates"]
    text = s["user_text"]
    chosen = None
    index = _ordinal_choice(text, len(cands))
    if index is not None:
        chosen = cands[index]["id"]
    else:
        chosen = _pick_by_distinguishing_words(text, cands)
    if chosen is None:
        return say(s, P.prompt_for(Step.C_PICK, s))
    found_property(s, c, chosen)


def h_c5_budget(s: BotState, u: Understanding, c: Ctx) -> None:
    b = parse_budget(s["user_text"])
    if b.ambiguous:
        return invalid(s, "budget", "Could you give me one figure for your budget, please?", Step.C5_BUDGET)
    if b.value is None:
        return invalid(s, "budget", "Sorry, I didn't catch a budget (for example £1.5m or 950k).", Step.C5_BUDGET)
    s.update(budget_gbp=b.value, invalid={})
    if s["category"] == "new_build":                     # C6: type comes from the listing itself
        return run_a2(s, c)                              # < £1M -> M1, else A3 (postcode)
    enter_b2(s, c)                                       # resale -> B2 (postcode), any budget


def enter_lane(s: BotState, c: Ctx, prefix: str = "") -> None:
    return enter_a1(s, c, prefix) if s["category"] == "new_build" else enter_b1(s, c, prefix)


# ---- lane A: new-build buyer -----------------------------------------------
def enter_a1(s: BotState, c: Ctx, prefix: str = "") -> None:
    if s.get("budget_gbp") is not None and not c.strict_flow:
        return run_a2(s, c)
    goto(s, Step.A1_BUDGET, prefix=prefix)


def run_a2(s: BotState, c: Ctx) -> None:
    """A2: budget >= £1M? (M1 before the postcode is even asked, as in the flowchart)."""
    if s["budget_gbp"] < MIN_NEW_BUILD_BUDGET:
        return end_chat(s, "M1")
    if s.get("postcode") and not c.strict_flow:
        return run_a4(s, c)
    goto(s, Step.A3_POSTCODE)


def run_a4(s: BotState, c: Ctx) -> None:
    """A4 listed at postcode? / A5 available within budget? / A6 show matches."""
    conclude(s, qualify_new_build(s["budget_gbp"], s["postcode"], c.listings), c)


def h_a1_budget(s: BotState, u: Understanding, c: Ctx) -> None:
    b = parse_budget(s["user_text"])
    if b.ambiguous:
        return invalid(s, "budget", "Could you give me one figure for your budget, please?", Step.A1_BUDGET)
    if b.value is None:
        return invalid(s, "budget", "Sorry, I didn't catch a budget (for example £1.5m or 950k).", Step.A1_BUDGET)
    s.update(budget_gbp=b.value, invalid={})
    run_a2(s, c)


def h_a3_postcode(s: BotState, u: Understanding, c: Ctx) -> None:
    pc = extract_postcode(s["user_text"])
    if pc is None:
        return invalid(s, "postcode", "Sorry, that doesn't look like a full UK postcode (for example SW3 3RB).",
                       Step.A3_POSTCODE)
    s.update(postcode=pc, invalid={})
    run_a4(s, c)


# ---- lane B: resale buyer --------------------------------------------------
def enter_b1(s: BotState, c: Ctx, prefix: str = "") -> None:
    if s.get("budget_gbp") is not None and not c.strict_flow:
        return enter_b2(s, c)
    goto(s, Step.B1_BUDGET, prefix=prefix)


def enter_b2(s: BotState, c: Ctx) -> None:
    if s.get("postcode") and not c.strict_flow:
        return run_b3(s, c)
    goto(s, Step.B2_POSTCODE)


def run_b3(s: BotState, c: Ctx) -> None:
    """B3 listed at postcode? / B4 any available? / B5 show matches. Budget never filters."""
    conclude(s, qualify_resale(s["postcode"], c.listings), c)


def h_b1_budget(s: BotState, u: Understanding, c: Ctx) -> None:
    b = parse_budget(s["user_text"])
    if b.ambiguous:
        return invalid(s, "budget", "Could you give me one figure for your budget, please?", Step.B1_BUDGET)
    if b.value is None:
        return invalid(s, "budget", "Sorry, I didn't catch a budget (for example £750k or £1.2m).", Step.B1_BUDGET)
    s.update(budget_gbp=b.value, invalid={})  # recorded for the sales team only
    enter_b2(s, c)


def h_b2_postcode(s: BotState, u: Understanding, c: Ctx) -> None:
    pc = extract_postcode(s["user_text"])
    if pc is None:
        return invalid(s, "postcode", "Sorry, that doesn't look like a full UK postcode (for example SW3 3RB).",
                       Step.B2_POSTCODE)
    s.update(postcode=pc, invalid={})
    run_b3(s, c)


# ---- lane D: seller --------------------------------------------------------
def enter_d(s: BotState, c: Ctx, prefix: str = "") -> None:
    if s.get("postcode") and not c.strict_flow:
        return run_d2(s, c)
    goto(s, Step.D1_POSTCODE, prefix=prefix)


def run_d2(s: BotState, c: Ctx) -> None:
    conclude(s, qualify_seller(s["postcode"], c.coverage), c)


def h_d1_postcode(s: BotState, u: Understanding, c: Ctx) -> None:
    pc = extract_postcode(s["user_text"])
    if pc is None:
        return invalid(s, "postcode", "Sorry, that doesn't look like a full UK postcode (for example SW3 3RB).",
                       Step.D1_POSTCODE)
    s.update(postcode=pc, invalid={})
    run_d2(s, c)


# ---- outcome ---------------------------------------------------------------
def conclude(s: BotState, out: Outcome, c: Ctx) -> None:
    """The ONLY way into lane E. A visitor who is not qualified gets a closing message and
    no contact details are ever requested."""
    if not out.qualified:
        return end_chat(s, out.kind)
    s.update(outcome="QUALIFIED", qualified=True, invalid={},
             matches=[{"property_id": m.property_id, "name": m.name, "bedrooms": m.bedrooms,
                       "price": m.price} for m in out.matches])
    if out.matches:
        say(s, P.format_matches(s["matches"]))
    enter_e(s, c)


# ---- lane E: contact details & close --------------------------------------
def enter_e(s: BotState, c: Ctx) -> None:
    assert s["qualified"], "lane E is only reachable by qualified leads"
    if s.get("phone") and s.get("email"):      # second qualified journey: details already held
        return finish_e(s, c)
    goto(s, Step.E2_EMAIL if s.get("phone") else Step.E1_PHONE)


def finish_e(s: BotState, c: Ctx) -> None:
    s["step"] = Step.E3_ANYTHING_ELSE.value
    say(s, P.CALL_BACK)
    c.on_lead(s)


def h_e1_phone(s: BotState, u: Understanding, c: Ctx) -> None:
    phone = validate_phone(s["user_text"])
    if phone is None:
        return invalid(s, "phone", "Sorry, that doesn't look like a valid phone number. UK numbers work as they "
                                   "are (e.g. 07911 123456); for a number from another country please start "
                                   "with + and the country code (e.g. +91 …).", Step.E1_PHONE)
    s.update(phone=phone, invalid={})
    goto(s, Step.E2_EMAIL)


def h_e2_email(s: BotState, u: Understanding, c: Ctx) -> None:
    email = validate_email_address(s["user_text"])
    if email is None:
        return invalid(s, "email", "Sorry, that doesn't look like a valid email address.", Step.E2_EMAIL)
    s.update(email=email, invalid={})
    finish_e(s, c)


def h_e3_anything_else(s: BotState, u: Understanding, c: Ctx) -> None:
    if u.yes_no is False and not u.intent:
        s.update(ended=True, step=Step.ENDED.value)
        return say(s, P.GOODBYE)
    if u.yes_no is True or u.intent:
        reset_journey(s)
        s["intent"] = None
        absorb_volunteered(s, u, c)
        if s.get("intent"):
            return after_identification(s, c)
        return goto(s, Step.BUY_OR_SELL, prefix=P.REASSIST + " ")  # details kept
    say(s, P.prompt_for(Step.E3_ANYTHING_ELSE, s))


# ---- start of conversation ---------------------------------------------------
def h_home(s: BotState, u: Understanding, c: Ctx) -> None:
    absorb_volunteered(s, u, c)
    goto(s, Step.ASK_NAME)


def h_ask_name(s: BotState, u: Understanding, c: Ctx) -> None:
    if not u.name:
        return say(s, f"Sorry, I didn't catch that. {P.ASK_NAME}")
    s["name"] = u.name
    absorb_volunteered(s, u, c)
    after_identification(s, c, prefix=f"Thanks, {u.name}. ")


def h_buy_or_sell(s: BotState, u: Understanding, c: Ctx) -> None:
    if not u.intent:
        return say(s, P.prompt_for(Step.BUY_OR_SELL, s))
    s["intent"] = u.intent
    absorb_volunteered(s, u, c)
    after_identification(s, c)


def h_specific_property(s: BotState, u: Understanding, c: Ctx) -> None:
    """'Yes' enters lane C (skipping C1 if they already named the property). With lane C
    disabled, 'yes' is treated as 'no', exactly as the flowchart footnote says."""
    absorb_volunteered(s, u, c)
    if s.get("property_candidates"):
        return resolve_candidates(s, c)
    if s.get("category"):
        return enter_lane(s, c)
    if u.yes_no is None:
        return say(s, P.prompt_for(Step.SPECIFIC_PROPERTY, s))
    if u.yes_no and c.lane_c and c.props is not None:
        return goto(s, Step.C1_PROPERTY)
    goto(s, Step.NEW_OR_RESALE)


def h_new_or_resale(s: BotState, u: Understanding, c: Ctx) -> None:
    if not u.category:
        return say(s, P.prompt_for(Step.NEW_OR_RESALE, s))
    s["category"] = u.category
    enter_lane(s, c)


HANDLERS: dict[Step, Callable[[BotState, Understanding, Ctx], None]] = {
    Step.HOME: h_home,
    Step.ASK_NAME: h_ask_name,
    Step.BUY_OR_SELL: h_buy_or_sell,
    Step.SPECIFIC_PROPERTY: h_specific_property,
    Step.NEW_OR_RESALE: h_new_or_resale,
    Step.A1_BUDGET: h_a1_budget,
    Step.A3_POSTCODE: h_a3_postcode,
    Step.B1_BUDGET: h_b1_budget,
    Step.B2_POSTCODE: h_b2_postcode,
    Step.D1_POSTCODE: h_d1_postcode,
    Step.C1_PROPERTY: h_c1_property,
    Step.C1_RETRY: h_c1_retry,
    Step.C_PICK: h_c_pick,
    Step.C5_BUDGET: h_c5_budget,
    Step.E1_PHONE: h_e1_phone,
    Step.E2_EMAIL: h_e2_email,
    Step.E3_ANYTHING_ELSE: h_e3_anything_else,
}


def apply_switch(s: BotState, u: Understanding, c: Ctx) -> None:
    """Change of mind mid-journey: restart at the new journey's entry, keep name & contacts."""
    reset_journey(s)
    s["intent"] = u.switch_intent
    after_identification(s, c, prefix="No problem. ")


def answer_side_question(s: BotState, u: Understanding, c: Ctx) -> None:
    """Answer, then re-ask the SAME step. `step` is never changed here."""
    say(s, c.side.answer(u.side, s["user_text"], s))
    if s["step"] == Step.HOME.value:
        # First message was a question: answer it, then the flowchart's next box, 'Ask name'.
        # 'How long does it take to SELL a home?' does not say the visitor wants to sell (they may be
        # a buyer researching), so intent, postcode and budget are NOT taken from a question. Questions
        # that open a journey ('any homes to BUY at SW3 5WU?', 'tell me about <property>') are routed
        # to the flow before they get here.
        return goto(s, Step.ASK_NAME)       # nothing is kept from a question: it states no intent
    say(s, P.prompt_for(s["step"], s))
