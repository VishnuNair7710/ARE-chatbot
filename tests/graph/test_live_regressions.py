"""Defects found only by running a real model (gpt-oss-20b on Groq). Each test pins one down
using a fake that behaves like the real model did."""
import pytest

from arbot.graph import prompts as P
from arbot.graph.steps import Step
from arbot.llm.resilience import ResilientRunner, is_retryable
from arbot.nlu.keyword import looks_like_question
from arbot.nlu.llm import LLMExtraction, LLMNLU
from arbot.nlu.schema import Understanding

from .conftest import Chat


class FixedChain:
    def __init__(self, **fields):
        self.result = LLMExtraction(**fields)

    def invoke(self, messages):
        return self.result


def llm_nlu(**fields):
    return LLMNLU(FixedChain(**fields), runner=ResilientRunner(max_retries=0, sleep=lambda s: None))


@pytest.mark.parametrize("step,text", [
    (Step.C1_PROPERTY, "Apartment 1, Aldermoor Place"),     # a property name typed as the answer
    (Step.C1_RETRY, "Castle Anthrax"),
    (Step.E2_EMAIL, "not-an-email"),                         # model called it an FAQ about email
    (Step.E1_PHONE, "12345"),
    (Step.A3_POSTCODE, "SW3"),
])
def test_a_model_cannot_turn_a_plain_answer_into_a_side_question(step, text):
    for label in ("faq", "property", "off_topic"):
        assert llm_nlu(side=label).understand(text, step.value, {}).side == "none"


@pytest.mark.parametrize("step,text", [
    (Step.C1_PROPERTY, "What are your opening hours?"),
    (Step.A1_BUDGET, "tell me about your fees"),
    (Step.E1_PHONE, "Can I speak to a human?"),
])
def test_real_questions_and_requests_still_count_as_side_questions(step, text):
    assert llm_nlu(side="faq").understand(text, step.value, {}).side == "faq"


def test_looks_like_question():
    for q in ["What time?", "how much", "Do you charge a fee", "tell me about Chelsea", "I'd like to know the fees",
              "can you tell me more"]:
        assert looks_like_question(q), q
    for a in ["Castle Anthrax", "not-an-email", "£2m", "sam@example.com", "Apartment 1, Aldermoor Place"]:
        assert not looks_like_question(a), a


def test_malformed_tool_call_is_retried():
    class ToolUseFailed(Exception):
        status_code = 400

        def __str__(self):
            return "Error code: 400 - {'code': 'tool_use_failed', 'failed_generation': 'get_property<|channel|>'}"

    assert is_retryable(ToolUseFailed())
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ToolUseFailed()
        return "ok"

    assert ResilientRunner(max_retries=2, sleep=lambda s: None).call(flaky) == "ok" and calls["n"] == 2


class SideLabellingNLU:
    """Behaves like the live model did: calls a first message about a listing a side question."""

    def understand(self, text, step, state):
        from arbot.nlu.keyword import KeywordNLU
        u = KeywordNLU().understand(text, step, state)
        return Understanding(**{**u.model_dump(), "side": "property"}) if step == "HOME" else u


def test_naming_a_property_in_the_first_message_is_not_answered_twice(homes):
    h = homes[("new_build", "available")]
    c = Chat(nlu=SideLabellingNLU())
    first = c.say(f"give me information on {h.property_name}")
    assert first == P.ASK_NAME                                   # no side answer, straight to the flowchart
    reply = c.say("Vish")
    assert h.property_name in reply and f"£{h.price_gbp:,}" in reply and reply.endswith(P.ASK_PHONE)
    assert reply.count(h.property_name) == 1


def test_postcode_availability_question_as_first_message_starts_the_buyer_journey(data):
    c = Chat(nlu=SideLabellingNLU())
    first = c.say(f"Are any properties available to buy on {data['nb_ok']} postal code?")
    assert first == P.ASK_NAME and "don't have that information" not in first
    assert c.state["postcode"] == data["nb_ok"] and c.state["intent"] == "buy"
    assert c.say("Sam") == f"Thanks, Sam. {P.SPECIFIC_PROPERTY}"
    c.say("no", "resale")
    assert c.state["step"] == Step.B1_BUDGET.value


def test_a_faq_question_that_mentions_a_postcode_but_no_intent_is_still_answered():
    c = Chat()
    reply = c.say("Which branch covers SW3 5WU?")
    assert reply.endswith(P.ASK_NAME) and reply != P.ASK_NAME     # answered (or "not in data"), then name


def test_model_gaps_are_filled_from_the_words_but_never_overridden():
    # the model said nothing about intent although the text plainly says 'buy'
    u = llm_nlu(side="property").understand("any properties to buy on SW3 5WU?", Step.HOME.value, {})
    assert u.intent == "buy"
    # ...but if the model did give an answer, the rules do not overrule it
    u = llm_nlu(intent="sell").understand("I want to buy and sell things, whatever", Step.BUY_OR_SELL.value, {})
    assert u.intent == "sell"


def test_model_fills_intent_so_a_postcode_question_opens_the_journey(data):
    # FixedChain(side="property") is exactly what the live model returned for this message
    c = Chat(nlu=LLMNLU(FixedChain(side="property")))
    first = c.say(f"Are any properties available to buy on {data['nb_ok']} postal code?")
    assert first == P.ASK_NAME and c.state["intent"] == "buy"


@pytest.mark.parametrize("text", ["Give me your background", "Tell me about your history", "Tell me about yourselves"])
def test_rule_based_mode_treats_tell_me_requests_as_questions(text):
    c = Chat()
    reply = c.say(text)
    assert reply.endswith(P.ASK_NAME) and "didn't catch" not in reply


@pytest.mark.parametrize("question", [
    "How long does it take to sell a home?",          # mentions selling but states no intent
    "What are the fees if I buy a new-build?",
    "Which branch covers SW3 5WU?",                    # a postcode inside a question is not their postcode
])
def test_a_first_message_question_keeps_nothing_and_the_buy_or_sell_question_is_still_asked(question):
    for nlu in (None, SideLabellingNLU()):
        c = Chat(nlu=nlu) if nlu else Chat()
        first = c.say(question)
        assert first.endswith(P.ASK_NAME)
        assert c.state["intent"] is None and c.state["postcode"] is None and c.state["category"] is None
        assert c.say("Vish") == f"Thanks, Vish. {P.BUY_OR_SELL}"


def test_a_statement_of_intent_in_the_first_message_is_still_kept():
    c = Chat()
    assert c.say("I want to sell my flat").endswith(P.ASK_NAME)
    assert c.say("Vish") == f"Thanks, Vish. {P.POSTCODE_SELLER}"      # statement: skips buy-or-sell, by design


def test_a_foreign_phone_number_is_explained_not_just_rejected(data):
    c = Chat()
    c.say("hi", "Vish", "sell", data["covered"])
    reply = c.say("8921478343")                                       # 10 digits, no country code: not a UK number
    assert "valid phone number" in reply and "+91" in reply and reply.endswith(P.ASK_PHONE)
    assert c.say("+91 89214 78343") == P.ASK_EMAIL                    # with the country code it is accepted
