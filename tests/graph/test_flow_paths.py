"""Walks the paths of flowchart.png end to end with the deterministic keyword NLU."""
import pytest

from arbot.graph import prompts as P
from arbot.graph.steps import Step

from .conftest import Chat

PHONE, EMAIL = "07911 123456", "sam@example.com"
GOOD_POSTCODE_NB_BUDGET = "2m"


def assert_ended_with(c: Chat, kind: str):
    assert c.state["ended"] and c.state["outcome"] == kind
    assert P.ENDINGS[kind] in c.last.reply
    assert P.OFFICE_PHONE in c.last.reply and c.last.reply.endswith("Goodbye.")
    # no contact details are ever collected from an unqualified lead
    assert c.state["phone"] is None and c.state["email"] is None and not c.state["qualified"]


# ---- Start --------------------------------------------------------------------
def test_opening_and_start_sequence(chat):
    assert ChatBotGreeting() == "How may I help you?"
    assert chat.say("hello") == P.ASK_NAME
    reply = chat.say("My name is Sam")
    assert reply == f"Thanks, Sam. {P.BUY_OR_SELL}" and chat.state["name"] == "Sam"
    assert chat.say("I want to buy") == P.SPECIFIC_PROPERTY
    assert chat.say("no") == P.NEW_OR_RESALE


def ChatBotGreeting():
    from arbot.graph.bot import ChatBot
    return ChatBot.opening()


def test_name_is_reasked_until_given(chat):
    chat.say("hello")
    assert chat.say("why do you need that??").endswith(P.ASK_NAME)
    assert chat.state["step"] == Step.ASK_NAME.value


# ---- Lane A: new-build buyer ------------------------------------------------------
def test_lane_a_qualified_then_contact_details_then_goodbye(chat, data):
    chat.to_buy_question().say("no")
    assert chat.say("a new-build please") == P.BUDGET
    assert chat.say("£2m") == P.POSTCODE_INTEREST
    reply = chat.say(data["nb_ok"].lower())
    assert "Here are some homes" in reply and reply.endswith(P.ASK_PHONE)
    assert chat.state["qualified"] and 1 <= len(chat.state["matches"]) <= 5
    assert all(m["price"] <= 2_000_000 for m in chat.state["matches"])
    assert chat.say(PHONE) == P.ASK_EMAIL
    assert chat.say(EMAIL) == P.CALL_BACK
    assert chat.say("no thanks") == P.GOODBYE and chat.state["ended"]
    assert chat.state["phone"] == "+447911123456" and chat.state["email"] == EMAIL


def test_lane_a_m1_budget_under_a_million_before_postcode_is_asked(chat):
    chat.to_buy_question().say("no", "new-build")
    assert chat.say("£950k") == P.ending("M1")
    assert_ended_with(chat, "M1")


def test_lane_a_m2_no_new_build_at_postcode(chat, data):
    chat.to_buy_question().say("no", "new-build", "£2m")
    chat.say(data["nb_absent"])
    assert_ended_with(chat, "M2")


def test_lane_a_m3_listed_but_none_available(chat, data):
    chat.to_buy_question().say("no", "new-build", "£5m")
    chat.say(data["nb_listed_none_available"])
    assert_ended_with(chat, "M3")


def test_lane_a_m3_available_but_over_budget(chat, data):
    chat.to_buy_question().say("no", "new-build", "1,000,000")  # passes the gate exactly
    chat.say(data["nb_ok"])
    # ok only if some listing is <= £1M, which the data never has for new-builds
    assert_ended_with(chat, "M3")


# ---- Lane B: resale buyer -----------------------------------------------------------
def test_lane_b_qualified_ignores_budget(chat, data):
    chat.to_buy_question().say("no", "resale")
    assert chat.say("£100k") == P.POSTCODE_INTEREST  # no minimum
    reply = chat.say(data["rs_ok"])
    assert "Here are some homes" in reply and reply.endswith(P.ASK_PHONE)
    assert chat.state["budget_gbp"] == 100_000 and chat.state["qualified"]


def test_lane_b_m2_and_m3(chat, data):
    chat.to_buy_question().say("no", "resale", "1m", data["rs_absent"])
    assert_ended_with(chat, "M2")
    other = Chat()
    other.to_buy_question().say("no", "resale", "1m", data["rs_listed_none_available"])
    assert_ended_with(other, "M3")


# ---- Lane D: seller -------------------------------------------------------------------
def test_lane_d_covered_postcode_qualifies(chat, data):
    chat.say("hi", "Sam")
    assert chat.say("I'd like to sell my house") == P.POSTCODE_SELLER
    assert chat.say(data["covered"].lower().replace(" ", "")) == P.ASK_PHONE
    assert chat.state["qualified"] and chat.state["matches"] == []


def test_lane_d_uncovered_postcode_is_m2(chat, data):
    chat.say("hi", "Sam", "sell")
    chat.say(data["uncovered"])
    assert_ended_with(chat, "M2")


def test_seller_check_uses_seller_file_not_listings(chat, data):
    # a postcode with buyer listings but absent from seller coverage must NOT qualify
    chat.say("hi", "Sam", "sell", data["uncovered"])
    assert chat.state["outcome"] == "M2"


# ---- Lane E ---------------------------------------------------------------------------
def _qualified_seller(data) -> Chat:
    c = Chat()
    c.say("hi", "Sam", "sell", data["covered"])
    return c


def test_invalid_phone_and_email_are_reasked(data):
    c = _qualified_seller(data)
    assert c.say("12345").startswith("Sorry") and c.state["step"] == Step.E1_PHONE.value
    assert c.say(PHONE) == P.ASK_EMAIL
    assert c.say("not-an-email").endswith(P.ASK_EMAIL) and c.state["step"] == Step.E2_EMAIL.value
    assert c.say(EMAIL) == P.CALL_BACK


def test_office_number_offered_after_repeated_invalid_input(data):
    c = _qualified_seller(data)
    c.say("abc", "xyz")
    assert P.OFFICE_PHONE in c.say("still no")


def test_reassist_keeps_details_and_skips_contact_questions(data):
    c = _qualified_seller(data)
    c.say(PHONE, EMAIL)
    assert c.say("yes please") == f"{P.REASSIST} {P.BUY_OR_SELL}"
    assert c.state["name"] == "Sam" and c.state["phone"] and c.state["email"]
    assert c.state["step"] == Step.BUY_OR_SELL.value and c.state["postcode"] is None
    c.say("sell")
    assert c.say(data["covered"]) == P.CALL_BACK  # already have phone + email
    assert c.say("no") == P.GOODBYE


# ---- Contact details are only for qualified leads ---------------------------------------
@pytest.mark.parametrize("setup", ["m1", "m2", "m3"])
def test_no_contact_question_after_a_closing_message(setup, data):
    c = Chat()
    c.to_buy_question().say("no", "new-build")
    if setup == "m1":
        c.say("500k")
    else:
        c.say("2m", data["nb_absent" if setup == "m2" else "nb_listed_none_available"])
    assert c.state["step"] == Step.ENDED.value
    reply = c.say(PHONE)  # a late phone number must not be stored
    assert reply == P.CHAT_ENDED and c.state["phone"] is None


# ---- Side questions: answer, then resume the SAME step -----------------------------------
STEP_SETUPS = {
    Step.ASK_NAME: ["hi"],
    Step.BUY_OR_SELL: ["hi", "Sam"],
    Step.SPECIFIC_PROPERTY: ["hi", "Sam", "buy"],
    Step.NEW_OR_RESALE: ["hi", "Sam", "buy", "no"],
    Step.A1_BUDGET: ["hi", "Sam", "buy", "no", "new-build"],
    Step.B1_BUDGET: ["hi", "Sam", "buy", "no", "resale"],
    Step.A3_POSTCODE: ["hi", "Sam", "buy", "no", "new-build", "2m"],
    Step.B2_POSTCODE: ["hi", "Sam", "buy", "no", "resale", "2m"],
    Step.D1_POSTCODE: ["hi", "Sam", "sell"],
}


@pytest.mark.parametrize("step", list(STEP_SETUPS))
def test_side_question_returns_to_same_step(step):
    c = Chat()
    c.say(*STEP_SETUPS[step])
    assert c.state["step"] == step.value
    before = {k: c.state[k] for k in ("name", "intent", "category", "budget_gbp", "postcode")}
    reply = c.say("What are your opening hours?")
    assert reply.endswith(P.prompt_for(step)) and c.state["step"] == step.value
    assert {k: c.state[k] for k in before} == before


def test_side_question_at_contact_steps(data):
    c = _qualified_seller(data)
    assert c.say("Where is your head office?").endswith(P.ASK_PHONE)
    c.say(PHONE)
    assert c.say("How quickly will someone call me?").endswith(P.ASK_EMAIL)


def test_off_topic_request_is_declined_and_flow_resumes(chat):
    chat.say("hi", "Sam")
    reply = chat.say("tell me a joke?")
    assert "only able to help" in reply and reply.endswith(P.BUY_OR_SELL)


# ---- Change of mind -----------------------------------------------------------------------
def test_buy_to_sell_switch_keeps_name_and_clears_lane_data(chat, data):
    chat.to_buy_question().say("no", "new-build", "2m")
    assert chat.state["step"] == Step.A3_POSTCODE.value
    reply = chat.say("actually I want to sell my home instead")
    assert reply.endswith(P.POSTCODE_SELLER)
    assert chat.state["name"] == "Sam" and chat.state["budget_gbp"] is None
    assert chat.state["intent"] == "sell"
    assert chat.say(data["covered"]) == P.ASK_PHONE


def test_sell_to_buy_switch(chat):
    chat.say("hi", "Sam", "sell")
    assert chat.say("sorry, I'd rather buy").endswith(P.SPECIFIC_PROPERTY)


# ---- Input handling ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [("£1.5m", 1_500_000), ("1,200,000", 1_200_000),
                                           ("1.2 million", 1_200_000), ("1.2m", 1_200_000)])
def test_budget_formats_accepted_in_lane_a(chat, text, expected):
    chat.to_buy_question().say("no", "new-build")
    assert chat.say(text) == P.POSTCODE_INTEREST and chat.state["budget_gbp"] == expected


def test_budget_unparseable_or_ambiguous_is_reasked(chat):
    chat.to_buy_question().say("no", "new-build")
    assert chat.say("quite a lot").endswith(P.BUDGET)
    assert chat.say("between 1m and 2m").endswith(P.BUDGET)
    assert chat.state["budget_gbp"] is None


def test_partial_postcode_is_reasked(chat):
    chat.say("hi", "Sam", "sell")
    assert chat.say("SW3").endswith(P.POSTCODE_SELLER) and chat.state["step"] == Step.D1_POSTCODE.value


# ---- Natural conversation vs strict flow ---------------------------------------------------------
def test_volunteered_answers_skip_questions_but_checks_still_run_in_order(data):
    c = Chat()
    c.say(f"I'd like a new-build in {data['nb_ok']} with a budget of £2m")
    reply = c.say("Sam")
    assert "Here are some homes" in reply and reply.endswith(P.ASK_PHONE)
    low = Chat()
    low.say(f"new-build in {data['nb_ok']}, budget £500k", "Sam")
    assert_ended_with(low, "M1")  # M1 gate still applies even though postcode was volunteered


def test_strict_flow_asks_every_question_verbatim(data):
    c = Chat(strict_flow=True)
    c.say(f"I'd like a new-build in {data['nb_ok']} with a budget of £2m")
    assert c.say("Sam") == f"Thanks, Sam. {P.BUY_OR_SELL}"
    assert c.say("buy") == P.SPECIFIC_PROPERTY
    assert c.say("no") == P.NEW_OR_RESALE
    assert c.say("new-build") == P.BUDGET
    assert c.say("2m") == P.POSTCODE_INTEREST


def test_specific_property_yes_is_treated_as_no_when_lane_c_is_disabled():
    c = Chat(lane_c=False)
    c.say("hi", "Sam", "buy")
    assert c.say("yes") == P.NEW_OR_RESALE


# ---- Robustness --------------------------------------------------------------------------------------
def test_threads_are_isolated(data):
    a, b = Chat(), Chat()
    a.say("hi", "Ann", "sell")
    b.say("hi", "Bob")
    assert a.state["name"] == "Ann" and b.state["name"] == "Bob"
    assert a.state["step"] == Step.D1_POSTCODE.value and b.state["step"] == Step.BUY_OR_SELL.value


def test_overlong_input_is_truncated(chat):
    chat.say("hi")
    chat.say("x" * 5000)
    assert len(chat.state["user_text"]) <= 1000
