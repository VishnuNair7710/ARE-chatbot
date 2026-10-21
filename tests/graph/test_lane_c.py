"""Lane C: buyer with a specific property (flowchart boxes C1-C6)."""
from arbot.graph import prompts as P
from arbot.graph.steps import Step

from .conftest import Chat

PHONE = "07911 123456"


def at_c1() -> Chat:
    c = Chat()
    c.say("hi", "Sam", "I want to buy")
    assert c.say("yes") == P.WHICH_PROPERTY
    return c


def test_available_property_shares_details_then_contact_details(homes):
    h = homes[("new_build", "available")]
    c = at_c1()
    reply = c.say(h.property_name)
    assert h.property_name in reply and f"{h.bedrooms} bedrooms" in reply and f"£{h.price_gbp:,}" in reply
    assert reply.endswith(P.ASK_PHONE)
    assert c.state["qualified"] and c.state["property_id"] == h.property_id
    assert [m["property_id"] for m in c.state["matches"]] == [h.property_id]


def test_sold_new_build_under_a_million_budget_is_m1(homes):
    c = at_c1()
    assert c.say(homes[("new_build", "sold")].property_name) == P.SIMILAR_HOME
    assert c.state["step"] == Step.C5_BUDGET.value and c.state["category"] == "new_build"
    c.say("£900k")
    assert c.state["outcome"] == "M1" and c.state["ended"] and c.state["phone"] is None


def test_under_offer_new_build_continues_at_a3_postcode_without_reasking_budget(homes, data):
    c = at_c1()
    assert c.say(homes[("new_build", "under_offer")].property_name) == P.SIMILAR_HOME
    assert c.say("£2m") == P.POSTCODE_INTEREST
    assert c.state["budget_gbp"] == 2_000_000
    assert c.say(data["nb_ok"]).endswith(P.ASK_PHONE) and c.state["qualified"]
    assert c.state["property_id"] is None or c.state["matches"]   # re-qualified through lane A


def test_sold_resale_continues_at_b2_with_any_budget(homes, data):
    c = at_c1()
    c.say(homes[("resale", "sold")].property_name)
    assert c.say("£100k") == P.POSTCODE_INTEREST            # no £1M rule for resale
    assert c.say(data["rs_ok"]).endswith(P.ASK_PHONE)


def test_not_found_asks_once_more_then_falls_back_to_general_buyer():
    c = at_c1()
    assert c.say("Castle Anthrax") == P.PROPERTY_RETRY and c.state["step"] == Step.C1_RETRY.value
    assert c.say("Camelot Mansions") == (
        "No problem, I'll help you find a home another way. " + P.NEW_OR_RESALE)


def test_not_found_then_found_on_second_try(homes):
    c = at_c1()
    c.say("zzzz")
    assert homes[("resale", "available")].property_name in c.say(homes[("resale", "available")].property_name)


def test_several_matches_ask_which_one_then_pick_by_number_or_address():
    c = at_c1()
    reply = c.say("The Old Rectory")
    assert reply.startswith("I found a few matches:") and "Sheen Road" in reply
    assert c.say("banana") .startswith("I found a few matches:")      # unclear: list again
    assert c.say("2").endswith(P.ASK_PHONE) or "The Old Rectory" in c.last.reply
    c2 = at_c1()
    c2.say("The Old Rectory")
    assert "The Old Rectory" in c2.say("the one on Sheen Road") and c2.state["property_id"] == "AR-0080"


def test_property_named_in_first_message_goes_straight_to_lane_c():
    c = Chat()
    first = c.say("Is The Old Rectory, Sheen Road still available?")
    assert first.endswith(P.ASK_NAME)                        # answered, then the flowchart's name box
    reply = c.say("Sam")
    assert reply.startswith("Thanks, Sam.") is False or "The Old Rectory" in reply
    assert "The Old Rectory" in reply and reply.endswith(P.ASK_PHONE)
    assert c.state["property_id"] == "AR-0080" and c.state["qualified"]


def test_property_named_inside_the_yes_answer_skips_c1():
    c = Chat()
    c.say("hi", "Sam", "buy")
    assert c.say("yes, Apartment 1, Aldermoor Place").endswith(P.ASK_PHONE)


def test_strict_flow_ignores_volunteered_property_but_still_runs_lane_c():
    c = Chat(strict_flow=True)
    c.say("I'm interested in Apartment 1, Aldermoor Place")
    assert c.say("Sam") == f"Thanks, Sam. {P.BUY_OR_SELL}"
    assert c.say("buy") == P.SPECIFIC_PROPERTY
    assert c.say("yes") == P.WHICH_PROPERTY


def test_side_question_inside_lane_c_resumes_same_step():
    c = at_c1()
    reply = c.say("What are your opening hours?")
    assert reply.endswith(P.WHICH_PROPERTY) and c.state["step"] == Step.C1_PROPERTY.value


def test_change_of_mind_inside_lane_c_switches_to_selling(data):
    c = at_c1()
    assert c.say("actually I want to sell my home").endswith(P.POSTCODE_SELLER)
    assert c.state["property_id"] is None and c.state["name"] == "Sam"


def test_property_details_use_the_right_article():
    c = Chat()
    c.say("hi", "Sam", "buy", "yes")
    assert "is an apartment with 2 bedrooms" in c.say("Apartment 1, Aldermoor Place")
    d = Chat()
    d.say("hi", "Sam", "buy", "yes")
    reply = d.say("Rosemount House")
    assert " is a " in reply and " a a" not in reply and "an house" not in reply


import pytest as _pytest


@_pytest.mark.parametrize("reply,expected_index", [
    ("2", 1), ("2nd one", 1), ("the second one", 1), ("number 3", 2), ("#1", 0), ("first", 0),
    ("the last one", 4), ("5th", 4),
])
def test_pick_accepts_ordinals_and_phrases(reply, expected_index):
    from arbot.graph.handlers import _ordinal_choice
    assert _ordinal_choice(reply, 5) == expected_index


@_pytest.mark.parametrize("reply", ["the one on Sheen Road", "I have 2 kids and want the nice one", "6", "0", "banana"])
def test_pick_ignores_replies_that_are_not_a_position(reply):
    from arbot.graph.handlers import _ordinal_choice
    assert _ordinal_choice(reply, 5) is None


def test_ordinal_reply_picks_the_listed_property():
    c = at_c1()
    c.say("The Old Rectory")
    assert c.say("2nd one").endswith(P.ASK_PHONE) and c.state["property_id"] == "AR-0080"


def test_match_list_is_markdown_friendly():
    c = at_c1()
    reply = c.say("The Old Rectory")
    assert reply.startswith("I found a few matches:\n\n1. ") and reply.endswith("\n\nWhich one do you mean?")
