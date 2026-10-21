"""The real side-question stack (FAQ RAG + property tools, no LLM) wired into the flow."""
import pytest

from arbot.app import build_bot
from arbot.config import Settings
from arbot.data.hash_embedder import HashEmbedder
from arbot.graph import prompts as P
from arbot.graph.steps import Step

from .conftest import Chat


@pytest.fixture(scope="module")
def bot():
    return build_bot(Settings(llm_api_key="", chroma_dir=""), embedder=HashEmbedder())


def chat_with(bot) -> Chat:
    c = Chat.__new__(Chat)
    c.bot, c.tid, c.last = bot, bot.new_thread_id(), None
    return c


def test_faq_question_mid_flow_is_answered_then_same_step_resumes(bot):
    c = chat_with(bot)
    c.say("hi", "Sam", "buy", "no", "new-build")
    reply = c.say("What is your head office address?")
    assert "22 Halden Mews" in reply and reply.endswith(P.BUDGET)
    assert c.state["step"] == Step.A1_BUDGET.value
    assert c.say("£2m") == P.POSTCODE_INTEREST        # flow carries on unharmed


def test_property_question_mid_flow_uses_listing_data(bot):
    c = chat_with(bot)
    c.say("hi", "Sam")
    reply = c.say("How many bedrooms does Apartment 3, Aldermoor Place have?")
    assert "3 bedrooms" in reply and "£2,075,000" in reply and reply.endswith(P.BUY_OR_SELL)


def test_ambiguous_property_asks_which_one_and_keeps_step(bot):
    c = chat_with(bot)
    c.say("hi", "Sam")
    reply = c.say("What is the price of The Old Rectory?")
    assert "several matching listings" in reply and c.state["step"] == Step.BUY_OR_SELL.value


def test_unanswerable_question_gives_office_number_never_a_guess(bot):
    c = chat_with(bot)
    c.say("hi", "Sam")
    reply = c.say("What is the airspeed velocity of an unladen swallow?")
    assert P.OFFICE_PHONE in reply and reply.endswith(P.BUY_OR_SELL)
