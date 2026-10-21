import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from arbot.data.loaders import load_listings, load_seller_coverage
from arbot.graph.bot import ChatBot
from arbot.services.observability import (JsonlObserver, ListObserver, estimate_cost_usd, fan_out,
                                          langfuse_callbacks, track_turn)


def fake_model(n=3):
    msgs = [AIMessage("ok", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
            for _ in range(n)]
    return GenericFakeChatModel(messages=iter(msgs))


def test_track_turn_counts_calls_and_tokens_from_any_model_invoked_inside():
    model = fake_model()
    with track_turn() as usage:
        model.invoke("hello")
        model.invoke("again")
    assert usage.calls == 2 and usage.input_tokens == 200 and usage.output_tokens == 40


def test_usage_is_not_shared_between_turns():
    model = fake_model()
    with track_turn() as first:
        model.invoke("a")
    with track_turn() as second:
        model.invoke("b")
    assert first.calls == 1 and second.calls == 1 and second.input_tokens == 100


def test_cost_estimate():
    assert estimate_cost_usd(1_000_000, 500_000, 0.5, 1.5) == 1.25
    assert estimate_cost_usd(1000, 1000, 0.0, 0.0) == 0.0


def test_turn_records_describe_the_turn_without_the_visitors_words(tmp_path):
    path = tmp_path / "turns.jsonl"
    mem = ListObserver()
    bot = ChatBot(listings=load_listings(), coverage=load_seller_coverage(),
                  observer=fan_out(JsonlObserver(str(path)), mem), price_per_mtok=(1.0, 2.0),
                  degraded_probe=lambda: True)
    tid = bot.new_thread_id()
    bot.send(tid, "hi")
    text = "My name is Priya Sharma, email priya@secret.example"
    bot.send(tid, text)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2 and lines == mem.records
    first, second = lines
    assert (first["step_before"], first["step_after"], first["turn"]) == ("HOME", "ASK_NAME", 1)
    assert second["step_before"] == "ASK_NAME" and second["thread_id"] == tid
    assert first["latency_ms"] >= 0 and first["degraded"] is True and first["llm_calls"] == 0
    raw = path.read_text()
    assert "priya" not in raw.lower() and "secret.example" not in raw      # no PII in the log
    assert second["input_chars"] == len(text)


def test_observer_failure_does_not_break_the_chat(tmp_path):
    bot = ChatBot(listings=load_listings(), coverage=load_seller_coverage(),
                  observer=JsonlObserver(str(tmp_path / "missing-dir" / "turns.jsonl")))
    assert bot.send(bot.new_thread_id(), "hi").reply == "Can I get your name?"


def test_langfuse_is_off_without_keys_and_never_raises_with_bad_install():
    assert langfuse_callbacks({}) == []
    assert langfuse_callbacks({"LANGFUSE_PUBLIC_KEY": "pk"}) == []             # secret missing
    assert isinstance(langfuse_callbacks({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}), list)


def test_turn_log_says_why_the_bot_fell_back():
    from arbot.nlu.llm import LLMExtraction, LLMNLU
    from arbot.llm.resilience import ResilientRunner

    class Down:
        def invoke(self, messages):
            raise TimeoutError("model unreachable")

    nlu = LLMNLU(Down(), runner=ResilientRunner(max_retries=0, sleep=lambda s: None))
    mem = ListObserver()
    bot = ChatBot(nlu, listings=load_listings(), coverage=load_seller_coverage(), observer=mem,
                  degraded_probe=lambda: nlu.degraded, reason_probe=lambda: nlu.last_error)
    tid = bot.new_thread_id()
    bot.send(tid, "hi")
    bot.send(tid, "I'm thinking of putting my flat on the market")      # needs the model -> falls back
    healthy, degraded_turn = mem.records[0], mem.records[-1]
    assert degraded_turn["degraded"] and "TimeoutError" in degraded_turn["fallback_reason"]
    assert healthy["fallback_reason"] is None or healthy["degraded"]
