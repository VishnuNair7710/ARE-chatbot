"""Public entry point: a thin wrapper that owns the graph and the per-thread memory."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver

from arbot.agents.side_question import SideAnswerer, StubSideAnswerer
from arbot.data.loaders import load_listings, load_seller_coverage
from arbot.data.property_index import PropertyIndex
from arbot.graph import prompts as P
from arbot.graph.build import build_graph
from arbot.graph.handlers import Ctx
from arbot.graph.state import BotState, initial_state
from arbot.nlu.keyword import KeywordNLU
from arbot.nlu.schema import NLU
from arbot.services.observability import track_turn, turn_record

TURN_LIMIT_REPLY = f"We've reached the limit for this chat. Please call our office on {P.OFFICE_PHONE}."


@dataclass
class TurnResult:
    reply: str
    state: BotState

    @property
    def ended(self) -> bool:
        return bool(self.state.get("ended"))


class ChatBot:
    def __init__(self, nlu: NLU | None = None, side: SideAnswerer | None = None, *,
                 listings=None, coverage=None, strict_flow: bool = False, lane_c: bool = True,
                 on_lead=None, checkpointer=None, max_turns: int = 100,
                 observer: Callable[[dict], None] | None = None, callbacks: list | None = None,
                 degraded_probe: Callable[[], bool] | None = None,
                 reason_probe: Callable[[], str | None] | None = None,
                 price_per_mtok: tuple[float, float] = (0.0, 0.0)):
        listings = load_listings() if listings is None else listings
        self.max_turns = max_turns
        self.observer, self.callbacks = observer, list(callbacks or [])
        self.degraded_probe, self.price = degraded_probe or (lambda: False), price_per_mtok
        self.reason_probe = reason_probe or (lambda: None)
        self.ctx = Ctx(
            listings=listings,
            coverage=load_seller_coverage() if coverage is None else coverage,
            nlu=nlu or KeywordNLU(), side=side or StubSideAnswerer(),
            props=PropertyIndex(listings), lane_c=lane_c, strict_flow=strict_flow,
            **({"on_lead": on_lead} if on_lead else {}),
        )
        self.graph = build_graph(self.ctx, checkpointer or MemorySaver())

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def opening() -> str:
        return P.GREETING

    def has_thread(self, thread_id: str) -> bool:
        return bool(self.graph.get_state({"configurable": {"thread_id": thread_id}}).values)

    def send(self, thread_id: str, text: str) -> TurnResult:
        config = {"configurable": {"thread_id": thread_id}, "callbacks": self.callbacks,
                  "metadata": {"langfuse_session_id": thread_id}}
        current = self.graph.get_state(config).values
        if current and current.get("turns", 0) >= self.max_turns:   # abuse / runaway-loop cap
            return TurnResult(TURN_LIMIT_REPLY, current)
        update: dict = {"user_text": text}
        if not current:
            update = {**initial_state(), "user_text": text, "thread_id": thread_id}
        step_before = (current or {}).get("step", "HOME")
        started = time.perf_counter()
        with track_turn() as usage:
            final = self.graph.invoke(update, config)
        if self.observer:
            self.observer(turn_record(
                thread_id=thread_id, turn=final.get("turns", 0), step_before=step_before, state=final,
                latency_ms=(time.perf_counter() - started) * 1000, usage=usage,
                degraded=self.degraded_probe(), input_chars=len(text),
                price_in=self.price[0], price_out=self.price[1], fallback_reason=self.reason_probe()))
        return TurnResult("\n\n".join(final["replies"]), final)
