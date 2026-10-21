"""Per-turn observability that needs no external service.

* `track_turn()` collects LLM calls and token usage for everything a turn triggers, via a
  LangChain callback registered through a context variable (so the NLU and the side-question
  agent are measured without either knowing about it).
* `JsonlObserver` appends one JSON line per turn (latency, tokens, estimated cost, step
  transition, outcome, degraded flag). It never writes what the visitor typed.
* `langfuse_callbacks()` adds Langfuse tracing when its keys are present (LangSmith works
  natively through its own environment variables).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

log = logging.getLogger(__name__)


class TurnUsage(BaseCallbackHandler):
    """Counts model calls and tokens during one turn."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        self.calls += 1

    def on_llm_end(self, response, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.input_tokens += int(usage.get("input_tokens", 0))
                self.output_tokens += int(usage.get("output_tokens", 0))


_current: ContextVar[TurnUsage | None] = ContextVar("arbot_turn_usage", default=None)
register_configure_hook(_current, inheritable=True)


@contextmanager
def track_turn() -> Iterator[TurnUsage]:
    usage = TurnUsage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def estimate_cost_usd(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    """Prices are USD per million tokens; 0 for free tiers."""
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


def turn_record(*, thread_id: str, turn: int, step_before: str, state: dict, latency_ms: float,
                usage: TurnUsage, degraded: bool, input_chars: int,
                price_in: float = 0.0, price_out: float = 0.0, fallback_reason: str | None = None) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "thread_id": thread_id,
        "turn": turn,
        "step_before": step_before,
        "step_after": state.get("step"),
        "side": (state.get("understanding") or {}).get("side", "none"),
        "outcome": state.get("outcome"),
        "qualified": bool(state.get("qualified")),
        "ended": bool(state.get("ended")),
        "latency_ms": round(latency_ms, 1),
        "llm_calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "est_cost_usd": estimate_cost_usd(usage.input_tokens, usage.output_tokens, price_in, price_out),
        "degraded": degraded,
        "fallback_reason": fallback_reason if degraded else None,   # e.g. 'RateLimitError: ... TPD'
        "input_chars": input_chars,          # length only: the text itself is never logged
    }


class JsonlObserver:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, record: dict) -> None:
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:                       # logging must never break a chat
            log.warning("could not write turn log: %s", exc)


class ListObserver:
    """Keeps records in memory (tests and the eval harness)."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def __call__(self, record: dict) -> None:
        self.records.append(record)


def fan_out(*observers: Callable[[dict], None]) -> Callable[[dict], None]:
    def observe(record: dict) -> None:
        for obs in observers:
            obs(record)
    return observe


def langfuse_callbacks(env: dict[str, str] | None = None) -> list:
    """A Langfuse LangChain handler if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set,
    otherwise an empty list. Import errors are logged, never raised."""
    env = os.environ if env is None else env
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception as exc:  # noqa: BLE001
        log.warning("Langfuse keys are set but the handler could not load (%s); tracing off", exc)
        return []
