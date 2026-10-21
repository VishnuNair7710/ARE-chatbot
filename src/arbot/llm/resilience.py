"""Retries with exponential backoff + jitter, Retry-After support and a circuit breaker.

Free-tier providers (Groq especially) rate-limit aggressively, so a 429 is a normal event,
not an exception. When retries are exhausted or the circuit is open we raise
`LLMUnavailable`, and callers fall back to deterministic behaviour (degraded mode).
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class LLMUnavailable(Exception):
    """The model could not be used right now (rate limit, outage, timeout, circuit open)."""


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    status = _status(exc)
    if status == 400 and "tool_use_failed" in str(exc):
        return True      # the model emitted a malformed tool call (seen live); a retry usually succeeds
    if status is not None:
        return status in (408, 409, 425, 429) or status >= 500
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "ratelimit", "unavailable", "overloaded"))


def retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return None
    return None


class ResilientRunner:
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 8.0,
                 failure_threshold: int = 4, cooldown_s: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.max_retries, self.base_delay, self.max_delay = max_retries, base_delay, max_delay
        self.failure_threshold, self.cooldown_s = failure_threshold, cooldown_s
        self._sleep, self._clock = sleep, clock
        self._failures = 0
        self._opened_at: float | None = None

    def reset(self) -> None:
        """Close the circuit and forget past failures (e.g. after waiting out a rate limit)."""
        self._failures, self._opened_at = 0, None

    @property
    def circuit_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self.cooldown_s:  # half-open: allow a probe
            self._opened_at = None
            self._failures = self.failure_threshold - 1
            return False
        return True

    def call(self, fn: Callable[[], T]) -> T:
        if self.circuit_open:
            raise LLMUnavailable("circuit open")
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                if not is_retryable(exc):
                    raise
                last = exc
                if attempt < self.max_retries:
                    wait = retry_after_seconds(exc)
                    if wait is None:
                        wait = min(self.max_delay, self.base_delay * 2 ** attempt)
                        wait *= random.uniform(0.5, 1.0)
                    self._sleep(min(wait, self.max_delay))
                continue
            self._failures, self._opened_at = 0, None
            return result
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
        raise LLMUnavailable(f"{type(last).__name__}: {last}") from last
