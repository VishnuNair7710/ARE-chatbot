"""Small in-memory limiters for the API (single process; use Redis behind several workers)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable


class RateLimiter:
    """Sliding window: at most `limit` hits per `window_s` per key."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """0.0 if allowed (and counted); otherwise seconds until the next hit is allowed."""
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return max(0.001, self.window_s - (now - q[0]))
            q.append(now)
            if len(self._hits) > 50_000:                    # bound memory under key churn
                self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
            return 0.0


class SessionRegistry:
    """Server-issued session ids with idle expiry. Unknown ids are rejected, so a client
    cannot make up (or probe for) other visitors' sessions."""

    def __init__(self, ttl_s: float, max_sessions: int, clock: Callable[[], float] = time.monotonic):
        self.ttl_s, self.max_sessions, self._clock = ttl_s, max_sessions, clock
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        for sid in [k for k, t in self._seen.items() if now - t > self.ttl_s]:
            del self._seen[sid]

    def register(self, session_id: str) -> bool:
        now = self._clock()
        with self._lock:
            self._sweep(now)
            if len(self._seen) >= self.max_sessions:
                return False
            self._seen[session_id] = now
            return True

    def touch(self, session_id: str) -> bool:
        now = self._clock()
        with self._lock:
            self._sweep(now)
            if session_id not in self._seen:
                return False
            self._seen[session_id] = now
            return True
