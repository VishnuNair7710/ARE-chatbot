import pytest

from arbot.llm.resilience import LLMUnavailable, ResilientRunner, is_retryable


class HttpError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"http {status}")
        self.status_code = status
        if retry_after is not None:
            self.response = type("R", (), {"headers": {"retry-after": str(retry_after)}})()


def make_runner(**kw):
    sleeps = []
    clock = {"t": 0.0}
    r = ResilientRunner(sleep=sleeps.append, clock=lambda: clock["t"], **kw)
    return r, sleeps, clock


def flaky(failures, exc):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return "ok"
    return fn, calls


def test_retries_then_succeeds():
    r, sleeps, _ = make_runner(max_retries=3)
    fn, calls = flaky(2, HttpError(429))
    assert r.call(fn) == "ok" and calls["n"] == 3 and len(sleeps) == 2


def test_non_retryable_error_is_raised_immediately():
    r, sleeps, _ = make_runner()
    fn, calls = flaky(5, HttpError(401))
    with pytest.raises(HttpError):
        r.call(fn)
    assert calls["n"] == 1 and sleeps == []


def test_exhausted_retries_raise_llm_unavailable():
    r, _, _ = make_runner(max_retries=2)
    fn, calls = flaky(99, HttpError(503))
    with pytest.raises(LLMUnavailable):
        r.call(fn)
    assert calls["n"] == 3


def test_retry_after_header_is_honoured():
    r, sleeps, _ = make_runner(max_retries=1, max_delay=8.0)
    fn, _ = flaky(1, HttpError(429, retry_after=2))
    r.call(fn)
    assert sleeps == [2.0]


def test_circuit_opens_then_half_opens_after_cooldown():
    r, _, clock = make_runner(max_retries=0, failure_threshold=2, cooldown_s=30)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            r.call(flaky(99, HttpError(500))[0])
    fn, calls = flaky(0, None)
    with pytest.raises(LLMUnavailable, match="circuit"):
        r.call(fn)
    assert calls["n"] == 0                      # not even attempted while open
    clock["t"] = 31
    assert r.call(fn) == "ok"                   # probe succeeds, circuit closes
    assert not r.circuit_open


@pytest.mark.parametrize("exc,expected", [
    (HttpError(429), True), (HttpError(500), True), (HttpError(400), False), (HttpError(401), False),
    (TimeoutError(), True), (ValueError("x"), False),
])
def test_retryable_classification(exc, expected):
    assert is_retryable(exc) is expected
