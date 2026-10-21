from arbot.services.ratelimit import RateLimiter, SessionRegistry


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_sliding_window_allows_limit_then_blocks_then_recovers():
    clock = Clock()
    rl = RateLimiter(2, 10.0, clock)
    assert rl.check("ip") == 0 and rl.check("ip") == 0
    wait = rl.check("ip")
    assert 0 < wait <= 10
    clock.t = 10.5
    assert rl.check("ip") == 0


def test_keys_are_independent():
    rl = RateLimiter(1, 10.0, Clock())
    assert rl.check("a") == 0 and rl.check("b") == 0 and rl.check("a") > 0


def test_session_registry_expiry_unknown_ids_and_capacity():
    clock = Clock()
    reg = SessionRegistry(ttl_s=100, max_sessions=2, clock=clock)
    assert reg.register("s1") and reg.register("s2")
    assert not reg.register("s3")                 # full
    assert reg.touch("s1") and not reg.touch("nope")
    clock.t = 150                                  # idle sessions expire
    assert not reg.touch("s2")
    assert reg.register("s3")                      # room again after sweep
