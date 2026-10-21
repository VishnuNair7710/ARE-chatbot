import pytest
from fastapi.testclient import TestClient

from arbot.config import Settings
from arbot.data.loaders import load_listings, load_seller_coverage
from arbot.graph.bot import ChatBot
from arbot.services.api import create_app

LISTINGS, COVERAGE = load_listings(), load_seller_coverage()
COVERED = sorted(COVERAGE)[0]


def make_client(**settings_kw) -> TestClient:
    bot = ChatBot(listings=LISTINGS, coverage=COVERAGE)
    app = create_app(bot, Settings(api_rate_limit=1000, **settings_kw))
    return TestClient(app, raise_server_exceptions=False)


def start(client) -> str:
    r = client.post("/sessions")
    assert r.status_code == 200 and r.json()["reply"] == "How may I help you?"
    return r.json()["session_id"]


def test_health_is_cheap_and_does_not_build_the_bot():
    app = create_app(None, Settings(llm_api_key="", llm_provider="groq"))   # independent of any local .env
    r = TestClient(app).get("/health")
    body = r.json()
    assert body["status"] == "ok" and body["ready"] is False and body["llm"] is False
    assert body["degraded"] is False and body["last_fallback_reason"] is None
    assert body["llm_configured"] is False and body["model"] is None


def test_health_reports_a_configured_model_before_the_bot_has_loaded():
    app = create_app(None, Settings(llm_api_key="not-a-real-key", llm_model="some/model"))
    body = TestClient(app).get("/health").json()
    assert body["llm_configured"] is True and body["model"] == "some/model" and body["ready"] is False


def test_warm_start_builds_the_bot_when_the_server_starts():
    bot = ChatBot(listings=LISTINGS, coverage=COVERAGE)
    app = create_app(bot, Settings(), warm_start=True)
    with TestClient(app) as client:                      # entering the context runs the startup hook
        assert client.get("/health").json()["ready"] is True


def test_full_seller_conversation_over_http():
    client = make_client()
    sid = start(client)

    def say(text):
        r = client.post("/chat", json={"session_id": sid, "message": text})
        assert r.status_code == 200
        return r.json()

    say("hi"), say("Sam"), say("sell")
    body = say(COVERED)
    assert body["reply"] == "Can I get your phone number?" and body["qualified"] and body["step"] == "E1_PHONE"
    say("07911 123456"), say("sam@example.com")
    last = say("no thanks")
    assert last["ended"] and last["reply"] == "Thank you for chatting with us. Goodbye."


def test_closing_message_ends_the_chat_for_unqualified_visitors():
    client = make_client()
    sid = start(client)
    for text in ["hi", "Sam", "buy", "no", "new-build"]:
        client.post("/chat", json={"session_id": sid, "message": text})
    body = client.post("/chat", json={"session_id": sid, "message": "500k"}).json()
    assert body["ended"] and not body["qualified"] and "020 7946 0100" in body["reply"]


def test_sessions_are_isolated():
    client = make_client()
    a, b = start(client), start(client)
    assert a != b
    client.post("/chat", json={"session_id": a, "message": "hi"})
    client.post("/chat", json={"session_id": a, "message": "Ann"})
    r = client.post("/chat", json={"session_id": b, "message": "hi"}).json()
    assert r["step"] == "ASK_NAME"


def test_unknown_session_is_404_and_cannot_be_invented():
    client = make_client()
    r = client.post("/chat", json={"session_id": "a" * 32, "message": "hi"})
    assert r.status_code == 404


@pytest.mark.parametrize("payload", [
    {"session_id": "a" * 32, "message": ""},
    {"session_id": "a" * 32, "message": "x" * 1001},
    {"message": "hi"},
    {"session_id": "short", "message": "hi"},
    {},
])
def test_invalid_requests_are_rejected_with_422(payload):
    assert make_client().post("/chat", json=payload).status_code == 422


def test_rate_limit_returns_429_with_retry_after():
    bot = ChatBot(listings=LISTINGS, coverage=COVERAGE)
    client = TestClient(create_app(bot, Settings(api_rate_limit=3, api_rate_window_s=60)))
    codes = [client.post("/sessions").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = client.post("/sessions")
    assert int(r.headers["Retry-After"]) >= 1


def test_session_capacity_limit_returns_503():
    client = make_client(max_sessions=1)
    assert client.post("/sessions").status_code == 200
    assert client.post("/sessions").status_code == 503


def test_internal_errors_do_not_leak_details():
    class Boom(ChatBot):
        def send(self, thread_id, text):
            raise RuntimeError("secret internal detail: db password hunter2")

    app = create_app(Boom(listings=LISTINGS, coverage=COVERAGE), Settings(api_rate_limit=1000))
    client = TestClient(app, raise_server_exceptions=False)
    sid = client.post("/sessions").json()["session_id"]
    r = client.post("/chat", json={"session_id": sid, "message": "hi"})
    assert r.status_code == 500 and "hunter2" not in r.text and "020 7946 0100" in r.text
