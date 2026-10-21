from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py")


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.ok = status, body or {}, 200 <= status < 300

    def json(self):
        return self._body


def fake_post(routes):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url, json))
        handler = routes[url.rsplit("/", 1)[-1]]
        return handler(json) if callable(handler) else handler
    post.calls = calls
    return post


def sessions_ok():
    return Resp(200, {"session_id": "s" * 32, "reply": "How may I help you?", "step": "HOME",
                      "ended": False, "qualified": False})


def test_ui_shows_greeting_and_relays_messages_to_the_api(monkeypatch):
    import requests
    post = fake_post({"sessions": sessions_ok(),
                      "chat": lambda p: Resp(200, {"session_id": p["session_id"], "reply": "Can I get your name?",
                                                   "step": "ASK_NAME", "ended": False, "qualified": False})})
    monkeypatch.setattr(requests, "post", post)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.chat_message[0].markdown[0].value == "How may I help you?"
    at.chat_input[0].set_value("hello").run()
    texts = [m.markdown[0].value for m in at.chat_message]
    assert texts == ["How may I help you?", "hello", "Can I get your name?"]
    assert post.calls[-1][1] == {"session_id": "s" * 32, "message": "hello"}


def test_ui_degrades_gracefully_when_the_backend_is_down(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "020 7946 0100" in at.chat_message[0].markdown[0].value


def test_ui_stops_accepting_input_when_the_chat_has_ended(monkeypatch):
    import requests
    post = fake_post({"sessions": sessions_ok(),
                      "chat": Resp(200, {"session_id": "s" * 32, "reply": "Goodbye.", "step": "ENDED",
                                         "ended": True, "qualified": False})})
    monkeypatch.setattr(requests, "post", post)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("bye").run()
    assert len(at.chat_input) == 0 and any("chat has ended" in i.value for i in at.info)


def test_ui_starts_a_new_chat_when_the_session_expired(monkeypatch):
    import requests
    post = fake_post({"sessions": sessions_ok(), "chat": Resp(404)})
    monkeypatch.setattr(requests, "post", post)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello").run()
    assert "expired" in at.chat_message[-1].markdown[0].value
