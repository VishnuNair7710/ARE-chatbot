"""Streamlit chat UI. It only talks to the FastAPI backend (API_URL), never to the bot directly.

Run: streamlit run ui/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 60
FALLBACK = "Sorry, I'm having trouble right now. Please call our office on 020 7946 0100."


def api_post(path: str, payload: dict | None = None) -> dict:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT_S)
    except requests.RequestException:
        return {"error": FALLBACK}
    if r.status_code == 404:
        return {"error": "expired"}
    if r.status_code == 429:
        return {"error": "You're sending messages too quickly. Please wait a moment."}
    if not r.ok:
        return {"error": FALLBACK}
    return r.json()


def start_new_chat() -> None:
    data = api_post("/sessions")
    st.session_state.update(
        session_id=data.get("session_id"), ended=False, debug={},
        messages=[{"role": "assistant", "content": data.get("reply") or data.get("error", FALLBACK)}],
    )


st.set_page_config(page_title="Ashcombe & Rowe Estates", page_icon="🏠")
st.title("Ashcombe & Rowe Estates")

if "messages" not in st.session_state:
    start_new_chat()

with st.sidebar:
    if st.button("New chat"):
        start_new_chat()
        st.rerun()
    show_debug = st.toggle("Show flow state (demo)")
    if show_debug:
        st.json(st.session_state.get("debug", {}))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if st.session_state.get("ended"):
    st.info("This chat has ended. Use **New chat** in the sidebar to start again.")
elif prompt := st.chat_input("Type your message…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    data = api_post("/chat", {"session_id": st.session_state.session_id, "message": prompt})
    if data.get("error") == "expired":
        reply = "Your session expired, so I've started a new chat. How may I help you?"
        start_new_chat()
        st.session_state.messages.append({"role": "assistant", "content": reply})
    elif "error" in data:
        st.session_state.messages.append({"role": "assistant", "content": data["error"]})
    else:
        st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
        st.session_state.ended = data["ended"]
        st.session_state.debug = {k: data[k] for k in ("step", "qualified", "ended")}
    st.rerun()
