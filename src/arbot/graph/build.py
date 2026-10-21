"""The LangGraph: one invocation per visitor message.

    begin -> understand -> route --+-> closed        (chat already ended)
                                   +-> side_question (answer, re-ask same step)
                                   +-> switch_intent (changed their mind)
                                   +-> <one node per flowchart step>
    every branch -> END

State lives in the checkpointer (keyed by thread_id), so the server stays stateless.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from arbot.graph import handlers as H
from arbot.graph import prompts as P
from arbot.domain.postcode import extract_postcode
from arbot.graph.guard import looks_like_injection, normalise_spaces
from arbot.graph.state import BotState
from arbot.graph.steps import Step
from arbot.nlu.schema import Understanding

MAX_INPUT_CHARS = 1000
HISTORY_WINDOW = 10


def build_graph(ctx: H.Ctx, checkpointer=None):
    def _is_flow_request(text: str, u: Understanding) -> bool:
        """'Tell me about Townhouse 1, Riverside Quarter' or 'any homes to buy at SW3 5WU?' name a
        property or a buying/selling need: they open the journey, which answers them properly."""
        if ctx.lane_c and ctx.props is not None and ctx.props.mentioned(text) and not ctx.strict_flow:
            return True
        return bool(extract_postcode(text) and u.intent and not ctx.strict_flow)

    g = StateGraph(BotState)

    def begin(state: BotState) -> dict:
        text = "".join(ch for ch in normalise_spaces(state.get("user_text", "")) if ch.isprintable() or ch in "\n\t")
        return {"user_text": text.strip()[:MAX_INPUT_CHARS], "replies": [],
                "turns": state.get("turns", 0) + 1}

    def understand(state: BotState) -> dict:
        if looks_like_injection(state["user_text"]):
            # Decline without involving any model: nothing to exploit, and it saves a call.
            u = Understanding(side="off_topic")
        else:
            u = ctx.nlu.understand(state["user_text"], state["step"], dict(state))
            if u.side in ("faq", "property") and state["step"] == "HOME" and _is_flow_request(state["user_text"], u):
                u.side = "none"   # first message starts a journey: lane C / slot-filling, not a side question
        history = (state.get("history", []) + [{"role": "user", "content": state["user_text"]}])
        return {"understanding": u.model_dump(), "history": history[-HISTORY_WINDOW:]}

    def run(fn):
        """Wrap a handler: copy state, mutate the copy, return it as the update."""
        def node(state: BotState) -> dict:
            s: BotState = {**state, "replies": list(state.get("replies", [])),
                           "invalid": dict(state.get("invalid", {}))}
            fn(s, Understanding(**state["understanding"]), ctx)
            history = s.get("history", []) + [{"role": "assistant", "content": "\n".join(s["replies"])}]
            s["history"] = history[-HISTORY_WINDOW:]
            return s
        return node

    def closed(state: BotState) -> dict:
        return {"replies": [P.CHAT_ENDED]}

    def route(state: BotState) -> str:
        if state.get("ended"):
            return "closed"
        u = state["understanding"]
        if u.get("side", "none") != "none":
            return "side_question"
        if u.get("switch_intent"):
            return "switch_intent"
        return state["step"]

    g.add_node("begin", begin)
    g.add_node("understand", understand)
    g.add_node("closed", closed)
    g.add_node("side_question", run(H.answer_side_question))
    g.add_node("switch_intent", run(H.apply_switch))
    for step, handler in H.HANDLERS.items():
        g.add_node(step.value, run(handler))

    g.add_edge(START, "begin")
    g.add_edge("begin", "understand")
    targets = ["closed", "side_question", "switch_intent"] + [st.value for st in H.HANDLERS]
    g.add_conditional_edges("understand", route, {t: t for t in targets})
    for t in targets:
        g.add_edge(t, END)
    return g.compile(checkpointer=checkpointer)
