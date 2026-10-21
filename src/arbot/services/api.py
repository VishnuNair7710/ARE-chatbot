"""FastAPI backend: POST /sessions starts a chat, POST /chat sends a message.

Run: uvicorn arbot.services.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from arbot.config import Settings
from arbot.graph.bot import ChatBot
from arbot.services.ratelimit import RateLimiter, SessionRegistry

log = logging.getLogger(__name__)
MAX_MESSAGE_CHARS = 1000


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    step: str
    ended: bool
    qualified: bool


def create_app(bot: ChatBot | None = None, settings: Settings | None = None, warm_start: bool = False) -> FastAPI:
    """`warm_start=True` builds the bot (models, index) when the server starts, so the first visitor
    does not wait and /health reports the real mode immediately. Off by default for tests."""
    s = settings or Settings()
    holder: dict = {"bot": bot}

    def get_bot() -> ChatBot:
        if holder["bot"] is None:                  # built lazily so importing the module is cheap
            from arbot.app import build_bot
            holder["bot"] = build_bot(s)
        return holder["bot"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warm_start:
            await run_in_threadpool(get_bot)
            b = holder["bot"]
            log.warning("chatbot ready: %s (model=%s)",
                        "LLM" if hasattr(b.ctx.nlu, "chain") else "RULE-BASED, no LLM configured",
                        s.llm_model if s.llm_enabled else "none")
        yield

    app = FastAPI(title="Ashcombe & Rowe chatbot", version="0.1.0", lifespan=lifespan)
    limiter = RateLimiter(s.api_rate_limit, s.api_rate_window_s)
    sessions = SessionRegistry(s.session_ttl_s, s.max_sessions)

    def rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client)
        if wait:
            raise HTTPException(429, "Too many requests, please slow down.",
                                headers={"Retry-After": str(int(wait) + 1)})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")           # details stay in the server log only
        return JSONResponse(status_code=500, content={"detail": "Something went wrong. "
                                                      "Please call our office on 020 7946 0100."})

    @app.get("/health")
    def health() -> dict:
        b = holder["bot"]                          # liveness only: never triggers a model/index load
        return {"status": "ok", "ready": b is not None,
                "llm_configured": s.llm_enabled,   # a key (or Ollama) is set; known even before the bot loads
                "model": s.llm_model if s.llm_enabled else None,
                "llm": bool(b is not None and hasattr(b.ctx.nlu, "chain")),
                "degraded": bool(b is not None and b.degraded_probe()),
                "last_fallback_reason": (b.reason_probe() if b is not None and b.degraded_probe() else None)}

    @app.post("/sessions", response_model=ChatResponse)
    def start_session(request: Request) -> ChatResponse:
        rate_limit(request)
        sid = ChatBot.new_thread_id()
        if not sessions.register(sid):
            raise HTTPException(503, "The service is busy, please try again shortly.")
        return ChatResponse(session_id=sid, reply=ChatBot.opening(), step="HOME", ended=False, qualified=False)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, request: Request) -> ChatResponse:
        rate_limit(request)
        if not sessions.touch(req.session_id):
            raise HTTPException(404, "Unknown or expired session. Start a new chat.")
        result = get_bot().send(req.session_id, req.message)
        return ChatResponse(session_id=req.session_id, reply=result.reply, step=result.state["step"],
                            ended=result.ended, qualified=bool(result.state.get("qualified")))

    return app


app = create_app(warm_start=True)
