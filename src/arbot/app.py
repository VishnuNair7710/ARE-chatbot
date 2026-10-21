"""Composition root: builds a ChatBot from environment settings."""
from __future__ import annotations

import logging
import sqlite3

from arbot.agents.side_question import AgentSideAnswerer
from arbot.config import Settings
from arbot.data.faq_store import FaqStore
from arbot.data.hash_embedder import HashEmbedder
from arbot.data.loaders import load_faq, load_listings, load_seller_coverage
from arbot.data.property_index import PropertyIndex
from arbot.graph.bot import ChatBot
from arbot.graph.guard import OutputGuard
from arbot.llm.factory import build_chat_model, build_embedder, structured_method
from arbot.net import use_system_certs
from arbot.llm.resilience import ResilientRunner
from arbot.nlu.keyword import KeywordNLU
from arbot.nlu.llm import LLMExtraction, LLMNLU
from arbot.services.leads import SqliteLeadSink, make_on_lead
from arbot.services.observability import JsonlObserver, langfuse_callbacks

log = logging.getLogger(__name__)


def build_embedder_with_fallback(s: Settings):
    """Neural embeddings if they load; otherwise the lexical fallback (logged, not silent)."""
    embedder = build_embedder(s)
    try:
        embedder.embed_query("warm-up")
        return embedder
    except Exception as exc:  # noqa: BLE001 - e.g. model download blocked
        log.warning("embedding model unavailable (%s: %s); using lexical fallback", type(exc).__name__, exc)
        return HashEmbedder()


def build_checkpointer(s: Settings):
    """In-memory sessions by default; SQLite when CHECKPOINT_DB is set (survives restarts)."""
    if not s.checkpoint_db:
        return None
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(s.checkpoint_db, check_same_thread=False))


def build_bot(settings: Settings | None = None, *, embedder=None, model=None, **bot_kwargs) -> ChatBot:
    s = settings or Settings()
    if s.use_system_certs:
        use_system_certs()                      # before any network use (LLM, model download)
    listings, coverage = load_listings(), load_seller_coverage()
    embedder = embedder or build_embedder_with_fallback(s)
    faq_df = load_faq()
    faq = FaqStore(faq_df, embedder, path=s.chroma_dir or None)
    props = PropertyIndex(listings)
    if model is None and s.llm_enabled:
        model = build_chat_model(s)
    runner = ResilientRunner(max_retries=s.llm_max_retries)
    if model is not None:
        nlu = LLMNLU(model.with_structured_output(LLMExtraction, method=structured_method(s)), runner=runner)
    else:
        log.warning("no LLM configured: running in degraded (rule-based) mode")
        nlu = KeywordNLU()
    fallback_embedder = embedder.__class__ is HashEmbedder        # has its own similarity scale
    side = AgentSideAnswerer(
        faq, props, model, runner, output_guard=OutputGuard(list(faq_df["answer"])),
        direct_threshold=None if fallback_embedder else s.faq_direct_threshold,
        min_threshold=None if fallback_embedder else s.faq_min_threshold,
    )
    if "on_lead" not in bot_kwargs and s.lead_db_path:
        bot_kwargs["on_lead"] = make_on_lead(SqliteLeadSink(s.lead_db_path))
    if "observer" not in bot_kwargs and s.turn_log_path:
        bot_kwargs["observer"] = JsonlObserver(s.turn_log_path)
    if "callbacks" not in bot_kwargs:
        bot_kwargs["callbacks"] = langfuse_callbacks()
    if "checkpointer" not in bot_kwargs:
        bot_kwargs["checkpointer"] = build_checkpointer(s)
    bot_kwargs.setdefault("reason_probe", lambda: getattr(nlu, "last_error", None) or getattr(side, "last_error", None))
    bot_kwargs.setdefault("degraded_probe", lambda: bool(getattr(nlu, "degraded", model is None))
                          or bool(getattr(side, "degraded", False)))
    return ChatBot(nlu, side, listings=listings, coverage=coverage, strict_flow=s.strict_flow,
                   lane_c=s.enable_lane_c, max_turns=s.max_turns,
                   price_per_mtok=(s.llm_price_in_per_mtok, s.llm_price_out_per_mtok), **bot_kwargs)
