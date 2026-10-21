"""Hand-off of qualified leads to the sales team.

`LeadSink` is the seam: SQLite today, a CRM webhook tomorrow. A sink failure must never break
the visitor's chat, so `safe_save` swallows and logs errors (without the visitor's details)."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Protocol

from arbot.graph.state import BotState

log = logging.getLogger(__name__)


def lane_of(s: BotState) -> str:
    if s.get("intent") == "sell":
        return "D"
    if s.get("property_id"):
        return "C"
    return "A" if s.get("category") == "new_build" else "B"


def lead_record(s: BotState) -> dict:
    return {
        "lead_id": uuid.uuid4().hex,
        "thread_id": s.get("thread_id", ""),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": lane_of(s),
        "name": s.get("name"),
        "phone": s.get("phone"),
        "email": s.get("email"),
        "intent": s.get("intent"),
        "category": s.get("category"),
        "budget_gbp": s.get("budget_gbp"),
        "postcode": s.get("postcode"),
        "property_id": s.get("property_id"),
        "matches": s.get("matches", []),
    }


class LeadSink(Protocol):
    def save(self, record: dict) -> None: ...


class InMemoryLeadSink:
    def __init__(self):
        self.records: list[dict] = []

    def save(self, record: dict) -> None:
        self.records.append(record)


class SqliteLeadSink:
    def __init__(self, path: str = "leads.db"):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS leads (
                lead_id TEXT PRIMARY KEY, thread_id TEXT, created_at TEXT, lane TEXT, name TEXT,
                phone TEXT, email TEXT, intent TEXT, category TEXT, budget_gbp INTEGER,
                postcode TEXT, property_id TEXT, matches TEXT)""")

    def save(self, record: dict) -> None:
        row = {**record, "matches": json.dumps(record["matches"])}
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with sqlite3.connect(self.path) as db:
            db.execute(f"INSERT INTO leads ({cols}) VALUES ({marks})", row)


def make_on_lead(sink: LeadSink):
    def on_lead(s: BotState) -> None:
        try:
            sink.save(lead_record(s))
        except Exception as exc:  # noqa: BLE001 - the chat must survive a failing CRM
            log.error("lead hand-off failed (%s); lead for thread %s not saved",
                      type(exc).__name__, s.get("thread_id"))
    return on_lead
