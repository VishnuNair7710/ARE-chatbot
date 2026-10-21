"""Read-only tools for the side-question agent. They return data and never touch the
conversation state, so a side question can never move the flow."""
from __future__ import annotations

from langchain_core.tools import tool

from arbot.data.faq_store import FaqStore
from arbot.data.property_index import PropertyIndex


def make_tools(faq: FaqStore, props: PropertyIndex) -> list:
    @tool
    def faq_search(query: str) -> list[dict]:
        """Search the company FAQ (company, branches, opening hours, fees, buying/selling
        process, contact details). Returns the best matching Q&A entries with similarity scores."""
        return [{"faq_id": h.faq_id, "question": h.question, "answer": h.answer,
                 "score": round(h.score, 3)} for h in faq.search(query, k=3)]

    @tool
    def find_property(query: str) -> list[dict]:
        """Find property listings by name or address. Returns candidate listings (id, name,
        address, status). Use get_property for the full details of one."""
        return [{"property_id": h.property_id, "name": h.name, "address": h.address,
                 "status": h.status} for h in props.find(query)]

    @tool
    def get_property(property_id: str) -> dict:
        """Get all details of one listing by property_id (type, bedrooms, bathrooms, price,
        status, description, branch...)."""
        return props.get(property_id) or {"error": "no such property"}

    return [faq_search, find_property, get_property]
