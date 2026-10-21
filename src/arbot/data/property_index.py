"""Property lookup by name/address (structured data -> exact id + fuzzy text match).

Not a vector search on purpose: names like 'Apartment 2, Aldermoor Place' and
'Apartment 3, Aldermoor Place' are near-identical to an embedding model but are different
homes. Token-based fuzzy matching keeps the distinguishing token ('2' vs '3') decisive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

_ID = re.compile(r"\bAR-\d{4}\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")

DETAIL_FIELDS = ("property_id", "property_name", "address", "town", "postcode", "listing_category",
                 "property_type", "bedrooms", "bathrooms", "price_gbp", "status", "description", "branch")


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", str(text).lower()).split())


@dataclass(frozen=True)
class PropertyHit:
    property_id: str
    name: str
    address: str
    status: str
    score: float


class PropertyIndex:
    def __init__(self, listings: pd.DataFrame, threshold: float = 85.0):
        self.df = listings.set_index("property_id", drop=False)
        self.threshold = threshold
        self._names = {pid: _norm(n) for pid, n in self.df["property_name"].items()}
        self._addrs = {pid: _norm(a) for pid, a in self.df["address"].items()}

    def get(self, property_id: str) -> dict | None:
        pid = property_id.strip().upper()
        if pid not in self.df.index:
            return None
        row = self.df.loc[pid]
        return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in DETAIL_FIELDS}

    def find(self, text: str, limit: int = 5) -> list[PropertyHit]:
        """Best candidates for `text`. One hit if the match is unambiguous, several if the
        visitor must choose (e.g. two homes called 'The Old Rectory')."""
        m = _ID.search(text)
        if m and (row := self.get(m.group(0))):
            return [self._hit(row["property_id"], 100.0)]
        q = _norm(text)
        if not q:
            return []
        scored = [(pid, max(fuzz.token_set_ratio(q, self._names[pid]),
                            fuzz.token_set_ratio(q, self._addrs[pid]))) for pid in self.df.index]
        scored = [(pid, s) for pid, s in scored if s >= self.threshold]
        if not scored:
            return []
        best = max(s for _, s in scored)
        perfect = [(pid, s) for pid, s in scored if s >= 100.0]
        if perfect:
            # Several homes can contain every query word (two 'The Old Rectory'); prefer the one
            # that explains the most query words (the address part, 'Sheen Road').
            cover = {pid: self._coverage(q, pid) for pid, _ in perfect}
            top = max(cover.values())
            keep = [(pid, s) for pid, s in perfect if cover[pid] >= top]
        else:
            keep = [(pid, s) for pid, s in scored if s >= best - 2]
        keep.sort(key=lambda t: (-t[1], t[0]))
        return [self._hit(pid, s) for pid, s in keep[:limit]]

    def mentioned(self, text: str) -> list[PropertyHit]:
        """Homes the visitor *named* in free text ('Is The Old Rectory, Sheen Road available?').
        Stricter than find(): the full multi-word name, address or id must appear, so a
        sentence like 'I want an apartment' is never mistaken for a property."""
        m = _ID.search(text)
        if m and (row := self.get(m.group(0))):
            return [self._hit(row["property_id"], 100.0)]
        padded = f" {_norm(text)} "
        hits = [pid for pid in self.df.index
                if (len(self._names[pid].split()) >= 2 and f" {self._names[pid]} " in padded)
                or f" {self._addrs[pid]} " in padded]
        by_address = [pid for pid in hits if f" {self._addrs[pid]} " in padded]
        return [self._hit(pid, 100.0) for pid in (by_address or hits)[:5]]

    def _coverage(self, q: str, pid: str) -> float:
        words = set(q.split())
        known = set(self._names[pid].split()) | set(self._addrs[pid].split())
        return len(words & known) / len(words)

    def _hit(self, pid: str, score: float) -> PropertyHit:
        r = self.df.loc[pid]
        return PropertyHit(pid, r["property_name"], r["address"], r["status"], float(score))


def describe_property(p: dict) -> str:
    """Deterministic fact sheet, used when the LLM is unavailable."""
    kind = "new-build" if p["listing_category"] == "new_build" else "resale"
    status = {"available": "available", "under_offer": "currently under offer", "sold": "sold"}[p["status"]]
    return (f"{p['property_name']} ({p['address']}, {p['postcode']}) is a {kind} "
            f"{p['property_type'].lower()} with {p['bedrooms']} bedrooms and {p['bathrooms']} bathrooms, "
            f"priced at £{p['price_gbp']:,}. It is {status}.")
