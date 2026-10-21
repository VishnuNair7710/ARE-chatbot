"""Deterministic lead-qualification rules (no LLM). One function per journey.

Outcome kinds follow the flowchart: M1 (new-build under £1M), M2 (postcode not
catered for), M3 (nothing available), QUALIFIED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import pandas as pd

MIN_NEW_BUILD_BUDGET = 1_000_000
MAX_MATCHES = 5

Kind = Literal["M1", "M2", "M3", "QUALIFIED"]


@dataclass(frozen=True)
class Match:
    property_id: str
    name: str
    bedrooms: int
    price: int


@dataclass(frozen=True)
class Outcome:
    kind: Kind
    matches: tuple[Match, ...] = field(default_factory=tuple)

    @property
    def qualified(self) -> bool:
        return self.kind == "QUALIFIED"


def _at(listings: pd.DataFrame, category: str, postcode: str) -> pd.DataFrame:
    return listings[(listings["listing_category"] == category) & (listings["postcode"] == postcode)]


def _matches(df: pd.DataFrame) -> tuple[Match, ...]:
    # Cheapest first, then id: deterministic output (brief only says "up to 5").
    top = df.sort_values(["price_gbp", "property_id"]).head(MAX_MATCHES)
    return tuple(
        Match(r.property_id, r.property_name, int(r.bedrooms), int(r.price_gbp))
        for r in top.itertuples()
    )


def qualify_new_build(budget: int, postcode: str, listings: pd.DataFrame) -> Outcome:
    if budget < MIN_NEW_BUILD_BUDGET:
        return Outcome("M1")
    here = _at(listings, "new_build", postcode)  # any status: sold/under offer still "listed"
    if here.empty:
        return Outcome("M2")
    options = here[(here["status"] == "available") & (here["price_gbp"] <= budget)]
    if options.empty:
        return Outcome("M3")
    return Outcome("QUALIFIED", _matches(options))


def qualify_resale(postcode: str, listings: pd.DataFrame) -> Outcome:
    """Budget is recorded elsewhere for the sales team; it never filters resale."""
    here = _at(listings, "resale", postcode)
    if here.empty:
        return Outcome("M2")
    options = here[here["status"] == "available"]
    if options.empty:
        return Outcome("M3")
    return Outcome("QUALIFIED", _matches(options))


def qualify_seller(postcode: str, coverage: Iterable[str]) -> Outcome:
    return Outcome("QUALIFIED") if postcode in set(coverage) else Outcome("M2")
