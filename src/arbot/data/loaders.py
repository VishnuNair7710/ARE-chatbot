"""CSV loaders. Postcodes are normalised once here so lookups are exact matches."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from arbot.domain.postcode import normalise_postcode

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
STATUSES = {"available", "under_offer", "sold"}
CATEGORIES = {"new_build", "resale"}


def load_listings(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "properties_for_buyers.csv", encoding="utf-8-sig")
    df["postcode"] = df["postcode"].map(lambda p: normalise_postcode(p) or p)
    return df


def load_seller_coverage(path: Path | None = None) -> frozenset[str]:
    df = pd.read_csv(path or DATA_DIR / "seller_coverage.csv", encoding="utf-8-sig")
    return frozenset(p for p in df["postcode"].map(normalise_postcode) if p)


def load_faq(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "faq.csv", encoding="utf-8-sig")
