"""UK postcode normalisation and validation (full postcodes only)."""
from __future__ import annotations

import re

_FULL = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\d[A-Z]{2}$")
_IN_TEXT = re.compile(r"(?<![A-Za-z\d])([A-Za-z]{1,2}\d[A-Za-z\d]?)\s*(\d[A-Za-z]{2})(?![A-Za-z\d])")


def normalise_postcode(text: str) -> str | None:
    """'gu5  7aq' -> 'GU5 7AQ'; None if not a valid full postcode."""
    compact = re.sub(r"\s+", "", text or "").upper()
    if not _FULL.match(compact):
        return None
    return f"{compact[:-3]} {compact[-3:]}"


def extract_postcode(text: str) -> str | None:
    """Find a full postcode inside a sentence, in any case or spacing."""
    m = _IN_TEXT.search(text or "")
    return normalise_postcode(m.group(1) + m.group(2)) if m else None
