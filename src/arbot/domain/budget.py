"""Budget parsing: '£1.5m', '1,200,000', '1.2 million', '950k' -> int pounds."""
from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {
    "k": 1_000, "thousand": 1_000, "grand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "mil": 1_000_000, "million": 1_000_000,
}
_TOKEN = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|grand|mil|mn|k|m)?(?![\w])",
    re.IGNORECASE,
)
_RANGE = re.compile(r"\d\s*(?:-|–|—|to)\s*\d", re.IGNORECASE)
_WORDS = (
    (re.compile(r"\bhalf an?\s+million\b", re.I), 500_000),
)


_UNIT_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())
         if w != "_"}
_SCALES = {"thousand": 1_000, "million": 1_000_000}
_NUM = "|".join([*_UNIT_WORDS, *_TENS, "hundred", *_SCALES])
_START = "|".join([*_UNIT_WORDS, *_TENS, "hundred"])   # a span never starts at a bare 'million' ('1.2 million')
_SPAN = re.compile(
    rf"\b(?:(?:{_START})|(?:a|an)(?=\s+(?:hundred|thousand|million)))(?:[\s-]+(?:{_NUM}|and|a|an|point|half))*\b",
    re.IGNORECASE)


def _plain_words(tokens: list[str]) -> int | None:
    """'nine hundred and fifty thousand' -> 950000"""
    total = current = 0
    seen = False
    prev = ""
    for i, t in enumerate(tokens):
        if t == "and" and prev not in ("hundred", "thousand", "million"):
            return None        # 'one and two million' is a range, not 'nine hundred and fifty'
        prev = t
        if t in _UNIT_WORDS or t in _TENS:
            current += _UNIT_WORDS.get(t, _TENS.get(t, 0)); seen = True
        elif t == "hundred":
            current = (current or 1) * 100; seen = True
        elif t in _SCALES:
            total += (current or 1) * _SCALES[t]; current = 0; seen = True
        elif t not in ("and", "a", "an"):
            return None
    return total + current if seen else None


def _eval_words(tokens: list[str]) -> int | None:
    text = " ".join(tokens)
    if "point" in tokens:                                    # 'one point five million'
        i = tokens.index("point")
        left, right = tokens[:i], tokens[i + 1:]
        scale = _SCALES[right.pop()] if right and right[-1] in _SCALES else 1
        digits = "".join(str(_UNIT_WORDS[t]) for t in right if _UNIT_WORDS.get(t, 10) < 10)
        base = _plain_words(left)
        return round(float(f"{base}.{digits}") * scale) if base is not None and digits else None
    m = re.fullmatch(r"(?:(.+?) )?(thousand|million) and a half", text)      # 'a million and a half'
    if m:
        lead = m.group(1)
        base = 1 if lead in (None, "a", "an") else _plain_words(lead.split())
        return round((base + 0.5) * _SCALES[m.group(2)]) if base is not None else None
    m = re.fullmatch(r"(.+?) and a half (thousand|million)", text)           # 'two and a half million'
    if m:
        base = _plain_words(m.group(1).split())
        return round((base + 0.5) * _SCALES[m.group(2)]) if base is not None else None
    return _plain_words(tokens)


def numbers_from_words(text: str) -> str:
    """Replace spelled-out numbers with digits: 'one point five million' -> '1500000'."""
    def repl(m: re.Match) -> str:
        tokens = [t for t in re.split(r"[\s-]+", m.group(0).lower()) if t]
        while (tokens and tokens[-1] in ("and", "a", "an", "half", "point")
               and tokens[-3:] != ["and", "a", "half"]):                      # dangling filler
            tokens.pop()
        value = _eval_words(tokens) if tokens else None
        return f" {value} " if value is not None else m.group(0)
    return _SPAN.sub(repl, text)


@dataclass(frozen=True)
class BudgetParse:
    value: int | None
    ambiguous: bool = False  # several different figures (e.g. a range): ask which one


def parse_budget(text: str) -> BudgetParse:
    """Return the single budget figure in `text`, or flag it as ambiguous/absent.

    A bare number below 1,000 with no unit (e.g. '3') is ignored, because it is
    probably not money ('3 bedrooms'), and guessing would break the £1M rule.
    """
    cleaned = re.sub(r"£|gbp", " ", text, flags=re.IGNORECASE)
    if _RANGE.search(cleaned):  # '1-2m': the unit is shared, so a per-token parse would miss it
        return BudgetParse(None, ambiguous=True)
    found: set[int] = set()
    for pattern, amount in _WORDS:
        if pattern.search(cleaned):
            found.add(amount)
            cleaned = pattern.sub(" ", cleaned)
    cleaned = numbers_from_words(cleaned)
    for number, unit in _TOKEN.findall(cleaned):
        try:
            value = float(number.replace(",", ""))
        except ValueError:
            continue
        if unit:
            value *= _UNITS[unit.lower()]
        elif value < 1_000:
            continue
        amount = round(value)
        if amount > 0:
            found.add(amount)
    if len(found) == 1:
        return BudgetParse(next(iter(found)))
    return BudgetParse(None, ambiguous=len(found) > 1)


def format_gbp(amount: int) -> str:
    return f"£{amount:,}"
