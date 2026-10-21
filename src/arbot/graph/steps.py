"""Every point where the bot waits for the visitor. Names mirror flowchart.png boxes.

The flowchart's decision diamonds (A2, A4, A5, B3, B4, D2) are not steps: they run
inside the handler of the question that precedes them.
"""
from __future__ import annotations

from enum import Enum


class Step(str, Enum):
    HOME = "HOME"                            # greeted; waiting for the first message
    ASK_NAME = "ASK_NAME"
    BUY_OR_SELL = "BUY_OR_SELL"
    SPECIFIC_PROPERTY = "SPECIFIC_PROPERTY"
    NEW_OR_RESALE = "NEW_OR_RESALE"
    A1_BUDGET = "A1_BUDGET"
    A3_POSTCODE = "A3_POSTCODE"
    B1_BUDGET = "B1_BUDGET"
    B2_POSTCODE = "B2_POSTCODE"
    C1_PROPERTY = "C1_PROPERTY"              # lane C: which property?
    C1_RETRY = "C1_RETRY"                    # not found: ask once more
    C_PICK = "C_PICK"                        # several matches: which one?
    C5_BUDGET = "C5_BUDGET"                  # sold / under offer: find a similar home
    D1_POSTCODE = "D1_POSTCODE"
    E1_PHONE = "E1_PHONE"
    E2_EMAIL = "E2_EMAIL"
    E3_ANYTHING_ELSE = "E3_ANYTHING_ELSE"
    ENDED = "ENDED"
