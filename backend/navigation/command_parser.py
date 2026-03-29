"""
Voice / text command parsing for navigation and direction queries.
"""

from __future__ import annotations

import re
from typing import Any


_NAV_PREFIXES = (
    r"navigate\s+to",
    r"navigation\s+to",
    r"take\s+me\s+to",
    r"directions\s+to",
    r"drive\s+me\s+to",
    r"walk\s+to",
    r"go\s+to",
    r"head\s+to",
    r"lead\s+me\s+to",
    r"guide\s+me\s+to",
)

_NAV_PATTERN = re.compile(
    rf"^\s*(?:({'|'.join(_NAV_PREFIXES)}))\s+(.+)$",
    re.IGNORECASE,
)

_QUERY_NEXT_PHRASES = (
    "what's next",
    "whats next",
    "what is next",
    "next direction",
    "next step",
    "where do i go",
    "what do i do",
    "what should i do now",
    "what should i do",
    "for my navigation",
    "for navigation",
    "which way",
    "where should i go",
    "what direction",
)

_CONFIRM_YES = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok|okay|confirm|start|go ahead|do it|please)\s*[!.]?\s*$",
    re.IGNORECASE,
)

_CONFIRM_NO = re.compile(
    r"^\s*(no|nope|nah|cancel|stop|never mind|nevermind|forget it|abort)\s*[!.]?\s*$",
    re.IGNORECASE,
)


def parse_command(text: str) -> dict[str, Any]:
    """
    Detect navigation intent, confirmation, or next-direction queries.

    Returns a dict with at least ``intent`` (str). Optional keys: ``destination``.
    """
    raw = (text or "").strip()
    low = raw.lower()

    if not low:
        return {"intent": "UNKNOWN", "raw": raw}

    for phrase in _QUERY_NEXT_PHRASES:
        if phrase in low:
            return {"intent": "QUERY_NEXT_DIRECTION", "raw": raw}

    if _CONFIRM_YES.match(raw):
        return {"intent": "CONFIRM_YES", "raw": raw}

    if _CONFIRM_NO.match(raw):
        return {"intent": "CONFIRM_NO", "raw": raw}

    m = _NAV_PATTERN.match(raw)
    if m:
        dest = (m.group(2) or "").strip()
        dest = re.sub(r"[.!?]+$", "", dest).strip()
        if dest:
            return {"intent": "NAVIGATE", "destination": dest, "raw": raw}

    # Transcripts often include filler before the command, e.g. "I want to navigate to Yale".
    for key in (
        "navigate to ",
        "navigation to ",
        "directions to ",
        "walk to ",
        "take me to ",
        "head to ",
        "guide me to ",
        "lead me to ",
    ):
        idx = low.find(key)
        if idx >= 0:
            dest = raw[idx + len(key) :].strip()
            dest = re.sub(r"[.!?]+$", "", dest).strip()
            if dest:
                return {"intent": "NAVIGATE", "destination": dest, "raw": raw}

    return {"intent": "UNKNOWN", "raw": raw}
