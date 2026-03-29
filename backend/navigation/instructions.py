"""
Turn verbose Google Directions HTML instructions into short spoken cues.
"""

from __future__ import annotations

import html as html_lib
import re


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(raw: str) -> str:
    text = _TAG_RE.sub("", raw or "")
    return html_lib.unescape(text).strip()


def simplify_instruction(clean_text: str) -> str:
    """
    Map verbose instruction text to a short command.

    Examples:
        "Turn right onto Chapel Street" -> "Turn right"
        "Head north" -> "Go straight"
        "Continue straight for 200 meters" -> "Continue straight"
    """
    t = re.sub(r"\s+", " ", (clean_text or "").strip())
    low = t.lower()

    m_turn = re.search(
        r"\b(turn\s+(?:slightly\s+)?(?:sharp\s+)?(?:left|right)|"
        r"keep\s+(?:left|right)|"
        r"bear\s+(?:left|right))\b",
        low,
    )
    if m_turn:
        phrase = m_turn.group(1)
        if phrase.startswith("turn"):
            parts = phrase.split()
            if len(parts) >= 2:
                return f"Turn {parts[-1]}"
        if phrase.startswith("keep"):
            return f"Keep {phrase.split()[-1]}"
        if phrase.startswith("bear"):
            return f"Bear {phrase.split()[-1]}"

    if re.search(r"\bhead\s+(north|south|east|west)\b", low):
        return "Go straight"

    if re.search(r"\bcontinue\s+straight\b", low) or re.search(r"\bstraight\s+for\b", low):
        return "Continue straight"

    if re.search(r"\bmerge\b", low):
        if "left" in low:
            return "Merge left"
        if "right" in low:
            return "Merge right"
        return "Merge"

    if "u-turn" in low or "uturn" in low.replace(" ", "") or "make a u" in low:
        return "Make a U-turn"

    if low.startswith("destination") or "arrive" in low or "your destination" in low:
        return "Arrive at destination"

    # Fallback: first clause before comma / "onto" / "on "
    cut = re.split(r",|\s+onto\s+|\s+on\s+", t, maxsplit=1)[0].strip()
    if len(cut) < 3:
        return t
    return cut[:1].upper() + cut[1:] if cut else t
