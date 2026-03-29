"""
Simple confirmation state machine for navigation start.

IDLE --(NAVIGATE)--> CONFIRMING --(YES)--> NAVIGATING
                        | (NO)
                        v
                      IDLE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NavConfirmState(str, Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"
    NAVIGATING = "NAVIGATING"


@dataclass
class NavigationConfirmationFSM:
    state: NavConfirmState = field(default=NavConfirmState.IDLE)
    pending_destination: Optional[str] = None

    def reset(self) -> None:
        self.state = NavConfirmState.IDLE
        self.pending_destination = None

    def handle_parsed(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """
        Feed a result from :func:`parse_command`.

        Returns a dict describing what happened, e.g.
        ``{"event": "need_confirmation", "destination": "..."}``,
        ``{"event": "confirmed", "destination": "..."}``,
        ``{"event": "cancelled"}``, ``{"event": "ignored"}``.
        """
        intent = parsed.get("intent", "UNKNOWN")

        if intent == "NAVIGATE":
            dest = (parsed.get("destination") or "").strip()
            if not dest:
                return {"event": "ignored", "reason": "empty_destination"}
            self.state = NavConfirmState.CONFIRMING
            self.pending_destination = dest
            return {"event": "need_confirmation", "destination": dest}

        if intent == "CONFIRM_YES":
            if self.state != NavConfirmState.CONFIRMING:
                return {"event": "ignored", "reason": "not_confirming"}
            dest = self.pending_destination or ""
            self.state = NavConfirmState.NAVIGATING
            self.pending_destination = None
            return {"event": "confirmed", "destination": dest}

        if intent == "CONFIRM_NO":
            if self.state != NavConfirmState.CONFIRMING:
                return {"event": "ignored", "reason": "not_confirming"}
            self.pending_destination = None
            self.state = NavConfirmState.IDLE
            return {"event": "cancelled"}

        return {"event": "ignored", "reason": "intent_not_handled", "intent": intent}

    def navigation_ended(self) -> None:
        """Call when route is finished or user stops navigation."""
        self.state = NavConfirmState.IDLE
        self.pending_destination = None
