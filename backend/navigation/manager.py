"""
NavigationManager: proximity-triggered spoken instructions and off-route rerouting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from backend.navigation.geo import cross_track_distance_meters, distance
from backend.navigation.maps_routes import Step, geocode_location, get_walking_route


class SpokenStage(str, Enum):
    NONE = "none"
    EARLY = "early"
    IMMEDIATE = "immediate"


EarlyTriggerM = 30.0
ImmediateTriggerM = 8.0
OffRouteThresholdM = 50.0
CompletionDistanceM = 15.0
RerouteCooldownSec = 12.0


def navigation_message(text: str) -> dict[str, str]:
    """Part 8 envelope for downstream TTS / priority mixing."""
    return {"text": text, "priority": "navigation"}


@dataclass
class NavigationManager:
    steps: list[Step] = field(default_factory=list)
    current_step_index: int = 0
    last_spoken_stage: SpokenStage = SpokenStage.NONE

    _dest_label: str = ""
    _dest_lat: Optional[float] = None
    _dest_lon: Optional[float] = None
    _session: Any = field(default=None, repr=False)
    _last_reroute_mono: float = field(default=0.0, repr=False)
    _completed: bool = field(default=False, repr=False)
    _last_lat: Optional[float] = field(default=None, init=False, repr=False)
    _last_lon: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._session is None:
            import requests

            self._session = requests.Session()

    def start_navigation(
        self,
        destination: str,
        current_location: tuple[float, float],
    ) -> None:
        """
        Geocode destination, fetch walking route from current GPS fix, reset state.
        """
        lat0, lon0 = current_location
        dlat, dlon = geocode_location(destination, session=self._session)
        self._dest_label = destination.strip()
        self._dest_lat = float(dlat)
        self._dest_lon = float(dlon)
        self.steps = get_walking_route(lat0, lon0, dlat, dlon, session=self._session)
        self.current_step_index = 0
        self.last_spoken_stage = SpokenStage.NONE
        self._completed = False
        self._last_reroute_mono = 0.0
        self._last_lat = None
        self._last_lon = None

    def mark_intro_announced(self) -> None:
        """
        After speaking the first step out loud at route start, set stage so we do not
        repeat the same \"ahead\" cue before the proximity \"now\" cue.
        """
        self.last_spoken_stage = SpokenStage.EARLY

    def load_route_from_steps(
        self,
        steps: list[Step],
        *,
        destination_label: str = "",
        dest_lat: Optional[float] = None,
        dest_lon: Optional[float] = None,
    ) -> None:
        """Test helper: bypass HTTP and inject steps directly."""
        self.steps = list(steps)
        self._dest_label = destination_label
        self._dest_lat = None if dest_lat is None else float(dest_lat)
        self._dest_lon = None if dest_lon is None else float(dest_lon)
        self.current_step_index = 0
        self.last_spoken_stage = SpokenStage.NONE
        self._completed = False
        self._last_reroute_mono = 0.0
        self._last_lat = None
        self._last_lon = None

    def is_navigating(self) -> bool:
        return bool(self.steps) and not self._completed

    def get_current_instruction(self) -> str:
        """
        Short answer for ``QUERY_NEXT_DIRECTION`` style prompts.
        """
        if not self.steps:
            return "No active navigation"
        if self._completed or self.current_step_index >= len(self.steps):
            return "You have arrived"
        return self.steps[self.current_step_index].instruction

    def nav_display_line(self) -> str:
        """Short line for the phone overlay; empty when navigation has never started."""
        if not self.steps:
            return ""
        if self._completed or self.current_step_index >= len(self.steps):
            return "You have arrived"
        return self.steps[self.current_step_index].instruction

    def update_location(self, lat: float, lon: float) -> None:
        """Remember last fix, reroute if off-segment, advance step after completing maneuver."""
        self._last_lat = lat
        self._last_lon = lon

        if not self.steps or self._completed:
            return

        if self.current_step_index >= len(self.steps):
            self._completed = True
            return

        step = self.steps[self.current_step_index]
        off = cross_track_distance_meters(lat, lon, step.start_lat, step.start_lon, step.lat, step.lon)
        now = time.monotonic()
        if (
            self._dest_lat is not None
            and self._dest_lon is not None
            and off > OffRouteThresholdM
            and (now - self._last_reroute_mono) >= RerouteCooldownSec
        ):
            try:
                self.steps = get_walking_route(
                    lat, lon, self._dest_lat, self._dest_lon, session=self._session
                )
                self.current_step_index = 0
                self.last_spoken_stage = SpokenStage.NONE
                self._last_reroute_mono = now
            except Exception:
                self._last_reroute_mono = now

        self._maybe_advance_step(lat, lon)

    def _maybe_advance_step(self, lat: float, lon: float) -> None:
        if not self.steps or self._completed:
            return
        if self.current_step_index >= len(self.steps):
            self._completed = True
            return

        step = self.steps[self.current_step_index]
        dist_end = distance(lat, lon, step.lat, step.lon)
        is_last = self.current_step_index >= len(self.steps) - 1

        if self.last_spoken_stage == SpokenStage.IMMEDIATE:
            if is_last and dist_end <= 10.0:
                self._completed = True
                return
            if not is_last and dist_end > CompletionDistanceM:
                self.current_step_index += 1
                self.last_spoken_stage = SpokenStage.NONE
                if self.current_step_index >= len(self.steps):
                    self._completed = True

    def check_instruction_trigger(self) -> Optional[dict[str, str]]:
        """
        If a cue should be spoken now, return ``navigation_message(...)``, else ``None``.

        Call after :meth:`update_location` with the same GPS fix. Uses ~30 m for
        \"ahead\" and ~8 m for \"now\"; each stage once per step (immediate can follow early).
        """
        if not self.steps or self._completed:
            return None
        if self.current_step_index >= len(self.steps):
            return None
        if self._last_lat is None or self._last_lon is None:
            return None

        step = self.steps[self.current_step_index]
        lat, lon = self._last_lat, self._last_lon
        dist_end = distance(lat, lon, step.lat, step.lon)

        if dist_end <= ImmediateTriggerM and self.last_spoken_stage != SpokenStage.IMMEDIATE:
            self.last_spoken_stage = SpokenStage.IMMEDIATE
            return navigation_message(f"{step.instruction} now")

        if dist_end <= EarlyTriggerM and self.last_spoken_stage == SpokenStage.NONE:
            self.last_spoken_stage = SpokenStage.EARLY
            return navigation_message(f"{step.instruction} ahead")

        return None
