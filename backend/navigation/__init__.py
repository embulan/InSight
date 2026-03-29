"""
GPS walking navigation helpers (Google Maps HTTP, parsing, proximity cues).

Does not integrate with the WebSocket server by default — import what you need.
"""

from backend.navigation.command_parser import parse_command
from backend.navigation.confirmation_fsm import NavConfirmState, NavigationConfirmationFSM
from backend.navigation.geo import cross_track_distance_meters, distance
from backend.navigation.instructions import simplify_instruction, strip_html
from backend.navigation.manager import (
    EarlyTriggerM,
    ImmediateTriggerM,
    NavigationManager,
    SpokenStage,
    navigation_message,
)
from backend.navigation.maps_routes import (
    Step,
    geocode_location,
    get_walking_route,
    plan_walking_navigation,
    step_to_dict,
)

__all__ = [
    "EarlyTriggerM",
    "ImmediateTriggerM",
    "NavConfirmState",
    "NavigationConfirmationFSM",
    "NavigationManager",
    "SpokenStage",
    "Step",
    "cross_track_distance_meters",
    "distance",
    "geocode_location",
    "get_walking_route",
    "plan_walking_navigation",
    "step_to_dict",
    "navigation_message",
    "parse_command",
    "simplify_instruction",
    "strip_html",
]
