"""
Google Maps Platform HTTP helpers: Geocoding + Directions (walking).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from backend.navigation.instructions import simplify_instruction, strip_html


GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

_DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class Step:
    """One walking leg step after cleaning."""

    instruction: str
    lat: float
    lon: float
    distance_meters: float
    start_lat: float
    start_lon: float


def _api_key() -> str:
    key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in the environment")
    return key


def geocode_location(destination: str, *, session: Optional[requests.Session] = None) -> tuple[float, float]:
    """
    Resolve a free-text place name to (lat, lon) using the Geocoding API.
    """
    sess = session or requests.Session()
    params = {"address": destination, "key": _api_key()}
    r = sess.get(f"{GEOCODE_URL}?{urlencode(params)}", timeout=_DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    status = data.get("status")
    if status != "OK" or not data.get("results"):
        raise RuntimeError(f"Geocoding failed: status={status!r} error={data.get('error_message')!r}")

    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])


def _fetch_directions_json(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    sess = session or requests.Session()
    origin = f"{origin_lat},{origin_lon}"
    destination = f"{dest_lat},{dest_lon}"
    params: dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "mode": "walking",
        "key": _api_key(),
    }
    r = sess.get(f"{DIRECTIONS_URL}?{urlencode(params)}", timeout=_DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    status = data.get("status")
    if status != "OK" or not data.get("routes"):
        raise RuntimeError(
            f"Directions failed: status={status!r} error={data.get('error_message')!r}"
        )
    return data


def _parse_leg_steps(data: dict[str, Any]) -> tuple[dict[str, Any], list[Step]]:
    leg = data["routes"][0]["legs"][0]
    dur = leg.get("duration") or {}
    dist = leg.get("distance") or {}
    meta = {
        "total_distance_meters": int(dist.get("value", 0)),
        "total_distance_text": dist.get("text", ""),
        "total_duration_seconds": int(dur.get("value", 0)),
        "total_duration_text": dur.get("text", ""),
        "start_address": leg.get("start_address"),
        "end_address": leg.get("end_address"),
    }
    out: list[Step] = []
    for s in leg.get("steps", []):
        html = s.get("html_instructions") or ""
        clean = strip_html(html)
        base = simplify_instruction(clean)
        end = s.get("end_location") or {}
        start = s.get("start_location") or {}
        sdist = s.get("distance") or {}
        meters = float(sdist.get("value", 0))
        out.append(
            Step(
                instruction=base,
                lat=float(end["lat"]),
                lon=float(end["lng"]),
                distance_meters=meters,
                start_lat=float(start["lat"]),
                start_lon=float(start["lng"]),
            )
        )
    return meta, out


def step_to_dict(step: Step) -> dict[str, Any]:
    """JSON-friendly step for API tests / logging."""
    return {
        "instruction": step.instruction,
        "distance_meters": step.distance_meters,
        "start_lat": step.start_lat,
        "start_lon": step.start_lon,
        "end_lat": step.lat,
        "end_lon": step.lon,
    }


def get_walking_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    session: Optional[requests.Session] = None,
) -> list[Step]:
    """
    Fetch a walking route and return parsed steps.

    Uses https://maps.googleapis.com/maps/api/directions/json with mode=walking.
    """
    data = _fetch_directions_json(
        origin_lat, origin_lon, dest_lat, dest_lon, session=session
    )
    _, steps = _parse_leg_steps(data)
    return steps


def plan_walking_navigation(
    destination: str,
    origin_lat: float,
    origin_lon: float,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """
    Geocode ``destination``, fetch walking directions from a fixed origin, return all steps.

    No GPS hardware required — use any lat/lon (e.g. from Google Maps right-click).

    Returns a JSON-serializable dict:

    - ``destination_query``, ``destination_lat``, ``destination_lon``
    - ``origin_lat``, ``origin_lon``
    - ``leg``: total distance/duration and addresses
    - ``steps``: list of dicts from :func:`step_to_dict`
    - ``step_count``
    """
    sess = session or requests.Session()
    dest_lat, dest_lon = geocode_location(destination, session=sess)
    data = _fetch_directions_json(
        origin_lat, origin_lon, dest_lat, dest_lon, session=sess
    )
    leg_meta, steps = _parse_leg_steps(data)
    return {
        "destination_query": destination,
        "destination_lat": dest_lat,
        "destination_lon": dest_lon,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "leg": leg_meta,
        "steps": [step_to_dict(s) for s in steps],
        "step_count": len(steps),
    }
