"""
Geodesy helpers: haversine distance and point-to-segment distance for off-route checks.
"""

from __future__ import annotations

import math


_EARTH_RADIUS_M = 6_371_000.0


def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return _EARTH_RADIUS_M * c


def _to_local_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Local east/north meters relative to (lat0, lon0); fine for short walking segments."""
    dy = (lat - lat0) * 111_320.0
    dx = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    return dx, dy


def cross_track_distance_meters(
    lat_p: float,
    lon_p: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Shortest distance from point P to the line segment A–B (meters).

    Uses a local tangent-plane approximation; accurate enough for walking-scale steps.
    """
    lat0 = (lat1 + lat2 + lat_p) / 3.0
    lon0 = (lon1 + lon2 + lon_p) / 3.0

    ax, ay = _to_local_m(lat1, lon1, lat0, lon0)
    bx, by = _to_local_m(lat2, lon2, lat0, lon0)
    px, py = _to_local_m(lat_p, lon_p, lat0, lon0)

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-6:
        return distance(lat_p, lon_p, lat1, lon1)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)
