#!/usr/bin/env python3
"""
Test Google Maps Geocoding + Directions (walking) without a phone.

Uses fixed origin lat/lon — get them from Google Maps (right-click / drop pin).

Examples (from repo root, one line):

  python scripts/nav_api_test.py -d "Yale University" --origin-lat 41.308 --origin-lon -72.928

  python scripts/nav_api_test.py -d "Times Square" --origin-lat 40.758 --origin-lon -73.985 --json

Requires GOOGLE_MAPS_API_KEY in .env or the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from backend.navigation.maps_routes import plan_walking_navigation


def main() -> None:
    p = argparse.ArgumentParser(description="Test Maps API: geocode + walking route + all steps.")
    p.add_argument("-d", "--destination", required=True, help="Place name to geocode")
    p.add_argument("--origin-lat", type=float, required=True, help="Start latitude (WGS84)")
    p.add_argument("--origin-lon", type=float, required=True, help="Start longitude (WGS84)")
    p.add_argument("--json", action="store_true", help="Print full result as JSON only")
    args = p.parse_args()

    if not __import__("os").getenv("GOOGLE_MAPS_API_KEY", "").strip():
        print("Error: GOOGLE_MAPS_API_KEY is not set.", file=sys.stderr)
        raise SystemExit(1)

    result = plan_walking_navigation(args.destination, args.origin_lat, args.origin_lon)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Geocoding + walking directions OK\n")
    print(f"  Query:        {result['destination_query']}")
    print(
        f"  Destination:  {result['destination_lat']:.6f}, {result['destination_lon']:.6f}"
    )
    print(f"  Origin:       {result['origin_lat']:.6f}, {result['origin_lon']:.6f}")
    leg = result["leg"]
    print(f"  Leg distance: {leg['total_distance_text']} ({leg['total_distance_meters']} m)")
    print(f"  Leg duration: {leg['total_duration_text']} ({leg['total_duration_seconds']} s)")
    if leg.get("start_address"):
        print(f"  Start addr:   {leg['start_address']}")
    if leg.get("end_address"):
        print(f"  End addr:     {leg['end_address']}")

    print(f"\nSteps ({result['step_count']}):\n")
    for i, st in enumerate(result["steps"], start=1):
        print(f"  {i}. {st['instruction']}")
        print(
            f"      segment ~{st['distance_meters']:.0f} m  "
            f"({st['start_lat']:.5f},{st['start_lon']:.5f}) → "
            f"({st['end_lat']:.5f},{st['end_lon']:.5f})"
        )

    print("\nTip: add --json to copy the full structure into other tools.")


if __name__ == "__main__":
    main()
