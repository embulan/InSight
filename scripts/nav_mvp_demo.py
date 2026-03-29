#!/usr/bin/env python3
"""
MVP demo: destination input, simulated walk along route, printed navigation cues.

Run from repository root:

    python scripts/nav_mvp_demo.py

Requires GOOGLE_MAPS_API_KEY in the environment (.env loaded automatically).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on sys.path (same pattern as running backend from project root)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from backend.navigation.command_parser import parse_command
from backend.navigation.confirmation_fsm import NavigationConfirmationFSM, NavConfirmState
from backend.navigation.manager import NavigationManager
from backend.navigation.maps_routes import Step


def optional_tts(text: str) -> None:
    """Print only unless TTS_NAV_DEMO=1, then try ElevenLabs if configured."""
    flag = os.getenv("TTS_NAV_DEMO", "").strip().lower() in ("1", "true", "yes")
    if not flag:
        print(f"  [TTS skipped] {text}")
        return
    try:
        from elevenlabs.client import ElevenLabs

        key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not key:
            print(f"  [TTS skipped — no ELEVENLABS_API_KEY] {text}")
            return
        client = ElevenLabs(api_key=key)
        stream = client.text_to_speech.stream(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        _ = b"".join(stream)
        print(f"  [TTS played] {text}")
    except Exception as exc:
        print(f"  [TTS error: {exc}] {text}")


def interpolate(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int
) -> list[tuple[float, float]]:
    if n < 2:
        return [(lat2, lon2)]
    out: list[tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        out.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
    return out


def simulate_route(nav: NavigationManager, *, steps_per_leg: int = 12) -> None:
    """Walk each step from start→end in straight-line lat/lon (demo only)."""
    for idx, step in enumerate(nav.steps):
        print(f"\n--- Simulating leg {idx + 1}/{len(nav.steps)}: {step.instruction!r} ---")
        path = interpolate(step.start_lat, step.start_lon, step.lat, step.lon, steps_per_leg)
        for lat, lon in path:
            nav.update_location(lat, lon)
            msg = nav.check_instruction_trigger()
            if msg:
                print(f"  TRIGGER {msg}")
                optional_tts(msg["text"])


def demo_parse_and_fsm() -> None:
    print("--- parse_command + confirmation FSM sample ---")
    fsm = NavigationConfirmationFSM()
    samples = [
        "navigate to Yale University",
        "yes",
        "what's next",
    ]
    for s in samples:
        p = parse_command(s)
        print(f"  text={s!r} -> {p}")
        if p.get("intent") == "QUERY_NEXT_DIRECTION":
            print("    (FSM ignores query intents)")
            continue
        ev = fsm.handle_parsed(p)
        print(f"    FSM -> {ev}  state={fsm.state}")


def main() -> None:
    demo_parse_and_fsm()

    dest = input("\nDestination (or Enter to skip live API demo): ").strip()
    if not dest:
        print("Skipping Google API calls.")
        return

    if not os.getenv("GOOGLE_MAPS_API_KEY", "").strip():
        print("GOOGLE_MAPS_API_KEY not set — cannot fetch route.")
        return

    try:
        olat = float(input("Origin latitude: ").strip())
        olon = float(input("Origin longitude: ").strip())
    except ValueError:
        print("Invalid origin.")
        raise SystemExit(1)

    nav = NavigationManager()
    print("\nFetching route...")
    nav.start_navigation(dest, (olat, olon))
    print(f"Steps: {len(nav.steps)}")
    for i, st in enumerate(nav.steps):
        print(f"  {i + 1}. {st.instruction}  (~{st.distance_meters:.0f} m)")

    print("\nQuery mid-walk:", nav.get_current_instruction())

    simulate_route(nav, steps_per_leg=16)

    print("\nFinal query:", nav.get_current_instruction())


def dry_run_no_api() -> None:
    """Used when --dry-run: synthetic steps, no keys."""
    print("--- Dry run (no HTTP) ---")
    nav = NavigationManager()
    steps = [
        Step(
            instruction="Turn right",
            lat=40.0,
            lon=-74.0,
            distance_meters=120.0,
            start_lat=40.001,
            start_lon=-74.0,
        ),
        Step(
            instruction="Continue straight",
            lat=39.999,
            lon=-73.999,
            distance_meters=80.0,
            start_lat=40.0,
            start_lon=-74.0,
        ),
    ]
    nav.load_route_from_steps(
        steps,
        destination_label="Synthetic",
        dest_lat=steps[-1].lat,
        dest_lon=steps[-1].lon,
    )
    simulate_route(nav, steps_per_leg=20)
    print("Final:", nav.get_current_instruction())


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        demo_parse_and_fsm()
        dry_run_no_api()
    else:
        main()
