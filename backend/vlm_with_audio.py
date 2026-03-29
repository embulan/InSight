"""
vlm_with_audio.py
Blind-assist pipeline: Gemini vision → priority-based audio alerts → ElevenLabs TTS.

Priority levels:
  CRITICAL  - loud alarm + stern short alert  (e.g. "Car ahead!")
  HIGH      - softer tone + short alert        (e.g. "Bicycle nearby.")
  MEDIUM    - soft chime + short alert         (e.g. "Door ahead.")
  AMBIENT   - no sound effect, TTS description spoken every ~5 seconds

Usage:
  python vlm_with_audio.py                           # single image: test-image2.jpg
  python vlm_with_audio.py img1.jpg img2.jpg ...     # multiple images in sequence
  python vlm_with_audio.py --loop img1.jpg img2.jpg  # loop continuously

Setup:
    pip install google-genai elevenlabs pillow python-dotenv pygame

Keys needed in .env:
    GEMINI_API_KEY     - https://aistudio.google.com/app/apikey
    ELEVENLABS_API_KEY - https://elevenlabs.io  (free account)
"""

from __future__ import annotations

import io
import os
import sys
import time
import math
import struct
import threading
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"

DEFAULT_PROMPT = '''You are the navigation assistant for a blind or low-vision user who is walking.

Analyze the scene and report hazards using this EXACT format for each one:
  [DIRECTION]: [OBJECT] — [ACTION]

Rules:
- DIRECTION must be one of: ahead, left, right, lower-left, lower-right, above
- OBJECT is the hazard (e.g. "pole", "car", "stairs going down", "person")
- ACTION is a short movement instruction: move left, move right, stop, slow down, step up, step down, use handrail

Examples of good output:
  ahead: car — stop immediately
  left: pole — move right to avoid
  right: sign — move left to avoid
  ahead: stairs going down — slow down, use handrail
  ahead: stairs going up — step up carefully
  ahead: person — slow down
  ahead: car and cyclist — stop, no clear path

STAIR DIRECTION rules (critical for safety):
- Steps rising away from camera (fronts visible, getting smaller) = "stairs going up"
- Steps descending away from camera (tops visible, drop ahead) = "stairs going down"

If multiple hazards exist, list each on its own line in this format.
If the path is completely blocked with no safe direction, end with: "no clear path — stop"
If nothing notable, reply exactly: Clear.
Keep it short — no extra explanation.'''

ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George" — works on all accounts

_DEFAULT_TIMEOUT_MS = 120_000
_MAX_RETRIES        = 3
_RETRY_DELAY_SEC    = 1.0

AMBIENT_INTERVAL_SEC = 5.0
_last_ambient_time   = 0.0

# ── Frame-rate gate (Change 5) ────────────────────────────────────────────────
LAST_RUN_TIME = 0.0
MIN_INTERVAL  = 0.4   # seconds — skip frames arriving faster than this

# ── Repeated-speech dedup (Change 6) ─────────────────────────────────────────
LAST_SPOKEN_SET: set[str] = set()

# ── Priority levels ───────────────────────────────────────────────────────────

CRITICAL = "critical"
HIGH     = "high"
MEDIUM   = "medium"
AMBIENT  = "ambient"

PRIORITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, AMBIENT: 3}
PRIORITY_ICON  = {CRITICAL: "🚨", HIGH: "⚠️ ", MEDIUM: "🔔", AMBIENT: "🗣 "}

# ── Hazard classification ─────────────────────────────────────────────────────
# Maps keywords found in the object part of a hazard line → priority level.
# First match wins. Order matters — more specific first.

OBJECT_PRIORITY: list[tuple[list[str], str]] = [
    # CRITICAL
    (["car"],          CRITICAL),
    (["vehicle"],      CRITICAL),
    (["truck"],        CRITICAL),
    (["bus"],          CRITICAL),
    (["motorcycle"],   CRITICAL),
    (["traffic"],      CRITICAL),
    (["stairs", "down"], CRITICAL),
    (["steps", "down"],  CRITICAL),
    (["descend"],      CRITICAL),
    (["stair"],        CRITICAL),
    (["step", "down"], CRITICAL),
    (["curb"],         CRITICAL),
    (["drop"],         CRITICAL),
    (["wet", "floor"], CRITICAL),
    (["low", "ceiling"], CRITICAL),
    # HIGH
    (["stairs", "up"], HIGH),
    (["steps", "up"],  HIGH),
    (["step", "up"],   HIGH),
    (["steps"],        HIGH),
    (["bicycle"],      HIGH),
    (["bike"],         HIGH),
    (["cyclist"],      HIGH),
    (["person"],       HIGH),
    (["people"],       HIGH),
    (["pedestrian"],   HIGH),
    (["scooter"],      HIGH),
    (["dog"],          HIGH),
    (["runner"],       HIGH),
    (["crowd"],        HIGH),
    # MEDIUM
    (["door"],         MEDIUM),
    (["pole"],         MEDIUM),
    (["bollard"],      MEDIUM),
    (["sign"],         MEDIUM),
    (["chair"],        MEDIUM),
    (["table"],        MEDIUM),
    (["bench"],        MEDIUM),
    (["barrier"],      MEDIUM),
    (["cone"],         MEDIUM),
    (["rock"],         MEDIUM),
    (["construction"], MEDIUM),
    (["shopping", "cart"], MEDIUM),
]

# ── Direction → spoken phrase ─────────────────────────────────────────────────

DIRECTION_PHRASE: dict[str, str] = {
    "ahead":       "directly ahead",
    "left":        "on your left",
    "right":       "on your right",
    "lower-left":  "to your lower left",
    "lower-right": "to your lower right",
    "above":       "above you",
}

# ── Action → spoken phrase ────────────────────────────────────────────────────

def spoken_action(action: str) -> str:
    """Convert a short action code into a natural spoken instruction."""
    a = action.lower().strip()
    if "no clear path" in a or "stop" in a:
        return "Stop — no clear path."
    if "move right" in a:
        return "Move right to avoid."
    if "move left" in a:
        return "Move left to avoid."
    if "slow down" in a and "handrail" in a:
        return "Slow down and use the handrail."
    if "slow down" in a:
        return "Slow down."
    if "step up" in a:
        return "Step up carefully."
    if "step down" in a:
        return "Step down carefully."
    if "use handrail" in a:
        return "Use the handrail."
    return action.strip().rstrip(".")  + "."


# ── Parse VLM structured output ───────────────────────────────────────────────

def classify_priority(object_text: str) -> str:
    """Return priority level based on keywords in the object description."""
    t = object_text.lower()
    for keywords, priority in OBJECT_PRIORITY:
        if all(kw in t for kw in keywords):
            return priority
    return MEDIUM  # default unknown objects to medium


def parse_hazard_lines(description: str) -> list[dict]:
    """
    Parse structured VLM output into hazard dicts.
    Each line expected: "direction: object — action"
    Returns list of {direction, object, action, priority, spoken}
    """
    hazards = []
    seen: set[str] = set()

    for line in description.strip().splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line or line.lower() == "clear.":
            continue

        # Parse "direction: object — action"
        if ":" not in line:
            continue

        colon_idx = line.index(":")
        direction_raw = line[:colon_idx].strip().lower()
        rest = line[colon_idx + 1:].strip()

        # Split on — or - for action
        if "—" in rest:
            obj_part, action_part = rest.split("—", 1)
        elif " - " in rest:
            obj_part, action_part = rest.split(" - ", 1)
        else:
            obj_part   = rest
            action_part = "be careful"

        obj_part    = obj_part.strip()
        action_part = action_part.strip()

        # Normalise direction
        direction = direction_raw
        for key in DIRECTION_PHRASE:
            if key in direction_raw:
                direction = key
                break

        priority = classify_priority(obj_part)
        dir_phrase = DIRECTION_PHRASE.get(direction, f"to your {direction}")
        action_spoken = spoken_action(action_part)

        # Build spoken alert
        spoken = f"Warning: {obj_part} {dir_phrase}. {action_spoken}"

        dedup_key = f"{direction}:{obj_part.lower()}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        hazards.append({
            "direction": direction,
            "object":    obj_part,
            "action":    action_part,
            "priority":  priority,
            "spoken":    spoken,
        })

    # Sort critical → high → medium
    hazards.sort(key=lambda h: PRIORITY_ORDER.get(h["priority"], 9))
    return hazards


# ── Synthesised sound effects (no files needed) ───────────────────────────────

def _make_wav_bytes(frames: bytes, sample_rate: int = 44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return buf.getvalue()


def _sine_wave(freq: float, duration: float, amplitude: float, sample_rate: int = 44100) -> bytes:
    n = int(sample_rate * duration)
    frames = b""
    for i in range(n):
        t = i / sample_rate
        fade = 1.0 if i < n * 0.9 else (n - i) / (n * 0.1)
        val = int(amplitude * fade * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", max(-32767, min(32767, val)))
    return frames


def make_critical_alarm() -> bytes:
    sr, amp = 44100, 28000
    t1  = _sine_wave(880, 0.18, amp, sr)
    t2  = _sine_wave(660, 0.18, amp, sr)
    sil = b"\x00\x00" * int(sr * 0.06)
    return _make_wav_bytes((t1 + sil + t2 + sil) * 2, sr)


def make_high_tone() -> bytes:
    return _make_wav_bytes(_sine_wave(520, 0.25, 18000), 44100)


def make_medium_chime() -> bytes:
    return _make_wav_bytes(_sine_wave(380, 0.3, 10000), 44100)


# ── TTS engine ────────────────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self._el_client  = None
        self._chars_used = 0

        # Initialise pygame mixer exactly once (Change 4)
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        if ELEVENLABS_API_KEY:
            try:
                from elevenlabs.client import ElevenLabs
                self._el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                print("TTS: ElevenLabs active")
            except ImportError:
                print("TTS: elevenlabs not installed — run: pip install elevenlabs")
        else:
            print("TTS: no ELEVENLABS_API_KEY in .env")

        self._sounds: dict[str, bytes] = {
            CRITICAL: make_critical_alarm(),
            HIGH:     make_high_tone(),
            MEDIUM:   make_medium_chime(),
        }

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        import pygame
        sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
        sound.play()
        while pygame.mixer.get_busy():
            time.sleep(0.01)

    def play_priority_tone(self, priority: str) -> None:
        if priority == AMBIENT:
            return
        wav = self._sounds.get(priority)
        if wav:
            try:
                self._play_wav_bytes(wav)
            except Exception as e:
                print(f"  [tone error: {e}]")

    def speak(self, text: str) -> None:
        """Synthesise and start playback. Non-blocking — returns immediately after play() (Change 1)."""
        if not self._el_client:
            print(f"  [no TTS] {text}")
            return
        if self._chars_used >= 9500:
            print("  [ElevenLabs quota nearly reached]")
            return
        try:
            audio_stream = self._el_client.text_to_speech.convert(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id="eleven_flash_v2_5",   # Change 7: faster model
            )
            audio_bytes = b"".join(audio_stream)
            import pygame
            pygame.mixer.music.load(io.BytesIO(audio_bytes))
            pygame.mixer.music.play()           # Change 1: no blocking wait
            self._chars_used += len(text)
        except Exception as e:
            print(f"  ElevenLabs error: {e}")

    def speak_async(self, text: str) -> None:
        """Fire-and-forget TTS on a daemon thread (Change 1)."""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()


# ── Gemini vision call ────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("Missing GEMINI_API_KEY")
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=_DEFAULT_TIMEOUT_MS),
    )


def analyze_image(
    image_path: str | Path,
    prompt: str = DEFAULT_PROMPT,
    *,
    model: str = MODEL,
    max_retries: int = _MAX_RETRIES,
) -> str:
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    client = _get_client()

    for attempt in range(max(1, max_retries)):
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
                image.thumbnail((768, 768))

            response = client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                )
            )

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ValueError("Empty response from model")
            return text

        except FileNotFoundError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(_RETRY_DELAY_SEC)
            else:
                raise RuntimeError(f"Gemini failed after {max_retries} attempt(s): {exc}") from exc


# ── Single frame pipeline ─────────────────────────────────────────────────────

def run_frame(image_path: str | Path, tts: TTSEngine, frame_num: int = 1) -> str:
    global _last_ambient_time, LAST_RUN_TIME, LAST_SPOKEN_SET

    # Change 5: frame-rate gate — skip frames arriving too quickly
    now = time.time()
    if now - LAST_RUN_TIME < MIN_INTERVAL:
        print("Skipping frame (rate limit)")
        return ""
    LAST_RUN_TIME = now

    print(f"\n{'═' * 55}")
    print(f"  Frame {frame_num}: {image_path}")
    print(f"{'═' * 55}")

    t_start   = time.perf_counter()
    result    = analyze_image(image_path)
    t_vlm_end = time.perf_counter()
    vlm_ms    = (t_vlm_end - t_start) * 1000

    print(f"VLM latency      : {vlm_ms:.1f} ms")
    print(f"result           : {result}")
    print("─" * 55)

    is_clear = result.strip().lower().startswith("clear")
    hazards  = [] if is_clear else parse_hazard_lines(result)

    t_audio_start = time.perf_counter()

    if hazards:
        print(f"hazards detected : {len(hazards)}")
        for h in hazards:
            icon = PRIORITY_ICON.get(h["priority"], "  ")
            print(f"  {icon} [{h['priority'].upper():<8}] {h['spoken']}")

        # Change 6: skip hazards that were already spoken in a recent frame
        new_hazards = []
        for h in hazards:
            if h["spoken"] not in LAST_SPOKEN_SET:
                new_hazards.append(h)
                LAST_SPOKEN_SET.add(h["spoken"])

        if not new_hazards:
            print("All hazards already spoken — skipping audio")
        else:
            # Change 2: one combined TTS call for the top two hazards
            top_hazards  = new_hazards[:2]
            combined_text = " ".join(h["spoken"] for h in top_hazards)
            tts.play_priority_tone(top_hazards[0]["priority"])
            tts.speak_async(combined_text)   # Change 1: non-blocking
    else:
        print("hazards detected : none — ambient")
        # Clear spoken set when scene is clear so stale hazards don't stay suppressed
        LAST_SPOKEN_SET.clear()
        now_t = time.time()
        if now_t - _last_ambient_time >= AMBIENT_INTERVAL_SEC:
            print(f"  🗣  [AMBIENT  ] {result}")
            tts.speak_async(result)          # Change 8: non-blocking ambient speech
            _last_ambient_time = now_t
        else:
            remaining = AMBIENT_INTERVAL_SEC - (now_t - _last_ambient_time)
            print(f"  🗣  [AMBIENT  ] skipped — next in {remaining:.1f}s")

    t_end    = time.perf_counter()
    audio_ms = (t_end - t_audio_start) * 1000
    total_ms = (t_end - t_start) * 1000

    print("─" * 55)
    print(f"VLM latency      : {vlm_ms:.1f} ms")
    print(f"audio latency    : {audio_ms:.1f} ms")
    print(f"TOTAL latency    : {total_ms:.1f} ms")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    loop_mode = "--loop" in args
    if loop_mode:
        args = [a for a in args if a != "--loop"]

    images = args if args else ["test-image2.jpg"]

    tts = TTSEngine()

    if loop_mode:
        print(f"Looping over {len(images)} image(s) continuously — Ctrl+C to stop.")
        frame = 1
        try:
            while True:
                for img in images:
                    run_frame(img, tts, frame_num=frame)
                    frame += 1
                    time.sleep(0.3)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        for i, img in enumerate(images, 1):
            run_frame(img, tts, frame_num=i)
            if i < len(images):
                time.sleep(0.3)


if __name__ == "__main__":
    main()