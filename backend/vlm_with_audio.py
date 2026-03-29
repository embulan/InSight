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
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"

DEFAULT_PROMPT = (
    '''You are assisting a blind or low-vision user who is walking.

Identify only the most important navigation-relevant obstacles or hazards in this scene.

Prioritize:
1. obstacles directly ahead in the walking path
2. stairs, step-downs, curbs, ledges, drop-offs
3. people, cars, bikes, or moving objects nearby
4. doors, poles, chairs, tables, boxes, or objects that may be bumped into
5. crosswalk state or traffic-related hazards if visible

Keep the response short and practical (2-3 sentences max).
Include brief location for each hazard: ahead, left, right, lower left, lower right.
Do not speculate. If nothing notable, reply: Clear.'''
)

ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George" — works on all accounts

_DEFAULT_TIMEOUT_MS = 120_000
_MAX_RETRIES        = 3
_RETRY_DELAY_SEC    = 1.0

AMBIENT_INTERVAL_SEC = 5.0
_last_ambient_time   = 0.0

# ── Priority levels ───────────────────────────────────────────────────────────

CRITICAL = "critical"
HIGH     = "high"
MEDIUM   = "medium"
AMBIENT  = "ambient"

PRIORITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, AMBIENT: 3}
PRIORITY_ICON  = {CRITICAL: "🚨", HIGH: "⚠️ ", MEDIUM: "🔔", AMBIENT: "🗣 "}

# ── Hazard keyword library ────────────────────────────────────────────────────
# Each entry: (trigger_words, alert_text, priority)
# ALL trigger_words must appear anywhere in the description to match.

HAZARD_KEYWORDS: list[tuple[list[str], str, str]] = [
    # ── CRITICAL ──────────────────────────────────────────────────────────────
    (["car"],                   "Car in your path!",           CRITICAL),
    (["vehicle"],               "Vehicle ahead!",              CRITICAL),
    (["truck"],                 "Truck ahead!",                CRITICAL),
    (["bus"],                   "Bus ahead!",                  CRITICAL),
    (["motorcycle"],            "Motorcycle ahead!",           CRITICAL),
    (["step", "down"],          "Step down ahead!",            CRITICAL),
    (["step", "up"],            "Step up ahead!",              CRITICAL),
    (["curb"],                  "Curb ahead!",                 CRITICAL),
    (["stair"],                 "Stairs ahead!",               CRITICAL),
    (["wet", "floor"],          "Wet floor!",                  CRITICAL),
    (["low", "ceiling"],        "Low ceiling ahead!",          CRITICAL),
    (["drop"],                  "Drop-off ahead!",             CRITICAL),
    (["traffic"],               "Traffic — stop!",             CRITICAL),
    # ── HIGH ──────────────────────────────────────────────────────────────────
    (["bicycle"],               "Bicycle nearby.",             HIGH),
    (["bike"],                  "Bike nearby.",                HIGH),
    (["cyclist"],               "Cyclist nearby.",             HIGH),
    (["pedestrian"],            "Pedestrian ahead.",           HIGH),
    (["person"],                "Person ahead.",               HIGH),
    (["people"],                "People ahead.",               HIGH),
    (["scooter"],               "Scooter nearby.",             HIGH),
    (["dog"],                   "Dog nearby.",                 HIGH),
    (["runner"],                "Runner nearby.",              HIGH),
    (["crowd"],                 "Crowd ahead.",                HIGH),
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    (["door"],                  "Door ahead.",                 MEDIUM),
    (["pole"],                  "Pole ahead.",                 MEDIUM),
    (["bollard"],               "Bollard ahead.",              MEDIUM),
    (["shopping", "cart"],      "Shopping cart ahead.",        MEDIUM),
    (["construction"],          "Construction nearby.",        MEDIUM),
    (["chair"],                 "Chair in path.",              MEDIUM),
    (["table"],                 "Table in path.",              MEDIUM),
    (["bench"],                 "Bench ahead.",                MEDIUM),
    (["sign"],                  "Sign in path.",               MEDIUM),
    (["rock"],                  "Rock in path.",               MEDIUM),
    (["barrier"],               "Barrier ahead.",              MEDIUM),
    (["cone"],                  "Cone in path.",               MEDIUM),
]


# ── Hazard matching ───────────────────────────────────────────────────────────

def find_hazards(description: str) -> list[tuple[str, str, str]]:
    """
    Returns ALL matched hazards sorted critical → high → medium.
    Each entry: (matched keywords string, alert text, priority)
    Deduplicates by alert text so the same warning isn't repeated.
    """
    text = description.lower()
    found = []
    seen_alerts: set[str] = set()

    for keywords, alert, priority in HAZARD_KEYWORDS:
        if alert in seen_alerts:
            continue
        if all(kw in text for kw in keywords):
            found.append((", ".join(keywords), alert, priority))
            seen_alerts.add(alert)

    # Sort: critical first, then high, then medium
    found.sort(key=lambda x: PRIORITY_ORDER.get(x[2], 9))
    return found


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
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
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
                model_id="eleven_turbo_v2",
            )
            audio_bytes = b"".join(audio_stream)
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            pygame.mixer.music.load(io.BytesIO(audio_bytes))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            self._chars_used += len(text)
        except Exception as e:
            print(f"  ElevenLabs error: {e}")


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
    global _last_ambient_time

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

    hazards = find_hazards(result)

    t_audio_start = time.perf_counter()

    if hazards:
        print(f"hazards detected : {len(hazards)}")
        for phrase, alert, priority in hazards:
            icon = PRIORITY_ICON.get(priority, "  ")
            print(f"  {icon} [{priority.upper():<8}] {alert}  (matched: '{phrase}')")
            tts.play_priority_tone(priority)
            tts.speak(alert)
    else:
        print("hazards detected : none — ambient")
        now = time.time()
        if now - _last_ambient_time >= AMBIENT_INTERVAL_SEC:
            print(f"  🗣  [AMBIENT  ] {result}")
            tts.speak(result)
            _last_ambient_time = now
        else:
            remaining = AMBIENT_INTERVAL_SEC - (now - _last_ambient_time)
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