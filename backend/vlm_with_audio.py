"""
vlm_with_audio.py
Blind-assist pipeline: Gemini vision → priority-based audio alerts → ElevenLabs TTS.

Priority levels:
  CRITICAL  - loud alarm + stern short alert  (e.g. "Car ahead!")
  HIGH      - softer tone + short alert        (e.g. "Bicycle nearby.")
  MEDIUM    - soft chime + short alert         (e.g. "Door ahead.")
  AMBIENT   - no sound effect, TTS description spoken every ~5 seconds

Setup:
    pip install google-genai elevenlabs pillow python-dotenv pygame numpy

Keys needed in .env:
    GEMINI_API_KEY     - https://aistudio.google.com/app/apikey
    ELEVENLABS_API_KEY - https://elevenlabs.io  (free account)
"""

from __future__ import annotations

import io
import os
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

AMBIENT_INTERVAL_SEC = 5.0   # how often ambient descriptions are spoken
_last_ambient_time   = 0.0


# Priority levels

CRITICAL = "critical"
HIGH     = "high"
MEDIUM   = "medium"
AMBIENT  = "ambient"

PRIORITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, AMBIENT: 3}


# ── Hazard library ────────────────────────────────────────────────────────────
# phrase → (short spoken alert, priority)

HAZARD_LIBRARY: dict[str, tuple[str, str]] = {
    # CRITICAL — alarm sound + stern alert
    "step down":        ("Step down ahead!",          CRITICAL),
    "step up":          ("Step up ahead!",            CRITICAL),
    "curb":             ("Curb ahead!",               CRITICAL),
    "stairs":           ("Stairs ahead!",             CRITICAL),
    "obstacle ahead":   ("Obstacle ahead!",           CRITICAL),
    "wet floor":        ("Wet floor!",                CRITICAL),
    "car ahead":        ("Car ahead!",                CRITICAL),
    "car approaching":  ("Stop! - Car approaching",   CRITICAL),
    "crossing traffic": ("Stop! - Traffic crossing",  CRITICAL),
    "low ceiling":      ("Low ceiling ahead!",        CRITICAL),
    "drop":             ("Drop-off ahead!",           CRITICAL),

    # HIGH — soft alert tone + short alert
    "bicycle":          ("Bicycle nearby.",           HIGH),
    "cyclist":          ("Cyclist nearby.",           HIGH),
    "person ahead":     ("Person ahead.",             HIGH),
    "person approaching": ("Person approaching.",     HIGH),
    "scooter":          ("Scooter nearby.",           HIGH),
    "dog":              ("Dog nearby.",               HIGH),

    # MEDIUM — soft chime + short alert
    "door":             ("Door ahead.",               MEDIUM),
    "pole":             ("Pole ahead.",               MEDIUM),
    "bollard":          ("Bollard ahead.",            MEDIUM),
    "chair":            ("Chair in path.",            MEDIUM),
    "table":            ("Table in path.",            MEDIUM),
    "shopping cart":    ("Shopping cart ahead.",      MEDIUM),
    "construction":     ("Construction nearby.",      MEDIUM),
    "sign":             ("Sign in path.",             MEDIUM),
}


# ── Synthesised sound effects (no audio files needed) ────────────────────────

def _make_wav_bytes(frames: bytes, sample_rate: int = 44100, n_channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return buf.getvalue()


def _sine_wave(freq: float, duration: float, amplitude: float, sample_rate: int = 44100) -> bytes:
    n = int(sample_rate * duration)
    frames = b""
    for i in range(n):
        t = i / sample_rate
        # Fade out last 10% to avoid clicks
        fade = 1.0 if i < n * 0.9 else (n - i) / (n * 0.1)
        val = int(amplitude * fade * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", max(-32767, min(32767, val)))
    return frames


def make_critical_alarm() -> bytes:
    """Urgent two-tone alarm."""
    sr = 44100
    amp = 28000
    tone1 = _sine_wave(880, 0.18, amp, sr)   # high A
    tone2 = _sine_wave(660, 0.18, amp, sr)   # E
    silence = b"\x00\x00" * int(sr * 0.06)
    pattern = tone1 + silence + tone2 + silence
    return _make_wav_bytes(pattern * 2, sr)   # repeat twice


def make_high_tone() -> bytes:
    """Single mid-pitched ping."""
    sr = 44100
    frames = _sine_wave(520, 0.25, 18000, sr)
    return _make_wav_bytes(frames, sr)


def make_medium_chime() -> bytes:
    """Soft low chime."""
    sr = 44100
    frames = _sine_wave(380, 0.3, 10000, sr)
    return _make_wav_bytes(frames, sr)


# ── TTS engine

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

        # Pre-generate sound effect bytes once
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
        """Play the sound effect for a given priority level. AMBIENT = silent."""
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


# ── Hazard matching ───────────────────────────────────────────────────────────

def find_hazards(description: str) -> list[tuple[str, str, str]]:
    """
    Returns list of (phrase, alert_text, priority) found in description,
    sorted critical → high → medium.
    """
    text = description.lower()
    found = []
    seen_alerts = set()
    for phrase, (alert, priority) in HAZARD_LIBRARY.items():
        if phrase in text and alert not in seen_alerts:
            found.append((phrase, alert, priority))
            seen_alerts.add(alert)
    found.sort(key=lambda x: PRIORITY_ORDER.get(x[2], 9))
    return found


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(image_path: str | Path, prompt: str = DEFAULT_PROMPT) -> str:
    global _last_ambient_time

    tts = TTSEngine()

    print(f"\nmodel  : {MODEL}")
    print(f"image  : {image_path}")
    print("─" * 50)

    # ── Timing: VLM call
    t_start       = time.perf_counter()
    t_vlm_start   = t_start
    result        = analyze_image(image_path, prompt)
    t_vlm_end     = time.perf_counter()
    vlm_ms        = (t_vlm_end - t_vlm_start) * 1000

    print(f"VLM latency      : {vlm_ms:.1f} ms")
    print(f"result           : {result}")
    print("─" * 50)

    # ── Hazard detection
    hazards = find_hazards(result)

    if hazards:
        print(f"hazards detected : {[h[0] for h in hazards]}")
    else:
        print("hazards detected : none — ambient description")

    # ── Audio: play tones + speak alerts for hazards
    t_audio_start = time.perf_counter()

    if hazards:
        for phrase, alert, priority in hazards:
            print(f"  [{priority.upper():<8}] 🔊 {alert}")
            tts.play_priority_tone(priority)   # sound effect first
            tts.speak(alert)                   # then stern short alert
    else:
        # No hazards — ambient: speak full description on interval
        now = time.time()
        if now - _last_ambient_time >= AMBIENT_INTERVAL_SEC:
            print(f"  [AMBIENT  ] 🗣  {result}")
            tts.speak(result)
            _last_ambient_time = now
        else:
            remaining = AMBIENT_INTERVAL_SEC - (now - _last_ambient_time)
            print(f"  [AMBIENT  ] skipped — next in {remaining:.1f}s")

    t_audio_end = time.perf_counter()
    audio_ms    = (t_audio_end - t_audio_start) * 1000
    total_ms    = (t_audio_end - t_start) * 1000

    print("─" * 50)
    print(f"audio latency    : {audio_ms:.1f} ms")
    print(f"TOTAL latency    : {total_ms:.1f} ms")

    return result


# def main() -> None:
#     import sys
#     image_arg = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
#     run(image_arg)


# if __name__ == "__main__":
#     main()
