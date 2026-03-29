"""
InSight WebSocket server.

Frame path
----------
  iPhone sends one JPEG frame every Config.frameInterval seconds (default 20 s).
  Each frame goes through backend.pipeline.run_pipeline():
    1. sim_check  — motion-compensated scene-novelty gate (fast, no API call)
    2. FrameCache — saves frame to per-connection temp dir on disk
    3. VLM        — calls Gemini only when sim_check says scene changed
  If the pipeline produced a caption, the server converts it to MP3 via
  ElevenLabs and sends both caption text and audio bytes back to the phone.

Submit path (voice query)
-------------------------
  Phone sends audio PCM chunks while mic is open, then a "submit" message.
  Server calls Gemini with the latest cached frame + WAV audio, then TTS.

Incoming messages from iPhone
-------------------------------
  {"type": "frame",  "timestampMs": <int>, "jpegBase64":  "<b64>"}
  {"type": "audio",  "timestampMs": <int>, "sampleRate":  <int>, "pcmBase64": "<b64>"}
  {"type": "submit", "timestampMs": <int>}
  {"type": "location", "timestampMs": <int>, "lat": <float>, "lon": <float>}

Outgoing messages to iPhone
----------------------------
  {"type": "caption", "message": "<text>", "data": null}
  {"type": "audio",   "message": null,     "data": "<base64 mp3>"}
  {"type": "status",  "message": "<text>", "data": null}
  {"type": "error",   "message": "<text>", "data": null}
  {"type": "nav_step", "message": "<current nav instruction>", "data": null}

Navigation (voice + GPS)
------------------------
  Requires GOOGLE_MAPS_API_KEY. After submit with speech \"navigate to …\", the server
  geocodes, fetches a walking route from the last ``location`` fix, and TTSes the
  acknowledgment plus the first step. Ongoing ``location`` messages advance steps and
  trigger proximity cues. Say \"what should I do\" (etc.) for a spoken current step.

Run from repo root
------------------
  uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import time
import wave
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from backend.navigation.command_parser import parse_command
from backend.navigation.gemini_transcribe import transcribe_wav
from backend.navigation.manager import NavigationManager
from backend.pipeline import run_pipeline
from backend.update_image_cache import FrameCache
from backend.vlm_with_audio import analyze_image, _get_client, MODEL

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EL_VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"
EL_MODEL_ID  = "eleven_flash_v2_5"
CACHE_MAX_SIZE = 3

# Seconds to pause all Gemini calls after a quota/rate-limit (429) error.
GEMINI_COOLDOWN_SECS: float = 60.0


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "quota" in s or "rate" in s

_AUDIO_PROMPT = (
    "You are assisting a blind or low-vision user. "
    "Only look at the bottom 60 percent of the image. "
    "The user has spoken a question — audio is attached. "
    "Answer their question about what you see, prioritising navigation hazards. "
    "Be concise (1-3 sentences)."
)

# ---------------------------------------------------------------------------
# ElevenLabs singleton
# ---------------------------------------------------------------------------

_el_client = None


def _get_el():
    global _el_client
    if _el_client is None:
        from elevenlabs.client import ElevenLabs
        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set in .env")
        _el_client = ElevenLabs(api_key=api_key)
    return _el_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jpeg_bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pcm_float32_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Raw mono Float32 little-endian PCM → WAV (Int16)."""
    arr_f32 = np.frombuffer(pcm_bytes, dtype="<f4")
    arr_i16 = (arr_f32 * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(arr_i16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Blocking calls (run in thread pool)
# ---------------------------------------------------------------------------

def _sync_tts(text: str) -> bytes:
    """ElevenLabs TTS → raw MP3 bytes."""
    return b"".join(
        _get_el().text_to_speech.stream(
            text=text,
            voice_id=EL_VOICE_ID,
            model_id=EL_MODEL_ID,
            output_format="mp3_44100_128",
        )
    )


def _sync_gemini_vision_audio(image: Image.Image, wav_bytes: bytes, prompt: str) -> str:
    """Gemini multimodal call with both image and audio."""
    from google.genai import types
    img = image.copy().convert("RGB")
    img.thumbnail((768, 768))
    audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=[prompt, img, audio_part],
    )
    return (getattr(response, "text", None) or "").strip()


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

async def _tts(text: str) -> bytes:
    return await asyncio.get_running_loop().run_in_executor(None, _sync_tts, text)


async def _gemini_vision_audio(image: Image.Image, wav_bytes: bytes, prompt: str) -> str:
    return await asyncio.get_running_loop().run_in_executor(
        None, _sync_gemini_vision_audio, image, wav_bytes, prompt
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="InSight Server")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Per-connection temp dir — keeps concurrent clients isolated
    cache_dir  = tempfile.mkdtemp(prefix="insight_ws_")
    frame_cache = FrameCache(cache_dir=cache_dir, max_size=CACHE_MAX_SIZE)
    print(f"[ws] connected  cache={cache_dir}")

    # Per-connection state
    curr_pil: Optional[Image.Image] = None
    frame_count: int = 0
    audio_buffer      = bytearray()
    audio_sample_rate = 44100
    # monotonic timestamp after which Gemini calls are allowed again (0 = always allowed)
    gemini_cooldown_until: float = 0.0
    # doubles on each 429; resets on success — caps at 10 minutes
    _gemini_cooldown_secs: float = GEMINI_COOLDOWN_SECS
    # set to True when a submit arrives so respond_frame bails immediately
    submit_pending: bool = False

    # Prevents concurrent heavy AI calls for the same client
    vision_lock = asyncio.Lock()
    nav_lock = asyncio.Lock()

    # Walking navigation (Google Maps + proximity cues)
    nav = NavigationManager()
    last_gps: Optional[tuple[float, float]] = None
    last_nav_ui_sent: str = "\n"  # force first push

    # -------------------------------------------------------------------------
    async def send(obj: dict) -> None:
        try:
            await websocket.send_text(json.dumps(obj))
        except Exception as exc:
            print(f"[ws] send error: {exc}")

    async def send_caption_and_audio(caption: str) -> None:
        """Send caption text then TTS audio to the phone."""
        print(f"[ai] caption → {caption[:100]}")
        await send({"type": "caption", "message": caption, "data": None})
        print("[ai] TTS...")
        mp3_bytes = await _tts(caption)
        print(f"[ai] audio ready  {len(mp3_bytes)} bytes")
        await send({
            "type": "audio",
            "message": None,
            "data": base64.b64encode(mp3_bytes).decode(),
        })

    async def push_nav_ui_if_changed() -> None:
        """Minimal nav overlay updates — only when the displayed line changes."""
        nonlocal last_nav_ui_sent
        line = nav.nav_display_line()
        if line == last_nav_ui_sent:
            return
        last_nav_ui_sent = line
        await send({"type": "nav_step", "message": line, "data": None})

    async def handle_gps_tick(lat: float, lon: float) -> None:
        """Update route progress and speak proximity cues (does not take vision_lock)."""
        nonlocal last_gps
        last_gps = (lat, lon)
        cues: list[str] = []
        async with nav_lock:
            nav.update_location(lat, lon)
            for _ in range(4):
                t = nav.check_instruction_trigger()
                if not t:
                    break
                text = (t.get("text") or "").strip()
                if text:
                    cues.append(text)
            await push_nav_ui_if_changed()
        for line in cues:
            await send_caption_and_audio(line)

    async def respond_frame(jpeg_bytes: bytes) -> None:
        """
        Run the full pipeline (sim_check → cache → VLM) for a frame.
        Bails immediately if a submit arrives while processing.
        """
        nonlocal gemini_cooldown_until
        async with vision_lock:
            # Submit arrived while we were waiting for the lock — hand off immediately
            if submit_pending:
                print("[frame] submit pending — skipping frame pipeline")
                return
            try:
                await send({"type": "status", "message": "Analyzing...", "data": None})

                # Run pipeline in thread pool — handles sim_check, cache, and Gemini
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: run_pipeline(jpeg_bytes, cache=frame_cache)
                )

                # Submit arrived while Gemini was running — drop this result
                if submit_pending:
                    print("[frame] submit arrived mid-pipeline — discarding frame result")
                    return

                sr = result["scene_result"]
                if sr:
                    print(
                        f"[sim_check] novelty={sr.novelty_score:.3f}"
                        f"  trigger={sr.should_trigger}"
                        f"  reason={sr.debug_reason}"
                    )

                if not result["send_to_vlm"]:
                    print("[pipeline] scene unchanged — no caption")
                    return

                caption = result["vlm_text"]
                if not caption:
                    print("[pipeline] VLM returned empty text")
                    return

                await send_caption_and_audio(caption)

            except Exception as exc:
                if _is_quota_error(exc):
                    gemini_cooldown_until = time.monotonic() + GEMINI_COOLDOWN_SECS
                    remaining = GEMINI_COOLDOWN_SECS
                    print(f"[frame ai] quota/rate-limit — cooldown {remaining:.0f}s")
                    await send({"type": "status", "message": f"Rate limited. Pausing {int(remaining)}s.", "data": None})
                else:
                    print(f"[frame ai] error: {exc}")
                    await send({"type": "error", "message": str(exc)[:200], "data": None})

    async def respond_submit(wav_bytes: Optional[bytes]) -> None:
        """
        Voice: transcribe first, then navigation commands, else vision+audio hazard Q&A.
        No audio: vision-only on latest cached frame.
        """
        nonlocal gemini_cooldown_until, last_nav_ui_sent, submit_pending
        async with vision_lock:
            submit_pending = False  # we now own the lock — clear the flag
            try:
                await send({"type": "status", "message": "Processing...", "data": None})

                latest_path = frame_cache.get_latest()

                if wav_bytes and len(wav_bytes) > 100:
                    print("[submit] transcribe audio (Gemini)")
                    transcript = await asyncio.get_running_loop().run_in_executor(
                        None, transcribe_wav, wav_bytes
                    )
                    transcript = (transcript or "").strip().strip('"').strip("'")
                    print(f"[submit] transcript → {transcript[:120]!r}")

                    parsed = parse_command(transcript)
                    intent = parsed.get("intent", "UNKNOWN")

                    if intent == "QUERY_NEXT_DIRECTION":
                        async with nav_lock:
                            answer = nav.get_current_instruction()
                            last_nav_ui_sent = "\n"
                            await push_nav_ui_if_changed()
                        await send_caption_and_audio(answer)
                        return

                    if intent == "NAVIGATE":
                        dest = (parsed.get("destination") or "").strip()
                        if not dest:
                            await send_caption_and_audio(
                                "I did not catch a destination. Try saying navigate to, followed by the place name."
                            )
                            return
                        if last_gps is None:
                            await send_caption_and_audio(
                                "I need your GPS location to navigate. "
                                "Enable location for this app, wait a few seconds, then try again."
                            )
                            return
                        if not (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip():
                            await send_caption_and_audio(
                                "Navigation is not configured on the server. Missing GOOGLE_MAPS_API_KEY."
                            )
                            return

                        lat0, lon0 = last_gps

                        def _start() -> None:
                            nav.start_navigation(dest, (lat0, lon0))

                        first_instr = ""
                        nav_err: Optional[str] = None
                        async with nav_lock:
                            try:
                                await asyncio.get_running_loop().run_in_executor(None, _start)
                            except Exception as exc:
                                print(f"[nav] start failed: {exc}")
                                nav_err = f"I could not build a walking route. {str(exc)[:120]}"
                            if nav_err is None and not nav.steps:
                                nav_err = "No walking steps were returned for that destination."
                            if nav_err is None:
                                nav.mark_intro_announced()
                                first_instr = nav.steps[0].instruction
                                last_nav_ui_sent = "\n"
                                await push_nav_ui_if_changed()

                        if nav_err is not None:
                            await send_caption_and_audio(nav_err)
                            return

                        ack = f"Okay, navigating to {dest}. First, {first_instr}."
                        await send_caption_and_audio(ack)
                        return

                    # Default: scene question — needs a frame
                    if curr_pil is None:
                        await send({"type": "error", "message": "No frames received yet.", "data": None})
                        return
                    print("[submit] Gemini vision+audio (hazard Q&A)")
                    caption = await _gemini_vision_audio(curr_pil, wav_bytes, _AUDIO_PROMPT)
                    await send_caption_and_audio(caption)
                    return

                if latest_path:
                    print(f"[submit] vision-only call on {latest_path}")
                    caption = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: analyze_image(str(latest_path))
                    )
                    if not caption:
                        print("[submit] VLM throttled or returned empty")
                        await send({"type": "error", "message": "Too soon — try again in a moment.", "data": None})
                        return
                    await send_caption_and_audio(caption)
                    return

                await send({"type": "error", "message": "No frames received yet.", "data": None})

            except Exception as exc:
                if _is_quota_error(exc):
                    nonlocal _gemini_cooldown_secs
                    gemini_cooldown_until = time.monotonic() + _gemini_cooldown_secs
                    remaining = _gemini_cooldown_secs
                    _gemini_cooldown_secs = min(_gemini_cooldown_secs * 2, 600)
                    print(f"[submit ai] quota/rate-limit — cooldown {remaining:.0f}s")
                    await send({"type": "error", "message": f"Rate limited. Try again in {int(remaining)}s.", "data": None})
                else:
                    print(f"[submit ai] error: {exc}")
                    await send({"type": "error", "message": str(exc)[:200], "data": None})

    # -------------------------------------------------------------------------
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            # ------------------------------------------------------------------
            if msg_type == "frame":
                b64 = msg.get("jpegBase64", "")
                if not b64:
                    continue

                frame_count += 1
                jpeg_bytes = base64.b64decode(b64)
                curr_pil   = _jpeg_bytes_to_pil(jpeg_bytes)   # always keep freshest

                now = time.monotonic()
                cooldown_left = gemini_cooldown_until - now
                lock_held     = vision_lock.locked()

                print(
                    f"[frame] #{frame_count}  {len(b64)} chars"
                    f"  lock={'held' if lock_held else 'free'}"
                    + (f"  cooldown={cooldown_left:.0f}s" if cooldown_left > 0 else "")
                )

                if cooldown_left > 0:
                    print(f"[frame] quota cooldown active ({cooldown_left:.0f}s left) — dropping frame")
                elif lock_held:
                    print("[frame] vision lock held — dropping frame")
                else:
                    asyncio.create_task(respond_frame(jpeg_bytes))

            # ------------------------------------------------------------------
            elif msg_type == "audio":
                b64 = msg.get("pcmBase64", "")
                if b64:
                    chunk = base64.b64decode(b64)
                    audio_buffer.extend(chunk)
                    print(f"[audio] +{len(chunk)} bytes  total={len(audio_buffer)}")
                audio_sample_rate = msg.get("sampleRate", 44100)

            # ------------------------------------------------------------------
            elif msg_type == "submit":
                print(f"[submit] frame={'yes' if curr_pil is not None else 'no'}  audio={len(audio_buffer)} bytes")

                cooldown_left = gemini_cooldown_until - time.monotonic()
                if cooldown_left > 0:
                    print(f"[submit] quota cooldown active ({cooldown_left:.0f}s left) — dropping submit")
                    await send({"type": "status", "message": f"Rate limited. Try again in {int(cooldown_left)}s.", "data": None})
                    audio_buffer.clear()
                    continue

                # Signal any running frame analysis to bail so we get the lock fast
                submit_pending = True

                wav_bytes: Optional[bytes] = None
                if len(audio_buffer) > 0:
                    wav_bytes = _pcm_float32_to_wav(bytes(audio_buffer), audio_sample_rate)
                    print(f"[submit] converted {len(audio_buffer)} PCM bytes → {len(wav_bytes)} WAV bytes")
                    audio_buffer.clear()

                asyncio.create_task(respond_submit(wav_bytes))

            # ------------------------------------------------------------------
            elif msg_type == "location":
                try:
                    lat = float(msg.get("lat"))
                    lon = float(msg.get("lon"))
                except (TypeError, ValueError):
                    continue
                asyncio.create_task(handle_gps_tick(lat, lon))

    except WebSocketDisconnect:
        print(f"[ws] disconnected  frames={frame_count}")
    except Exception as exc:
        print(f"[ws] unexpected error: {exc}")
    finally:
        frame_cache.clear()
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[ws] cache cleaned up: {cache_dir}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
