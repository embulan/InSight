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

Outgoing messages to iPhone
----------------------------
  {"type": "caption", "message": "<text>", "data": null, "hazard_level": "critical"|"high"|"medium"|"ambient"}
  {"type": "audio",   "message": null,     "data": "<base64 mp3>"}
  {"type": "status",  "message": "<text>", "data": null}
  {"type": "error",   "message": "<text>", "data": null}

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

from backend.pipeline import run_pipeline
from backend.update_image_cache import FrameCache
from backend.vlm_with_audio import (
    analyze_image, _get_client, MODEL,
    parse_hazard_lines, PRIORITY_ORDER, AMBIENT,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EL_VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"
EL_MODEL_ID  = "eleven_flash_v2_5"
CACHE_MAX_SIZE = 3

# Seconds to pause all Gemini calls after a quota/rate-limit (429) error.
GEMINI_COOLDOWN_SECS: float = 60.0

# Minimum seconds between successive Gemini calls (proactive throttle).
# Phone sends frames at 0.5 s cadence; sim_check + vision_lock naturally
# limit actual Gemini calls to ~1 per Gemini-latency-cycle (2–4 s).
# Raise to 6.0 if 429 rate-limit errors reappear.
MIN_GEMINI_INTERVAL_SEC: float = 0.0


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "quota" in s or "rate" in s


def _highest_priority(vlm_text: str) -> str:
    """Return the single highest-urgency hazard level found in the VLM output.

    Returns one of: "critical", "high", "medium", "ambient".
    If the scene is clear or no structured lines were found, returns "ambient".
    """
    hazards = parse_hazard_lines(vlm_text)
    if not hazards:
        return AMBIENT
    return min(
        (h["priority"] for h in hazards),
        key=lambda p: PRIORITY_ORDER.get(p, 99),
    )

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
    # monotonic timestamp of the last completed Gemini call (proactive throttle)
    last_gemini_call_time: float = 0.0
    # True from listening_start (or submit arrival) until respond_submit acquires the lock,
    # so that any in-flight frame pipeline or TTS is aborted cleanly.
    submit_pending: bool = False

    # Serializes the Gemini / pipeline phase (heavy AI call).
    vision_lock = asyncio.Lock()
    # Serializes TTS synthesis — runs OUTSIDE vision_lock so Gemini can start
    # on the next frame while the previous caption is still being spoken.
    tts_lock = asyncio.Lock()
    # True while the iOS app is playing a submit-response clip.
    # Frame TTS is suppressed during this window so it cannot interrupt the answer.
    submit_audio_active: bool = False
    # Safety expiry — cleared automatically if iOS never sends audio_done (e.g. crash).
    submit_audio_expires: float = 0.0

    # -------------------------------------------------------------------------
    async def send(obj: dict) -> None:
        try:
            await websocket.send_text(json.dumps(obj))
        except Exception as exc:
            print(f"[ws] send error: {exc}")

    async def send_caption_and_audio(caption: str, *, is_submit: bool = False) -> None:
        """Send caption text then TTS audio to the phone.

        Caption text is always sent immediately.
        TTS runs under tts_lock; if TTS is busy the audio is skipped (caption
        text still updates on-screen).
        Frame audio (is_submit=False) is additionally suppressed while a submit
        response is playing so it cannot interrupt the answer.
        """
        nonlocal submit_audio_active, submit_audio_expires

        if submit_pending:
            print("[ai] listening active — skipping stale caption")
            return

        # Auto-expire the submit guard if iOS never acknowledged
        if submit_audio_active and time.monotonic() > submit_audio_expires:
            submit_audio_active = False
            print("[ai] submit_audio_active expired — resuming frame TTS")

        # Frame audio must not interrupt a submit response that is still playing
        if not is_submit and submit_audio_active:
            hazard_level = _highest_priority(caption)
            print(f"[ai] caption [{hazard_level}] → {caption[:100]}  (audio suppressed — submit playing)")
            await send({"type": "caption", "message": caption, "data": None, "hazard_level": hazard_level})
            return

        hazard_level = _highest_priority(caption)
        print(f"[ai] caption [{hazard_level}] → {caption[:100]}")
        await send({"type": "caption", "message": caption, "data": None, "hazard_level": hazard_level})

        if submit_pending:
            print("[ai] listening started — skipping TTS")
            return

        if tts_lock.locked():
            print("[ai] TTS busy — skipping audio for older caption")
            return

        async with tts_lock:
            if submit_pending:
                print("[ai] listening started — skipping TTS")
                return
            print("[ai] TTS...")
            mp3_bytes = await _tts(caption)
            print(f"[ai] audio ready  {len(mp3_bytes)} bytes")
            if submit_pending:
                print("[ai] listening started during TTS — dropping audio")
                return

            if is_submit:
                # Mark submit audio as active; iOS will clear this via audio_done.
                # Safety expiry = generous 60 s in case iOS never confirms.
                submit_audio_active = True
                submit_audio_expires = time.monotonic() + 60.0
                audio_type = "submit_audio"
            else:
                audio_type = "audio"

            await send({
                "type": audio_type,
                "message": None,
                "data": base64.b64encode(mp3_bytes).decode(),
            })

    async def respond_frame(jpeg_bytes: bytes) -> None:
        """
        Phase 1 (under vision_lock): sim_check → cache → Gemini.
        Phase 2 (outside vision_lock): TTS → send audio.
        Releasing the lock before TTS lets the next Gemini call start immediately
        after the current caption is ready, cutting effective cadence from
        (Gemini + TTS) ≈ 6 s down to just Gemini latency ≈ 3 s.
        """
        nonlocal gemini_cooldown_until, last_gemini_call_time
        caption: Optional[str] = None

        # ── Phase 1: Gemini ───────────────────────────────────────────────────
        async with vision_lock:
            if submit_pending:
                print("[frame] listening active — skipping frame pipeline")
                return
            try:
                await send({"type": "status", "message": "Analyzing...", "data": None})
                last_gemini_call_time = time.monotonic()

                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: run_pipeline(jpeg_bytes, cache=frame_cache)
                )

                if submit_pending:
                    print("[frame] listening started mid-pipeline — discarding result")
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

                caption = result["vlm_text"] or None

            except Exception as exc:
                if _is_quota_error(exc):
                    gemini_cooldown_until = time.monotonic() + GEMINI_COOLDOWN_SECS
                    remaining = GEMINI_COOLDOWN_SECS
                    print(f"[frame ai] quota/rate-limit — cooldown {remaining:.0f}s")
                    await send({"type": "status", "message": f"Rate limited. Pausing {int(remaining)}s.", "data": None})
                else:
                    print(f"[frame ai] error: {exc}")
                    await send({"type": "error", "message": str(exc)[:200], "data": None})
                return

        # ── Phase 2: TTS (vision_lock already released) ───────────────────────
        if caption:
            await send_caption_and_audio(caption)

    async def respond_submit(wav_bytes: Optional[bytes]) -> None:
        """
        Phase 1 (under vision_lock): Gemini vision ± audio call.
        Phase 2 (outside vision_lock): TTS → send audio.
        Same lock-split as respond_frame so frame pipeline can resume immediately.
        """
        nonlocal gemini_cooldown_until, last_gemini_call_time, submit_pending
        caption: Optional[str] = None

        # ── Phase 1: Gemini ───────────────────────────────────────────────────
        async with vision_lock:
            submit_pending = False
            last_gemini_call_time = time.monotonic()
            try:
                await send({"type": "status", "message": "Processing...", "data": None})

                latest_path = frame_cache.get_latest()

                if wav_bytes and len(wav_bytes) > 100:
                    if curr_pil is None:
                        await send({"type": "error", "message": "No frames received yet.", "data": None})
                        return
                    print("[submit] Gemini vision+audio call")
                    caption = await _gemini_vision_audio(curr_pil, wav_bytes, _AUDIO_PROMPT)

                elif latest_path:
                    print(f"[submit] vision-only call on {latest_path}")
                    caption = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: analyze_image(str(latest_path))
                    )

                else:
                    await send({"type": "error", "message": "No frames received yet.", "data": None})
                    return

            except Exception as exc:
                if _is_quota_error(exc):
                    gemini_cooldown_until = time.monotonic() + GEMINI_COOLDOWN_SECS
                    remaining = GEMINI_COOLDOWN_SECS
                    print(f"[submit ai] quota/rate-limit — cooldown {remaining:.0f}s")
                    await send({"type": "status", "message": f"Rate limited. Pausing {int(remaining)}s.", "data": None})
                else:
                    print(f"[submit ai] error: {exc}")
                    await send({"type": "error", "message": str(exc)[:200], "data": None})
                return

        # ── Phase 2: TTS (vision_lock already released) ───────────────────────
        if caption:
            await send_caption_and_audio(caption, is_submit=True)

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
                throttle_left = MIN_GEMINI_INTERVAL_SEC - (now - last_gemini_call_time)
                lock_held     = vision_lock.locked()

                print(
                    f"[frame] #{frame_count}  {len(b64)} chars"
                    f"  lock={'held' if lock_held else 'free'}"
                    + (f"  cooldown={cooldown_left:.0f}s" if cooldown_left > 0 else "")
                    + (f"  throttle={throttle_left:.1f}s" if throttle_left > 0 else "")
                )

                if cooldown_left > 0:
                    print(f"[frame] quota cooldown active ({cooldown_left:.0f}s left) — dropping frame")
                elif throttle_left > 0:
                    print(f"[frame] throttle — {throttle_left:.1f}s until next Gemini call allowed")
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
            elif msg_type == "audio_done":
                # iOS finished playing the submit-response clip — frame TTS can resume
                submit_audio_active = False
                print("[audio_done] submit audio finished — frame TTS unblocked")

            # ------------------------------------------------------------------
            elif msg_type == "listening_start":
                # User tapped the mic — immediately cancel any in-flight frame work and TTS
                submit_pending = True
                submit_audio_active = False   # mic open means no audio is playing
                audio_buffer.clear()
                print("[listen] listening_start — submit_pending set, audio buffer cleared")

            # ------------------------------------------------------------------
            elif msg_type == "submit":
                print(f"[submit] frame={'yes' if curr_pil is not None else 'no'}  audio={len(audio_buffer)} bytes")

                cooldown_left = gemini_cooldown_until - time.monotonic()
                if cooldown_left > 0:
                    print(f"[submit] quota cooldown active ({cooldown_left:.0f}s left) — dropping submit")
                    await send({"type": "status", "message": f"Rate limited. Try again in {int(cooldown_left)}s.", "data": None})
                    audio_buffer.clear()
                    submit_pending = False
                    continue

                submit_pending = True   # ensure flag is set even if listening_start was missed
                wav_bytes: Optional[bytes] = None
                if len(audio_buffer) > 0:
                    wav_bytes = _pcm_float32_to_wav(bytes(audio_buffer), audio_sample_rate)
                    print(f"[submit] converted {len(audio_buffer)} PCM bytes → {len(wav_bytes)} WAV bytes")
                    audio_buffer.clear()

                asyncio.create_task(respond_submit(wav_bytes))

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
