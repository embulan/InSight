"""
InSight WebSocket server.

Frame cadence
-------------
  iOS Config.frameInterval controls how often the phone sends a frame (default 20 s).
  The server caches EVERY received frame into a per-connection FrameCache on disk.
  scene_change_gate compares the two most-recent cached frames; if it
  triggers and the vision lock is free, a Gemini + TTS pipeline fires.

Incoming messages from iPhone
-------------------------------
  {"type": "frame",  "timestampMs": <int>, "jpegBase64":  "<b64>"}
  {"type": "audio",  "timestampMs": <int>, "sampleRate":  <int>, "pcmBase64": "<b64>"}
  {"type": "submit", "timestampMs": <int>}

Outgoing messages to iPhone
----------------------------
  {"type": "caption", "message": "<text>", "data": null}
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
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from backend.sim_check import scene_change_gate
from backend.update_image_cache import FrameCache

load_dotenv()

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

GEMINI_MODEL   = "gemini-2.5-flash"
EL_VOICE_ID    = "JBFqnCBsd6RMkjVDRZzb"
EL_MODEL_ID    = "eleven_flash_v2_5"

# Keep the last N frames on disk per connection (for sim_check history).
CACHE_MAX_SIZE: int = 3

_VISION_PROMPT = """
You are assisting a blind or low-vision user who is walking.

Identify only the most important navigation-relevant obstacles or hazards in this scene.

Prioritize:
1. Obstacles directly ahead in the walking path
2. Stairs, step-downs, curbs, ledges, drop-offs
3. People, cars, bikes, or moving objects nearby
4. Doors, poles, chairs, tables, boxes that may be bumped into
5. Crosswalk state or traffic-related hazards if visible

Keep the response to 1-3 short sentences. Mention only the main hazards.
Include a brief location: ahead, left, right, lower-left, lower-right.
Do not speculate.
""".strip()

# ---------------------------------------------------------------------------
# Lazy AI client singletons
# ---------------------------------------------------------------------------

_gemini_client = None
_el_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


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


def _prep_image(image: Image.Image) -> Image.Image:
    img = image.copy().convert("RGB")
    img.thumbnail((768, 768))
    return img


# ---------------------------------------------------------------------------
# Blocking AI calls — run in thread pool so the event loop stays free
# ---------------------------------------------------------------------------

def _sync_gemini_vision(image: Image.Image, prompt: str) -> str:
    response = _get_gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, _prep_image(image)],
    )
    return (getattr(response, "text", None) or "").strip()


def _sync_gemini_vision_audio(image: Image.Image, wav_bytes: bytes, prompt: str) -> str:
    from google.genai import types
    audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
    combined = (
        prompt
        + "\n\nThe user has spoken a question — audio is attached. "
        "Answer their question in the context of what you see."
    )
    response = _get_gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=[combined, _prep_image(image), audio_part],
    )
    return (getattr(response, "text", None) or "").strip()


def _sync_tts(text: str) -> bytes:
    return b"".join(
        _get_el().text_to_speech.stream(
            text=text,
            voice_id=EL_VOICE_ID,
            model_id=EL_MODEL_ID,
            output_format="mp3_44100_128",
        )
    )


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

async def _gemini_vision(image: Image.Image, prompt: str) -> str:
    return await asyncio.get_running_loop().run_in_executor(
        None, _sync_gemini_vision, image, prompt
    )


async def _gemini_vision_audio(image: Image.Image, wav_bytes: bytes, prompt: str) -> str:
    return await asyncio.get_running_loop().run_in_executor(
        None, _sync_gemini_vision_audio, image, wav_bytes, prompt
    )


async def _tts(text: str) -> bytes:
    return await asyncio.get_running_loop().run_in_executor(None, _sync_tts, text)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="InSight Server")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Per-connection temp dir so concurrent connections don't share cache files
    cache_dir = tempfile.mkdtemp(prefix="insight_ws_")
    frame_cache = FrameCache(cache_dir=cache_dir, max_size=CACHE_MAX_SIZE)
    print(f"[ws] connected  cache={cache_dir}")

    # Per-connection state
    curr_pil: Optional[Image.Image] = None   # latest decoded frame (always fresh)
    frame_count: int = 0
    audio_buffer = bytearray()
    audio_sample_rate = 44100

    # One AI pipeline at a time per connection
    vision_lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    async def send(obj: dict) -> None:
        try:
            await websocket.send_text(json.dumps(obj))
        except Exception as exc:
            print(f"[ws] send error: {exc}")

    async def respond(image: Image.Image, wav_bytes: Optional[bytes] = None) -> None:
        """Gemini vision (+ optional audio) → TTS → send caption + audio."""
        async with vision_lock:
            try:
                has_audio = bool(wav_bytes and len(wav_bytes) > 100)
                print(f"[ai] Gemini call  has_audio={has_audio}")
                await send({"type": "status", "message": "Processing...", "data": None})

                if has_audio:
                    caption = await _gemini_vision_audio(image, wav_bytes, _VISION_PROMPT)
                else:
                    caption = await _gemini_vision(image, _VISION_PROMPT)

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

            except Exception as exc:
                print(f"[ai] error: {exc}")
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
                curr_pil = _jpeg_bytes_to_pil(base64.b64decode(b64))

                print(f"[frame] #{frame_count}  {len(b64)} chars  lock={'held' if vision_lock.locked() else 'free'}")

                # Always cache every received frame (phone already rate-limits via frameInterval)
                prev_paths = frame_cache.get_latest_n(1)
                frame_cache.add_image(curr_pil)
                print(f"[cache] saved  total={len(frame_cache)}")

                # Don't start a new AI call if one is already running
                if vision_lock.locked():
                    print("[frame] vision lock held — skipping Gemini trigger")
                    continue

                if not prev_paths:
                    # First ever frame — describe the scene immediately
                    print("[frame] first frame → Gemini")
                    asyncio.create_task(respond(curr_pil))
                else:
                    # Run sim_check: compare previous cached frame with this one
                    try:
                        prev_cached_pil = Image.open(prev_paths[0]).convert("RGB")
                        result = scene_change_gate(prev_cached_pil, curr_pil)
                        print(
                            f"[sim_check] novelty={result.novelty_score:.3f}"
                            f"  trigger={result.should_trigger}"
                            f"  reason={result.debug_reason}"
                        )
                        if result.should_trigger:
                            asyncio.create_task(respond(curr_pil))
                        else:
                            print("[sim_check] scene unchanged — skipping Gemini")
                    except Exception as exc:
                        print(f"[sim_check] error: {exc}")

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
                print(f"[submit] frame={curr_pil is not None}  audio={len(audio_buffer)} bytes")
                if curr_pil is None:
                    await send({"type": "error", "message": "No frames received yet.", "data": None})
                    continue

                wav_bytes: Optional[bytes] = None
                if len(audio_buffer) > 0:
                    wav_bytes = _pcm_float32_to_wav(bytes(audio_buffer), audio_sample_rate)
                    print(f"[submit] {len(audio_buffer)} PCM bytes → {len(wav_bytes)} WAV bytes")
                    audio_buffer.clear()

                asyncio.create_task(respond(curr_pil, wav_bytes))

    except WebSocketDisconnect:
        print(f"[ws] disconnected  frames={frame_count}  cache={cache_dir}")
    except Exception as exc:
        print(f"[ws] unexpected error: {exc}")
    finally:
        # Clean up this connection's disk cache
        frame_cache.clear()
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[ws] cache cleaned up: {cache_dir}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
