"""
Minimal Gemini vision call for MVP testing: one local image + one text prompt → short text.

Uses the Google Gen AI Python SDK (``google-genai``), not the deprecated package.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL = "gemini-2.5-flash"

DEFAULT_PROMPT = ('''
    You are assisting a blind or low-vision user who is walking.

    Your job is to identify only the most important navigation-relevant obstacles or hazards in this scene.

    Prioritize:
    1. obstacles directly ahead in the walking path
    2. stairs, step-downs, curbs, ledges, drop-offs
    3. people, cars, bikes, or moving objects nearby
    4. doors, poles, chairs, tables, boxes, or objects that may be bumped into
    5. crosswalk state or traffic-related hazards if visible

    Keep the response short and practical.
    Mention only the main hazards or obstacles.
    When you mention a hazard, include a brief location such as: ahead, left, right, lower left, lower right.

    Do not speculate.'''
)

# Total request timeout (ms) for vision calls — generous for large JPEGs on slow links
_DEFAULT_TIMEOUT_MS = 120_000
_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 1.0


def _resolve_image_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        here = Path(__file__).resolve().parent
        candidate = here / path
        if candidate.is_file():
            return candidate.resolve()
    return path


def _get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("Missing GEMINI_API_KEY in environment (.env or shell)")
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=_DEFAULT_TIMEOUT_MS),
    )


def analyze_image(
    image_path: str | Path,
    prompt: str,
    *,
    model: str = MODEL,
    max_retries: int = _MAX_RETRIES,
) -> str:
    """
    Send a local image and text prompt to Gemini; return stripped text or raise.

    Retries transient failures a few times with a short delay between attempts.
    """
    path = _resolve_image_path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    client = _get_client()  # fail fast on missing API key (no retry)

    for attempt in range(max(1, max_retries)):
        try:
            # PIL load; SDK accepts PIL Image in contents list
            with Image.open(path) as im:
                image = im.convert("RGB")

                # Resize for faster upload + inference
                image.thumbnail((768, 768))  # keeps aspect ratio

            response = client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                )
            )

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ValueError("Empty or missing response text from model")

            return text

        except FileNotFoundError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(_RETRY_DELAY_SEC)
            else:
                raise RuntimeError(
                    f"Gemini request failed after {max_retries} attempt(s): {exc}"
                ) from exc


def main() -> None:
    import sys

    # Default image path; override with first CLI arg (e.g. IMG_5271.JPG)
    image_arg = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    prompt = DEFAULT_PROMPT

    print(f"model: {MODEL}")
    print(f"image: {image_arg}")
    print("---")

    t0 = time.perf_counter()
    try:
        result = analyze_image(image_arg, prompt)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"latency_ms: {elapsed_ms:.1f}")
        print(f"error: {e}")
        raise SystemExit(1) from e

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"latency_ms: {elapsed_ms:.1f}")
    print("---")
    print(result)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Quick reference (run)
# -----------------------------------------------------------------------------

# Run from repo root (use any local image path):
#   python gemini_vision_test.py
#   python gemini_vision_test.py IMG_5271.JPG
#
