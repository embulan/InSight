"""
Frame pipeline: similarity gate → cache update → optional VLM text.

1. Compare the incoming frame to the latest image in ``vlm_cache`` via
   :func:`backend.sim_check.scene_change_gate`. If the cache is empty the
   frame always goes to the VLM.
2. Append the frame to the disk cache via :class:`backend.update_image_cache.FrameCache`.
3. If the gate triggered (or cache was empty), call
   :func:`backend.vlm_with_audio.analyze_image` and return the caption text.
   TTS is intentionally NOT called here — the server generates audio bytes
   from the returned text and streams them to the client.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from PIL import Image

from backend.sim_check import scene_change_gate
from backend.update_image_cache import FrameCache
from backend.vlm_with_audio import analyze_image

logger = logging.getLogger(__name__)

# Module-level cache singleton (used when the caller does not supply their own).
_default_cache: FrameCache | None = None

PipelineImageInput = Union[str, Path, bytes, "Image.Image"]


def get_default_cache() -> FrameCache:
    """Return the module-level :class:`FrameCache` (lazy singleton)."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FrameCache()
    return _default_cache


def reset_default_cache() -> None:
    """Clear and drop the default cache (e.g. on a new WebSocket session)."""
    global _default_cache
    if _default_cache is not None:
        _default_cache.clear()
    _default_cache = None


def _to_pil(image: PipelineImageInput) -> Image.Image | str | Path:
    """Normalise raw JPEG bytes to PIL; pass through paths and PIL images."""
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return image


def run_pipeline(
    image: PipelineImageInput,
    *,
    cache: FrameCache | None = None,
    prompt: str | None = None,
) -> dict:
    """
    Run sim_check, update cache, optionally run VLM and return text.

    Parameters
    ----------
    image
        ``Path``, file-path string, RGB :class:`PIL.Image.Image`, or raw
        JPEG/WebP ``bytes`` (from a WebSocket frame message).
    cache
        Per-connection :class:`FrameCache`.  If ``None`` the module singleton
        from :func:`get_default_cache` is used.
    prompt
        Optional override passed to :func:`~backend.vlm_with_audio.analyze_image`.

    Returns
    -------
    dict with keys:
        ``send_to_vlm`` (bool),
        ``vlm_text``   (str | None — None if sim_check said skip),
        ``cache_path`` (Path),
        ``scene_result`` (SceneChangeResult | None),
        ``frame_id``   (int)
    """
    c   = cache or get_default_cache()
    img = _to_pil(image)

    logger.info("[pipeline] start")

    # ── sim_check ────────────────────────────────────────────────────────────
    latest       = c.get_latest()
    scene_result = None

    if latest is None:
        send_to_vlm = True
        logger.info("[pipeline] sim_check skipped (cache empty) → send=YES")
    else:
        logger.info("[pipeline] sim_check comparing to %s", latest)
        scene_result = scene_change_gate(latest, img)
        send_to_vlm  = scene_result.should_trigger
        logger.info(
            "[pipeline] sim_check send=%s  novelty=%.4f  reason=%s  latency=%.1fms",
            "YES" if send_to_vlm else "NO",
            scene_result.novelty_score,
            scene_result.debug_reason,
            scene_result.latency_ms,
        )

    # ── cache update ─────────────────────────────────────────────────────────
    record = c.add_image(img)
    logger.info(
        "[pipeline] cached frame_id=%s  path=%s  queue_len=%s",
        record.frame_id, record.path, len(c),
    )

    # ── VLM (text only — no TTS here) ────────────────────────────────────────
    vlm_text = None
    if send_to_vlm:
        logger.info("[pipeline] calling VLM on %s", record.path)
        kwargs = {} if prompt is None else {"prompt": prompt}
        vlm_text = analyze_image(str(record.path), **kwargs)
        logger.info("[pipeline] VLM done  len=%s", len(vlm_text or ""))
    else:
        logger.info("[pipeline] VLM skipped (sim_check said NO)")

    logger.info("[pipeline] end")
    return {
        "send_to_vlm":  send_to_vlm,
        "vlm_text":     vlm_text,
        "cache_path":   record.path,
        "scene_result": scene_result,
        "frame_id":     record.frame_id,
    }


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python -m backend.pipeline <image_path>")
        raise SystemExit(1)
    result = run_pipeline(path)
    print("vlm_text:", result["vlm_text"])


if __name__ == "__main__":
    main()
