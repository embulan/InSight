"""
Frame pipeline: similarity gate → cache update → optional VLM + TTS.

1. Compare the incoming frame to the latest image in ``vlm_cache`` via
   :func:`sim_check.scene_change_gate`. If the cache is empty, the frame is
   always sent to the VLM.
2. Append the frame to the disk cache via :class:`update_image_cache.FrameCache`.
3. If the gate said to send (or cache was empty), run :func:`vlm_with_audio.run`.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from PIL import Image

from sim_check import scene_change_gate
from update_image_cache import FrameCache
import vlm_with_audio

logger = logging.getLogger(__name__)

# Shared cache across calls when the caller does not pass their own instance.
_default_cache: FrameCache | None = None

PipelineImageInput = Union[str, Path, bytes, "Image.Image"]


def get_default_cache() -> FrameCache:
    """Return the module-level :class:`FrameCache` (lazy singleton)."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FrameCache()
    return _default_cache


def reset_default_cache() -> None:
    """Clear and drop the default cache (e.g. new WebSocket session)."""
    global _default_cache
    if _default_cache is not None:
        _default_cache.clear()
    _default_cache = None


def _normalize_image(image: PipelineImageInput) -> Image.Image | str | Path:
    """Accept WebSocket JPEG bytes and normalize to types the gate/cache accept."""
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
    Run sim check vs latest cache entry, always update cache, optionally VLM+TTS.

    Parameters
    ----------
    image
        Path, ``Path``, RGB :class:`PIL.Image.Image`, or JPEG/WebP (etc.) ``bytes``.
    cache
        If ``None``, uses the module singleton from :func:`get_default_cache`.
    prompt
        Optional override passed to :func:`vlm_with_audio.run`.

    Returns
    -------
    dict
        ``send_to_vlm``, ``vlm_text`` (``None`` if skipped), ``cache_path``,
        ``scene_result`` (``None`` if cache was empty), ``frame_id``.
    """
    c = cache or get_default_cache()
    img = _normalize_image(image)

    logger.info("[pipeline] start")

    latest = c.get_latest()
    scene_result = None
    if latest is None:
        send_to_vlm = True
        logger.info(
            "[pipeline] sim_check: skipped (vlm_cache empty) -> send to VLM = YES"
        )
    else:
        logger.info("[pipeline] sim_check: comparing to latest cache file %s", latest)
        scene_result = scene_change_gate(latest, img)
        send_to_vlm = scene_result.should_trigger
        logger.info(
            "[pipeline] sim_check: send to VLM = %s (should_trigger=%s, novelty=%.4f, "
            "reason=%s, latency_ms=%.1f)",
            "YES" if send_to_vlm else "NO",
            scene_result.should_trigger,
            scene_result.novelty_score,
            scene_result.debug_reason,
            scene_result.latency_ms,
        )

    record = c.add_image(img)
    logger.info(
        "[pipeline] cache: added frame id=%s path=%s (queue len=%s)",
        record.frame_id,
        record.path,
        len(c),
    )

    vlm_text = None
    if send_to_vlm:
        logger.info("[pipeline] vlm_with_audio: running on %s", record.path)
        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        vlm_text = vlm_with_audio.run(str(record.path), **kwargs)
        logger.info("[pipeline] vlm_with_audio: done (result length=%s)", len(vlm_text or ""))
    else:
        logger.info("[pipeline] vlm_with_audio: skipped (sim_check said do not send)")

    logger.info("[pipeline] end")
    return {
        "send_to_vlm": send_to_vlm,
        "vlm_text": vlm_text,
        "cache_path": record.path,
        "scene_result": scene_result,
        "frame_id": record.frame_id,
    }


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python pipeline.py <image_path>")
        raise SystemExit(1)
    run_pipeline(path)


if __name__ == "__main__":
    main()
