"""
Rolling disk cache for VLM-bound frames (after ``should_trigger``).

Folder-backed queue: ``frame_000001.jpg``, ``frame_000002.jpg``, …  
On startup, scans the folder **once**, rebuilds an in-memory deque, and sets
``next_id``. Each ``add_image`` saves, appends metadata, then drops the oldest
file when over capacity.

API: :class:`FrameCache` — ``add_image``, ``get_latest``, ``get_latest_n``, ``clear``.

Default cache location: :data:`DEFAULT_CACHE_DIR` — a ``vlm_cache`` folder next to this file
(i.e. project root for a flat layout).
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "vlm_cache"

_FRAME_RE = re.compile(r"^frame_(\d+)\.(jpe?g)$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameRecord:
    """One cached frame: stable id and path on disk."""

    path: Path
    frame_id: int


def _load_bgr_uint8(image: ImageInput, *, bgr: bool = False) -> np.ndarray:
    """HxWx3 uint8 BGR for ``cv2.imwrite``."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError("numpy image must be uint8")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        if image.ndim == 3 and image.shape[2] == 3:
            return image if bgr else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise ValueError("expected HxW or HxWx3 (or 4) uint8 array")
    path = _resolve_image_path(Path(image))
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return arr


def _resolve_image_path(path: Path) -> Path:
    """
    Use path as-is if it exists. If relative and missing from CWD, try the folder
    that contains this module (same place as ``update_image_cache.py``).
    """
    path = Path(path)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = Path(__file__).resolve().parent / path
        if candidate.is_file():
            return candidate.resolve()
    return path


class FrameCache:
    """
    Fixed-capacity queue of frame files under ``cache_dir``.

    * Default ``cache_dir`` is :data:`DEFAULT_CACHE_DIR` (``vlm_cache`` beside this module).
    * Oldest entries sit at the **left** of the internal deque; newest at the **right**.
    * Ordering is by **numeric ``frame_id``**, not filesystem mtimes.
    * ``next_id`` increments in memory; no per-add full-directory scan.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        max_size: int = 3,
        jpeg_quality: int = 85,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.max_size = max_size
        self.jpeg_quality = int(jpeg_quality)
        self._queue: deque[FrameRecord] = deque()
        self._next_id: int = 0
        self._hydrate_from_disk()

    def _hydrate_from_disk(self) -> None:
        """Create folder if needed; load existing ``frame_*.jpg`` into deque; set ``_next_id``."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        found: list[tuple[int, Path]] = []
        for p in self.cache_dir.iterdir():
            if not p.is_file():
                continue
            m = _FRAME_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda t: t[0])
        for fid, path in found:
            self._queue.append(FrameRecord(path=path, frame_id=fid))
        self._next_id = found[-1][0] + 1 if found else 0
        # Disk had more than max_size (e.g. config changed): trim oldest
        while len(self._queue) > self.max_size:
            dropped = self._queue.popleft()
            try:
                dropped.path.unlink(missing_ok=True)
            except TypeError:
                if dropped.path.exists():
                    dropped.path.unlink()

    def add_image(self, image: ImageInput, *, bgr: bool = False) -> FrameRecord:
        """
        Save a new frame, append to the queue, delete oldest on disk if over ``max_size``.
        Returns the record for the file just written.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fid = self._next_id
        self._next_id += 1
        path = self.cache_dir / f"frame_{fid:06d}.jpg"
        bgr_img = _load_bgr_uint8(image, bgr=bgr)
        ok = cv2.imwrite(
            str(path),
            bgr_img,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            self._next_id -= 1
            raise RuntimeError(f"failed to write {path}")

        self._queue.append(FrameRecord(path=path, frame_id=fid))
        while len(self._queue) > self.max_size:
            dropped = self._queue.popleft()
            try:
                dropped.path.unlink(missing_ok=True)
            except TypeError:
                if dropped.path.exists():
                    dropped.path.unlink()
        return FrameRecord(path=path, frame_id=fid)

    def get_latest(self) -> Path | None:
        """Path to the newest cached frame, or ``None`` if empty."""
        if not self._queue:
            return None
        return self._queue[-1].path

    def get_latest_n(self, k: int) -> list[Path]:
        """Up to ``k`` paths, oldest-first among those ``k`` (i.e. chronological order)."""
        if k <= 0:
            return []
        items = list(self._queue)
        return [r.path for r in items[-k:]]

    def clear(self) -> None:
        """
        End-of-session (or manual) reset: delete every frame file currently in the
        deque, empty the deque, and set ``next_id`` back to ``0`` so the next
        ``add_image`` writes ``frame_000001.jpg`` again.
        """
        while self._queue:
            dropped = self._queue.popleft()
            try:
                dropped.path.unlink(missing_ok=True)
            except TypeError:
                if dropped.path.exists():
                    dropped.path.unlink()
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def records_oldest_to_newest(self) -> list[FrameRecord]:
        """Copy of metadata in queue order (oldest → newest)."""
        return list(self._queue)
