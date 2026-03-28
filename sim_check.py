"""
Scene novelty gate for a walking, handheld phone camera.

Plain SSIM compares fixed pixel locations, which breaks under translation,
rotation, and forward motion. This module *first* estimates dominant camera
motion (sparse KLT tracks + partial affine), warps the previous frame into the
current view, then measures *residual* change. That residual is a much better
proxy for “new stuff appeared” than raw frame similarity.

Primary API: :func:`scene_change_gate` → :class:`SceneChangeResult`.

Secondary / fallback: :func:`legacy_ssim_similarity` (raw SSIM, no motion comp).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as sk_ssim
from skimage.transform import resize as sk_resize

ImageInput = Union[str, Path, np.ndarray, Image.Image]

# ---------------------------------------------------------------------------
# Tunable defaults — adjust these first when calibrating on real walking video
# ---------------------------------------------------------------------------

@dataclass
class SceneChangeConfig:
    """All thresholds for :func:`scene_change_gate`."""

    max_dim: int = 320
    blur_ksize: int = 5  # Gaussian; must be odd; 0 to disable
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 3  # goodFeaturesToTrack

    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_max_error: float = 50.0  # drop tracks with LK error above this

    min_tracked_points: int = 25
    min_inlier_ratio: float = 0.35
    min_inlier_count: int = 8
    affine_ransac_threshold: float = 3.0
    affine_max_iters: int = 2000
    confidence_inlier_floor: float = 0.15  # below this → uncertain path

    use_full_affine: bool = False
    use_homography: bool = False  # force homography instead of affine for the whole pipeline

    # If partial/full affine has low RANSAC inliers (common when walking forward —
    # perspective zoom), try homography before giving up. Then residual novelty
    # + aligned SSIM can still run on a decent warp.
    fallback_homography_if_affine_inliers_low: bool = True
    # Homography fallback uses its own (looser) gates — stricter affine bar often
    # rejects forward-motion pairs where H still fits most tracks if RANSAC is
    # allowed more reprojection slack.
    homography_fallback_reproj_threshold: float = 8.0  # 2nd try; 1st uses affine_ransac_threshold
    homography_fallback_min_inlier_ratio: float = 0.22
    homography_fallback_min_inlier_count: int = 6

    diff_threshold: int = 32  # higher → fewer false “changed” pixels from slight misalignment
    morph_kernel_size: int = 3
    morph_open_iters: int = 1
    morph_close_iters: int = 1

    changed_pixel_trigger: float = 0.10
    largest_blob_trigger: float = 0.028
    center_changed_trigger: float = 0.055

    aligned_ssim_same_scene: float = 0.90
    compute_aligned_ssim: bool = True

    # Lower-center ROI: path / hazards often enter here
    center_roi_y0_frac: float = 0.45
    center_roi_y1_frac: float = 1.0
    center_roi_x0_frac: float = 0.20
    center_roi_x1_frac: float = 0.80

    min_working_side: int = 32  # below this after resize → uncertain

    # Below this inlier fraction, ignore residual diff/blob triggers (weak H /
    # affine → huge false “change”). None = always trust residual. ~0.30 works
    # for handheld zoom; set None to recover older recall-heavy behavior.
    min_inlier_ratio_to_trust_residual: float | None = 0.31

    bgr_input: bool = False


DEFAULT_SCENE_CONFIG = SceneChangeConfig()


@dataclass
class SceneChangeResult:
    novelty_score: float
    should_trigger: bool
    confidence: float
    changed_pixel_ratio: float
    largest_changed_blob_ratio: float
    center_changed_ratio: float
    aligned_ssim: float | None
    track_count: int
    inlier_ratio: float | None
    transform_found: bool
    latency_ms: float
    debug_reason: str
    # Optional extras for debugging / logging
    extras: dict[str, Any] = field(default_factory=dict)


# --- Legacy SSIM path (no motion compensation) --------------------------------

_DEFAULT_LEGACY_MAX_DIM = 512
_CENTER_KEEP_W = 0.72
_CENTER_KEEP_H = 0.78


def _load_array(image: ImageInput) -> np.ndarray:
    """Load to H×W or H×W×C uint8 (RGB order for color paths)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError("numpy inputs must be uint8 (0–255).")
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] in (3, 4):
            return image[..., :3]
        raise ValueError("numpy array must be H×W or H×W×3 (or 4).")
    path = Path(image)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _to_gray_uint8(arr: np.ndarray, *, bgr: bool = False) -> np.ndarray:
    """RGB/BGR or gray → uint8 grayscale (H,W)."""
    if arr.ndim == 2:
        return arr
    if bgr:
        return cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    rgb = arr[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _resize_max_edge_u8(gray: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray
    scale = max_dim / float(longest)
    nh = max(8, int(round(h * scale)))
    nw = max(8, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def _preprocess_for_gate(
    image: ImageInput,
    cfg: SceneChangeConfig,
) -> np.ndarray:
    """Resize + grayscale + optional blur. All gate math runs on this."""
    arr = _load_array(image)
    g = _to_gray_uint8(arr, bgr=cfg.bgr_input)
    g = _resize_max_edge_u8(g, cfg.max_dim)
    if cfg.blur_ksize and cfg.blur_ksize >= 3:
        k = cfg.blur_ksize
        if k % 2 == 0:
            k += 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    return g


def _center_crop_mask(shape: tuple[int, int], cfg: SceneChangeConfig) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * cfg.center_roi_y0_frac)
    y1 = int(h * cfg.center_roi_y1_frac)
    x0 = int(w * cfg.center_roi_x0_frac)
    x1 = int(w * cfg.center_roi_x1_frac)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = 1
    return m


def _novelty_from_diff(
    aligned_prev: np.ndarray,
    curr: np.ndarray,
    cfg: SceneChangeConfig,
) -> tuple[np.ndarray, float, float, float]:
    """Absolute diff after alignment → binary mask + ratios."""
    d = cv2.absdiff(aligned_prev, curr)
    mask = (d > cfg.diff_threshold).astype(np.uint8) * 255

    k = max(3, cfg.morph_kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if cfg.morph_open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iters)
    if cfg.morph_close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iters)

    bin01 = (mask > 0).astype(np.float32)
    h, w = bin01.shape
    total = float(h * w)
    changed_pixel_ratio = float(bin01.mean())

    largest_ratio = 0.0
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        largest_ratio = max(largest_ratio, area / total)

    roi = _center_crop_mask((h, w), cfg)
    roi_pixels = float(roi.sum()) + 1e-6
    center_changed_ratio = float((bin01 * roi).sum() / roi_pixels)

    return mask, changed_pixel_ratio, largest_ratio, center_changed_ratio


def _fuse_novelty_score(
    changed_pixel_ratio: float,
    largest_blob_ratio: float,
    center_changed_ratio: float,
    cfg: SceneChangeConfig,
) -> float:
    """Map raw ratios to [0,1]; higher = more residual change vs triggers."""
    a = changed_pixel_ratio / max(cfg.changed_pixel_trigger, 1e-6)
    b = largest_blob_ratio / max(cfg.largest_blob_trigger, 1e-6)
    c = center_changed_ratio / max(cfg.center_changed_trigger, 1e-6)
    return float(np.clip(max(a, b, c), 0.0, 1.0))


def _aligned_ssim_optional(
    aligned_prev: np.ndarray,
    curr: np.ndarray,
    cfg: SceneChangeConfig,
) -> float | None:
    if not cfg.compute_aligned_ssim:
        return None
    a = aligned_prev.astype(np.float64)
    b = curr.astype(np.float64)
    h, w = a.shape[:2]
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return None
    try:
        s = sk_ssim(a, b, data_range=255.0, win_size=win)
        return float(np.clip(s, 0.0, 1.0))
    except Exception:
        return None


def scene_change_gate(
    prev_image: ImageInput,
    curr_image: ImageInput,
    *,
    config: SceneChangeConfig | None = None,
) -> SceneChangeResult:
    """
    Motion-compensated scene novelty gate for consecutive frames.

    Estimates partial affine motion (translation + rotation + uniform scale),
    warps ``prev_image`` toward ``curr_image``, then thresholds |aligned−curr|.
    Biased toward **triggering** when tracking or the motion fit is unreliable.
    """
    cfg = config or DEFAULT_SCENE_CONFIG
    t0 = time.perf_counter()

    prev_g = _preprocess_for_gate(prev_image, cfg)
    curr_g = _preprocess_for_gate(curr_image, cfg)

    h, w = curr_g.shape[:2]
    if min(h, w) < cfg.min_working_side:
        ms = (time.perf_counter() - t0) * 1000.0
        return SceneChangeResult(
            novelty_score=1.0,
            should_trigger=True,
            confidence=0.15,
            changed_pixel_ratio=0.0,
            largest_changed_blob_ratio=0.0,
            center_changed_ratio=0.0,
            aligned_ssim=None,
            track_count=0,
            inlier_ratio=None,
            transform_found=False,
            latency_ms=ms,
            debug_reason="image_too_small_after_preprocess",
        )

    if prev_g.shape != curr_g.shape:
        prev_g = cv2.resize(prev_g, (w, h), interpolation=cv2.INTER_AREA)

    return _scene_change_klt_core(prev_g, curr_g, cfg, t0)


def build_debug_visualization(result: SceneChangeResult) -> np.ndarray | None:
    """
    Horizontal strip: previous (gray) | aligned previous | current | change mask.
    Requires ``result.extras`` from :func:`scene_change_gate` (contains aligned
    frame and mask). Returns BGR uint8 or None if missing.
    """
    ex = result.extras
    prev = ex.get("prev_gray")
    aligned = ex.get("aligned_prev")
    curr = ex.get("curr_gray")
    mask = ex.get("diff_mask")
    if prev is None or aligned is None or curr is None or mask is None:
        return None

    def _to_bgr(g: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    m_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return np.hstack([_to_bgr(prev), _to_bgr(aligned), _to_bgr(curr), m_color])


def scene_change_gate_on_gray(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    config: SceneChangeConfig | None = None,
) -> SceneChangeResult:
    """
    Like :func:`scene_change_gate` but both inputs are already preprocessed
    uint8 grayscale (same shape). Skips decode/resize/blur — use when you cache
    working frames between ticks. For full images/paths use
    :func:`scene_change_gate` instead.
    """
    cfg = config or DEFAULT_SCENE_CONFIG
    if prev_gray.dtype != np.uint8 or curr_gray.dtype != np.uint8:
        raise TypeError("prev_gray and curr_gray must be uint8")
    if prev_gray.ndim != 2 or curr_gray.ndim != 2:
        raise ValueError("expected HxW grayscale arrays")
    t0 = time.perf_counter()
    prev_g = prev_gray.copy()
    curr_g = curr_gray.copy()
    h, w = curr_g.shape[:2]
    if min(h, w) < cfg.min_working_side:
        ms = (time.perf_counter() - t0) * 1000.0
        return SceneChangeResult(
            novelty_score=1.0,
            should_trigger=True,
            confidence=0.15,
            changed_pixel_ratio=0.0,
            largest_changed_blob_ratio=0.0,
            center_changed_ratio=0.0,
            aligned_ssim=None,
            track_count=0,
            inlier_ratio=None,
            transform_found=False,
            latency_ms=ms,
            debug_reason="image_too_small",
        )
    if prev_g.shape != curr_g.shape:
        prev_g = cv2.resize(prev_g, (w, h), interpolation=cv2.INTER_AREA)
    return _scene_change_klt_core(prev_g, curr_g, cfg, t0)


def _scene_change_klt_core(
    prev_g: np.ndarray,
    curr_g: np.ndarray,
    cfg: SceneChangeConfig,
    t0: float,
) -> SceneChangeResult:
    """KLT + affine/homography + residual novelty. ``prev_g``/``curr_g`` same shape uint8."""
    h, w = curr_g.shape[:2]

    p0 = cv2.goodFeaturesToTrack(
        prev_g,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
        blockSize=cfg.block_size,
    )

    if p0 is None or len(p0) < cfg.min_tracked_points:
        ms = (time.perf_counter() - t0) * 1000.0
        return SceneChangeResult(
            novelty_score=1.0,
            should_trigger=True,
            confidence=0.2,
            changed_pixel_ratio=0.0,
            largest_changed_blob_ratio=0.0,
            center_changed_ratio=0.0,
            aligned_ssim=None,
            track_count=0 if p0 is None else len(p0),
            inlier_ratio=None,
            transform_found=False,
            latency_ms=ms,
            debug_reason="insufficient_features_to_track",
        )

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_g,
        curr_g,
        p0,
        None,
        winSize=cfg.lk_win_size,
        maxLevel=cfg.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    if p1 is None or st is None:
        ms = (time.perf_counter() - t0) * 1000.0
        return SceneChangeResult(
            novelty_score=1.0,
            should_trigger=True,
            confidence=0.2,
            changed_pixel_ratio=0.0,
            largest_changed_blob_ratio=0.0,
            center_changed_ratio=0.0,
            aligned_ssim=None,
            track_count=0,
            inlier_ratio=None,
            transform_found=False,
            latency_ms=ms,
            debug_reason="optical_flow_failed",
        )

    st = st.reshape(-1)
    err = err.reshape(-1) if err is not None else np.zeros(len(st))
    good = (st == 1) & (err < cfg.lk_max_error)
    prev_pts = p0.reshape(-1, 2)[good].astype(np.float32)
    curr_pts = p1.reshape(-1, 2)[good].astype(np.float32)
    track_count = int(len(prev_pts))

    if track_count < cfg.min_tracked_points:
        ms = (time.perf_counter() - t0) * 1000.0
        return SceneChangeResult(
            novelty_score=1.0,
            should_trigger=True,
            confidence=0.25,
            changed_pixel_ratio=0.0,
            largest_changed_blob_ratio=0.0,
            center_changed_ratio=0.0,
            aligned_ssim=None,
            track_count=track_count,
            inlier_ratio=None,
            transform_found=False,
            latency_ms=ms,
            debug_reason="too_few_tracks_after_lk",
        )

    method = cv2.RANSAC
    ransac_reproj = cfg.affine_ransac_threshold
    max_iters = cfg.affine_max_iters
    inlier_ratio: float | None = None
    align_meta: dict[str, Any] = {}

    if cfg.use_homography:
        H, mask_h = cv2.findHomography(
            prev_pts,
            curr_pts,
            method=method,
            ransacReprojThreshold=ransac_reproj,
            maxIters=max_iters,
        )
        if H is None or mask_h is None:
            ms = (time.perf_counter() - t0) * 1000.0
            return SceneChangeResult(
                novelty_score=1.0,
                should_trigger=True,
                confidence=0.2,
                changed_pixel_ratio=0.0,
                largest_changed_blob_ratio=0.0,
                center_changed_ratio=0.0,
                aligned_ssim=None,
                track_count=track_count,
                inlier_ratio=None,
                transform_found=False,
                latency_ms=ms,
                debug_reason="homography_estimation_failed",
            )
        inl = mask_h.ravel().astype(bool)
        inlier_ratio = float(inl.sum()) / float(len(inl))
        if inl.sum() < cfg.min_inlier_count or inlier_ratio < cfg.min_inlier_ratio:
            ms = (time.perf_counter() - t0) * 1000.0
            return SceneChangeResult(
                novelty_score=1.0,
                should_trigger=True,
                confidence=float(np.clip(inlier_ratio, 0.1, 0.5)),
                changed_pixel_ratio=0.0,
                largest_changed_blob_ratio=0.0,
                center_changed_ratio=0.0,
                aligned_ssim=None,
                track_count=track_count,
                inlier_ratio=inlier_ratio,
                transform_found=True,
                latency_ms=ms,
                debug_reason="low_inlier_ratio_homography",
                extras={"inlier_count": int(inl.sum())},
            )
        H_inv = np.linalg.inv(H)
        aligned = cv2.warpPerspective(
            prev_g, H_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        align_meta["align_model"] = "homography"
    else:
        est = cv2.estimateAffine2D if cfg.use_full_affine else cv2.estimateAffinePartial2D
        M_fwd, inliers = est(
            prev_pts,
            curr_pts,
            method=method,
            ransacReprojThreshold=ransac_reproj,
            maxIters=max_iters,
        )
        if M_fwd is None or inliers is None:
            ms = (time.perf_counter() - t0) * 1000.0
            return SceneChangeResult(
                novelty_score=1.0,
                should_trigger=True,
                confidence=0.2,
                changed_pixel_ratio=0.0,
                largest_changed_blob_ratio=0.0,
                center_changed_ratio=0.0,
                aligned_ssim=None,
                track_count=track_count,
                inlier_ratio=None,
                transform_found=False,
                latency_ms=ms,
                debug_reason="affine_estimation_failed",
            )
        inl = inliers.ravel().astype(bool)
        inlier_ratio = float(inl.sum()) / float(len(inl))
        aligned = None

        if inl.sum() >= cfg.min_inlier_count and inlier_ratio >= cfg.min_inlier_ratio:
            M_inv = cv2.invertAffineTransform(M_fwd)
            aligned = cv2.warpAffine(
                prev_g, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            align_meta["align_model"] = "affine"
        elif cfg.fallback_homography_if_affine_inliers_low and not cfg.use_homography:
            hb_min_n = cfg.homography_fallback_min_inlier_count
            hb_min_r = cfg.homography_fallback_min_inlier_ratio
            r1 = float(ransac_reproj)
            r2 = max(r1 * 2.5, float(cfg.homography_fallback_reproj_threshold))
            r3 = max(r1 * 4.0, 12.0)
            reproj_tries: list[float] = []
            for x in (r1, r2, r3):
                if not any(abs(x - y) < 1e-6 for y in reproj_tries):
                    reproj_tries.append(x)

            for reproj in reproj_tries:
                H, mask_h = cv2.findHomography(
                    prev_pts,
                    curr_pts,
                    method=method,
                    ransacReprojThreshold=reproj,
                    maxIters=max_iters,
                )
                if H is None or mask_h is None:
                    continue
                inh = mask_h.ravel().astype(bool)
                n_in = int(inh.sum())
                hr = float(n_in) / float(len(inh))
                if n_in >= hb_min_n and hr >= hb_min_r:
                    inlier_ratio = hr
                    H_inv = np.linalg.inv(H)
                    aligned = cv2.warpPerspective(
                        prev_g,
                        H_inv,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    align_meta["align_model"] = "homography_fallback"
                    align_meta["homography_ransac_reproj"] = reproj
                    break

        if aligned is None:
            ms = (time.perf_counter() - t0) * 1000.0
            return SceneChangeResult(
                novelty_score=1.0,
                should_trigger=True,
                confidence=float(np.clip(inlier_ratio, 0.1, 0.5)),
                changed_pixel_ratio=0.0,
                largest_changed_blob_ratio=0.0,
                center_changed_ratio=0.0,
                aligned_ssim=None,
                track_count=track_count,
                inlier_ratio=inlier_ratio,
                transform_found=True,
                latency_ms=ms,
                debug_reason="low_inlier_ratio_or_count",
                extras={"inlier_count": int(inl.sum()), **align_meta},
            )

    mask, cpr, lbr, ccr = _novelty_from_diff(aligned, curr_g, cfg)
    novelty = _fuse_novelty_score(cpr, lbr, ccr, cfg)
    aligned_ssim = _aligned_ssim_optional(aligned, curr_g, cfg)

    track_strength = min(1.0, track_count / max(cfg.min_tracked_points * 3, 1))
    inlier_part = inlier_ratio if inlier_ratio is not None else 0.5
    confidence = float(np.clip(0.35 + 0.35 * track_strength + 0.35 * inlier_part, 0.0, 0.98))

    novelty_any = (
        cpr >= cfg.changed_pixel_trigger
        or lbr >= cfg.largest_blob_trigger
        or ccr >= cfg.center_changed_trigger
    )
    residual_trusted = True
    if cfg.min_inlier_ratio_to_trust_residual is not None:
        residual_trusted = inlier_part >= cfg.min_inlier_ratio_to_trust_residual
    novelty_triggers = novelty_any and residual_trusted

    uncertain_geometry = inlier_part < cfg.confidence_inlier_floor

    if uncertain_geometry:
        should_trigger = True
        reason = "low_geometric_confidence_trigger_safe"
        confidence = min(confidence, 0.45)
    elif novelty_triggers:
        should_trigger = True
        reason = "novelty_above_threshold"
    elif novelty_any and not residual_trusted:
        should_trigger = False
        reason = "novelty_ignored_low_inlier_ratio"
        confidence = min(confidence, 0.55)
    elif (
        cfg.compute_aligned_ssim
        and aligned_ssim is not None
        and aligned_ssim >= cfg.aligned_ssim_same_scene
        and novelty < 0.35
    ):
        should_trigger = False
        reason = "stable_aligned_high_ssim"
        confidence = max(confidence, 0.82)
    else:
        should_trigger = False
        reason = "below_novelty_thresholds"

    ms = (time.perf_counter() - t0) * 1000.0
    return SceneChangeResult(
        novelty_score=novelty,
        should_trigger=should_trigger,
        confidence=confidence,
        changed_pixel_ratio=cpr,
        largest_changed_blob_ratio=lbr,
        center_changed_ratio=ccr,
        aligned_ssim=aligned_ssim,
        track_count=track_count,
        inlier_ratio=inlier_ratio,
        transform_found=True,
        latency_ms=ms,
        debug_reason=reason,
        extras={
            **align_meta,
            "diff_mask": mask,
            "prev_gray": prev_g,
            "aligned_prev": aligned,
            "curr_gray": curr_g,
        },
    )


# --- Legacy: raw SSIM (no motion compensation) --------------------------------


def _to_gray_f64(arr: np.ndarray, *, bgr: bool = False) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if bgr:
        b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return (rgb2gray(arr) * 255.0).astype(np.float64)


def _resize_max_edge_f64(gray: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray
    scale = max_dim / float(longest)
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
    out = sk_resize(gray, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    return out.astype(np.float64, copy=False)


def _center_crop(arr: np.ndarray, keep_w: float, keep_h: float) -> np.ndarray:
    if not (0 < keep_w <= 1.0 and 0 < keep_h <= 1.0):
        raise ValueError("keep_w and keep_h must be in (0, 1].")
    h, w = arr.shape[:2]
    nh = min(h, max(2, int(round(h * keep_h))))
    nw = min(w, max(2, int(round(w * keep_w))))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    if arr.ndim == 2:
        return arr[y0 : y0 + nh, x0 : x0 + nw]
    return arr[y0 : y0 + nh, x0 : x0 + nw, :]


def _prepare_gray_legacy(
    image: ImageInput, *, bgr: bool, center_crop: bool, max_dim: int
) -> np.ndarray:
    arr = _load_array(image)
    if center_crop:
        arr = _center_crop(arr, _CENTER_KEEP_W, _CENTER_KEEP_H)
    return _resize_max_edge_f64(_to_gray_f64(arr, bgr=bgr), max_dim)


def legacy_ssim_similarity(
    image_a: ImageInput,
    image_b: ImageInput,
    *,
    max_dim: int = _DEFAULT_LEGACY_MAX_DIM,
    bgr: bool = False,
    center_crop: bool = False,
    log_time: bool = False,
) -> float:
    """
    Raw structural similarity (no motion compensation). Higher = more alike.

    Prefer :func:`scene_change_gate` for handheld walking video; keep this for
    baselines or non-moving cameras.
    """
    if max_dim < 16:
        raise ValueError("max_dim must be at least 16.")

    t0 = time.perf_counter()
    g1 = _prepare_gray_legacy(image_a, bgr=bgr, center_crop=center_crop, max_dim=max_dim)
    g2 = _prepare_gray_legacy(image_b, bgr=bgr, center_crop=center_crop, max_dim=max_dim)

    if g1.shape != g2.shape:
        h1, w1 = g1.shape[:2]
        if g2.shape[:2] != (h1, w1):
            g2 = sk_resize(g2, (h1, w1), preserve_range=True, anti_aliasing=True).astype(
                np.float64, copy=False
            )

    win = min(7, g1.shape[0], g1.shape[1])
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3

    score = sk_ssim(g1, g2, data_range=255.0, win_size=win)
    out = float(np.clip(score, 0.0, 1.0))
    if log_time:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"sim_check: legacy_ssim_similarity took {elapsed_ms:.2f} ms",
            file=sys.stderr,
            flush=True,
        )
    return out


def save_debug_visualization(result: SceneChangeResult, path: str | Path) -> bool:
    """Write :func:`build_debug_visualization` to ``path`` (e.g. .png). Returns False if no viz."""
    vis = build_debug_visualization(result)
    if vis is None:
        return False
    return bool(cv2.imwrite(str(path), vis))
