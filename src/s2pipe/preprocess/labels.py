from __future__ import annotations

from typing import Any

import numpy as np

from .cfg import LabelConfig
from .raster import Raster, RasterGrid
from .resample import resample_raster


def _ensure_2d_label_array(a: Any) -> np.ndarray:
    """Ensure labels are a 2D array (H, W).

    Accepts:
      - (H, W)
      - (1, H, W)
      - (H, W, 1)
    """
    arr = np.asarray(a)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
    raise ValueError(
        f"Expected label array with shape (H,W) or singleton channel, got {arr.shape}."
    )


def _infer_label_dtype(mapping: dict[int, int] | None, ignore_index: int) -> np.dtype:
    """Infer a compact integer dtype for output labels."""
    max_val = int(ignore_index)
    if mapping:
        max_map = max(int(v) for v in mapping.values()) if mapping else 0
        max_val = max(max_val, max_map)

    if 0 <= max_val <= 255:
        return np.dtype(np.uint8)
    if 0 <= max_val <= 65535:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


def _sparse_counts_dict(counts: np.ndarray) -> dict[str, int]:
    """Convert dense counts to sparse dict with string keys (JSON-friendly)."""
    nz = np.nonzero(counts)[0]
    return {str(int(i)): int(counts[i]) for i in nz}


def _sparse_pct_dict(counts: np.ndarray, total: int) -> dict[str, float]:
    """Convert dense counts to sparse pct dict (JSON-friendly)."""
    if total <= 0:
        return {}
    nz = np.nonzero(counts)[0]
    tot = float(total)
    return {str(int(i)): float(counts[i]) / tot for i in nz}


def _hist_counts_fixed_range(a: np.ndarray, value_range: int) -> np.ndarray:
    """Fast histogram with fixed minlength (assumes values in [0, value_range-1])."""
    return np.bincount(a.ravel(), minlength=int(value_range))


def _apply_mapping_lut(
    a: np.ndarray,
    mapping: dict[int, int],
    *,
    ignore_index: int,
    src_value_range: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Apply mapping via LUT of fixed length `src_value_range`.

    Fast path when ignore_index < src_value_range (default SCL: 255 < 256):
        out = lut[a]   (expects values within range; config guarantees this)

    Safe path when ignore_index >= src_value_range:
        out filled with ignore; only in-range values are LUT-mapped.
    """
    if a.ndim != 2:
        raise ValueError(f"Expected 2D label array, got ndim={a.ndim}.")
    if src_value_range <= 0:
        raise ValueError(f"src_value_range must be > 0, got {src_value_range}.")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"Expected integer label dtype, got {a.dtype}.")

    lut = np.full((int(src_value_range),), int(ignore_index), dtype=out_dtype)
    for k, v in mapping.items():
        kk = int(k)
        if kk < 0 or kk >= src_value_range:
            raise ValueError(
                f"Mapping key {kk} is outside src_value_range={src_value_range}. "
                "Increase cfg.src_value_range or fix the mapping."
            )
        lut[kk] = int(v)

    if int(ignore_index) < int(src_value_range):
        # Fast path (SCL default)
        return lut[a]

    # Safe path
    out = np.full(a.shape, int(ignore_index), dtype=out_dtype)
    valid = (a >= 0) & (a < int(src_value_range))
    if np.any(valid):
        out[valid] = lut[a[valid]]
    return out


def _label_hist_len_for_dtype(
    out_dtype: np.dtype, mapping: dict[int, int] | None, ignore_index: int
) -> int:
    """Pick a histogram length for final labels without scanning y for max."""
    if out_dtype == np.dtype(np.uint8):
        return 256
    if out_dtype == np.dtype(np.uint16):
        return 65536
    max_val = int(ignore_index)
    if mapping:
        max_val = max(max_val, max(int(v) for v in mapping.values()))
    return max_val + 1


def resample_labels_with_meta(
    *,
    label_raster: Raster,
    dst_grid: RasterGrid,
    cfg: LabelConfig,
) -> tuple[Raster, dict]:
    """Convert label_raster to dst_grid, and return per-sample label metadata.

    Metadata includes:
      - valid_pixel_ratio, valid_pixel_count, total_pixel_count
      - label_counts/label_pct (final labels after mapping)
    """
    if cfg.resample != "nearest":
        raise ValueError(f"Label resampling must be 'nearest', got {cfg.resample!r}.")

    label_raster_2d = _ensure_2d_label_array(label_raster.array)
    label_raster_single = Raster(
        array=label_raster_2d,
        grid=label_raster.grid,
        nodata=label_raster.nodata,
        band_names=(label_raster.band_names[:1] if label_raster.band_names else None),
    )

    # Nearest for categorical data. Pixels outside coverage become ignore_index.
    label_raster_rs = resample_raster(
        label_raster_single,
        dst_grid,
        method="nearest",
        dst_nodata=int(cfg.ignore_index),
    )
    label_raster_on_target = _ensure_2d_label_array(label_raster_rs.array)

    out_dtype = _infer_label_dtype(cfg.mapping, int(cfg.ignore_index))

    if cfg.mapping is None:
        y = label_raster_on_target.astype(out_dtype, copy=False)
    else:
        y = _apply_mapping_lut(
            label_raster_on_target,
            cfg.mapping,
            ignore_index=int(cfg.ignore_index),
            src_value_range=int(cfg.src_value_range),
            out_dtype=out_dtype,
        )

    total_pixels = int(y.size)
    valid_pixels = int(np.count_nonzero(y != int(cfg.ignore_index)))
    valid_pixel_ratio = (
        float(valid_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
    )

    # Histograms: fixed-length where possible, without extra scans/conversions.
    label_hist_len = _label_hist_len_for_dtype(
        out_dtype, cfg.mapping, int(cfg.ignore_index)
    )
    label_counts = np.bincount(y.ravel(), minlength=int(label_hist_len)).astype(
        np.int64, copy=False
    )

    meta = {
        "ignore_index": int(cfg.ignore_index),
        "mapping": cfg.mapping,
        "src_value_range": int(cfg.src_value_range),
        "valid_pixel_ratio": valid_pixel_ratio,
        "valid_pixel_count": valid_pixels,
        "total_pixel_count": total_pixels,
        "label_counts": _sparse_counts_dict(label_counts),
        "label_pct": _sparse_pct_dict(label_counts, total_pixels),
    }

    raster_y = Raster(
        array=y,
        grid=dst_grid,
        nodata=int(cfg.ignore_index),
        band_names=["label"],
    )
    return raster_y, meta
