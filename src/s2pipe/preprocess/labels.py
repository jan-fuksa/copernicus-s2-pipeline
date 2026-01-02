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


def _apply_mapping_lut(
    a: np.ndarray,
    mapping: dict[int, int],
    *,
    ignore_index: int,
    src_value_range: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Apply mapping via LUT of fixed length `src_value_range`.

    Assumptions:
      - source labels are integers in [0, src_value_range-1]
      - unknown values (not present in mapping) become ignore_index (LUT default)

    This is intentionally optimized for SCL (uint8, range 256) and avoids:
      - a.max()
      - int64 conversion
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

    # Fast path: direct LUT indexing (expects values within range).
    # If config is wrong and values exceed the LUT size, NumPy will raise IndexError.
    return lut[a]


def scl_to_labels(
    *,
    scl: Raster,
    dst_grid: RasterGrid,
    cfg: LabelConfig,
) -> Raster:
    """Convert Sentinel-2 SCL raster to training labels on dst_grid.

    Steps:
      1) Resample SCL to dst_grid using nearest (categorical).
      2) Optionally apply cfg.mapping via LUT; unknown -> ignore_index.
      3) Output is single-band label raster as 2D (H, W).
    """
    if cfg.resample != "nearest":
        raise ValueError(f"Label resampling must be 'nearest', got {cfg.resample!r}.")

    scl_2d = _ensure_2d_label_array(scl.array)
    scl_single = Raster(
        array=scl_2d,
        grid=scl.grid,
        nodata=scl.nodata,
        band_names=(scl.band_names[:1] if scl.band_names else None),
    )

    # Nearest is required for categorical labels.
    scl_rs = resample_raster(
        scl_single,
        dst_grid,
        method="nearest",
        dst_nodata=int(cfg.ignore_index),
    )
    a = _ensure_2d_label_array(scl_rs.array)

    out_dtype = _infer_label_dtype(cfg.mapping, int(cfg.ignore_index))

    if cfg.mapping is None:
        y = a.astype(out_dtype, copy=False)
    else:
        y = _apply_mapping_lut(
            a,
            cfg.mapping,
            ignore_index=int(cfg.ignore_index),
            src_value_range=int(cfg.src_value_range),
            out_dtype=out_dtype,
        )

    return Raster(
        array=y,
        grid=dst_grid,
        nodata=int(cfg.ignore_index),
        band_names=["label"],
    )
