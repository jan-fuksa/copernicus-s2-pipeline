from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .raster import Raster, RasterGrid


_RESAMPLING_MAP: dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
    "sum": Resampling.sum,
}


def _normalize_method(method: str) -> Resampling:
    m = (method or "").strip().lower()
    if m not in _RESAMPLING_MAP:
        raise ValueError(f"Unsupported resample method={method!r}. Supported: {sorted(_RESAMPLING_MAP.keys())}")
    return _RESAMPLING_MAP[m]


def _as_chw_if_3d(arr: np.ndarray, grid: RasterGrid) -> np.ndarray:
    """If arr is 3D, normalize to (C,H,W). Accepts CHW or HWC."""
    if arr.ndim != 3:
        return arr
    H, W = grid.height, grid.width
    # CHW?
    if arr.shape[1] == H and arr.shape[2] == W:
        return arr
    # HWC?
    if arr.shape[0] == H and arr.shape[1] == W:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Cannot infer channel axis for array shape {arr.shape} and grid (H,W)=({H},{W}).")


def _resample_to_grid(
    *,
    src_array: Any,
    src_grid: RasterGrid,
    dst_grid: RasterGrid,
    method: str,
    src_nodata: float | int | None = None,
    dst_nodata: float | int | None = None,
) -> Any:
    """Warp/resample an array to dst_grid.

    Accepts:
      - 2D (H, W)
      - 3D (C, H, W) or (H, W, C)

    Returns:
      - 2D if input was 2D
      - 3D (C, H, W) if input was 3D
    """
    resampling = _normalize_method(method)

    arr = src_array
    if np.ma.isMaskedArray(arr):
        fill = src_nodata if src_nodata is not None else 0
        arr = arr.filled(fill)

    arr = np.asarray(arr)
    input_was_2d = (arr.ndim == 2)

    if arr.ndim == 3:
        arr = _as_chw_if_3d(arr, src_grid)

    if arr.ndim == 2:
        bands = 1
        src_bands = [arr]
    elif arr.ndim == 3:
        bands = int(arr.shape[0])
        src_bands = [arr[i] for i in range(bands)]
    else:
        raise ValueError(f"Unsupported src_array ndim={arr.ndim}; expected 2 or 3.")

    # For interpolating resampling on integer inputs, prefer float32 output.
    src_dtype = np.asarray(src_bands[0]).dtype
    if resampling in {Resampling.bilinear, Resampling.cubic, Resampling.average} and np.issubdtype(src_dtype, np.integer):
        dst_dtype = np.float32
    else:
        dst_dtype = src_dtype

    out_bands: list[np.ndarray] = []
    for b in range(bands):
        dst = np.empty((dst_grid.height, dst_grid.width), dtype=dst_dtype)
        reproject(
            source=src_bands[b],
            destination=dst,
            src_transform=src_grid.transform,
            src_crs=src_grid.crs,
            dst_transform=dst_grid.transform,
            dst_crs=dst_grid.crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
        out_bands.append(dst)

    if input_was_2d:
        return out_bands[0]
    return np.stack(out_bands, axis=0)  # (C,H,W)


def resample_raster(
    raster: Raster,
    dst_grid: RasterGrid,
    *,
    method: str,
    dst_nodata: float | int | None = None,
) -> Raster:
    """Resample a Raster to dst_grid.

    Output convention:
      - if input is 2D, output remains 2D
      - if input is 3D (CHW or HWC), output is CHW
    """
    a = np.asarray(raster.array)
    if a.ndim == 2:
        out = _resample_to_grid(
            src_array=a,
            src_grid=raster.grid,
            dst_grid=dst_grid,
            method=method,
            src_nodata=raster.nodata,
            dst_nodata=dst_nodata,
        )
        return Raster(
            array=out,  # 2D
            grid=dst_grid,
            nodata=dst_nodata if dst_nodata is not None else raster.nodata,
            band_names=raster.band_names[:1] if raster.band_names else None,
        )

    out = _resample_to_grid(
        src_array=raster.to_chw(),  # normalize to CHW
        src_grid=raster.grid,
        dst_grid=dst_grid,
        method=method,
        src_nodata=raster.nodata,
        dst_nodata=dst_nodata,
    )
    return Raster(
        array=out,  # CHW
        grid=dst_grid,
        nodata=dst_nodata if dst_nodata is not None else raster.nodata,
        band_names=raster.band_names,
    )
