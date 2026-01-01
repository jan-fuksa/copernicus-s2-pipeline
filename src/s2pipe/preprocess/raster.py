from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterGrid:
    """Definition of a raster grid."""

    crs: str
    transform: Any  # affine.Affine
    width: int
    height: int
    res: tuple[float, float]  # (xres, yres) in CRS units (UTM: meters)

    @property
    def res_m(self) -> float:
        rx, ry = self.res
        return float((abs(rx) + abs(ry)) / 2.0)

    def shape_hw(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class Raster:
    """Raster array with its georeferencing.

    Conventions:
      - single-band: array is (H, W)
      - multi-band:  array is (C, H, W)  [preferred internal layout]
      - (H, W, C) is tolerated but should be normalized via to_chw().
    """

    array: Any
    grid: RasterGrid
    nodata: float | int | None = None
    band_names: list[str] | None = None

    def to_chw(self) -> np.ndarray:
        """Return the raster array as (C, H, W)."""
        a = np.asarray(self.array)
        H, W = self.grid.height, self.grid.width

        if a.ndim == 2:
            if a.shape != (H, W):
                raise ValueError(f"Expected (H,W)=({H},{W}), got {a.shape}.")
            return a[np.newaxis, :, :]  # (1, H, W)

        if a.ndim == 3:
            # CHW?
            if a.shape[1] == H and a.shape[2] == W:
                return a
            # HWC?
            if a.shape[0] == H and a.shape[1] == W:
                return np.transpose(a, (2, 0, 1))  # (H,W,C)->(C,H,W)

            raise ValueError(
                f"Cannot infer channel axis for array shape {a.shape} and grid (H,W)=({H},{W})."
            )

        raise ValueError(f"Unsupported array ndim={a.ndim}; expected 2 or 3.")

    @property
    def nbands(self) -> int:
        a = np.asarray(self.array)
        if a.ndim == 2:
            return 1
        if a.ndim == 3:
            return int(self.to_chw().shape[0])
        raise ValueError(f"Unsupported array ndim={a.ndim}.")


def build_target_grid(*, tile_id: str, target_res_m: int, target_crs: str | None = None) -> RasterGrid:
    """Derive a target CRS/grid for the tile and requested resolution.

    Intentionally not implemented: robust derivation from tile_id alone requires an MGRS/tile-extent utility.

    Preferred approach in this pipeline:
        target_grid = grid_from_reference_raster(<path to 10m ref band, e.g. B02>)
    """
    raise NotImplementedError(
        "build_target_grid() is intentionally not implemented. "
        "Use grid_from_reference_raster(<path to a 10m reference band>, e.g. B02)."
    )


def grid_from_reference_raster(path: Path) -> RasterGrid:
    """Build a RasterGrid from a reference raster file (recommended approach)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with rasterio.open(p) as ds:
        if ds.crs is None:
            raise ValueError(f"Reference raster has no CRS: {p}")

        resx, resy = ds.res
        return RasterGrid(
            crs=ds.crs.to_string(),
            transform=ds.transform,
            width=int(ds.width),
            height=int(ds.height),
            res=(float(resx), float(resy)),
        )


def read_raster(path: Path) -> Raster:
    """Read a raster file (JP2/GeoTIFF) and return Raster(array, grid, nodata).

    Returns:
      - 2D array (H, W) if the raster has a single band
      - 3D array (C, H, W) if it has multiple bands
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with rasterio.open(p) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {p}")

        resx, resy = ds.res
        grid = RasterGrid(
            crs=ds.crs.to_string(),
            transform=ds.transform,
            width=int(ds.width),
            height=int(ds.height),
            res=(float(resx), float(resy)),
        )

        nodata = ds.nodata
        if ds.count == 1:
            arr = ds.read(1)   # (H, W)
        else:
            arr = ds.read()    # (C, H, W)

    return Raster(array=arr, grid=grid, nodata=nodata)


def write_geotiff(
    path: Path,
    raster: Raster,
    *,
    nodata: float | int | None = None,
    dtype: str | None = None,
    compress: str = "deflate",
) -> None:
    """Write Raster to GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = raster.grid
    arr_chw = raster.to_chw()  # always (C,H,W)
    C, H, W = arr_chw.shape

    if (H, W) != (grid.height, grid.width):
        raise ValueError(
            f"Array shape does not match grid: array(H,W)=({H},{W}) vs grid(H,W)=({grid.height},{grid.width})"
        )

    out_nodata = nodata if nodata is not None else raster.nodata
    out_dtype = dtype or str(arr_chw.dtype)

    profile = {
        "driver": "GTiff",
        "height": int(grid.height),
        "width": int(grid.width),
        "count": int(C),
        "dtype": out_dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": out_nodata,
        "tiled": True,
        "compress": compress,
        "bigtiff": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_chw.astype(out_dtype, copy=False))


def iter_windows(path: Path, *, blocksize: int = 1024) -> Iterator[Window]:
    """Iterate over dataset windows for streaming computations (mean/std, histograms, etc.)."""
    p = Path(path)
    with rasterio.open(p) as ds:
        width = ds.width
        height = ds.height

        for row_off in range(0, height, blocksize):
            h = min(blocksize, height - row_off)
            for col_off in range(0, width, blocksize):
                w = min(blocksize, width - col_off)
                yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def read_window(path: Path, window: Window, *, indexes: int | list[int] | None = None) -> np.ndarray:
    """Read a window from a raster.

    Returns:
      - 2D (H, W) if indexes is int or dataset has a single band
      - 3D (C, H, W) if indexes is list or None for multi-band
    """
    p = Path(path)
    with rasterio.open(p) as ds:
        if indexes is None:
            if ds.count == 1:
                return ds.read(1, window=window)
            return ds.read(window=window)
        return ds.read(indexes, window=window)


def assert_same_grid(grids: Sequence[RasterGrid]) -> None:
    """Raise if any grid differs (CRS/transform/shape/res)."""
    if not grids:
        return
    g0 = grids[0]
    for gi in grids[1:]:
        if (
            gi.crs != g0.crs
            or gi.transform != g0.transform
            or gi.width != g0.width
            or gi.height != g0.height
            or gi.res != g0.res
        ):
            raise ValueError("Grids are not identical; cannot stack/merge safely.")


def stack_rasters(rasters: Sequence[Raster], *, band_names: list[str] | None = None) -> Raster:
    """Stack rasters along channel axis, returning a multi-band Raster (C,H,W).

    Inputs may be 2D (H,W) or 3D; all are normalized via to_chw().
    """
    if not rasters:
        raise ValueError("No rasters provided to stack.")

    assert_same_grid([r.grid for r in rasters])
    grid = rasters[0].grid

    arrays = [r.to_chw() for r in rasters]
    stacked = np.concatenate(arrays, axis=0)  # (C,H,W)

    # Collect band names if not explicitly provided
    if band_names is None:
        names: list[str] = []
        for r in rasters:
            if r.band_names:
                names.extend(r.band_names)
            else:
                # If missing, leave as None overall (avoid partial mismatch)
                names = []
                break
        band_names = names if names else None

    return Raster(
        array=stacked,
        grid=grid,
        nodata=rasters[0].nodata,
        band_names=band_names,
    )
