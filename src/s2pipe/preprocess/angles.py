from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from affine import Affine

from .cfg import AngleAssetConfig
from .raster import Raster, RasterGrid
from .resample import resample_raster


"""
Sentinel-2 tile angles parsing and feature generation (Sun + View).

This module parses native/coarse angle grids from Sentinel-2 L1C tile metadata (MTD_TL.xml)
and converts them into feature rasters (sin/cos or degrees).

Correct georeferencing for angle grids (based on actual MTD_TL.xml structure)
---------------------------------------------------------------------------
In the Sentinel-2 tile metadata:

- `Geometric_Info/Tile_Geocoding/Geoposition` provides tile ULX/ULY (and XDIM/YDIM for pixel grids)
  at resolutions 10/20/60. ULX/ULY are identical across those resolutions in practice.

- `Geometric_Info/Tile_Angles/Sun_Angles_Grid` and each `Viewing_Incidence_Angles_Grids`
  provide:
    - COL_STEP, ROW_STEP  (in CRS units, typically meters; e.g. 5000 m)
    - Values_List         (H rows x W cols)

Angle grids do NOT carry their own Geoposition. Therefore the angle-grid RasterGrid is reconstructed as:

    transform = Affine(col_step, 0, ULX,
                       0, -row_step, ULY)

with width/height taken from Values_List shape.

Step-2 usage
------------
- `angles_to_sin_cos_features(..., dst_grid=None)` returns features on the native coarse angle grid
  (typically ~23x23). In Step-2, this is exported as a separate asset `angles.tif` (NOT appended to x.tif).
- Warping to a dense grid (dst_grid provided) is supported mainly for debugging. Warping uses
  a validity-mask strategy to avoid interpolation bleeding into invalid (NaN) regions.

View angles and detectors
-------------------------
View angles are provided per band and per detector (bandId, detectorId). At most nodes, only one detector is valid;
near boundaries, two can overlap. We aggregate detectors (default: NaN-aware mean):
- zenith: nanmean in degrees
- azimuth: nanmean in sin/cos space (to handle circular wrap-around)
"""


# Sentinel-2 bandId -> band name mapping used in MTD_TL.xml.
_BAND_ID_TO_NAME: dict[int, str] = {
    0: "B01",
    1: "B02",
    2: "B03",
    3: "B04",
    4: "B05",
    5: "B06",
    6: "B07",
    7: "B08",
    8: "B8A",
    9: "B09",
    10: "B10",
    11: "B11",
    12: "B12",
}

# Stable Sentinel-2 band order.
_BAND_ORDER: tuple[str, ...] = tuple(_BAND_ID_TO_NAME.values())
_BAND_ORDER_INDEX: dict[str, int] = {b: i for i, b in enumerate(_BAND_ORDER)}


def _band_order_key(b: str) -> int:
    return _BAND_ORDER_INDEX.get(b, 10_000)


@dataclass(frozen=True)
class AngleFields:
    """Parsed angle grids on the native (coarse) angle grid.

    All arrays are (H, W) float32 in degrees.
    View angles are stored per band, and each band has a list of detector grids.

    src_grid:
      The reconstructed georeferencing for the coarse angle grid.
    """

    src_grid: RasterGrid
    sun_zenith_deg: np.ndarray
    sun_azimuth_deg: np.ndarray
    view_zenith_deg_by_band: dict[str, list[np.ndarray]]
    view_azimuth_deg_by_band: dict[str, list[np.ndarray]]


def _strip_ns(tag: str) -> str:
    """Strip '{namespace}' prefix from an XML tag."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _find_first_child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _strip_ns(c.tag) == name:
            return c
    return None


def _find_first_path(el: ET.Element, path: Sequence[str]) -> ET.Element | None:
    """Follow a simple direct-child path (namespace-agnostic)."""
    cur: ET.Element | None = el
    for name in path:
        if cur is None:
            return None
        cur = _find_first_child(cur, name)
    return cur


def _iter_children(el: ET.Element, name: str) -> Iterable[ET.Element]:
    for c in el:
        if _strip_ns(c.tag) == name:
            yield c


def _parse_values_list(values_list: ET.Element) -> np.ndarray:
    """Parse <Values_List><VALUES> ... </VALUES> ...</Values_List> into (H,W) float32."""
    rows: list[np.ndarray] = []
    for v in values_list:
        if _strip_ns(v.tag) != "VALUES":
            continue
        if v.text is None:
            continue
        # float('NaN') is supported and preserves NaNs.
        row = np.asarray([float(x) for x in v.text.strip().split()], dtype=np.float32)
        rows.append(row)
    if not rows:
        raise ValueError("Empty Values_List")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _parse_float_text(el: ET.Element | None, what: str) -> float:
    if el is None or el.text is None:
        raise ValueError(f"Missing XML text for {what}")
    return float(el.text.strip())


def _parse_int(x: str | None) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _read_crs(root: ET.Element) -> str:
    el = _find_first_path(
        root, ("Geometric_Info", "Tile_Geocoding", "HORIZONTAL_CS_CODE")
    )
    if el is not None and el.text:
        return el.text.strip()
    return "unknown"


def _read_ulx_uly_from_tile_geocoding(
    root: ET.Element, *, prefer_resolution: str = "10"
) -> tuple[float, float]:
    """Read ULX/ULY from Geometric_Info/Tile_Geocoding/Geoposition.

    There are typically multiple Geoposition elements with attribute resolution=10/20/60.
    ULX/ULY are usually identical; we prefer resolution=10 if present.
    """
    tg = _find_first_path(root, ("Geometric_Info", "Tile_Geocoding"))
    if tg is None:
        raise ValueError("Missing Geometric_Info/Tile_Geocoding")

    geopos_all = [gp for gp in _iter_children(tg, "Geoposition")]
    if not geopos_all:
        raise ValueError("Missing Tile_Geocoding/Geoposition")

    gp = None
    for cand in geopos_all:
        if cand.attrib.get("resolution") == prefer_resolution:
            gp = cand
            break
    if gp is None:
        gp = geopos_all[0]

    ulx_el = _find_first_child(gp, "ULX")
    uly_el = _find_first_child(gp, "ULY")
    ulx = _parse_float_text(ulx_el, "Tile_Geocoding/Geoposition/ULX")
    uly = _parse_float_text(uly_el, "Tile_Geocoding/Geoposition/ULY")
    return ulx, uly


def _read_angle_steps_from_angle_grid(angle_grid: ET.Element) -> tuple[float, float]:
    """Read COL_STEP/ROW_STEP from an angle grid container (Zenith preferred)."""
    zen = _find_first_child(angle_grid, "Zenith")
    if zen is None:
        raise ValueError(
            "Angle grid missing Zenith element (cannot read COL_STEP/ROW_STEP)."
        )
    col_step_el = _find_first_child(zen, "COL_STEP")
    row_step_el = _find_first_child(zen, "ROW_STEP")
    col_step = _parse_float_text(col_step_el, "COL_STEP")
    row_step = _parse_float_text(row_step_el, "ROW_STEP")
    return col_step, row_step


def _reconstruct_angle_grid(
    *,
    ulx: float,
    uly: float,
    col_step: float,
    row_step: float,
    shape_hw: tuple[int, int],
    crs: str,
) -> RasterGrid:
    """Construct RasterGrid for coarse angle grid.

    COL_STEP/ROW_STEP are positive in the XML; y pixel size should be negative for north-up.
    """
    h, w = shape_hw
    transform = Affine(col_step, 0.0, ulx, 0.0, -row_step, uly)
    return RasterGrid(
        crs=crs,
        transform=transform,
        width=int(w),
        height=int(h),
        res=(float(col_step), float(-row_step)),
    )


def parse_tile_metadata_angles(
    tile_metadata_xml: Path, *, cfg: AngleAssetConfig
) -> AngleFields:
    """Parse sun/view angle grids from MTD_TL.xml (native/coarse angle grid).

    - ULX/ULY are read from Tile_Geocoding/Geoposition.
    - COL_STEP/ROW_STEP and Values_List shape are read from Tile_Angles/Sun_Angles_Grid.
    - Viewing_Incidence_Angles_Grids are parsed under Tile_Angles (per band, per detector).
    """
    tile_metadata_xml = Path(tile_metadata_xml)
    root = ET.parse(tile_metadata_xml).getroot()

    crs = _read_crs(root)
    ulx, uly = _read_ulx_uly_from_tile_geocoding(root, prefer_resolution="10")

    tile_angles = _find_first_path(root, ("Geometric_Info", "Tile_Angles"))
    if tile_angles is None:
        raise ValueError("Missing Geometric_Info/Tile_Angles")

    # --- Sun angles ---
    sun = _find_first_child(tile_angles, "Sun_Angles_Grid")
    if sun is None:
        raise ValueError("Missing Tile_Angles/Sun_Angles_Grid")

    sun_zen_el = _find_first_path(sun, ("Zenith", "Values_List"))
    sun_azi_el = _find_first_path(sun, ("Azimuth", "Values_List"))
    if sun_zen_el is None or sun_azi_el is None:
        raise ValueError("Missing Sun_Angles_Grid Zenith/Azimuth Values_List")

    sun_zen = _parse_values_list(sun_zen_el)
    sun_azi = _parse_values_list(sun_azi_el)

    col_step, row_step = _read_angle_steps_from_angle_grid(sun)
    src_grid = _reconstruct_angle_grid(
        ulx=ulx,
        uly=uly,
        col_step=col_step,
        row_step=row_step,
        shape_hw=(int(sun_zen.shape[0]), int(sun_zen.shape[1])),
        crs=crs,
    )

    if sun_zen.shape != (src_grid.height, src_grid.width) or sun_azi.shape != (
        src_grid.height,
        src_grid.width,
    ):
        raise ValueError(
            "Sun angle grid shape mismatch: "
            f"zen={sun_zen.shape}, azi={sun_azi.shape}, expected={(src_grid.height, src_grid.width)}"
        )

    # --- View angles (per band, per detector) ---
    want_bands = set(cfg.view_bands) if getattr(cfg, "view_bands", ()) else None

    view_zen: dict[str, list[np.ndarray]] = {}
    view_azi: dict[str, list[np.ndarray]] = {}

    for v in _iter_children(tile_angles, "Viewing_Incidence_Angles_Grids"):
        band_id_txt = v.attrib.get("bandId")
        band_id = _parse_int(band_id_txt)
        if band_id is None or band_id not in _BAND_ID_TO_NAME:
            continue
        band_name = _BAND_ID_TO_NAME[band_id]

        if want_bands is not None and band_name not in want_bands:
            continue

        # Optional consistency check: steps should match sun steps.
        # (In your XML they do: 5000/5000.)
        try:
            v_col_step, v_row_step = _read_angle_steps_from_angle_grid(v)
            if abs(v_col_step - col_step) > 1e-6 or abs(v_row_step - row_step) > 1e-6:
                raise ValueError(
                    f"View grid steps differ from sun steps for band={band_name}: "
                    f"view=({v_col_step},{v_row_step}) sun=({col_step},{row_step})"
                )
        except Exception:
            # Do not hard-fail on missing steps; Values_List shape check below is the real guard.
            pass

        zen_el = _find_first_path(v, ("Zenith", "Values_List"))
        azi_el = _find_first_path(v, ("Azimuth", "Values_List"))
        if zen_el is None or azi_el is None:
            continue

        z = _parse_values_list(zen_el)
        a = _parse_values_list(azi_el)

        if z.shape != (src_grid.height, src_grid.width) or a.shape != (
            src_grid.height,
            src_grid.width,
        ):
            raise ValueError(
                f"View grid shape mismatch for band={band_name}: "
                f"zen={z.shape}, azi={a.shape}, expected={(src_grid.height, src_grid.width)}"
            )

        view_zen.setdefault(band_name, []).append(z)
        view_azi.setdefault(band_name, []).append(a)

    view_zen = dict(sorted(view_zen.items(), key=lambda kv: _band_order_key(kv[0])))
    view_azi = dict(sorted(view_azi.items(), key=lambda kv: _band_order_key(kv[0])))

    return AngleFields(
        src_grid=src_grid,
        sun_zenith_deg=sun_zen,
        sun_azimuth_deg=sun_azi,
        view_zenith_deg_by_band=view_zen,
        view_azimuth_deg_by_band=view_azi,
    )


def _finite_counts_across_detectors(detector_grids: Sequence[np.ndarray]) -> np.ndarray:
    """Internal helper: per-node detector coverage counts across detectors.

    Output:
      counts: (H, W) int16 where counts[r,c] is number of detectors with finite value at that node.
    """
    if not detector_grids:
        raise ValueError("detector_grids is empty")
    stack = np.stack(
        [np.isfinite(np.asarray(g)) for g in detector_grids], axis=0
    )  # (D,H,W)
    return np.sum(stack, axis=0).astype(np.int16, copy=False)


def _aggregate_detectors_nanmean(
    zen_grids_deg: Sequence[np.ndarray],
    azi_grids_deg: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-detector view grids into a single per-band grid.

    Returns:
      zen_mean_deg: (H,W) float32 degrees
      azi_sin:      (H,W) float32
      azi_cos:      (H,W) float32

    Zenith is averaged in degrees (nanmean).
    Azimuth is averaged in sin/cos space (nanmean) to handle wrap-around.
    """
    if len(zen_grids_deg) != len(azi_grids_deg):
        raise ValueError("zen_grids_deg and azi_grids_deg must have the same length.")

    z_stack = np.stack(
        [np.asarray(z, dtype=np.float32) for z in zen_grids_deg], axis=0
    )  # (D,H,W)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=RuntimeWarning
        )  # mean of empty slice for all-NaN nodes
        zen_mean = np.nanmean(z_stack, axis=0).astype(np.float32, copy=False)

    sin_stack = []
    cos_stack = []
    for a in azi_grids_deg:
        a = np.asarray(a, dtype=np.float32)
        rad = np.deg2rad(a)
        sin_stack.append(np.sin(rad))
        cos_stack.append(np.cos(rad))

    sin_stack_arr = np.stack(sin_stack, axis=0)
    cos_stack_arr = np.stack(cos_stack, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sin_m = np.nanmean(sin_stack_arr, axis=0).astype(np.float32, copy=False)
        cos_m = np.nanmean(cos_stack_arr, axis=0).astype(np.float32, copy=False)

    return zen_mean, sin_m, cos_m


def _deg_to_sin_cos(deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(np.asarray(deg, dtype=np.float32))
    return np.sin(rad).astype(np.float32, copy=False), np.cos(rad).astype(
        np.float32, copy=False
    )


def _azi_sin_cos_to_deg(sin_a: np.ndarray, cos_a: np.ndarray) -> np.ndarray:
    rad = np.arctan2(
        np.asarray(sin_a, dtype=np.float32), np.asarray(cos_a, dtype=np.float32)
    )
    deg = np.rad2deg(rad)
    return np.mod(deg, 360.0).astype(np.float32, copy=False)


def angles_to_sin_cos_features(
    *,
    angles: AngleFields,
    cfg: AngleAssetConfig,
    dst_grid: RasterGrid | None,
) -> Raster:
    """Convert parsed angles to a feature raster.

    Parameters
    ----------
    angles:
      Output from `parse_tile_metadata_angles`.
    cfg:
      AngleAssetConfig controlling channel composition and encoding.
    dst_grid:
      - None: output on angles.src_grid (native coarse angle grid; recommended for Step-2 angles.tif).
      - RasterGrid: warp/resample features to this grid (debug/legacy).

    Returns
    -------
    Raster:
      float32 (C,H,W) with nodata=np.nan and band_names describing channels.
    """
    src_grid = angles.src_grid
    do_warp = dst_grid is not None and dst_grid != src_grid
    out_grid = dst_grid if do_warp else src_grid

    encode_sin_cos = bool(getattr(cfg, "encode_sin_cos", True))

    def _warp_2d(name: str, src_2d: np.ndarray, *, method: str) -> np.ndarray:
        """Warp a 2D float array src_grid -> out_grid with NaN-safe handling."""
        src_2d = np.asarray(src_2d, dtype=np.float32)
        if not do_warp:
            return src_2d

        valid = np.isfinite(src_2d).astype(np.uint8)
        val = np.where(np.isfinite(src_2d), src_2d, 0.0).astype(np.float32)

        r_val = Raster(array=val, grid=src_grid, nodata=None, band_names=[name])
        r_msk = Raster(
            array=valid, grid=src_grid, nodata=None, band_names=[f"{name}_valid"]
        )

        r_val_w = resample_raster(r_val, out_grid, method=method, dst_nodata=0.0)
        r_msk_w = resample_raster(r_msk, out_grid, method="nearest", dst_nodata=0.0)

        val_w = np.asarray(r_val_w.array, dtype=np.float32)
        msk_w = np.asarray(r_msk_w.array, dtype=np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = val_w / msk_w
        out[msk_w <= 0.0] = np.nan
        return out.astype(np.float32, copy=False)

    feats: list[np.ndarray] = []
    names: list[str] = []

    # --- Sun ---
    if cfg.include_sun:
        z = angles.sun_zenith_deg
        a = angles.sun_azimuth_deg
        if encode_sin_cos:
            z_sin, z_cos = _deg_to_sin_cos(z)
            a_sin, a_cos = _deg_to_sin_cos(a)
            feats += [
                _warp_2d("sun_zen_sin", z_sin, method="bilinear"),
                _warp_2d("sun_zen_cos", z_cos, method="bilinear"),
                _warp_2d("sun_azi_sin", a_sin, method="bilinear"),
                _warp_2d("sun_azi_cos", a_cos, method="bilinear"),
            ]
            names += ["sun_zen_sin", "sun_zen_cos", "sun_azi_sin", "sun_azi_cos"]
        else:
            feats += [
                _warp_2d("sun_zen_deg", z, method="bilinear"),
                _warp_2d("sun_azi_deg", a, method="bilinear"),
            ]
            names += ["sun_zen_deg", "sun_azi_deg"]

    # --- View ---
    if cfg.include_view:
        view_mode = getattr(cfg, "view_mode", "per_band")
        requested = list(getattr(cfg, "view_bands", ()) or [])
        parsed_bands = list(angles.view_zenith_deg_by_band.keys())

        if requested:
            bands = [b for b in requested if b in angles.view_zenith_deg_by_band]
        else:
            bands = parsed_bands

        if view_mode == "single":
            per_band = []
            for b in bands:
                z_list = angles.view_zenith_deg_by_band[b]
                a_list = angles.view_azimuth_deg_by_band[b]
                zen_mean, azi_sin, azi_cos = _aggregate_detectors_nanmean(
                    z_list, a_list
                )
                per_band.append((zen_mean, azi_sin, azi_cos))

            if per_band:
                zen_stack = np.stack([p[0] for p in per_band], axis=0)
                sin_stack = np.stack([p[1] for p in per_band], axis=0)
                cos_stack = np.stack([p[2] for p in per_band], axis=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    zen_mean = np.nanmean(zen_stack, axis=0).astype(
                        np.float32, copy=False
                    )
                    sin_m = np.nanmean(sin_stack, axis=0).astype(np.float32, copy=False)
                    cos_m = np.nanmean(cos_stack, axis=0).astype(np.float32, copy=False)

                if encode_sin_cos:
                    z_sin, z_cos = _deg_to_sin_cos(zen_mean)
                    feats += [
                        _warp_2d("view_zen_sin", z_sin, method="bilinear"),
                        _warp_2d("view_zen_cos", z_cos, method="bilinear"),
                        _warp_2d("view_azi_sin", sin_m, method="bilinear"),
                        _warp_2d("view_azi_cos", cos_m, method="bilinear"),
                    ]
                    names += [
                        "view_zen_sin",
                        "view_zen_cos",
                        "view_azi_sin",
                        "view_azi_cos",
                    ]
                else:
                    azi_deg = _azi_sin_cos_to_deg(sin_m, cos_m)
                    feats += [
                        _warp_2d("view_zen_deg", zen_mean, method="bilinear"),
                        _warp_2d("view_azi_deg", azi_deg, method="bilinear"),
                    ]
                    names += ["view_zen_deg", "view_azi_deg"]

        elif view_mode == "per_band":
            for b in bands:
                z_list = angles.view_zenith_deg_by_band[b]
                a_list = angles.view_azimuth_deg_by_band[b]
                zen_mean, azi_sin, azi_cos = _aggregate_detectors_nanmean(
                    z_list, a_list
                )

                if encode_sin_cos:
                    z_sin, z_cos = _deg_to_sin_cos(zen_mean)
                    feats += [
                        _warp_2d(f"view_{b}_zen_sin", z_sin, method="bilinear"),
                        _warp_2d(f"view_{b}_zen_cos", z_cos, method="bilinear"),
                        _warp_2d(f"view_{b}_azi_sin", azi_sin, method="bilinear"),
                        _warp_2d(f"view_{b}_azi_cos", azi_cos, method="bilinear"),
                    ]
                    names += [
                        f"view_{b}_zen_sin",
                        f"view_{b}_zen_cos",
                        f"view_{b}_azi_sin",
                        f"view_{b}_azi_cos",
                    ]
                else:
                    azi_deg = _azi_sin_cos_to_deg(azi_sin, azi_cos)
                    feats += [
                        _warp_2d(f"view_{b}_zen_deg", zen_mean, method="bilinear"),
                        _warp_2d(f"view_{b}_azi_deg", azi_deg, method="bilinear"),
                    ]
                    names += [f"view_{b}_zen_deg", f"view_{b}_azi_deg"]
        else:
            raise ValueError(f"Unsupported view_mode={view_mode!r}")

    if not feats:
        raise ValueError(
            "No angle features generated (check cfg.include_sun/include_view and metadata contents)."
        )

    arr = np.stack(feats, axis=0).astype(np.float32, copy=False)  # (C,H,W)
    return Raster(array=arr, grid=out_grid, nodata=np.nan, band_names=names)
