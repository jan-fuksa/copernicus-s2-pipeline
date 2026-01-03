from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from affine import Affine

from .cfg import AngleFeatureConfig
from .raster import Raster, RasterGrid
from .resample import resample_raster


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


@dataclass(frozen=True)
class AngleFields:
    """Angle rasters (units: degrees unless stated otherwise)."""

    src_grid: RasterGrid

    sun_zenith: np.ndarray | None
    sun_azimuth: np.ndarray | None

    # Per band: list of per-detector grids (values are often NaN outside detector footprint).
    view_zenith: dict[str, list[np.ndarray]] | None
    view_azimuth: dict[str, list[np.ndarray]] | None


def _local(tag: str) -> str:
    """Return localname of an XML tag (strip namespace)."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_first(root: ET.Element, name: str) -> ET.Element | None:
    for el in root.iter():
        if _local(el.tag) == name:
            return el
    return None


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    out: list[ET.Element] = []
    for el in root.iter():
        if _local(el.tag) == name:
            out.append(el)
    return out


def _text_float(el: ET.Element | None) -> float | None:
    if el is None or el.text is None:
        return None
    s = el.text.strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _text_int(el: ET.Element | None) -> int | None:
    v = _text_float(el)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _parse_values_list(values_parent: ET.Element) -> np.ndarray:
    """Parse <Values_List><VALUES>...</VALUES>...</Values_List> into (H,W) float32 array."""
    vl = None
    for ch in list(values_parent):
        if _local(ch.tag) == "Values_List":
            vl = ch
            break
    if vl is None:
        vl = values_parent

    rows: list[list[float]] = []
    for v in list(vl):
        if _local(v.tag) != "VALUES":
            continue
        if v.text is None:
            continue
        parts = v.text.strip().split()
        if not parts:
            continue
        row: list[float] = []
        for p in parts:
            # float("NaN") is valid and becomes np.nan
            try:
                row.append(float(p))
            except Exception:
                row.append(float("nan"))
        rows.append(row)

    if not rows:
        raise ValueError("Angle grid: no <VALUES> rows found.")

    w = max(len(r) for r in rows)
    for r in rows:
        if len(r) != w:
            r.extend([float("nan")] * (w - len(r)))

    return np.asarray(rows, dtype=np.float32)


def _parse_crs_code(root: ET.Element) -> str | None:
    el = _find_first(root, "HORIZONTAL_CS_CODE")
    if el is None or el.text is None:
        return None
    s = el.text.strip()
    return s if s else None


def _parse_geoposition(
    root: ET.Element, resolution_preference: list[int]
) -> tuple[float, float, float, float, int] | None:
    """Return (ulx, uly, xdim, ydim, resolution_used) from <Geoposition>."""
    geopos = _find_all(root, "Geoposition")
    if not geopos:
        return None

    by_res: dict[int, ET.Element] = {}
    for g in geopos:
        res_attr = g.attrib.get("resolution") or g.attrib.get("Resolution")
        if res_attr is None:
            continue
        try:
            by_res[int(float(res_attr))] = g
        except Exception:
            continue

    chosen: ET.Element | None = None
    chosen_res: int | None = None

    for r in resolution_preference:
        if r in by_res:
            chosen = by_res[r]
            chosen_res = r
            break

    if chosen is None:
        chosen = geopos[0]
        res_attr = chosen.attrib.get("resolution") or chosen.attrib.get("Resolution")
        chosen_res = int(float(res_attr)) if res_attr else 0

    ulx = _text_float(_find_first(chosen, "ULX"))
    uly = _text_float(_find_first(chosen, "ULY"))
    xdim = _text_float(_find_first(chosen, "XDIM"))
    ydim = _text_float(_find_first(chosen, "YDIM"))

    if ulx is None or uly is None or xdim is None or ydim is None:
        return None

    return float(ulx), float(uly), float(xdim), float(ydim), int(chosen_res)


def _parse_tile_size(root: ET.Element, resolution: int) -> tuple[int, int] | None:
    """Return (nrows, ncols) from <Size resolution="...">."""
    sizes = _find_all(root, "Size")
    for s in sizes:
        res_attr = s.attrib.get("resolution") or s.attrib.get("Resolution")
        if res_attr is None:
            continue
        try:
            r = int(float(res_attr))
        except Exception:
            continue
        if r != int(resolution):
            continue
        nrows = _text_int(_find_first(s, "NROWS"))
        ncols = _text_int(_find_first(s, "NCOLS"))
        if nrows is not None and ncols is not None:
            return int(nrows), int(ncols)
    return None


def _parse_sampling_params(
    root: ET.Element,
) -> tuple[float | None, float | None, float, float]:
    """Parse COL_STEP, ROW_STEP, COL_START, ROW_START from the XML (best-effort)."""
    col_step = _text_float(_find_first(root, "COL_STEP"))
    row_step = _text_float(_find_first(root, "ROW_STEP"))
    col_start = _text_float(_find_first(root, "COL_START"))
    row_start = _text_float(_find_first(root, "ROW_START"))

    if col_start is None:
        col_start = 0.0
    if row_start is None:
        row_start = 0.0

    return col_step, row_step, float(col_start), float(row_start)


def _build_angle_grid(
    *,
    crs: str,
    ulx: float,
    uly: float,
    xdim: float,
    ydim: float,
    col_step: float,
    row_step: float,
    col_start: float,
    row_start: float,
    width: int,
    height: int,
) -> RasterGrid:
    xres = float(xdim) * float(col_step)
    yres = float(ydim) * float(row_step)
    x0 = float(ulx) + float(col_start) * float(xdim)
    y0 = float(uly) + float(row_start) * float(ydim)
    transform = Affine(xres, 0.0, x0, 0.0, yres, y0)
    res = (abs(xres), abs(yres))
    return RasterGrid(
        crs=str(crs), transform=transform, width=int(width), height=int(height), res=res
    )


def _parse_band_name(band_id_raw: str | None) -> str:
    if band_id_raw is None:
        return "band_unknown"
    try:
        return _BAND_ID_TO_NAME.get(int(float(band_id_raw)), f"band_{band_id_raw}")
    except Exception:
        return f"band_{band_id_raw}"


def _parse_view_grids_from_repeated_elements(root: ET.Element) -> list[ET.Element]:
    """Return per-detector view elements when they are encoded as repeated Viewing_Incidence_Angles_Grids."""
    # In many MTD_TL.xml files, each detector grid is itself an element named Viewing_Incidence_Angles_Grids
    # with attributes bandId=... detectorId=..., and has children Zenith/Azimuth.
    out: list[ET.Element] = []
    for el in root.iter():
        if _local(el.tag) != "Viewing_Incidence_Angles_Grids":
            continue
        # Heuristic: treat only elements that actually contain Zenith/Azimuth as per-detector grids.
        if _find_first(el, "Zenith") is None or _find_first(el, "Azimuth") is None:
            continue
        out.append(el)
    return out


def _parse_view_grids_from_container(root: ET.Element) -> list[ET.Element]:
    """Fallback for schemas that use a container with child Viewing_Incidence_Angles_Grid elements."""
    container = _find_first(root, "Viewing_Incidence_Angles_Grids")
    if container is None:
        container = _find_first(root, "Viewing_Incidence_Angles_Grid_List")
    if container is None:
        return []
    return [
        ch
        for ch in list(container)
        if _local(ch.tag) == "Viewing_Incidence_Angles_Grid"
    ]


def parse_tile_metadata_angles(
    tile_metadata_xml: Path, *, cfg: AngleFeatureConfig
) -> AngleFields:
    """Parse sun/view angles from Sentinel-2 tile metadata XML (MTD_TL.xml).

    Sun angles:
      - Sun_Angles_Grid (Zenith/Azimuth) as coarse grid.

    View angles:
      - Prefer repeated elements Viewing_Incidence_Angles_Grids (bandId, detectorId).
      - Fallback to container Viewing_Incidence_Angles_Grids -> Viewing_Incidence_Angles_Grid.
    """
    tile_metadata_xml = Path(tile_metadata_xml)
    root = ET.fromstring(tile_metadata_xml.read_text(encoding="utf-8"))

    crs = _parse_crs_code(root) or "UNKNOWN"
    geo = _parse_geoposition(root, resolution_preference=[10, 20, 60])
    if geo is None:
        raise ValueError(
            "Cannot parse Geoposition (ULX/ULY/XDIM/YDIM) from tile metadata XML."
        )

    ulx, uly, xdim, ydim, geo_res = geo
    tile_size = _parse_tile_size(root, resolution=geo_res)

    # --- Sun angles ---
    sun_zen: np.ndarray | None = None
    sun_azi: np.ndarray | None = None
    sun_grid_w: int | None = None
    sun_grid_h: int | None = None

    if cfg.include_sun:
        sun_grid = _find_first(root, "Sun_Angles_Grid")
        if sun_grid is None:
            raise ValueError("Missing Sun_Angles_Grid in tile metadata XML.")

        zen_el = _find_first(sun_grid, "Zenith")
        azi_el = _find_first(sun_grid, "Azimuth")
        if zen_el is None or azi_el is None:
            raise ValueError("Sun_Angles_Grid: missing Zenith/Azimuth elements.")

        sun_zen = _parse_values_list(zen_el)
        sun_azi = _parse_values_list(azi_el)
        if sun_zen.shape != sun_azi.shape:
            raise ValueError(
                f"Sun angles shape mismatch: zenith={sun_zen.shape} azimuth={sun_azi.shape}"
            )

        sun_grid_h, sun_grid_w = sun_zen.shape

    # --- Sampling params (for coarse angle grid) ---
    col_step, row_step, col_start, row_start = _parse_sampling_params(root)

    # If missing, infer from tile size and sun grid (best-effort).
    if (
        (col_step is None or row_step is None)
        and tile_size is not None
        and sun_grid_w
        and sun_grid_h
    ):
        nrows, ncols = tile_size
        if col_step is None:
            col_step = (
                (float(ncols) - 1.0) / (float(sun_grid_w) - 1.0)
                if sun_grid_w > 1
                else 1.0
            )
        if row_step is None:
            row_step = (
                (float(nrows) - 1.0) / (float(sun_grid_h) - 1.0)
                if sun_grid_h > 1
                else 1.0
            )

    if col_step is None or row_step is None:
        col_step = 1.0
        row_step = 1.0

    width = int(sun_grid_w) if sun_grid_w is not None else None
    height = int(sun_grid_h) if sun_grid_h is not None else None

    # --- View angles ---
    view_zen: dict[str, list[np.ndarray]] | None = None
    view_azi: dict[str, list[np.ndarray]] | None = None

    if cfg.include_view:
        view_elems = _parse_view_grids_from_repeated_elements(root)
        if not view_elems:
            view_elems = _parse_view_grids_from_container(root)

        if view_elems:
            view_zen = {}
            view_azi = {}
            for g in view_elems:
                band_id_raw = (
                    g.attrib.get("bandId")
                    or g.attrib.get("band_id")
                    or g.attrib.get("BAND_ID")
                )
                if band_id_raw is None:
                    bid_el = _find_first(g, "BAND_ID")
                    band_id_raw = (
                        bid_el.text.strip()
                        if (bid_el is not None and bid_el.text)
                        else None
                    )

                band_name = _parse_band_name(band_id_raw)

                zen_el = _find_first(g, "Zenith")
                azi_el = _find_first(g, "Azimuth")
                if zen_el is None or azi_el is None:
                    continue

                z = _parse_values_list(zen_el)
                a = _parse_values_list(azi_el)
                if z.shape != a.shape:
                    raise ValueError(
                        f"View angles shape mismatch for {band_name}: zen={z.shape} azi={a.shape}"
                    )

                if width is None or height is None:
                    height, width = z.shape

                view_zen.setdefault(band_name, []).append(z)
                view_azi.setdefault(band_name, []).append(a)

    if width is None or height is None:
        raise ValueError(
            "Could not determine angle grid dimensions (no sun/view grids parsed)."
        )

    src_grid = _build_angle_grid(
        crs=crs,
        ulx=ulx,
        uly=uly,
        xdim=xdim,
        ydim=ydim,
        col_step=float(col_step),
        row_step=float(row_step),
        col_start=float(col_start),
        row_start=float(row_start),
        width=int(width),
        height=int(height),
    )

    return AngleFields(
        src_grid=src_grid,
        sun_zenith=sun_zen,
        sun_azimuth=sun_azi,
        view_zenith=view_zen,
        view_azimuth=view_azi,
    )


def _deg2rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x.astype(np.float32, copy=False))


def _mean_zenith_deg_nanmean(arrs: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrs, axis=0).astype(np.float32, copy=False)
    return np.nanmean(stack, axis=0).astype(np.float32, copy=False)


def _mean_azimuth_sin_cos_nanmean(
    arrs_deg: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    # Vector mean to handle wrap-around at 360 degrees, ignoring NaNs.
    stack = np.stack([_deg2rad(a) for a in arrs_deg], axis=0)
    s = np.nanmean(np.sin(stack), axis=0).astype(np.float32, copy=False)
    c = np.nanmean(np.cos(stack), axis=0).astype(np.float32, copy=False)
    return s, c


def _aggregate_detectors(
    *,
    zen_grids_deg: list[np.ndarray],
    azi_grids_deg: list[np.ndarray],
    detector_aggregate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-detector view grids into a single grid.

    Returns:
        zen_deg, zen_sin, zen_cos, (azi_sin, azi_cos) are computed by caller;
        here we return zen_deg and (azi_sin, azi_cos) building blocks.
    """
    if detector_aggregate == "nanmean":
        zen_deg = _mean_zenith_deg_nanmean(zen_grids_deg)
        azi_sin, azi_cos = _mean_azimuth_sin_cos_nanmean(azi_grids_deg)
        return zen_deg, azi_sin, azi_cos
    if detector_aggregate == "mosaic":
        # Requires MSK_DETFOO (or equivalent) to pick the correct detector per pixel.
        raise NotImplementedError(
            "detector_aggregate='mosaic' requires MSK_DETFOO; not implemented yet."
        )
    raise ValueError(f"Unknown detector_aggregate: {detector_aggregate!r}")


def _finite_detector_count_map(det_grid: np.ndarray) -> np.ndarray:
    """Internal diagnostic: number of finite detectors per grid cell.

    Args:
        det_grid: array of shape (D, H, W)

    Returns:
        count_map: integer array of shape (H, W), values in [0..D]
    """
    a = np.asarray(det_grid)
    if a.ndim != 3:
        raise ValueError(f"Expected det_grid with shape (D,H,W), got {a.shape}.")
    count_map = np.sum(np.isfinite(a), axis=0)
    return count_map.astype(np.uint8, copy=False)


def angles_to_sin_cos_features(
    *,
    angles: AngleFields,
    dst_grid: RasterGrid,
    cfg: AngleFeatureConfig,
) -> Raster:
    """Convert angles to feature rasters (sin/cos) and resample to dst_grid.

    Returns:
        Raster with array shape (C,H,W), dtype float32, and band_names.
    """
    channels: list[np.ndarray] = []
    names: list[str] = []

    src_grid = angles.src_grid

    def _warp(name: str, arr2d: np.ndarray) -> np.ndarray:
        r = Raster(
            array=arr2d.astype(np.float32, copy=False),
            grid=src_grid,
            nodata=None,
            band_names=[name],
        )
        r2 = resample_raster(r, dst_grid, method="bilinear")
        out = np.asarray(r2.array, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"Resample produced unexpected ndim={out.ndim} for {name}")
        return out

    # --- Sun ---
    if cfg.include_sun:
        if angles.sun_zenith is None or angles.sun_azimuth is None:
            raise ValueError("include_sun=True but sun angles are missing.")

        sun_zen_deg = angles.sun_zenith
        sun_azi_deg = angles.sun_azimuth

        if cfg.encode_sin_cos:
            zen_rad = _deg2rad(sun_zen_deg)
            azi_rad = _deg2rad(sun_azi_deg)
            sun_zen_sin = np.sin(zen_rad).astype(np.float32, copy=False)
            sun_zen_cos = np.cos(zen_rad).astype(np.float32, copy=False)
            sun_azi_sin = np.sin(azi_rad).astype(np.float32, copy=False)
            sun_azi_cos = np.cos(azi_rad).astype(np.float32, copy=False)

            channels.extend(
                [_warp("sun_zen_sin", sun_zen_sin), _warp("sun_zen_cos", sun_zen_cos)]
            )
            channels.extend(
                [_warp("sun_azi_sin", sun_azi_sin), _warp("sun_azi_cos", sun_azi_cos)]
            )
            names.extend(["sun_zen_sin", "sun_zen_cos", "sun_azi_sin", "sun_azi_cos"])
        else:
            channels.extend(
                [_warp("sun_zen_deg", sun_zen_deg), _warp("sun_azi_deg", sun_azi_deg)]
            )
            names.extend(["sun_zen_deg", "sun_azi_deg"])

    # --- View ---
    if cfg.include_view:
        if angles.view_zenith is None or angles.view_azimuth is None:
            raise ValueError("include_view=True but view angles are missing.")

        view_zen = angles.view_zenith
        view_azi = angles.view_azimuth

        band_order = list(cfg.view_bands) if cfg.view_bands else sorted(view_zen.keys())

        if cfg.view_mode == "per_band":
            for b in band_order:
                if b not in view_zen or b not in view_azi:
                    continue

                zen_deg, azi_sin_mean, azi_cos_mean = _aggregate_detectors(
                    zen_grids_deg=view_zen[b],
                    azi_grids_deg=view_azi[b],
                    detector_aggregate=cfg.detector_aggregate,
                )

                if cfg.encode_sin_cos:
                    zen_rad = _deg2rad(zen_deg)
                    zen_sin = np.sin(zen_rad).astype(np.float32, copy=False)
                    zen_cos = np.cos(zen_rad).astype(np.float32, copy=False)

                    channels.extend(
                        [
                            _warp(f"view_{b}_zen_sin", zen_sin),
                            _warp(f"view_{b}_zen_cos", zen_cos),
                        ]
                    )
                    channels.extend(
                        [
                            _warp(f"view_{b}_azi_sin", azi_sin_mean),
                            _warp(f"view_{b}_azi_cos", azi_cos_mean),
                        ]
                    )
                    names.extend(
                        [
                            f"view_{b}_zen_sin",
                            f"view_{b}_zen_cos",
                            f"view_{b}_azi_sin",
                            f"view_{b}_azi_cos",
                        ]
                    )
                else:
                    azi_deg = (
                        np.rad2deg(np.arctan2(azi_sin_mean, azi_cos_mean)) % 360.0
                    ).astype(np.float32)
                    channels.extend(
                        [
                            _warp(f"view_{b}_zen_deg", zen_deg),
                            _warp(f"view_{b}_azi_deg", azi_deg),
                        ]
                    )
                    names.extend([f"view_{b}_zen_deg", f"view_{b}_azi_deg"])

        elif cfg.view_mode == "single":
            zen_arrs: list[np.ndarray] = []
            azi_arrs: list[np.ndarray] = []
            for b in band_order:
                if b in view_zen:
                    zen_arrs.extend(view_zen[b])
                if b in view_azi:
                    azi_arrs.extend(view_azi[b])

            if not zen_arrs or not azi_arrs:
                raise ValueError(
                    "view_mode='single' requested but no view angle grids matched the selection."
                )

            zen_deg, azi_sin_mean, azi_cos_mean = _aggregate_detectors(
                zen_grids_deg=zen_arrs,
                azi_grids_deg=azi_arrs,
                detector_aggregate=cfg.detector_aggregate,
            )

            if cfg.encode_sin_cos:
                zen_rad = _deg2rad(zen_deg)
                zen_sin = np.sin(zen_rad).astype(np.float32, copy=False)
                zen_cos = np.cos(zen_rad).astype(np.float32, copy=False)

                channels.extend(
                    [_warp("view_zen_sin", zen_sin), _warp("view_zen_cos", zen_cos)]
                )
                channels.extend(
                    [
                        _warp("view_azi_sin", azi_sin_mean),
                        _warp("view_azi_cos", azi_cos_mean),
                    ]
                )
                names.extend(
                    ["view_zen_sin", "view_zen_cos", "view_azi_sin", "view_azi_cos"]
                )
            else:
                azi_deg = (
                    np.rad2deg(np.arctan2(azi_sin_mean, azi_cos_mean)) % 360.0
                ).astype(np.float32)
                channels.extend(
                    [_warp("view_zen_deg", zen_deg), _warp("view_azi_deg", azi_deg)]
                )
                names.extend(["view_zen_deg", "view_azi_deg"])
        else:
            raise ValueError(f"Unknown view_mode: {cfg.view_mode!r}")

    if not channels:
        raise ValueError("No angle features produced (check AngleFeatureConfig).")

    x = np.stack(channels, axis=0).astype(np.float32, copy=False)
    return Raster(array=x, grid=dst_grid, nodata=None, band_names=names)
