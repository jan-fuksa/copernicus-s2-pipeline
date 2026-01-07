from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from s2pipe.preprocess.angles import (
    angles_to_sin_cos_features,
    parse_tile_metadata_angles,
)
from s2pipe.preprocess.cfg import AngleAssetConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_root() -> Path:
    root = _repo_root()
    p = root / "tests" / "fixtures" / "single_tile"
    if not p.exists():
        pytest.skip(f"Fixture directory not found: {p}")
    return p


def _find_mtd_tl_xml() -> Path:
    base = _fixture_root()
    hits = sorted(base.rglob("MTD_TL.xml"))
    if not hits:
        pytest.skip(f"MTD_TL.xml not found under fixture root: {base}")
    return hits[0]


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _find_first_path(el: ET.Element, path: tuple[str, ...]) -> ET.Element | None:
    cur = el
    for name in path:
        nxt = None
        for c in cur:
            if _strip_ns(c.tag) == name:
                nxt = c
                break
        if nxt is None:
            return None
        cur = nxt
    return cur


def _iter_children(el: ET.Element, name: str):
    for c in el:
        if _strip_ns(c.tag) == name:
            yield c


def _read_expected_ulx_uly(
    root: ET.Element, prefer_resolution: str = "10"
) -> tuple[float, float]:
    tg = _find_first_path(root, ("Geometric_Info", "Tile_Geocoding"))
    assert tg is not None, "Missing Geometric_Info/Tile_Geocoding"

    geopos = list(_iter_children(tg, "Geoposition"))
    assert geopos, "Missing Tile_Geocoding/Geoposition"

    gp = None
    for cand in geopos:
        if cand.attrib.get("resolution") == prefer_resolution:
            gp = cand
            break
    if gp is None:
        gp = geopos[0]

    ulx_el = _find_first_path(gp, ("ULX",))
    uly_el = _find_first_path(gp, ("ULY",))
    assert ulx_el is not None and ulx_el.text, "Missing ULX"
    assert uly_el is not None and uly_el.text, "Missing ULY"
    return float(ulx_el.text.strip()), float(uly_el.text.strip())


def _read_expected_steps_and_shape_from_sun(
    root: ET.Element,
) -> tuple[float, float, tuple[int, int]]:
    tile_angles = _find_first_path(root, ("Geometric_Info", "Tile_Angles"))
    assert tile_angles is not None, "Missing Geometric_Info/Tile_Angles"

    sun = None
    for c in tile_angles:
        if _strip_ns(c.tag) == "Sun_Angles_Grid":
            sun = c
            break
    assert sun is not None, "Missing Sun_Angles_Grid"

    zen = _find_first_path(sun, ("Zenith",))
    assert zen is not None, "Missing Sun_Angles_Grid/Zenith"

    col_step_el = _find_first_path(zen, ("COL_STEP",))
    row_step_el = _find_first_path(zen, ("ROW_STEP",))
    assert col_step_el is not None and col_step_el.text, "Missing COL_STEP"
    assert row_step_el is not None and row_step_el.text, "Missing ROW_STEP"

    col_step = float(col_step_el.text.strip())
    row_step = float(row_step_el.text.strip())

    vl = _find_first_path(zen, ("Values_List",))
    assert vl is not None, "Missing Sun_Angles_Grid/Zenith/Values_List"

    rows = [r for r in vl if _strip_ns(r.tag) == "VALUES" and (r.text or "").strip()]
    assert rows, "Empty Values_List"

    h = len(rows)
    w = len(rows[0].text.strip().split())
    return col_step, row_step, (h, w)


@pytest.mark.integration
def test_parse_tile_metadata_angles_reconstructs_grid_correctly_from_tile_geocoding_and_steps() -> (
    None
):
    xml_path = _find_mtd_tl_xml()
    root = ET.parse(xml_path).getroot()

    ulx, uly = _read_expected_ulx_uly(root, prefer_resolution="10")
    col_step, row_step, (h, w) = _read_expected_steps_and_shape_from_sun(root)

    cfg = AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="per_band",
        view_bands=("B02",),
        detector_aggregate="nanmean",
    )

    angles = parse_tile_metadata_angles(xml_path, cfg=cfg)

    # expected transform: Affine(col_step, 0, ULX, 0, -row_step, ULY)
    exp = Affine(col_step, 0.0, ulx, 0.0, -row_step, uly)
    got = angles.src_grid.transform

    assert got.a == pytest.approx(exp.a)
    assert got.b == pytest.approx(exp.b)
    assert got.c == pytest.approx(exp.c)
    assert got.d == pytest.approx(exp.d)
    assert got.e == pytest.approx(exp.e)
    assert got.f == pytest.approx(exp.f)

    assert angles.src_grid.width == w
    assert angles.src_grid.height == h

    # The provided XML should be 23x23 (coarse angle grid).
    assert (h, w) == (23, 23)
    assert angles.sun_zenith_deg.shape == (23, 23)
    assert angles.sun_azimuth_deg.shape == (23, 23)


@pytest.mark.integration
def test_parse_tile_metadata_angles_loads_view_grids_for_b02_and_shapes_match() -> None:
    xml_path = _find_mtd_tl_xml()

    cfg = AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="per_band",
        view_bands=("B02",),
        detector_aggregate="nanmean",
    )

    angles = parse_tile_metadata_angles(xml_path, cfg=cfg)

    assert "B02" in angles.view_zenith_deg_by_band
    assert "B02" in angles.view_azimuth_deg_by_band

    z_list = angles.view_zenith_deg_by_band["B02"]
    a_list = angles.view_azimuth_deg_by_band["B02"]

    assert isinstance(z_list, list) and len(z_list) > 0
    assert isinstance(a_list, list) and len(a_list) == len(z_list)

    H, W = angles.src_grid.height, angles.src_grid.width
    for z in z_list:
        assert z.shape == (H, W)
    for a in a_list:
        assert a.shape == (H, W)


@pytest.mark.integration
def test_angles_to_sin_cos_features_returns_expected_channels_on_native_coarse_grid() -> (
    None
):
    xml_path = _find_mtd_tl_xml()

    cfg = AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="per_band",
        view_bands=("B02", "B03"),  # deterministic: 4 + 4*B = 12 channels
        detector_aggregate="nanmean",
    )

    angles = parse_tile_metadata_angles(xml_path, cfg=cfg)
    r = angles_to_sin_cos_features(angles=angles, cfg=cfg, dst_grid=None)

    arr = r.to_chw()
    assert arr.dtype == np.float32
    assert arr.shape == (12, 23, 23)

    assert r.band_names is not None
    assert r.band_names == [
        "sun_zen_sin",
        "sun_zen_cos",
        "sun_azi_sin",
        "sun_azi_cos",
        "view_B02_zen_sin",
        "view_B02_zen_cos",
        "view_B02_azi_sin",
        "view_B02_azi_cos",
        "view_B03_zen_sin",
        "view_B03_zen_cos",
        "view_B03_azi_sin",
        "view_B03_azi_cos",
    ]

    # nodata is NaN
    assert r.nodata is not None and np.isnan(float(r.nodata))
