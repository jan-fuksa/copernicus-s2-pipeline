from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine

from s2pipe.preprocess.angles import (
    angles_to_sin_cos_features,
    parse_tile_metadata_angles,
)
from s2pipe.preprocess.cfg import AngleAssetConfig
from s2pipe.preprocess.raster import RasterGrid


def _write_xml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "MTD_TL.xml"
    p.write_text(text, encoding="utf-8")
    return p


def _grid(
    crs: str, transform: Affine, width: int, height: int, res: tuple[float, float]
) -> RasterGrid:
    return RasterGrid(crs=crs, transform=transform, width=width, height=height, res=res)


def test_angles_sun_parse_and_features_identity_grid_matches_mtd_structure(
    tmp_path: Path,
):
    """Sun angles: test matches the real MTD_TL.xml nesting.

    In MTD_TL.xml the angle steps (COL_STEP/ROW_STEP) live inside:
      Geometric_Info/Tile_Angles/Sun_Angles_Grid/Zenith (and Azimuth)
    and the grid ULX/ULY live inside:
      Geometric_Info/Tile_Geocoding/Geoposition
    """

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root xmlns:n1="urn:test">
  <n1:Geometric_Info>
    <n1:Tile_Geocoding>
      <n1:HORIZONTAL_CS_CODE>EPSG:32633</n1:HORIZONTAL_CS_CODE>
      <n1:Geoposition resolution="20">
        <n1:ULX>0</n1:ULX>
        <n1:ULY>40</n1:ULY>
        <n1:XDIM>20</n1:XDIM>
        <n1:YDIM>-20</n1:YDIM>
      </n1:Geoposition>
    </n1:Tile_Geocoding>

    <n1:Tile_Angles>
      <n1:Sun_Angles_Grid>
        <n1:Zenith>
          <n1:COL_STEP unit="m">20</n1:COL_STEP>
          <n1:ROW_STEP unit="m">20</n1:ROW_STEP>
          <n1:Values_List>
            <n1:VALUES>0 90</n1:VALUES>
            <n1:VALUES>45 60</n1:VALUES>
          </n1:Values_List>
        </n1:Zenith>
        <n1:Azimuth>
          <n1:COL_STEP unit="m">20</n1:COL_STEP>
          <n1:ROW_STEP unit="m">20</n1:ROW_STEP>
          <n1:Values_List>
            <n1:VALUES>0 180</n1:VALUES>
            <n1:VALUES>90 270</n1:VALUES>
          </n1:Values_List>
        </n1:Azimuth>
      </n1:Sun_Angles_Grid>
    </n1:Tile_Angles>
  </n1:Geometric_Info>
</root>
"""
    xml_path = _write_xml(tmp_path, xml)

    cfg = AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=False,
        encode="sin_cos",
    )
    ang = parse_tile_metadata_angles(xml_path, cfg=cfg)

    # Transform is derived from ULX/ULY and the *angle grid* COL_STEP/ROW_STEP.
    expected_grid = _grid(
        "EPSG:32633",
        Affine(20.0, 0.0, 0.0, 0.0, -20.0, 40.0),
        width=2,
        height=2,
        # RasterGrid.res follows angles.py: (col_step, -row_step)
        res=(20.0, -20.0),
    )
    assert ang.src_grid.crs == expected_grid.crs
    assert ang.src_grid.width == expected_grid.width
    assert ang.src_grid.height == expected_grid.height
    assert ang.src_grid.transform == expected_grid.transform
    assert ang.src_grid.res == expected_grid.res

    feat = angles_to_sin_cos_features(angles=ang, cfg=cfg, dst_grid=None)
    x = np.asarray(feat.to_chw())
    assert x.shape == (4, 2, 2)
    assert feat.band_names == [
        "sun_zen_sin",
        "sun_zen_cos",
        "sun_azi_sin",
        "sun_azi_cos",
    ]
    assert feat.grid == ang.src_grid

    zen = np.array([[0, 90], [45, 60]], dtype=np.float32)
    azi = np.array([[0, 180], [90, 270]], dtype=np.float32)
    z = np.deg2rad(zen)
    a = np.deg2rad(azi)

    np.testing.assert_allclose(x[0], np.sin(z), rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[1], np.cos(z), rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[2], np.sin(a), rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[3], np.cos(a), rtol=0, atol=1e-6)


def test_view_azimuth_wrap_detector_average_matches_mtd_structure(tmp_path: Path):
    """Two detectors for B02 (bandId=1).

    Azimuths near 359° and 1° should average to ~0° via vector mean
    (i.e. sin ~ 0 and cos ~ 1).

    Note: parse_tile_metadata_angles always requires Sun_Angles_Grid, because
    it defines the coarse angle grid geometry.
    """

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32633</HORIZONTAL_CS_CODE>
      <Geoposition resolution="20">
        <ULX>0</ULX><ULY>40</ULY>
        <XDIM>20</XDIM><YDIM>-20</YDIM>
      </Geoposition>
    </Tile_Geocoding>

    <Tile_Angles>
      <Sun_Angles_Grid>
        <Zenith>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List>
        </Azimuth>
      </Sun_Angles_Grid>

      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="0">
        <Zenith>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>10 10</VALUES><VALUES>10 10</VALUES></Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>359 359</VALUES><VALUES>359 359</VALUES></Values_List>
        </Azimuth>
      </Viewing_Incidence_Angles_Grids>

      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="1">
        <Zenith>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>10 10</VALUES><VALUES>10 10</VALUES></Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>1 1</VALUES><VALUES>1 1</VALUES></Values_List>
        </Azimuth>
      </Viewing_Incidence_Angles_Grids>
    </Tile_Angles>
  </Geometric_Info>
</root>
"""
    xml_path = _write_xml(tmp_path, xml)

    cfg = AngleAssetConfig(
        enabled=True,
        include_sun=False,
        include_view=True,
        encode="sin_cos",
        view_mode="single",
        view_bands=("B02",),
        detector_aggregate="nanmean",
    )
    ang = parse_tile_metadata_angles(xml_path, cfg=cfg)

    feat = angles_to_sin_cos_features(angles=ang, cfg=cfg, dst_grid=None)
    x = np.asarray(feat.to_chw())

    assert x.shape == (4, 2, 2)
    assert feat.band_names == [
        "view_zen_sin",
        "view_zen_cos",
        "view_azi_sin",
        "view_azi_cos",
    ]
    assert feat.grid == ang.src_grid

    # Azimuth should average to ~0 degrees -> sin ~ 0, cos ~ 1
    assert np.all(np.abs(x[2]) < 1e-3)
    assert np.all(x[3] > 0.999)


def test_view_mode_per_band_channel_naming_matches_mtd_structure(tmp_path: Path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32633</HORIZONTAL_CS_CODE>
      <Geoposition resolution="20"><ULX>0</ULX><ULY>40</ULY></Geoposition>
    </Tile_Geocoding>

    <Tile_Angles>
      <Sun_Angles_Grid>
        <Zenith>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List>
        </Azimuth>
      </Sun_Angles_Grid>

      <!-- B02 (bandId=1) -->
      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="0">
        <Zenith>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>10 10</VALUES><VALUES>10 10</VALUES></Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>20 20</VALUES><VALUES>20 20</VALUES></Values_List>
        </Azimuth>
      </Viewing_Incidence_Angles_Grids>

      <!-- B01 (bandId=0) -->
      <Viewing_Incidence_Angles_Grids bandId="0" detectorId="0">
        <Zenith>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>30 30</VALUES><VALUES>30 30</VALUES></Values_List>
        </Zenith>
        <Azimuth>
          <COL_STEP unit="m">20</COL_STEP>
          <ROW_STEP unit="m">20</ROW_STEP>
          <Values_List><VALUES>40 40</VALUES><VALUES>40 40</VALUES></Values_List>
        </Azimuth>
      </Viewing_Incidence_Angles_Grids>
    </Tile_Angles>
  </Geometric_Info>
</root>
"""
    xml_path = _write_xml(tmp_path, xml)

    cfg = AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="per_band",
        view_bands=("B02", "B01"),
        detector_aggregate="nanmean",
    )
    ang = parse_tile_metadata_angles(xml_path, cfg=cfg)

    feat = angles_to_sin_cos_features(angles=ang, cfg=cfg, dst_grid=None)
    x = np.asarray(feat.to_chw())

    # 4 sun channels + 4 channels per band
    assert x.shape == (4 + 4 * 2, 2, 2)

    expected = [
        "sun_zen_sin",
        "sun_zen_cos",
        "sun_azi_sin",
        "sun_azi_cos",
        "view_B02_zen_sin",
        "view_B02_zen_cos",
        "view_B02_azi_sin",
        "view_B02_azi_cos",
        "view_B01_zen_sin",
        "view_B01_zen_cos",
        "view_B01_azi_sin",
        "view_B01_azi_cos",
    ]
    assert feat.band_names == expected
