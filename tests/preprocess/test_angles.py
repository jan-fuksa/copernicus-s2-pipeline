from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine

from s2pipe.preprocess.angles import (
    parse_tile_metadata_angles,
    angles_to_sin_cos_features,
)
from s2pipe.preprocess.cfg import AngleFeatureConfig
from s2pipe.preprocess.raster import RasterGrid


def _write_xml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "MTD_TL.xml"
    p.write_text(text, encoding="utf-8")
    return p


def _grid(
    crs: str, transform: Affine, width: int, height: int, res: tuple[float, float]
) -> RasterGrid:
    return RasterGrid(crs=crs, transform=transform, width=width, height=height, res=res)


def test_angles_sun_parse_and_features_identity_grid(tmp_path: Path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root xmlns:n1="urn:test">
  <n1:Geometric_Info>
    <n1:Tile_Geocoding>
      <n1:HORIZONTAL_CS_CODE>EPSG:32633</n1:HORIZONTAL_CS_CODE>
      <n1:Size resolution="20"><n1:NROWS>2</n1:NROWS><n1:NCOLS>2</n1:NCOLS></n1:Size>
      <n1:Geoposition resolution="20">
        <n1:ULX>0</n1:ULX><n1:ULY>40</n1:ULY><n1:XDIM>20</n1:XDIM><n1:YDIM>-20</n1:YDIM>
      </n1:Geoposition>
    </n1:Tile_Geocoding>
    <n1:Tile_Angles>
      <n1:COL_STEP>1</n1:COL_STEP><n1:ROW_STEP>1</n1:ROW_STEP>
      <n1:COL_START>0</n1:COL_START><n1:ROW_START>0</n1:ROW_START>
      <n1:Sun_Angles_Grid>
        <n1:Zenith>
          <n1:Values_List>
            <n1:VALUES>0 90</n1:VALUES>
            <n1:VALUES>45 60</n1:VALUES>
          </n1:Values_List>
        </n1:Zenith>
        <n1:Azimuth>
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

    cfg = AngleFeatureConfig(include_sun=True, include_view=False, encode_sin_cos=True)
    ang = parse_tile_metadata_angles(xml_path, cfg=cfg)

    dst = _grid(
        "EPSG:32633",
        Affine(20, 0, 0, 0, -20, 40),
        width=2,
        height=2,
        res=(20.0, 20.0),
    )

    feat = angles_to_sin_cos_features(angles=ang, dst_grid=dst, cfg=cfg)
    x = np.asarray(feat.array)
    assert x.shape == (4, 2, 2)
    assert feat.band_names == [
        "sun_zen_sin",
        "sun_zen_cos",
        "sun_azi_sin",
        "sun_azi_cos",
    ]

    zen = np.array([[0, 90], [45, 60]], dtype=np.float32)
    azi = np.array([[0, 180], [90, 270]], dtype=np.float32)
    z = np.deg2rad(zen)
    a = np.deg2rad(azi)

    np.testing.assert_allclose(x[0], np.sin(z), rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[1], np.cos(z), rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[2], np.sin(a), rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[3], np.cos(a), rtol=0, atol=1e-6)


def test_view_azimuth_wrap_detector_average_repeated_elements(tmp_path: Path):
    # Two detectors for B02 (bandId=1). Azimuths near 0/360 should average to ~0 using vector mean.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32633</HORIZONTAL_CS_CODE>
      <Size resolution="20"><NROWS>2</NROWS><NCOLS>2</NCOLS></Size>
      <Geoposition resolution="20"><ULX>0</ULX><ULY>40</ULY><XDIM>20</XDIM><YDIM>-20</YDIM></Geoposition>
    </Tile_Geocoding>
    <Tile_Angles>
      <COL_STEP>1</COL_STEP><ROW_STEP>1</ROW_STEP>
      <Sun_Angles_Grid>
        <Zenith><Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List></Zenith>
        <Azimuth><Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List></Azimuth>
      </Sun_Angles_Grid>

      <!-- Repeated per-detector elements -->
      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="0">
        <Zenith><Values_List><VALUES>10 10</VALUES><VALUES>10 10</VALUES></Values_List></Zenith>
        <Azimuth><Values_List><VALUES>359 359</VALUES><VALUES>359 359</VALUES></Values_List></Azimuth>
      </Viewing_Incidence_Angles_Grids>

      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="1">
        <Zenith><Values_List><VALUES>10 10</VALUES><VALUES>10 10</VALUES></Values_List></Zenith>
        <Azimuth><Values_List><VALUES>1 1</VALUES><VALUES>1 1</VALUES></Values_List></Azimuth>
      </Viewing_Incidence_Angles_Grids>
    </Tile_Angles>
  </Geometric_Info>
</root>
"""
    xml_path = _write_xml(tmp_path, xml)

    cfg = AngleFeatureConfig(
        include_sun=False,
        include_view=True,
        encode_sin_cos=True,
        view_mode="single",
        view_bands=("B02",),
        detector_aggregate="nanmean",
    )
    ang = parse_tile_metadata_angles(xml_path, cfg=cfg)

    dst = _grid(
        "EPSG:32633", Affine(20, 0, 0, 0, -20, 40), width=2, height=2, res=(20.0, 20.0)
    )
    feat = angles_to_sin_cos_features(angles=ang, dst_grid=dst, cfg=cfg)

    x = np.asarray(feat.array)
    assert feat.band_names == [
        "view_zen_sin",
        "view_zen_cos",
        "view_azi_sin",
        "view_azi_cos",
    ]

    # Azimuth should average to ~0 degrees -> sin ~ 0, cos ~ 1
    assert np.all(np.abs(x[2]) < 1e-3)
    assert np.all(x[3] > 0.999)


def test_view_mode_per_band_channel_naming_repeated_elements(tmp_path: Path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32633</HORIZONTAL_CS_CODE>
      <Size resolution="20"><NROWS>2</NROWS><NCOLS>2</NCOLS></Size>
      <Geoposition resolution="20"><ULX>0</ULX><ULY>40</ULY><XDIM>20</XDIM><YDIM>-20</YDIM></Geoposition>
    </Tile_Geocoding>
    <Tile_Angles>
      <COL_STEP>1</COL_STEP><ROW_STEP>1</ROW_STEP>
      <Sun_Angles_Grid>
        <Zenith><Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List></Zenith>
        <Azimuth><Values_List><VALUES>0 0</VALUES><VALUES>0 0</VALUES></Values_List></Azimuth>
      </Sun_Angles_Grid>

      <!-- B02 (bandId=1) -->
      <Viewing_Incidence_Angles_Grids bandId="1" detectorId="0">
        <Zenith><Values_List><VALUES>10 10</VALUES><VALUES>10 10</VALUES></Values_List></Zenith>
        <Azimuth><Values_List><VALUES>20 20</VALUES><VALUES>20 20</VALUES></Values_List></Azimuth>
      </Viewing_Incidence_Angles_Grids>

      <!-- B01 (bandId=0) -->
      <Viewing_Incidence_Angles_Grids bandId="0" detectorId="0">
        <Zenith><Values_List><VALUES>30 30</VALUES><VALUES>30 30</VALUES></Values_List></Zenith>
        <Azimuth><Values_List><VALUES>40 40</VALUES><VALUES>40 40</VALUES></Values_List></Azimuth>
      </Viewing_Incidence_Angles_Grids>
    </Tile_Angles>
  </Geometric_Info>
</root>
"""
    xml_path = _write_xml(tmp_path, xml)

    cfg = AngleFeatureConfig(
        include_sun=True,
        include_view=True,
        encode_sin_cos=True,
        view_mode="per_band",
        view_bands=("B02", "B01"),
        detector_aggregate="nanmean",
    )
    ang = parse_tile_metadata_angles(xml_path, cfg=cfg)

    dst = _grid(
        "EPSG:32633", Affine(20, 0, 0, 0, -20, 40), width=2, height=2, res=(20.0, 20.0)
    )
    feat = angles_to_sin_cos_features(angles=ang, dst_grid=dst, cfg=cfg)

    # Sun: 4 channels + per-band view: 2 bands * 4 channels = 8 -> total 12
    assert np.asarray(feat.array).shape == (12, 2, 2)

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
