import os
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from s2pipe.preprocess.cfg import LabelConfig
from s2pipe.preprocess.inputs import load_download_index, select_assets
from s2pipe.preprocess.labels import scl_to_labels_with_meta, scl_to_labels
from s2pipe.preprocess.raster import (
    Raster,
    RasterGrid,
    read_raster,
    grid_from_reference_raster,
)


def _make_dummy_grid(*, width: int, height: int, res: float = 20.0) -> RasterGrid:
    transform = from_origin(0.0, float(height) * res, res, res)
    return RasterGrid(
        crs="EPSG:32633",
        transform=transform,
        width=width,
        height=height,
        res=(float(res), float(res)),
    )


def test_scl_to_labels_mapping_unknown_to_ignore_and_nodata_to_ignore_with_meta():
    grid = _make_dummy_grid(width=4, height=4, res=20.0)

    scl = np.array(
        [
            [0, 1, 2, 3],
            [4, 0, 2, 99],
            [1, 2, 0, 4],
            [3, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    scl_r = Raster(array=scl, grid=grid, nodata=0)

    cfg = LabelConfig(ignore_index=255, mapping={1: 10, 2: 11}, src_value_range=256)

    y_r, meta = scl_to_labels_with_meta(scl=scl_r, dst_grid=grid, cfg=cfg)
    a = np.asarray(y_r.array)

    expected = np.array(
        [
            [255, 10, 11, 255],
            [255, 255, 11, 255],
            [10, 11, 255, 255],
            [255, 11, 10, 255],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(a, expected)

    assert meta["valid_pixel_count"] == 7
    assert meta["total_pixel_count"] == 16
    assert meta["valid_pixel_ratio"] == 7.0 / 16.0

    lp = meta["label_pct"]
    assert lp["10"] == 3.0 / 16.0
    assert lp["11"] == 4.0 / 16.0
    assert lp["255"] == 9.0 / 16.0

    sp = meta["scl_pct"]
    assert sp["1"] == 3.0 / 16.0
    assert sp["2"] == 4.0 / 16.0
    assert (
        sp["255"] == 4.0 / 16.0
    )  # nodata=0 -> ignore_index via reproject nodata handling


def test_scl_to_labels_no_mapping_preserves_classes_but_applies_nodata_with_meta():
    grid = _make_dummy_grid(width=3, height=2, res=20.0)

    scl = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    scl_r = Raster(array=scl, grid=grid, nodata=0)

    cfg = LabelConfig(ignore_index=255, mapping=None, src_value_range=256)

    y_r, meta = scl_to_labels_with_meta(scl=scl_r, dst_grid=grid, cfg=cfg)
    a = np.asarray(y_r.array)

    expected = np.array([[255, 1, 2], [2, 1, 255]], dtype=np.uint8)
    assert np.array_equal(a, expected)

    assert meta["valid_pixel_count"] == 4
    assert meta["total_pixel_count"] == 6
    assert meta["valid_pixel_ratio"] == 4.0 / 6.0

    lp = meta["label_pct"]
    assert lp["1"] == 2.0 / 6.0
    assert lp["2"] == 2.0 / 6.0
    assert lp["255"] == 2.0 / 6.0


def test_scl_mapping_key_outside_src_value_range_raises():
    grid = _make_dummy_grid(width=2, height=2, res=20.0)

    scl = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    scl_r = Raster(array=scl, grid=grid, nodata=None)

    cfg = LabelConfig(ignore_index=255, mapping={300: 1}, src_value_range=256)

    with pytest.raises(ValueError):
        _ = scl_to_labels(scl=scl_r, dst_grid=grid, cfg=cfg)


def test_labels_support_uint16_with_larger_src_value_range_and_meta():
    grid = _make_dummy_grid(width=2, height=1, res=20.0)

    lab = np.array([[1000, 1001]], dtype=np.uint16)
    lab_r = Raster(array=lab, grid=grid, nodata=None)

    cfg = LabelConfig(ignore_index=65535, mapping={1000: 7}, src_value_range=2048)

    y_r, meta = scl_to_labels_with_meta(scl=lab_r, dst_grid=grid, cfg=cfg)
    a = np.asarray(y_r.array)

    assert a.dtype == np.uint16
    assert a[0, 0] == 7
    assert a[0, 1] == 65535

    assert meta["valid_pixel_count"] == 1
    assert meta["total_pixel_count"] == 2
    assert meta["valid_pixel_ratio"] == 0.5
    assert meta["label_pct"]["7"] == 0.5
    assert meta["label_pct"]["65535"] == 0.5


def _testdata_root() -> Path:
    env = os.environ.get("S2PIPE_TESTDATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[1] / "fixtures" / "step1_single_tile"
    ).resolve()


@pytest.mark.integration
def test_scl_to_labels_integration_identity_on_20m_grid_with_meta():
    root = _testdata_root()
    index_path = root / "meta" / "manifest" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    index = load_download_index(index_path)
    assert len(index.pairs) >= 1
    pair = index.pairs[0]

    assets = select_assets(
        pair,
        index,
        l1c_bands=[],
        need_l1c_tile_metadata=False,
        need_l2a_tile_metadata=False,
        need_scl_20m=True,
        require_present=True,
    )
    assert assets.scl_20m is not None

    scl = read_raster(assets.scl_20m)
    dst_grid = grid_from_reference_raster(assets.scl_20m)

    identity = {i: i for i in range(0, 12)}
    cfg = LabelConfig(ignore_index=255, mapping=identity, src_value_range=256)

    y_r, meta = scl_to_labels_with_meta(scl=scl, dst_grid=dst_grid, cfg=cfg)
    a = np.asarray(y_r.array)

    assert a.shape == (dst_grid.height, dst_grid.width)
    uniq = set(np.unique(a).tolist())
    allowed = set(range(0, 12)) | {255}
    assert uniq.issubset(allowed)

    assert "valid_pixel_ratio" in meta
    assert "scl_pct" in meta
    assert "label_pct" in meta
