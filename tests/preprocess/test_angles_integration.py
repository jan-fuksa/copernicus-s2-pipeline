from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from s2pipe.preprocess.angles import (
    parse_tile_metadata_angles,
    angles_to_sin_cos_features,
)
from s2pipe.preprocess.cfg import AngleFeatureConfig
from s2pipe.preprocess.inputs import load_download_index, select_assets
from s2pipe.preprocess.raster import grid_from_reference_raster


def _testdata_root() -> Path:
    env = os.environ.get("S2PIPE_TESTDATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[1] / "fixtures" / "step1_single_tile"
    ).resolve()


@pytest.mark.integration
def test_angles_real_tile_to_scl20m_grid():
    root = _testdata_root()
    index_path = root / "meta" / "manifest" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    index = load_download_index(index_path)
    if not index.pairs:
        pytest.skip("No pairs in Step-1 index.json")

    pair = index.pairs[0]
    assets = select_assets(
        pair,
        index,
        l1c_bands=(),
        need_l1c_tile_metadata=True,
        need_l2a_tile_metadata=False,
        need_scl_20m=True,
        require_present=True,
    )

    if assets.l1c_tile_metadata is None or not assets.l1c_tile_metadata.exists():
        pytest.skip("Missing L1C MTD_TL.xml in fixtures.")
    if assets.scl_20m is None or not assets.scl_20m.exists():
        pytest.skip("Missing SCL_20m in fixtures.")

    dst_grid = grid_from_reference_raster(assets.scl_20m)

    cfg = AngleFeatureConfig(
        include_sun=True,
        include_view=True,
        encode_sin_cos=True,
        view_mode="single",
        view_bands=("B02",),
        detector_aggregate="nanmean",
    )

    ang = parse_tile_metadata_angles(assets.l1c_tile_metadata, cfg=cfg)

    if ang.src_grid.crs != "UNKNOWN":
        assert ang.src_grid.crs == dst_grid.crs

    feat = angles_to_sin_cos_features(angles=ang, dst_grid=dst_grid, cfg=cfg)
    x = np.asarray(feat.array)

    assert x.dtype == np.float32
    assert x.shape[0] == 8  # 4 sun + 4 view (single)
    assert x.shape[1] == dst_grid.height
    assert x.shape[2] == dst_grid.width

    assert feat.band_names == [
        "sun_zen_sin",
        "sun_zen_cos",
        "sun_azi_sin",
        "sun_azi_cos",
        "view_zen_sin",
        "view_zen_cos",
        "view_azi_sin",
        "view_azi_cos",
    ]

    absmax = float(np.nanmax(np.abs(x)))
    assert np.isfinite(absmax)
    assert absmax <= 1.0001

    finite_ratio = float(np.isfinite(x).mean())
    assert finite_ratio > 0.95
    assert float(np.nanstd(x)) > 0.0
