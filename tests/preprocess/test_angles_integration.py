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

    for pair in index.pairs:
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
        )

        ang = parse_tile_metadata_angles(assets.l1c_tile_metadata, cfg=cfg)

        # Preventing previous error (×10): we typically expect coarse grid step in kilometers.
        assert 1000.0 <= ang.src_grid.res[0] <= 20000.0
        assert 1000.0 <= ang.src_grid.res[1] <= 20000.0

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

        finite = np.isfinite(x)
        assert (
            int(finite.sum()) > 0
        )  # there may be a lot of NaN for low coverage_ration, but not all

        # sin/cos in [-1, 1] for finite pixels
        absmax = float(np.max(np.abs(x[finite])))
        assert absmax <= 1.0001
