import os
from pathlib import Path

import numpy as np
import pytest

from s2pipe.preprocess.inputs import load_download_index, select_assets
from s2pipe.preprocess.raster import read_raster, grid_from_reference_raster
from s2pipe.preprocess.resample import resample_raster


def _testdata_root() -> Path:
    """Return the root directory for Step-1 fixture data.

    Default:
        tests/fixtures/step1_single_tile

    Override:
        export S2PIPE_TESTDATA_DIR=/abs/path/to/step1_single_tile
    """
    env = os.environ.get("S2PIPE_TESTDATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "fixtures" / "step1_single_tile").resolve()


@pytest.mark.integration
def test_resample_10m_and_60m_to_20m_grid_from_scl():
    root = _testdata_root()
    index_path = root / "meta" / "manifest" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    index = load_download_index(index_path)
    assert len(index.pairs) >= 1, "Expected at least one pair in index.json"
    pair = index.pairs[0]

    assets = select_assets(
        pair,
        index,
        l1c_bands=["B02", "B01"],       # B02=10m, B01=60m
        need_l1c_tile_metadata=False,
        need_l2a_tile_metadata=False,
        need_scl_20m=True,
        require_present=True,
    )
    assert assets.scl_20m is not None, "SCL 20m is required for this test."

    # Load rasters
    b02 = read_raster(assets.l1c_bands["B02"])
    b01 = read_raster(assets.l1c_bands["B01"])

    # Target grid: use SCL (20m) as reference
    target_grid = grid_from_reference_raster(assets.scl_20m)

    # Sanity: expected approximate resolutions (UTM meters)
    assert abs(target_grid.res_m - 20.0) < 0.5, f"Unexpected target res_m={target_grid.res_m}"
    assert abs(b02.grid.res_m - 10.0) < 0.5, f"Unexpected B02 res_m={b02.grid.res_m}"
    assert abs(b01.grid.res_m - 60.0) < 1.0, f"Unexpected B01 res_m={b01.grid.res_m}"

    # Resample:
    # - 10m -> 20m: average is typically appropriate for downsampling reflectance
    # - 60m -> 20m: bilinear is fine for upsampling
    b02_20 = resample_raster(b02, target_grid, method="average")
    b01_20 = resample_raster(b01, target_grid, method="bilinear")

    # Check shapes
    H, W = target_grid.height, target_grid.width
    assert np.asarray(b02_20.array).shape == (H, W)
    assert np.asarray(b01_20.array).shape == (H, W)

    # Optional: for integer inputs with interpolating/downsampling methods, output becomes float32
    assert np.asarray(b02_20.array).dtype == np.float32
    assert np.asarray(b01_20.array).dtype == np.float32


@pytest.mark.integration
def test_scl_nearest_preserves_integer_classes_on_20m_grid():
    root = _testdata_root()
    index_path = root / "meta" / "manifest" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    index = load_download_index(index_path)
    assert len(index.pairs) >= 1, "Expected at least one pair in index.json"
    pair = index.pairs[0]

    assets = select_assets(
        pair,
        index,
        l1c_bands=[],                   # no L1C bands needed for this test
        need_l1c_tile_metadata=False,
        need_l2a_tile_metadata=False,
        need_scl_20m=True,
        require_present=True,
    )
    assert assets.scl_20m is not None, "SCL 20m is required for this test."

    scl = read_raster(assets.scl_20m)
    target_grid = grid_from_reference_raster(assets.scl_20m)  # 20m grid

    # Resample SCL to the same grid using nearest (should be a no-op in practice)
    scl_20 = resample_raster(scl, target_grid, method="nearest")

    a = np.asarray(scl_20.array)
    assert a.ndim == 2
    assert a.shape == (target_grid.height, target_grid.width)

    # Nearest on categorical labels must preserve integer-valued classes.
    # (SCL is typically uint8, but we just enforce "integer dtype" and "no fractional values".)
    assert np.issubdtype(a.dtype, np.integer), f"Expected integer dtype for SCL, got {a.dtype}"

    # Defensive: if something upstream caused float conversion, ensure values are integral anyway.
    if np.issubdtype(a.dtype, np.floating):
        assert np.all(np.isclose(a, np.round(a))), "SCL values are not integral after nearest resampling."
