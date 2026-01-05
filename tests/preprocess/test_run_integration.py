from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import rasterio

from s2pipe.preprocess.cfg import PreprocessConfig, AngleFeatureConfig, NormalizeConfig
from s2pipe.preprocess.inputs import load_download_index, select_assets
from s2pipe.preprocess.raster import grid_from_reference_raster
from s2pipe.preprocess.run import run_preprocess


def _testdata_root() -> Path:
    env = os.environ.get("S2PIPE_TESTDATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[1] / "fixtures" / "step1_single_tile"
    ).resolve()


@pytest.mark.integration
def test_run_preprocess_single_tile_end_to_end(tmp_path: Path) -> None:
    root = _testdata_root()
    index_path = root / "meta" / "manifest" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    index = load_download_index(index_path)
    if not index.pairs:
        pytest.skip("No pairs in Step-1 index.json")

    pair = index.pairs[0]

    # Select just enough assets to validate the pipeline.
    # We target 60m to keep the test light, and use a native 60m band (B01) as reference.
    assets = select_assets(
        pair,
        index,
        l1c_bands=("B01",),
        need_l1c_tile_metadata=False,
        need_l2a_tile_metadata=False,
        need_scl_20m=True,
        require_present=True,
    )

    if "B01" not in assets.l1c_bands or not assets.l1c_bands["B01"].exists():
        pytest.skip("Fixture does not contain B01.")
    if assets.scl_20m is None or not assets.scl_20m.exists():
        pytest.skip("Fixture does not contain SCL_20m.")

    # Expected target grid from the reference 60m band (B01)
    expected_grid = grid_from_reference_raster(assets.l1c_bands["B01"])

    out_dir = tmp_path / "out"

    cfg = PreprocessConfig(
        index_json=index_path,
        out_dir=out_dir,
        run_id="test",
        max_pairs=1,
        target_res_m=60,
        l1c_bands=("B01",),
        angles=AngleFeatureConfig(
            include_sun=False, include_view=False, encode_sin_cos=True
        ),
        normalize=NormalizeConfig(mode="none"),
    )

    result = run_preprocess(cfg)

    # Basic result checks
    assert result.run_id == "test"
    assert result.processed_count == 1
    assert result.failed_count == 0
    assert result.step2_index_path is not None
    assert result.run_manifest_path is not None

    # Run manifest exists and contains one OK record
    assert result.run_manifest_path.exists()
    lines = result.run_manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec["status"] == "ok"
    assert rec["key"]["tile_id"] == pair.tile_id
    assert rec["key"]["sensing_start_utc"] == pair.sensing_start_utc
    assert "paths" in rec and set(rec["paths"].keys()) == {"x", "y", "meta"}

    # Global Step-2 index exists and contains the sample
    step2_index_path = result.step2_index_path
    assert step2_index_path.exists()
    step2 = json.loads(step2_index_path.read_text(encoding="utf-8"))
    assert step2.get("schema") == "s2pipe.step2.index.v1"
    assert isinstance(step2.get("samples"), list)
    assert len(step2["samples"]) >= 1

    # Find the sample entry for our key
    sample = None
    for s in step2["samples"]:
        k = s.get("key", {})
        if (
            k.get("tile_id") == pair.tile_id
            and k.get("sensing_start_utc") == pair.sensing_start_utc
        ):
            sample = s
            break
    assert sample is not None, "Sample not found in Step-2 index.json"
    assert sample.get("status") == "ok"
    assert "paths" in sample

    # The referenced files must exist
    x_path = out_dir / sample["paths"]["x"]
    y_path = out_dir / sample["paths"]["y"]
    meta_path = out_dir / sample["paths"]["meta"]

    assert x_path.exists()
    assert y_path.exists()
    assert meta_path.exists()

    # Validate X GeoTIFF basic properties (grid match and single band)
    with rasterio.open(x_path) as ds:
        assert ds.count == 1  # only B01 (angles disabled)
        assert ds.width == expected_grid.width
        assert ds.height == expected_grid.height
        assert ds.crs is not None and ds.crs.to_string() == expected_grid.crs
        assert ds.transform == expected_grid.transform

    # Validate Y GeoTIFF basic properties (same grid)
    with rasterio.open(y_path) as ds:
        assert ds.count == 1
        assert ds.width == expected_grid.width
        assert ds.height == expected_grid.height
        assert ds.crs is not None and ds.crs.to_string() == expected_grid.crs
        assert ds.transform == expected_grid.transform

    # Validate meta.json contains key fields
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("schema") == "s2pipe.sample.meta.v1"
    assert meta.get("s2pipe_version") is not None
    assert meta.get("key", {}).get("tile_id") == pair.tile_id
    assert meta.get("key", {}).get("sensing_start_utc") == pair.sensing_start_utc
    assert "geofootprint" in meta
    assert "coverage" in meta
    assert "cloud_cover" in meta

    # NEW: label stats must be present (scl_to_labels returns them)
    assert "labels" in meta, (
        "Expected label statistics in meta.json under key 'labels'."
    )
    assert isinstance(meta["labels"], dict)
    # Keep this minimal: just check that something meaningful is present.
    # (Exact keys depend on your labels.py implementation.)
    assert len(meta["labels"]) > 0

    # Channel names should reflect only B01 (angles disabled)
    ch = meta.get("channels", {})
    assert ch.get("x") == ["B01"]
    assert isinstance(ch.get("y"), list) and len(ch.get("y")) == 1
