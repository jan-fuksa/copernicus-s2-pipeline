from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import rasterio

from s2pipe.preprocess.cfg import AngleAssetConfig, PreprocessConfig
from s2pipe.preprocess.inputs import load_download_index, select_assets
from s2pipe.preprocess.run import run_preprocess


def _repo_root() -> Path:
    # tests/preprocess/<this_file>.py -> parents[2] is repo root in the usual layout
    return Path(__file__).resolve().parents[2]


def _fixture_index_path() -> Path:
    root = _repo_root()
    candidates = [
        root / "tests" / "fixtures" / "single_tile" / "meta" / "step1" / "index.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip(
        "Step-1 fixture index.json not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )
    raise AssertionError("unreachable")


def _choose_available_l1c_band(index_path: Path) -> str:
    # Keep it robust: try a few common bands against the actual fixture contents.
    idx = load_download_index(index_path)
    if not idx.pairs:
        pytest.skip("Fixture index.json contains no pairs.")
    pair0 = idx.pairs[0]

    candidates = ("B02", "B01", "B03", "B04", "B08", "B11", "B12")
    for b in candidates:
        try:
            _ = select_assets(
                pair0,
                idx,
                l1c_bands=(b,),
                need_l1c_tile_metadata=False,
                need_l2a_tile_metadata=False,
                need_scl_20m=False,
                require_present=True,
            )
            return b
        except Exception:
            continue

    pytest.skip(
        "No candidate L1C band found in fixture. Tried: " + ", ".join(candidates)
    )
    raise AssertionError("unreachable")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _find_first_sample(step2_index: dict) -> dict:
    samples = step2_index.get("samples", [])
    assert isinstance(samples, list) and samples, "Step-2 index.json has no samples"
    s0 = samples[0]
    assert isinstance(s0, dict), "Step-2 index.json sample record is not a dict"
    return s0


def _find_ok_manifest_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    oks = [r for r in records if isinstance(r, dict) and r.get("status") == "ok"]
    assert oks, "No status='ok' record found in run manifest"
    # max_pairs=1 => a single ok record expected
    return oks[0]


@pytest.mark.integration
def test_step2_export_writes_angles_tif_and_records_it_in_meta_and_indexes(
    tmp_path: Path,
) -> None:
    index_path = _fixture_index_path()
    band = _choose_available_l1c_band(index_path)

    cfg = PreprocessConfig(
        index_json=index_path,
        out_dir=tmp_path,
        run_id="pytest_angles_on",
        max_pairs=1,
        target_grid_ref="scl_20m",
        l1c_bands=(band,),
        to_toa_reflectance=True,
        upsample_method="bilinear",
        downsample_method="average",
        valid_pixel_mask="all_in_one",
        angles=AngleAssetConfig(
            enabled=True,
            include_sun=True,
            include_view=True,
            encode="sin_cos",
            view_mode="per_band",
            view_bands=(band,),  # keep deterministic channel count
            output_name="angles.tif",
        ),
    )

    result = run_preprocess(cfg)
    assert result.processed_count == 1
    assert result.failed_count == 0

    step2_index_path = tmp_path / "meta" / "step2" / "index.json"
    assert step2_index_path.exists()

    step2_index = _read_json(step2_index_path)
    s0 = _find_first_sample(step2_index)

    paths = s0.get("paths", {})
    assert isinstance(paths, dict)
    assert "x" in paths and "y" in paths and "meta" in paths
    assert "angles" in paths, "angles path missing from Step-2 index sample record"

    x_path = tmp_path / paths["x"]
    y_path = tmp_path / paths["y"]
    meta_path = tmp_path / paths["meta"]
    angles_path = tmp_path / paths["angles"]

    assert x_path.exists()
    assert y_path.exists()
    assert meta_path.exists()
    assert angles_path.exists()

    # x.tif should contain ONLY the selected L1C bands
    with rasterio.open(x_path) as ds:
        assert ds.count == 2
        assert ds.width > 0 and ds.height > 0

    # angles.tif should be coarse (typically ~23x23) and have expected channel count.
    # With include_sun + include_view + sin/cos + per_band (B=1): 4 + 4*1 = 8.
    with rasterio.open(angles_path) as ds:
        assert ds.count == 8
        assert ds.width <= 64 and ds.height <= 64
        assert ds.width > 0 and ds.height > 0

    # meta.json should reference angles as a local file name, and include assets.angles
    meta = _read_json(meta_path)
    assert meta.get("paths", {}).get("angles") == "angles.tif"
    assert "angles" in meta.get("assets", {}), "meta.assets.angles missing"
    ainfo = meta["assets"]["angles"]
    assert isinstance(ainfo, dict)
    assert ainfo.get("shape_chw", [0, 0, 0])[0] == 8


@pytest.mark.integration
def test_step2_export_without_angles_does_not_write_angles_or_reference_it(
    tmp_path: Path,
) -> None:
    index_path = _fixture_index_path()
    band = _choose_available_l1c_band(index_path)

    cfg = PreprocessConfig(
        index_json=index_path,
        out_dir=tmp_path,
        run_id="pytest_angles_off",
        max_pairs=1,
        target_grid_ref="scl_20m",
        l1c_bands=(band,),
        angles=AngleAssetConfig(enabled=False),
    )

    result = run_preprocess(cfg)
    assert result.processed_count == 1
    assert result.failed_count == 0

    step2_index_path = tmp_path / "meta" / "step2" / "index.json"
    assert step2_index_path.exists()

    step2_index = _read_json(step2_index_path)
    s0 = _find_first_sample(step2_index)

    paths = s0.get("paths", {})
    assert isinstance(paths, dict)
    assert "x" in paths and "y" in paths and "meta" in paths
    assert "angles" not in paths, (
        "angles path should not be present when angles are disabled"
    )

    meta_path = tmp_path / paths["meta"]
    meta = _read_json(meta_path)

    assert "angles" not in meta.get("paths", {}), (
        "meta.paths.angles should not exist when disabled"
    )
    assert "angles" not in meta.get("assets", {}), (
        "meta.assets.angles should not exist when disabled"
    )


@pytest.mark.integration
def test_step2_run_manifest_records_angles_path_when_enabled_and_omits_when_disabled(
    tmp_path: Path,
) -> None:
    index_path = _fixture_index_path()
    band = _choose_available_l1c_band(index_path)

    # --- enabled ---
    cfg_on = PreprocessConfig(
        index_json=index_path,
        out_dir=tmp_path / "on",
        run_id="pytest_manifest_on",
        max_pairs=1,
        target_grid_ref="scl_20m",
        l1c_bands=(band,),
        angles=AngleAssetConfig(
            enabled=True,
            include_sun=True,
            include_view=True,
            encode="sin_cos",
            view_mode="per_band",
            view_bands=(band,),
            output_name="angles.tif",
        ),
    )
    res_on = run_preprocess(cfg_on)
    assert res_on.processed_count == 1
    assert res_on.failed_count == 0

    manifest_on = (
        tmp_path / "on" / "meta" / "step2" / "runs" / "run=pytest_manifest_on.jsonl"
    )
    assert manifest_on.exists(), "Run manifest was not created for enabled run"

    recs_on = _read_jsonl(manifest_on)
    ok_on = _find_ok_manifest_record(recs_on)
    paths_on = ok_on.get("paths", {})
    assert isinstance(paths_on, dict)
    assert "angles" in paths_on, (
        "Run manifest should include paths.angles when angles are enabled"
    )

    # Ensure the referenced file exists
    angles_on_path = (tmp_path / "on") / paths_on["angles"]
    assert angles_on_path.exists(), (
        f"angles path in manifest does not exist: {angles_on_path}"
    )

    # --- disabled ---
    cfg_off = PreprocessConfig(
        index_json=index_path,
        out_dir=tmp_path / "off",
        run_id="pytest_manifest_off",
        max_pairs=1,
        target_grid_ref="scl_20m",
        l1c_bands=(band,),
        angles=AngleAssetConfig(enabled=False),
    )
    res_off = run_preprocess(cfg_off)
    assert res_off.processed_count == 1
    assert res_off.failed_count == 0

    manifest_off = (
        tmp_path / "off" / "meta" / "step2" / "runs" / "run=pytest_manifest_off.jsonl"
    )
    assert manifest_off.exists(), "Run manifest was not created for disabled run"

    recs_off = _read_jsonl(manifest_off)
    ok_off = _find_ok_manifest_record(recs_off)
    paths_off = ok_off.get("paths", {})
    assert isinstance(paths_off, dict)
    assert "angles" not in paths_off, (
        "Run manifest must not include paths.angles when angles are disabled"
    )
