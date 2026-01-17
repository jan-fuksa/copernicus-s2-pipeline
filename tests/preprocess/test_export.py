from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from s2pipe.preprocess.export import (
    ProcessedSample,
    _append_jsonl,
    _atomic_write_json,
    _deep_merge,
    _grid_to_meta,
    _raster_asset_info,
    _safe_sensing_for_path,
    update_preprocess_manifest,
    update_step2_index,
    write_processed_sample,
)
from s2pipe.preprocess.raster import Raster, RasterGrid


ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _mk_grid() -> RasterGrid:
    return RasterGrid(
        crs="EPSG:32633",
        transform=Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0),
        width=4,
        height=3,
        res=(10.0, -10.0),
    )


def test_safe_sensing_for_path_sanitizes_colons_slashes_spaces():
    assert _safe_sensing_for_path("2020-01-01T12:34:56Z") == "2020-01-01T123456Z"
    assert _safe_sensing_for_path("2020/01/01 12:34:56") == "2020_01_01_123456"


def test_atomic_write_json_creates_parent_and_writes_newline(tmp_path: Path):
    p = tmp_path / "a" / "b" / "meta.json"
    obj = {"a": 1, "b": {"c": 2}}

    _atomic_write_json(p, obj)

    raw = p.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw) == obj


def test_append_jsonl_appends_records(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    _append_jsonl(p, {"i": 1})
    _append_jsonl(p, {"i": 2, "x": "y"})

    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln) for ln in lines] == [{"i": 1}, {"i": 2, "x": "y"}]


def test_deep_merge_recursive_dicts():
    a = {"a": 1, "b": {"x": 1, "y": 2}}
    b = {"b": {"y": 999, "z": 3}, "c": 4}
    out = _deep_merge(a, b)
    assert out == {"a": 1, "b": {"x": 1, "y": 999, "z": 3}, "c": 4}


def test_grid_to_meta_serializes_affine_and_res():
    g = _mk_grid()
    meta = _grid_to_meta(g)

    assert meta["crs"] == "EPSG:32633"
    assert meta["width"] == 4
    assert meta["height"] == 3
    assert meta["res"] == [10.0, -10.0]

    # Affine is serialized to the 6-parameter GDAL-ish form (a,b,c,d,e,f)
    assert meta["transform"] == [10.0, 0.0, 100.0, 0.0, -10.0, 200.0]


def test_raster_asset_info_includes_dtype_shape_grid_nodata_and_band_names():
    g = _mk_grid()
    arr = np.arange(g.height * g.width, dtype=np.uint16).reshape(g.height, g.width)
    r = Raster(array=arr, grid=g, nodata=0, band_names=["b1"])

    info = _raster_asset_info(r)

    assert info["dtype"] == "uint16"
    assert info["shape_chw"] == [1, g.height, g.width]
    assert info["nodata"] == 0
    assert info["band_names"] == ["b1"]
    assert info["grid"]["crs"] == g.crs


def test_write_processed_sample_writes_layout_and_meta_and_extra_assets(tmp_path: Path):
    g = _mk_grid()

    x_arr = np.arange(g.height * g.width, dtype=np.uint16).reshape(g.height, g.width)
    y_arr = (x_arr % 3).astype(np.uint8)
    ang_arr = np.stack(
        [x_arr.astype(np.float32), (x_arr + 1).astype(np.float32)], axis=0
    )  # (C,H,W)

    x = Raster(array=x_arr, grid=g, nodata=0, band_names=["x"])
    y = Raster(array=y_arr, grid=g, nodata=255, band_names=["y"])
    angles = Raster(array=ang_arr, grid=g, nodata=None, band_names=["a0", "a1"])

    sensing = "2020-01-01T12:34:56Z"
    ps = write_processed_sample(
        tmp_path,
        tile_id="33UVP",
        sensing_start_utc=sensing,
        x=x,
        y=y,
        meta_extra={
            "schema": "should_not_override",
            "paths": {"x": "should_not_override"},
            "extra": {"nested": 1},
        },
        extra_assets={"angles": angles},
        extra_asset_filenames={"angles": "angles.tif"},
    )

    assert isinstance(ps, ProcessedSample)
    assert ps.tile_id == "33UVP"
    assert ps.sensing_start_utc == sensing

    sensing_safe = _safe_sensing_for_path(sensing)
    expected_dir = tmp_path / "processed" / "tile=33UVP" / f"sensing={sensing_safe}"
    assert ps.sample_dir == expected_dir

    # Files exist
    assert ps.asset_paths["x"].exists()
    assert ps.asset_paths["y"].exists()
    assert ps.asset_paths["meta"].exists()
    assert ps.asset_paths["angles"].exists()
    assert ps.asset_paths["angles"].name == "angles.tif"

    # meta.json content
    meta = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == "s2pipe.step2.sample_meta.v2"
    assert ISO_Z_RE.match(meta["created_utc"]) is not None
    assert meta["key"] == {"tile_id": "33UVP", "sensing_start_utc": sensing}

    # Paths are relative to sample_dir
    assert meta["paths"]["x"] == "x.tif"
    assert meta["paths"]["y"] == "y.tif"
    assert meta["paths"]["meta"] == "meta.json"
    assert meta["paths"]["angles"] == "angles.tif"

    # Asset info present for x/y/angles
    assert meta["assets"]["x"]["shape_chw"] == [1, g.height, g.width]
    assert meta["assets"]["y"]["shape_chw"] == [1, g.height, g.width]
    assert meta["assets"]["angles"]["shape_chw"] == [2, g.height, g.width]

    # meta_extra merged, but protected keys are not overridden
    assert meta["extra"]["nested"] == 1

    # Basic GeoTIFF smoke-check (no need to test raster.py in depth)
    import rasterio

    with rasterio.open(ps.asset_paths["x"]) as ds:
        assert ds.count == 1
        assert ds.width == g.width
        assert ds.height == g.height
        assert ds.crs is not None and ds.crs.to_string() == g.crs

    with rasterio.open(ps.asset_paths["angles"]) as ds:
        assert ds.count == 2


def test_write_processed_sample_rejects_reserved_extra_asset_names(tmp_path: Path):
    g = _mk_grid()
    x = Raster(array=np.zeros((g.height, g.width), dtype=np.uint8), grid=g)
    y = Raster(array=np.zeros((g.height, g.width), dtype=np.uint8), grid=g)

    with pytest.raises(ValueError, match=r"reserved names"):
        write_processed_sample(
            tmp_path,
            tile_id="33UVP",
            sensing_start_utc="2020-01-01T00:00:00Z",
            x=x,
            y=y,
            extra_assets={"x": x},
        )


def test_update_preprocess_manifest_appends_jsonl(tmp_path: Path):
    p = tmp_path / "run_manifest.jsonl"
    update_preprocess_manifest(p, record={"a": 1})
    update_preprocess_manifest(p, record={"b": 2})

    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln) for ln in lines] == [{"a": 1}, {"b": 2}]


def test_update_step2_index_creates_and_upserts_and_merges_output(tmp_path: Path):
    p = tmp_path / "index.json"
    rec1 = {
        "key": {"tile_id": "33UVP", "sensing_start_utc": "2020-01-01T00:00:00Z"},
        "sample": {"path": "processed/tile=33UVP/..."},
    }

    update_step2_index(p, sample_rec=rec1)
    doc = json.loads(p.read_text(encoding="utf-8"))

    assert doc["schema"] == "s2pipe.step2.index.v2"
    assert ISO_Z_RE.match(doc["created_utc"]) is not None
    assert ISO_Z_RE.match(doc["updated_utc"]) is not None
    assert doc["samples"] == [rec1]

    # Upsert replaces record with same (tile_id, sensing_start_utc)
    rec2 = {
        "key": {"tile_id": "33UVP", "sensing_start_utc": "2020-01-01T00:00:00Z"},
        "sample": {"path": "processed/tile=33UVP/NEW"},
        "extra": 123,
    }
    update_step2_index(p, sample_rec=rec2, output={"n_samples": 1})
    doc2 = json.loads(p.read_text(encoding="utf-8"))

    assert doc2["samples"] == [rec2]
    assert doc2["output"]["n_samples"] == 1
    assert ISO_Z_RE.match(doc2["updated_utc"]) is not None

    # New key appends
    rec3 = {
        "key": {"tile_id": "33UVP", "sensing_start_utc": "2020-01-02T00:00:00Z"},
        "sample": {"path": "processed/tile=33UVP/SECOND"},
    }
    update_step2_index(p, sample_rec=rec3)
    doc3 = json.loads(p.read_text(encoding="utf-8"))
    assert len(doc3["samples"]) == 2
    assert rec2 in doc3["samples"]
    assert rec3 in doc3["samples"]


def test_update_step2_index_handles_malformed_samples_and_output(tmp_path: Path):
    p = tmp_path / "index.json"

    # Malformed doc: samples is not a list; output is not a dict
    p.write_text(
        json.dumps(
            {
                "schema": "s2pipe.step2.index.v2",
                "created_utc": "2020-01-01T00:00:00Z",
                "updated_utc": "2020-01-01T00:00:00Z",
                "output": "bad",
                "samples": {"bad": True},
            }
        ),
        encoding="utf-8",
    )

    rec = {
        "key": {"tile_id": "33UVP", "sensing_start_utc": "2020-01-01T00:00:00Z"},
        "ok": True,
    }
    update_step2_index(p, sample_rec=rec, output={"ok": True})
    doc = json.loads(p.read_text(encoding="utf-8"))

    assert doc["samples"] == [rec]
    assert doc["output"] == {"ok": True}


def test_update_step2_index_validates_key(tmp_path: Path):
    p = tmp_path / "index.json"

    with pytest.raises(ValueError, match=r"sample_rec\.key must be a dict"):
        update_step2_index(p, sample_rec={"key": "nope"})

    with pytest.raises(ValueError, match=r"must contain tile_id"):
        update_step2_index(p, sample_rec={"key": {"tile_id": "33UVP"}})
