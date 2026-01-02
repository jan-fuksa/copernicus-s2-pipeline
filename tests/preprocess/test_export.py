import json
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
import rasterio

from s2pipe.preprocess.export import (
    write_processed_sample,
    update_preprocess_manifest,
    update_step2_index,
)
from s2pipe.preprocess.raster import Raster, RasterGrid


def _dummy_grid(width: int, height: int, res: float = 20.0) -> RasterGrid:
    transform = from_origin(0.0, float(height) * res, res, res)
    return RasterGrid(
        crs="EPSG:32633",
        transform=transform,
        width=width,
        height=height,
        res=(float(res), float(res)),
    )


def test_write_processed_sample_writes_x_y_and_meta(tmp_path: Path):
    out_dir = tmp_path / "dataset"
    grid = _dummy_grid(16, 12, 20.0)

    x = Raster(
        array=np.random.rand(3, 12, 16).astype(np.float32),
        grid=grid,
        nodata=None,
        band_names=["b1", "b2", "b3"],
    )
    y = Raster(
        array=np.random.randint(0, 5, size=(12, 16), dtype=np.uint8),
        grid=grid,
        nodata=255,
        band_names=["label"],
    )

    meta = {
        "spatial": {
            "footprint_geojson": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        },
        "quality": {"cloud_cover": {"value": 12.3}},
        "labels": {"ignore_index": 255, "valid_pixel_ratio": 0.5},
    }

    sample = write_processed_sample(
        out_dir,
        tile_id="33UWQ",
        sensing_start_utc="2025-12-01T10:12:33Z",
        x=x,
        y=y,
        meta=meta,
    )

    assert sample.x_path.exists()
    assert sample.y_path.exists()
    assert sample.meta_path.exists()

    with rasterio.open(sample.x_path) as ds:
        assert ds.count == 3
        assert ds.width == 16
        assert ds.height == 12
        assert ds.crs.to_string() == "EPSG:32633"

    with rasterio.open(sample.y_path) as ds:
        assert ds.count == 1
        assert ds.width == 16
        assert ds.height == 12

    m = json.loads(sample.meta_path.read_text(encoding="utf-8"))
    assert m["tile_id"] == "33UWQ"
    assert m["sensing_start_utc"] == "2025-12-01T10:12:33Z"
    assert "s2pipe_version" in m
    assert "spatial" in m and "target_grid" in m["spatial"]
    assert m["spatial"]["footprint_geojson"]["type"] == "Polygon"


def test_update_run_manifest_and_global_index(tmp_path: Path):
    out_dir = tmp_path / "dataset"
    grid = _dummy_grid(8, 6, 20.0)

    def mk_sample(ts: str) -> tuple[Path, Path, Path]:
        x = Raster(
            array=np.random.rand(1, 6, 8).astype(np.float32),
            grid=grid,
            nodata=None,
            band_names=["b1"],
        )
        y = Raster(
            array=np.zeros((6, 8), dtype=np.uint8),
            grid=grid,
            nodata=255,
            band_names=["label"],
        )
        meta = {"labels": {"ignore_index": 255}}
        s = write_processed_sample(
            out_dir, tile_id="33UWQ", sensing_start_utc=ts, x=x, y=y, meta=meta
        )
        return s

    s1 = mk_sample("2025-12-01T10:12:33Z")
    s2 = mk_sample("2025-12-02T10:12:33Z")

    manifest_dir = out_dir / "processed" / "manifest"
    run_manifest = manifest_dir / "run_test.json"
    step2_index = manifest_dir / "index.json"

    update_preprocess_manifest(run_manifest, sample=s1, meta={"run_id": "test"})
    update_preprocess_manifest(run_manifest, sample=s2, meta={"run_id": "test"})
    update_step2_index(step2_index, sample=s1, dataset_meta={"target_res_m": 20})
    update_step2_index(step2_index, sample=s2, dataset_meta={"target_res_m": 20})

    run_doc = json.loads(run_manifest.read_text(encoding="utf-8"))
    assert len(run_doc["samples"]) == 2

    idx_doc = json.loads(step2_index.read_text(encoding="utf-8"))
    assert "dataset" in idx_doc
    assert idx_doc["dataset"]["target_res_m"] == 20
    assert len(idx_doc["samples"]) == 2

    # Paths in index should be relative to out_dir (not absolute)
    for rec in idx_doc["samples"]:
        assert not str(rec["paths"]["x"]).startswith(str(out_dir))
        assert rec["paths"]["x"].startswith("processed/")
