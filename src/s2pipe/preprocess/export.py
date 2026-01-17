from __future__ import annotations

import json
import os
import numpy as np
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .raster import Raster, RasterGrid, write_geotiff


@dataclass(frozen=True)
class ProcessedSample:
    tile_id: str
    sensing_start_utc: str
    sample_dir: Path
    # Keys: x, y, meta, and any extra assets (e.g. angles).
    # y may be None when labels are disabled.
    asset_paths: dict[str, Path | None]


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_sensing_for_path(s: str) -> str:
    # Avoid ':' on Windows/macOS and keep deterministic.
    return s.replace(":", "").replace("/", "_").replace(" ", "_")


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(
                obj, f, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False
            )
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _grid_to_meta(grid: RasterGrid) -> dict[str, Any]:
    t = grid.transform
    return {
        "crs": grid.crs,
        "width": int(grid.width),
        "height": int(grid.height),
        "res": [float(grid.res[0]), float(grid.res[1])],
        "transform": [
            float(t.a),
            float(t.b),
            float(t.c),
            float(t.d),
            float(t.e),
            float(t.f),
        ],
    }


def _raster_asset_info(r: Raster) -> dict[str, Any]:
    arr = r.to_chw()

    # JSON must not contain NaN/Infinity. Preserve semantics via nodata_kind.
    # - nodata_kind="none": nodata is not defined
    # - nodata_kind="value": nodata is a concrete numeric (or other) value
    # - nodata_kind="nan": nodata is represented by NaN in the raster values
    nodata = r.nodata
    nodata_kind = "none"
    nodata_json: Any = None
    if nodata is None:
        nodata_kind = "none"
        nodata_json = None
    else:
        # Typical for angle rasters: NaN indicates invalid pixels.
        if isinstance(nodata, (float, np.floating)) and np.isnan(float(nodata)):
            nodata_kind = "nan"
            nodata_json = None
        elif isinstance(nodata, (int, np.integer)):
            nodata_kind = "value"
            nodata_json = int(nodata)
        elif isinstance(nodata, (np.number,)):
            # Other numpy scalar types
            nodata_kind = "value"
            nodata_json = float(nodata)
        elif isinstance(nodata, float):
            nodata_kind = "value"
            nodata_json = float(nodata)
        else:
            # Rare: non-numeric nodata (kept as-is, but still valid JSON)
            nodata_kind = "value"
            nodata_json = nodata

    info: dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape_chw": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
        "grid": _grid_to_meta(r.grid),
        "nodata": nodata_json,
        "nodata_kind": nodata_kind,
    }
    if r.band_names is not None:
        info["band_names"] = list(r.band_names)
    return info


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def write_processed_sample(
    out_dir: Path,
    *,
    tile_id: str,
    sensing_start_utc: str,
    x: Raster,
    y: Raster | None,
    meta_extra: dict[str, Any] | None = None,
    extra_assets: dict[str, Raster] | None = None,
    extra_asset_filenames: dict[str, str] | None = None,
) -> ProcessedSample:
    """Write training-ready sample (x/y/meta + extra assets) and return paths.

    Layout:
      <out_dir>/processed/tile=<TILE>/sensing=<SENSING>/x.tif
      <out_dir>/processed/tile=<TILE>/sensing=<SENSING>/y.tif
      <out_dir>/processed/tile=<TILE>/sensing=<SENSING>/meta.json
      <out_dir>/processed/tile=<TILE>/sensing=<SENSING>/<filename>  (extra assets)

    By default, extra assets are written as '<name>.tif'. If you need a fixed filename
    (e.g. 'angles.tif'), pass extra_asset_filenames={'angles': 'angles.tif'}.
    """
    out_dir = Path(out_dir)
    sensing_safe = _safe_sensing_for_path(sensing_start_utc)
    sample_dir = out_dir / "processed" / f"tile={tile_id}" / f"sensing={sensing_safe}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Write rasters
    x_path = sample_dir / "x.tif"
    meta_path = sample_dir / "meta.json"

    write_geotiff(x_path, x)

    y_path: Path | None
    if y is None:
        y_path = None
    else:
        y_path = sample_dir / "y.tif"
        write_geotiff(y_path, y)

    asset_paths: dict[str, Path | None] = {
        "x": x_path,
        "y": y_path,
        "meta": meta_path,
    }

    extra_assets = dict(extra_assets or {})
    extra_asset_filenames = dict(extra_asset_filenames or {})

    reserved = {"x", "y", "meta"}
    bad = reserved.intersection(extra_assets.keys())
    if bad:
        raise ValueError(f"extra_assets uses reserved names: {sorted(bad)}")

    for name, raster in extra_assets.items():
        filename = extra_asset_filenames.get(name, f"{name}.tif")
        p = sample_dir / filename
        write_geotiff(p, raster)
        asset_paths[name] = p

    # Build meta.json
    rel_paths: dict[str, str | None] = {}
    for k, v in asset_paths.items():
        rel_paths[k] = Path(v).name if v is not None else None

    meta: dict[str, Any] = {
        "schema": "s2pipe.step2.sample_meta.v2",
        "created_utc": _utc_now_iso(),
        "key": {"tile_id": tile_id, "sensing_start_utc": sensing_start_utc},
        # relative to sample_dir
        "paths": rel_paths,
        "assets": {
            "x": _raster_asset_info(x),
            "y": _raster_asset_info(y) if y is not None else None,
        },
    }
    for name, raster in extra_assets.items():
        meta["assets"][name] = _raster_asset_info(raster)

    if meta_extra:
        # Do not allow overriding core keys.
        protected = {
            "schema",
            "created_utc",
            "key",
            "paths",
            "assets",
        }
        extra_filtered = {k: v for k, v in meta_extra.items() if k not in protected}
        meta = _deep_merge(meta, extra_filtered)

    _atomic_write_json(meta_path, meta)

    return ProcessedSample(
        tile_id=tile_id,
        sensing_start_utc=sensing_start_utc,
        sample_dir=sample_dir,
        asset_paths=asset_paths,
    )


def update_preprocess_manifest(
    run_manifest_path: Path, *, record: dict[str, Any]
) -> None:
    """Append one JSONL record to the Step-2 run manifest."""
    _append_jsonl(run_manifest_path, record)


def update_step2_index(
    step2_index_path: Path,
    *,
    sample_rec: dict[str, Any],
    output: dict[str, Any] | None = None,
) -> None:
    """Upsert a sample record into the global Step-2 index.json.

    If `output` is provided, it is merged into doc['output'] (shallow merge) and the
    'updated_utc' is refreshed.
    """
    step2_index_path = Path(step2_index_path)
    if step2_index_path.exists():
        doc = json.loads(step2_index_path.read_text(encoding="utf-8"))
        if doc.get("schema") != "s2pipe.step2.index.v2":
            raise ValueError(
                f"Unsupported Step-2 index schema: {doc.get('schema')!r} (expected 's2pipe.step2.index.v2')"
            )
    else:
        doc = {
            "schema": "s2pipe.step2.index.v2",
            "created_utc": _utc_now_iso(),
            "updated_utc": _utc_now_iso(),
            "output": {},
            "samples": [],
        }

    samples = doc.get("samples", [])
    if not isinstance(samples, list):
        samples = []

    key = sample_rec.get("key", {})
    if not isinstance(key, dict):
        raise ValueError("sample_rec.key must be a dict")

    tile_id = key.get("tile_id")
    sensing = key.get("sensing_start_utc")
    if tile_id is None or sensing is None:
        raise ValueError("sample_rec.key must contain tile_id and sensing_start_utc")

    replaced = False
    out_samples: list[dict[str, Any]] = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        k = s.get("key", {})
        if (
            isinstance(k, dict)
            and k.get("tile_id") == tile_id
            and k.get("sensing_start_utc") == sensing
        ):
            out_samples.append(sample_rec)
            replaced = True
        else:
            out_samples.append(s)

    if not replaced:
        out_samples.append(sample_rec)

    doc["samples"] = out_samples
    if output:
        out0 = doc.get("output", {})
        if not isinstance(out0, dict):
            out0 = {}
        out0.update(output)
        doc["output"] = out0

    doc["updated_utc"] = _utc_now_iso()
    _atomic_write_json(step2_index_path, doc)
