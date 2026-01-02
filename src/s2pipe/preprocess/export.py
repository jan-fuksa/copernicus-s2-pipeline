from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .raster import Raster
from .raster import RasterGrid  # type: ignore
from .raster import write_geotiff


@dataclass(frozen=True)
class ProcessedSample:
    sample_id: str
    tile_id: str
    sensing_start_utc: str
    sample_dir: Path
    x_path: Path
    y_path: Path
    meta_path: Path


# -------------------------
# Helpers
# -------------------------


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_sensing_for_path(sensing_start_utc: str) -> str:
    """Make a filesystem-safe timestamp token."""
    s = sensing_start_utc.strip()
    # common: 2025-12-01T10:12:33Z -> 2025-12-01T10-12-33Z
    s = s.replace(":", "-")
    s = s.replace(".", "-")
    return s


def make_sample_id(tile_id: str, sensing_start_utc: str) -> str:
    return f"{tile_id}__{_safe_sensing_for_path(sensing_start_utc)}"


def _get_s2pipe_version() -> str:
    """Best-effort version string for meta.json."""
    try:
        from importlib.metadata import version  # py3.8+

        return version("s2pipe")
    except Exception:
        # Fallback: try package attribute
        try:
            import s2pipe  # type: ignore

            v = getattr(s2pipe, "__version__", None)
            return str(v) if v else "unknown"
        except Exception:
            return "unknown"


def _affine_to_list(transform: Any) -> list[float]:
    # rasterio Affine is iterable of 6 coefficients.
    try:
        return [float(x) for x in transform]
    except Exception:
        # last resort
        return [
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ]


def _jsonify(obj: Any) -> Any:
    """Convert common non-JSON types to JSON-friendly ones."""
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return str(obj)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(_jsonify(data), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_same_grid(g1: RasterGrid, g2: RasterGrid) -> None:
    if (
        g1.crs != g2.crs
        or g1.transform != g2.transform
        or g1.width != g2.width
        or g1.height != g2.height
        or getattr(g1, "res", None) != getattr(g2, "res", None)
    ):
        raise ValueError("x.grid and y.grid differ; cannot export inconsistent sample.")


def _grid_to_meta(grid: RasterGrid) -> dict[str, Any]:
    return {
        "crs": grid.crs,
        "res": [float(grid.res[0]), float(grid.res[1])]
        if getattr(grid, "res", None) is not None
        else None,
        "width": int(grid.width),
        "height": int(grid.height),
        "transform": _affine_to_list(grid.transform),
    }


def _x_shape_chw(x: Raster) -> list[int]:
    a = np.asarray(x.array)
    if a.ndim == 2:
        return [1, int(a.shape[0]), int(a.shape[1])]
    if a.ndim == 3:
        # expect CHW or HWC; use to_chw if available
        if hasattr(x, "to_chw"):
            chw = x.to_chw()
            return [int(chw.shape[0]), int(chw.shape[1]), int(chw.shape[2])]
        # fallback: assume CHW
        return [int(a.shape[0]), int(a.shape[1]), int(a.shape[2])]
    raise ValueError(f"Unsupported x array ndim={a.ndim}")


def _y_shape_hw(y: Raster) -> list[int]:
    a = np.asarray(y.array)
    if a.ndim == 2:
        return [int(a.shape[0]), int(a.shape[1])]
    if a.ndim == 3:
        # singleton channels tolerated
        if a.shape[0] == 1:
            return [int(a.shape[1]), int(a.shape[2])]
        if a.shape[-1] == 1:
            return [int(a.shape[0]), int(a.shape[1])]
    raise ValueError(
        f"Expected y as 2D labels (or singleton channel), got shape={a.shape}."
    )


# -------------------------
# Public API
# -------------------------


def write_processed_sample(
    out_dir: Path,
    *,
    tile_id: str,
    sensing_start_utc: str,
    x: Raster,
    y: Raster,
    meta: dict[str, Any],
) -> ProcessedSample:
    """Write training-ready sample (x/y/meta) and return paths.

    Output layout:
        <out_dir>/processed/tile=<TILE>/sensing=<SENSING_SAFE>/{x.tif,y.tif,meta.json}
    """
    out_dir = Path(out_dir)
    sample_id = make_sample_id(tile_id, sensing_start_utc)
    sensing_safe = _safe_sensing_for_path(sensing_start_utc)

    sample_dir = out_dir / "processed" / f"tile={tile_id}" / f"sensing={sensing_safe}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Validate grids
    _assert_same_grid(x.grid, y.grid)

    x_path = sample_dir / "x.tif"
    y_path = sample_dir / "y.tif"
    meta_path = sample_dir / "meta.json"

    # Write rasters
    write_geotiff(x_path, x)
    write_geotiff(y_path, y)

    # Build meta.json
    m: dict[str, Any] = dict(meta) if meta is not None else {}

    # Required top-level fields (export-controlled)
    m["schema"] = m.get("schema", "s2pipe.step2.sample_meta")
    m["schema_version"] = int(m.get("schema_version", 1))
    m["s2pipe_version"] = m.get("s2pipe_version", _get_s2pipe_version())

    m["sample_id"] = sample_id
    m["tile_id"] = tile_id
    m["sensing_start_utc"] = sensing_start_utc

    # Paths: keep local (within sample dir) for portability
    m["paths"] = {
        "x": "x.tif",
        "y": "y.tif",
        "meta": "meta.json",
    }

    # Spatial info: ensure GeoJSON lives here (run.py should pass it under spatial.footprint_geojson)
    spatial = dict(m.get("spatial", {}))
    spatial["target_grid"] = _grid_to_meta(x.grid)
    m["spatial"] = spatial

    # x/y descriptors
    x_info = dict(m.get("x", {}))
    x_info.setdefault("band_names", getattr(x, "band_names", None))
    x_info["dtype"] = str(np.asarray(x.array).dtype)
    x_info["shape_chw"] = _x_shape_chw(x)
    m["x"] = x_info

    y_info = dict(m.get("y", {}))
    y_info["dtype"] = str(np.asarray(y.array).dtype)
    y_info["shape_hw"] = _y_shape_hw(y)
    # common label properties (if present)
    if "labels" in m and isinstance(m["labels"], dict):
        y_info.setdefault("ignore_index", m["labels"].get("ignore_index"))
    m["y"] = y_info

    # Write meta.json atomically
    _atomic_write_json(meta_path, m)

    return ProcessedSample(
        sample_id=sample_id,
        tile_id=tile_id,
        sensing_start_utc=sensing_start_utc,
        sample_dir=sample_dir,
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path,
    )


def update_preprocess_manifest(
    run_manifest_path: Path, *, sample: ProcessedSample, meta: dict[str, Any]
) -> None:
    """Append (atomic) entry to Step-2 run manifest (per-run)."""
    run_manifest_path = Path(run_manifest_path)
    now = _utc_now_iso()

    doc = _load_json_if_exists(run_manifest_path)
    if doc is None:
        doc = {
            "schema": "s2pipe.step2.run_manifest",
            "schema_version": 1,
            "created_utc": now,
            "updated_utc": now,
            "run_id": run_manifest_path.stem.replace("run_", ""),
            "samples": [],
            "meta": {},
        }

    doc["updated_utc"] = now

    # Merge run-level meta (shallow)
    if isinstance(meta, dict):
        run_meta = dict(doc.get("meta", {}))
        for k, v in meta.items():
            # do not overwrite existing keys unless explicitly desired
            if k not in run_meta:
                run_meta[k] = v
        doc["meta"] = run_meta

    # Append sample record (relative to processed root if possible)
    out_root = (
        run_manifest_path.parent.parent.parent
        if run_manifest_path.parent.name == "manifest"
        else run_manifest_path.parent
    )
    try:
        rel_x = sample.x_path.relative_to(out_root)
        rel_y = sample.y_path.relative_to(out_root)
        rel_m = sample.meta_path.relative_to(out_root)
    except Exception:
        rel_x, rel_y, rel_m = sample.x_path, sample.y_path, sample.meta_path

    rec = {
        "sample_id": sample.sample_id,
        "tile_id": sample.tile_id,
        "sensing_start_utc": sample.sensing_start_utc,
        "paths": {
            "x": str(rel_x),
            "y": str(rel_y),
            "meta": str(rel_m),
        },
    }

    doc.setdefault("samples", [])
    doc["samples"].append(rec)

    _atomic_write_json(run_manifest_path, doc)


def update_step2_index(
    step2_index_path: Path,
    *,
    sample: ProcessedSample,
    dataset_meta: dict[str, Any] | None = None,
) -> None:
    """Update (atomic) global Step-2 index.json (dataset-level).

    The index is intended as the canonical entrypoint for Step 3.
    Samples are keyed by sample_id; updates overwrite existing sample_id.
    """
    step2_index_path = Path(step2_index_path)
    now = _utc_now_iso()

    doc = _load_json_if_exists(step2_index_path)
    if doc is None:
        doc = {
            "schema": "s2pipe.step2.index",
            "schema_version": 1,
            "created_utc": now,
            "updated_utc": now,
            "dataset": {},
            "samples": [],
        }

    doc["updated_utc"] = now

    # Merge dataset-level metadata (shallow, but do not clobber existing keys)
    if dataset_meta:
        ds = dict(doc.get("dataset", {}))
        for k, v in dataset_meta.items():
            if k not in ds:
                ds[k] = v
        doc["dataset"] = ds

    # Store sample record (relative to out_dir root)
    out_dir = (
        step2_index_path.parent.parent.parent
        if step2_index_path.parent.name == "manifest"
        else step2_index_path.parent
    )
    try:
        rel_x = sample.x_path.relative_to(out_dir)
        rel_y = sample.y_path.relative_to(out_dir)
        rel_m = sample.meta_path.relative_to(out_dir)
    except Exception:
        rel_x, rel_y, rel_m = sample.x_path, sample.y_path, sample.meta_path

    sample_rec = {
        "sample_id": sample.sample_id,
        "tile_id": sample.tile_id,
        "sensing_start_utc": sample.sensing_start_utc,
        "paths": {
            "x": str(rel_x),
            "y": str(rel_y),
            "meta": str(rel_m),
        },
    }

    # Upsert by sample_id
    samples = list(doc.get("samples", []))
    by_id = {
        s.get("sample_id"): s
        for s in samples
        if isinstance(s, dict) and "sample_id" in s
    }
    by_id[sample.sample_id] = sample_rec
    doc["samples"] = list(by_id.values())

    _atomic_write_json(step2_index_path, doc)
