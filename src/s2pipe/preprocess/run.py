from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cfg import PreprocessConfig
from .inputs import DownloadIndex, IndexPair, load_download_index, select_assets
from .raster import (
    Raster,
    RasterGrid,
    grid_from_reference_raster,
    read_raster,
    stack_rasters,
)
from .resample import resample_raster
from .angles import parse_tile_metadata_angles, angles_to_sin_cos_features
from .labels import scl_to_labels_with_meta
from .export import write_processed_sample

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessResult:
    run_id: str
    run_manifest_path: Path | None
    stats_path: Path | None
    processed_count: int
    failed_count: int = 0
    skipped_count: int = 0
    step2_index_path: Path | None = None


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _default_run_id() -> str:
    # e.g. 20260105T142533Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_sensing_id(s: str) -> str:
    # Avoid ":" in filenames. Keep it deterministic.
    return s.replace(":", "").replace("/", "_")


def _relpath_str(root: Path, p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(Path(p).resolve())


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_step2_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": "s2pipe.step2.index.v1",
            "created_utc": _utc_now_iso(),
            "output": {},
            "samples": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    _ensure_parent(path)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _build_target_grid(
    *,
    pair: IndexPair,
    assets: Any,
    cfg: PreprocessConfig,
) -> RasterGrid:
    """Choose a target grid, preferring a reference raster whose resolution matches cfg.target_res_m.

    Preference order:
      1) If target_res_m == 20 and SCL_20m exists -> use SCL grid (common choice for SCL-driven training).
      2) Otherwise, try to find a reference band among selected L1C bands that matches target_res_m.
      3) Fallback: use the first selected L1C band.
    """
    target_res = int(getattr(cfg, "target_res_m", 10))

    # 1) Prefer SCL for 20m
    if target_res == 20 and getattr(assets, "scl_20m", None) is not None:
        return grid_from_reference_raster(assets.scl_20m)

    # 2) Try to pick a band whose native res matches the target
    band_paths: dict[str, Path] = getattr(assets, "l1c_bands", {}) or {}
    for b, p in band_paths.items():
        try:
            g = grid_from_reference_raster(p)
            if int(round(g.res_m)) == target_res:
                return g
        except Exception:
            continue

    # 3) Fallback to first band (deterministic order)
    if band_paths:
        b0 = sorted(band_paths.keys())[0]
        return grid_from_reference_raster(band_paths[b0])

    raise ValueError(
        f"Cannot determine target grid for tile={pair.tile_id} sensing={pair.sensing_start_utc}: "
        f"no reference rasters available."
    )


def _build_x(
    *,
    assets: Any,
    dst_grid: RasterGrid,
    cfg: PreprocessConfig,
) -> Raster:
    """Build input tensor X as Raster (C,H,W) float32.

    Channel order:
      - L1C bands in cfg.l1c_bands order
      - angle feature channels (as returned by angles_to_sin_cos_features)
    """
    rasters: list[Raster] = []
    names: list[str] = []

    # L1C bands
    for b in getattr(cfg, "l1c_bands", ()):
        p = assets.l1c_bands[str(b)]
        r = read_raster(p)

        # Always cast to float32 to allow stacking with float angle channels
        arr = r.to_chw().astype(np.float32, copy=False)  # (1,H,W) or (C,H,W)
        r = Raster(array=arr, grid=r.grid, nodata=None, band_names=[str(b)])

        if r.grid != dst_grid:
            # For reflectance/intensity bands, bilinear is fine; outside areas are typically 0.
            r = resample_raster(r, dst_grid, method="bilinear", dst_nodata=0.0)

        rasters.append(r)
        names.extend(r.band_names or [str(b)])

    # Angles
    angles_cfg = getattr(cfg, "angles", None)
    if angles_cfg is not None and (
        getattr(angles_cfg, "include_sun", False)
        or getattr(angles_cfg, "include_view", False)
    ):
        mtd = getattr(assets, "l1c_tile_metadata", None)
        if mtd is None:
            raise FileNotFoundError(
                "Angles requested but L1C tile metadata XML is missing."
            )
        ang = parse_tile_metadata_angles(mtd, cfg=angles_cfg)
        ang_r = angles_to_sin_cos_features(
            angles=ang, dst_grid=dst_grid, cfg=angles_cfg
        )
        rasters.append(ang_r)
        if ang_r.band_names:
            names.extend(list(ang_r.band_names))

    x = stack_rasters(rasters, band_names=names if names else None)
    # Ensure float32
    x = Raster(
        array=x.to_chw().astype(np.float32, copy=False),
        grid=dst_grid,
        nodata=None,
        band_names=names if names else None,
    )
    return x


def _build_y_and_label_stats(
    *,
    assets: Any,
    dst_grid: RasterGrid,
    cfg: PreprocessConfig,
) -> tuple[Raster, dict[str, Any] | None]:
    """Build label raster Y on dst_grid and optional label stats."""
    scl_path = getattr(assets, "scl_20m", None)
    if scl_path is None:
        raise FileNotFoundError("SCL_20m is required for labels but is missing.")

    scl = read_raster(scl_path)

    labels_cfg = getattr(cfg, "labels", None)
    out = scl_to_labels_with_meta(
        scl=scl,
        dst_grid=dst_grid,
        cfg=labels_cfg,
    )

    label_stats: dict[str, Any] | None = None

    # Be permissive: scl_to_labels may return either array/Raster or (array/Raster, stats)
    y_arr: Any
    if isinstance(out, tuple) and len(out) == 2:
        y_arr, label_stats = out
    else:
        y_arr = out

    if isinstance(y_arr, Raster):
        y = y_arr
    else:
        # Assume 2D labels (H,W)
        y = Raster(
            array=np.asarray(y_arr),
            grid=dst_grid,
            nodata=getattr(labels_cfg, "ignore_index", 255),
            band_names=["labels"],
        )

    return y, label_stats


def _get_s2pipe_version() -> str:
    # Best-effort; do not hard-fail if packaging metadata is missing.
    try:
        import s2pipe  # type: ignore

        v = getattr(s2pipe, "__version__", None)
        if isinstance(v, str) and v:
            return v
    except Exception:
        pass

    try:
        import importlib.metadata as im

        return im.version("s2pipe")
    except Exception:
        return "unknown"


def _build_meta(
    *,
    cfg: PreprocessConfig,
    pair: IndexPair,
    assets: Any,
    x: Raster,
    y: Raster,
    label_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble meta.json (sample-level)."""
    geo = pair.l1c.geofootprint or pair.l2a.geofootprint

    meta: dict[str, Any] = {
        "schema": "s2pipe.sample.meta.v1",
        "created_utc": _utc_now_iso(),
        "s2pipe_version": _get_s2pipe_version(),
        "key": {
            "tile_id": pair.tile_id,
            "sensing_start_utc": pair.sensing_start_utc,
        },
        "grid": {
            "crs": x.grid.crs,
            "width": x.grid.width,
            "height": x.grid.height,
            "res": list(x.grid.res),
            # affine as 6-tuple (a,b,c,d,e,f)
            "transform": [
                x.grid.transform.a,
                x.grid.transform.b,
                x.grid.transform.c,
                x.grid.transform.d,
                x.grid.transform.e,
                x.grid.transform.f,
            ],
        },
        "channels": {
            "x": list(x.band_names or []),
            "y": list(y.band_names or []),
        },
        # Preferred minimal coverage key (renamed from coverage_ratio)
        "coverage": getattr(assets, "coverage_ratio", None),
        "cloud_cover": getattr(assets, "cloud_cover", None),
        "geofootprint": geo,  # GeoJSON (best-effort)
        "step1": {
            # Keep original names and provenance if needed later
            "coverage_ratio": getattr(assets, "coverage_ratio", None),
            "scl_percentages": getattr(assets, "scl_percentages", None),
        },
    }

    if label_stats:
        meta["labels"] = label_stats

    return meta


def run_preprocess(cfg: PreprocessConfig) -> PreprocessResult:
    """Step 2 orchestrator (sequential).

    Normalization is intentionally not implemented yet; cfg.normalize is ignored for now.
    """
    index: DownloadIndex = load_download_index(cfg.index_json)

    out_dir = Path(cfg.out_dir).resolve()
    run_id = (cfg.run_id or _default_run_id()).strip()

    run_manifest_path = out_dir / "meta" / "step2" / "runs" / f"run={run_id}.jsonl"
    step2_index_path = out_dir / "meta" / "step2" / "index.json"

    processed = 0
    failed = 0
    skipped = 0

    # Load/merge existing global Step-2 index
    step2_index = _load_step2_index(step2_index_path)
    step2_index["output"] = {"out_dir": str(out_dir)}
    samples: list[dict[str, Any]] = (
        step2_index.get("samples")
        if isinstance(step2_index.get("samples"), list)
        else []
    )
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for s in samples:
        try:
            k = s.get("key", {})
            kk = (str(k.get("tile_id")), str(k.get("sensing_start_utc")))
            by_key[kk] = s
        except Exception:
            continue

    max_pairs = getattr(cfg, "max_pairs", None)

    for i, pair in enumerate(index.pairs):
        if max_pairs is not None and i >= int(max_pairs):
            break

        try:
            # Select assets
            assets = select_assets(
                pair,
                index,
                l1c_bands=list(getattr(cfg, "l1c_bands", ())),
                need_l1c_tile_metadata=True,
                need_l2a_tile_metadata=False,
                need_scl_20m=True,
                require_present=True,
            )

            # Target grid
            dst_grid = _build_target_grid(pair=pair, assets=assets, cfg=cfg)

            # X/Y
            x = _build_x(assets=assets, dst_grid=dst_grid, cfg=cfg)
            y, label_stats = _build_y_and_label_stats(
                assets=assets, dst_grid=dst_grid, cfg=cfg
            )

            # Meta
            meta = _build_meta(
                cfg=cfg, pair=pair, assets=assets, x=x, y=y, label_stats=label_stats
            )

            # Write sample
            sample = write_processed_sample(
                out_dir=out_dir,
                tile_id=pair.tile_id,
                sensing_start_utc=pair.sensing_start_utc,
                x=x,
                y=y,
                meta=meta,
            )

            processed += 1

            # Run manifest record
            rec_ok = {
                "ts_utc": _utc_now_iso(),
                "run_id": run_id,
                "status": "ok",
                "key": {
                    "tile_id": pair.tile_id,
                    "sensing_start_utc": pair.sensing_start_utc,
                },
                "paths": {
                    "x": _relpath_str(out_dir, sample.x_path),
                    "y": _relpath_str(out_dir, sample.y_path),
                    "meta": _relpath_str(out_dir, sample.meta_path),
                },
            }
            _append_jsonl(run_manifest_path, rec_ok)

            # Update in-memory global index
            kk = (pair.tile_id, pair.sensing_start_utc)
            by_key[kk] = {
                "key": {
                    "tile_id": pair.tile_id,
                    "sensing_start_utc": pair.sensing_start_utc,
                },
                "paths": rec_ok["paths"],
                "run_id": run_id,
                "status": "ok",
            }

        except Exception as e:
            failed += 1
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))

            rec_fail = {
                "ts_utc": _utc_now_iso(),
                "run_id": run_id,
                "status": "failed",
                "key": {
                    "tile_id": pair.tile_id,
                    "sensing_start_utc": pair.sensing_start_utc,
                },
                "error": str(e),
                "traceback": err,
            }
            _append_jsonl(run_manifest_path, rec_fail)
            log.exception(
                "Preprocess failed for tile=%s sensing=%s",
                pair.tile_id,
                pair.sensing_start_utc,
            )

    # Write global Step-2 index (atomic)
    step2_index["updated_utc"] = _utc_now_iso()
    step2_index["samples"] = list(by_key.values())
    _write_json_atomic(step2_index_path, step2_index)

    return PreprocessResult(
        run_id=run_id,
        run_manifest_path=run_manifest_path,
        stats_path=None,
        processed_count=processed,
        failed_count=failed,
        skipped_count=skipped,
        step2_index_path=step2_index_path,
    )
