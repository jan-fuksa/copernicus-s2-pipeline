from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .angles import angles_to_sin_cos_features, parse_tile_metadata_angles
from .cfg import PreprocessConfig
from .export import (
    update_preprocess_manifest,
    update_step2_index,
    write_processed_sample,
)
from .inputs import load_download_index, select_assets
from .labels import scl_to_labels_with_meta
from .raster import (
    Raster,
    RasterGrid,
    grid_from_reference_raster,
    read_raster,
)
from .resample import resample_raster
from .radiometry import parse_l1c_radiometry
from .normalize import (
    HistogramAccumulator,
    apply_stats_to_bands,
    hist_init,
    hist_update,
    stats_finalize_from_hist,
    stats_load,
    stats_save,
    validate_stats,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessResult:
    run_id: str
    run_manifest_path: Path
    step2_index_path: Path
    processed_count: int
    failed_count: int = 0
    skipped_count: int = 0


@dataclass(frozen=True)
class BuildXResult:
    bands: np.ndarray  # (C,H,W) float32
    valid_masks: np.ndarray  # (C,H,W) uint8 (0/1)
    band_names: list[str]
    grid: RasterGrid

    def to_raster(self, cfg: PreprocessConfig) -> Raster:
        """Convert bands + masks into a final export Raster (CC,H,W)."""
        bands = np.asarray(self.bands, dtype=np.float32)
        masks = np.asarray(self.valid_masks)

        if bands.ndim != 3:
            raise ValueError(
                f"BuildXResult.bands must be (C,H,W), got shape={bands.shape}"
            )
        if masks.shape != bands.shape:
            raise ValueError(
                f"BuildXResult.valid_masks must match bands shape, "
                f"got masks={masks.shape} vs bands={bands.shape}"
            )

        mask_names: list[str] = []
        mask_layers: list[np.ndarray] = []

        mode = str(cfg.valid_pixel_mask)
        if mode == "all_in_one":
            # AND across bands
            m = np.all(masks.astype(bool), axis=0).astype(
                np.float32, copy=False
            )  # (H,W)
            mask_layers = [m[np.newaxis, :, :]]  # (1,H,W)
            mask_names = ["valid"]
        elif mode == "per_band":
            # One mask per band
            m = masks.astype(np.float32, copy=False)  # (C,H,W)
            mask_layers = [m]
            mask_names = [f"valid_{b}" for b in self.band_names]
        else:
            raise ValueError(f"Invalid valid_pixel_mask: {mode!r}")

        out = np.concatenate([bands] + mask_layers, axis=0).astype(
            np.float32, copy=False
        )
        out_names = list(self.band_names) + mask_names

        return Raster(
            array=out,
            grid=self.grid,
            nodata=None,
            band_names=out_names,
        )


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _relpath_str(root: Path, p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(Path(p).resolve())


def _choose_target_grid_ref_path(assets: Any, cfg: PreprocessConfig) -> Path:
    ref = (cfg.target_grid_ref or "").strip()
    if ref.lower() == "scl_20m":
        if assets.scl_20m is None:
            raise FileNotFoundError("target_grid_ref='scl_20m' but SCL_20m is missing.")
        return Path(assets.scl_20m)

    if ref not in assets.l1c_bands:
        raise KeyError(
            f"target_grid_ref={ref!r} not found among selected L1C bands: {sorted(assets.l1c_bands.keys())}"
        )
    return Path(assets.l1c_bands[ref])


def _resolve_stats_path(cfg: PreprocessConfig, out_dir: Path) -> Path:
    """Resolve stats.json path for normalization."""
    if cfg.normalize.stats_path is not None:
        return Path(cfg.normalize.stats_path)
    return Path(out_dir) / "normalize" / "stats.json"


def _build_x(
    *, assets: Any, dst_grid: RasterGrid, cfg: PreprocessConfig
) -> BuildXResult:
    bands: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    names: list[str] = []

    rad = None
    if cfg.to_toa_reflectance:
        if assets.l1c_product_metadata is None:
            raise ValueError(
                "Missing L1C product metadata (product_metadata) required for TOA reflectance."
            )
        rad = parse_l1c_radiometry(Path(assets.l1c_product_metadata))

    for b in cfg.l1c_bands:
        p = Path(assets.l1c_bands[str(b)])
        r = read_raster(p)

        # L1C convention: DN==0 means nodata. Enforce nodata=0 for correct resampling.
        r0 = replace(r, nodata=0)

        # Resample only if grids differ.
        if r0.grid == dst_grid:
            r_w = r0
        else:
            src_m = float(r0.grid.res_m)
            dst_m = float(dst_grid.res_m)

            if src_m < dst_m:
                method = str(cfg.downsample_method)
            elif src_m > dst_m:
                method = str(cfg.upsample_method)
            else:
                # Same nominal resolution but different grid => treat as "upsample" strategy.
                method = str(cfg.upsample_method)

            r_w = resample_raster(r0, dst_grid, method=method, dst_nodata=0.0)

        a_dn = r_w.to_chw()[0]  # uint16/float32 according to resampling
        m = (a_dn != 0).astype(np.uint8)

        a = a_dn.astype(np.float32, copy=False)
        if rad is not None:
            offset, quant = rad.get_band_params(str(b))
            valid = m.astype(bool)
            a[valid] = (a[valid] + float(offset)) / float(quant)

        bands.append(a)
        masks.append(m)
        names.append(str(b))

    bands_chw = np.stack(bands, axis=0).astype(np.float32, copy=False)  # (C,H,W)
    masks_chw = np.stack(masks, axis=0)  # (C,H,W) uint8

    return BuildXResult(
        bands=bands_chw,
        valid_masks=masks_chw,
        band_names=names,
        grid=dst_grid,
    )


def _build_y_and_label_stats(
    *, assets: Any, dst_grid: RasterGrid, cfg: PreprocessConfig
) -> tuple[Raster, dict[str, Any]]:
    if assets.scl_20m is None:
        raise FileNotFoundError("Missing SCL_20m; cannot build labels (y).")

    scl = read_raster(Path(assets.scl_20m))
    y, label_stats = scl_to_labels_with_meta(scl=scl, dst_grid=dst_grid, cfg=cfg.labels)
    return y, label_stats


def _build_angles_asset(*, assets: Any, cfg: PreprocessConfig) -> Raster | None:
    angles_cfg = cfg.angles
    if not angles_cfg.enabled:
        return None

    if assets.l1c_tile_metadata is None:
        raise FileNotFoundError(
            "Angles enabled but L1C tile metadata (MTD_TL.xml) is missing."
        )

    ang = parse_tile_metadata_angles(Path(assets.l1c_tile_metadata), cfg=angles_cfg)
    # Step-2 export: keep angles on the native coarse grid => dst_grid=None
    r = angles_to_sin_cos_features(angles=ang, cfg=angles_cfg, dst_grid=None)

    return Raster(
        array=r.to_chw().astype(np.float32, copy=False),
        grid=r.grid,
        nodata=np.nan,
        band_names=r.band_names,
    )


def run_preprocess(cfg: PreprocessConfig) -> PreprocessResult:
    """Step-2 orchestrator.

    Per (tile_id, sensing_start_utc) pair:
      1) read Step-1 index.json
      2) select required assets
      3) choose target grid from a reference raster (cfg.target_grid_ref)
      4) build X by resampling selected L1C bands to target grid
      5) build Y + label_stats from L2A SCL_20m (nearest)
      6) optionally build coarse-grid angles asset (angles.tif)
      7) export x/y/meta (+ extra assets) and update run manifest + global Step-2 index.json

    Note: normalization behavior is controlled by cfg.normalize.mode ("none" | "compute_only" | "apply_with_stats").
    """
    index = load_download_index(cfg.index_json)
    out_dir = Path(cfg.out_dir).resolve()

    run_id = (cfg.run_id or _default_run_id()).strip()
    run_manifest_path = out_dir / "meta" / "step2" / "runs" / f"run={run_id}.jsonl"
    step2_index_path = out_dir / "meta" / "step2" / "index.json"

    norm_mode = str(cfg.normalize.mode)
    stats_path = _resolve_stats_path(cfg, out_dir)

    processed = 0
    failed = 0
    skipped = 0

    need_angles = bool(cfg.angles.enabled)

    if norm_mode == "compute_only":
        acc: HistogramAccumulator = hist_init(list(cfg.l1c_bands), cfg.normalize)

        for i, pair in enumerate(index.pairs):
            if cfg.max_pairs is not None and i >= int(cfg.max_pairs):
                break

            try:
                # For stats-only run we do not need angles metadata.
                assets = select_assets(
                    pair,
                    index,
                    l1c_bands=tuple(cfg.l1c_bands),
                    need_l1c_product_metadata=bool(cfg.to_toa_reflectance),
                    need_l1c_tile_metadata=False,
                    need_l2a_tile_metadata=False,
                    need_scl_20m=True,
                    require_present=True,
                )

                ref_path = _choose_target_grid_ref_path(assets, cfg)
                dst_grid = grid_from_reference_raster(ref_path)

                bx = _build_x(assets=assets, dst_grid=dst_grid, cfg=cfg)

                hist_update(
                    acc,
                    bx.bands,
                    bx.valid_masks,
                    max_pixels_per_scene=cfg.normalize.max_pixels_per_scene,
                )

            except Exception as e:
                acc.scenes_skipped += 1
                log.warning(
                    "Skipping scene in compute_only mode: tile=%s sensing=%s error=%s",
                    pair.tile_id,
                    pair.sensing_start_utc,
                    str(e),
                )
                continue

        stats = stats_finalize_from_hist(acc, cfg.normalize)
        # Persist preprocessing-relevant metadata for validation on apply.
        stats["to_toa_reflectance"] = bool(cfg.to_toa_reflectance)

        stats_save(
            stats,
            stats_path,
            save_histograms=bool(cfg.normalize.save_histograms),
            acc=acc,
        )

        log.info("Saved normalization stats: %s", stats_path)

        return PreprocessResult(
            run_id=run_id,
            run_manifest_path=run_manifest_path,
            step2_index_path=step2_index_path,
            processed_count=0,
            failed_count=0,
            skipped_count=int(acc.scenes_skipped),
        )

    stats: dict[str, Any] | None = None
    if norm_mode == "apply_with_stats":
        stats = stats_load(stats_path)
        validate_stats(
            stats,
            band_names=list(cfg.l1c_bands),
            cfg=cfg.normalize,
            to_toa_reflectance=bool(cfg.to_toa_reflectance),
        )

    for i, pair in enumerate(index.pairs):
        if cfg.max_pairs is not None and i >= int(cfg.max_pairs):
            break

        key = {"tile_id": pair.tile_id, "sensing_start_utc": pair.sensing_start_utc}

        try:
            assets = select_assets(
                pair,
                index,
                l1c_bands=tuple(cfg.l1c_bands),
                need_l1c_product_metadata=bool(cfg.to_toa_reflectance),
                need_l1c_tile_metadata=need_angles,
                need_l2a_tile_metadata=False,
                need_scl_20m=True,
                require_present=True,
            )

            ref_path = _choose_target_grid_ref_path(assets, cfg)
            dst_grid = grid_from_reference_raster(ref_path)

            bx = _build_x(assets=assets, dst_grid=dst_grid, cfg=cfg)
            if stats is not None:
                bands_norm = apply_stats_to_bands(
                    bx.bands,
                    bx.valid_masks,
                    bx.band_names,
                    stats,
                )
                bx = BuildXResult(
                    bands=bands_norm,
                    valid_masks=bx.valid_masks,
                    band_names=bx.band_names,
                    grid=bx.grid,
                )
            x = bx.to_raster(cfg)

            y, label_stats = _build_y_and_label_stats(
                assets=assets, dst_grid=dst_grid, cfg=cfg
            )

            angles_r = (
                _build_angles_asset(assets=assets, cfg=cfg) if need_angles else None
            )

            extra_assets: dict[str, Raster] = {}
            extra_filenames: dict[str, str] = {}
            if angles_r is not None:
                extra_assets["angles"] = angles_r
                extra_filenames["angles"] = cfg.angles.output_name

            # Scene-level metadata (stored in meta.json)
            geo = pair.l2a.geofootprint or pair.l1c.geofootprint
            meta_extra = {
                "scene": {
                    "cloud_cover": assets.cloud_cover,
                    "coverage_fraction": assets.coverage_ratio,
                    "geofootprint": geo,  # GeoJSON if present in Step-1 index
                    "scl_percentages": assets.scl_percentages,
                },
                "label_stats": label_stats,
            }

            sample = write_processed_sample(
                out_dir,
                tile_id=pair.tile_id,
                sensing_start_utc=pair.sensing_start_utc,
                x=x,
                y=y,
                meta_extra=meta_extra,
                extra_assets=extra_assets or None,
                extra_asset_filenames=extra_filenames or None,
            )

            paths_rel = {
                name: _relpath_str(out_dir, p) for name, p in sample.asset_paths.items()
            }

            ok_rec = {
                "ts_utc": _utc_now_iso(),
                "run_id": run_id,
                "status": "ok",
                "key": key,
                "paths": paths_rel,
            }
            update_preprocess_manifest(run_manifest_path, record=ok_rec)

            # Upsert Step-2 global index
            sample_rec = {
                "key": key,
                "run_id": run_id,
                "status": "ok",
                "paths": paths_rel,
            }
            update_step2_index(
                step2_index_path,
                sample_rec=sample_rec,
                output={
                    "out_dir": str(out_dir),
                    "last_run_id": run_id,
                    "last_run_manifest": _relpath_str(out_dir, run_manifest_path),
                },
            )

            processed += 1

        except Exception as e:
            failed += 1
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))

            err_rec = {
                "ts_utc": _utc_now_iso(),
                "run_id": run_id,
                "status": "error",
                "key": key,
                "error": str(e),
                "traceback": tb,
            }
            update_preprocess_manifest(run_manifest_path, record=err_rec)

            # Also upsert into global index to make failures visible.
            update_step2_index(
                step2_index_path,
                sample_rec={
                    "key": key,
                    "run_id": run_id,
                    "status": "error",
                    "error": str(e),
                },
                output={
                    "out_dir": str(out_dir),
                    "last_run_id": run_id,
                    "last_run_manifest": _relpath_str(out_dir, run_manifest_path),
                },
            )

            log.exception(
                "Step-2 preprocess failed for tile=%s sensing=%s",
                pair.tile_id,
                pair.sensing_start_utc,
            )

    return PreprocessResult(
        run_id=run_id,
        run_manifest_path=run_manifest_path,
        step2_index_path=step2_index_path,
        processed_count=processed,
        failed_count=failed,
        skipped_count=skipped,
    )
