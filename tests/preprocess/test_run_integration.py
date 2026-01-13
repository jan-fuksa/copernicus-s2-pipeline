from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from dataclasses import replace

import numpy as np
import pytest
import rasterio

from s2pipe.preprocess.inputs import load_download_index, select_assets
from s2pipe.preprocess.run import run_preprocess
from s2pipe.preprocess.cfg import PreprocessConfig, AngleAssetConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_step1_index_json() -> Path:
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _join_if_relative(root: Path, p: str | Path) -> Path:
    pp = Path(p)
    return (root / pp) if not pp.is_absolute() else pp


def _choose_available_l1c_band(index_path: Path) -> str:
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

    pytest.skip("No candidate L1C band found in fixture.")
    raise AssertionError("unreachable")


def _make_cfg(index_json: Path, out_dir: Path, run_id: str, band: str):
    angles_cfg = AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="per_band",
        view_bands=(band,),
        detector_aggregate="nanmean",
        output_name="angles.tif",
    )

    sig = inspect.signature(PreprocessConfig)
    kwargs: dict[str, Any] = {}

    for name, p in sig.parameters.items():
        if name == "index_json":
            kwargs[name] = index_json
        elif name == "out_dir":
            kwargs[name] = out_dir
        elif name == "run_id":
            kwargs[name] = run_id
        elif name == "max_pairs":
            kwargs[name] = 1
        elif name == "l1c_bands":
            kwargs[name] = (band,)
        elif name == "angles" and angles_cfg is not None:
            kwargs[name] = angles_cfg
        elif name == "target_grid_ref":
            # preferred newer style: derive grid from SCL_20m
            kwargs[name] = "scl_20m"
        elif name == "target_res_m":
            # older style: target resolution (SCL is 20 m)
            kwargs[name] = 20

    # Ensure all required parameters are present.
    for name, p in sig.parameters.items():
        if p.default is inspect._empty and name not in kwargs:
            raise TypeError(
                f"PreprocessConfig requires parameter {name!r} without default; "
                f"test helper _make_cfg does not know how to set it."
            )

    return PreprocessConfig(**kwargs)


def _find_step2_index_json(out_dir: Path) -> Path:
    candidates = [
        out_dir / "meta" / "step2" / "index.json",
        out_dir / "meta" / "manifest" / "step2" / "index.json",
        out_dir / "meta" / "step2" / "manifest" / "index.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # As a last resort, search.
    hits = (
        list((out_dir / "meta").rglob("step2/index.json"))
        if (out_dir / "meta").exists()
        else []
    )
    if hits:
        return hits[0]

    raise AssertionError(f"Step-2 global index.json not found under {out_dir}/meta")


def _extract_first_sample_record(step2_index: dict[str, Any]) -> dict[str, Any]:
    samples = step2_index.get("samples")
    if isinstance(samples, list) and samples:
        assert isinstance(samples[0], dict)
        return samples[0]

    # Some variants might store records under another key; try a few fallbacks.
    for k in ("items", "records", "pairs"):
        v = step2_index.get(k)
        if isinstance(v, list) and v:
            assert isinstance(v[0], dict)
            return v[0]

    raise AssertionError(
        "Step-2 index.json does not contain any sample records (no 'samples' / fallback lists)."
    )


def _with_normalize(
    cfg: PreprocessConfig, *, mode: str, stats_path: Path
) -> PreprocessConfig:
    """Return cfg with cfg.normalize updated (mode + stats path).

    This test helper is intentionally defensive to support multiple in-repo variants.
    """
    if not hasattr(cfg, "normalize"):
        pytest.skip(
            "PreprocessConfig has no 'normalize' field; normalization modes are not supported."
        )

    norm = getattr(cfg, "normalize")
    if norm is None:
        pytest.skip(
            "PreprocessConfig.normalize is None; normalization modes are not supported."
        )

    kwargs: dict[str, object] = {}
    if hasattr(norm, "mode"):
        kwargs["mode"] = mode
    else:
        pytest.skip("NormalizeConfig has no 'mode' field.")

    if hasattr(norm, "stats_path"):
        kwargs["stats_path"] = Path(stats_path)
    elif hasattr(norm, "stats_json"):
        # Legacy name
        kwargs["stats_json"] = Path(stats_path)
    else:
        pytest.skip(
            "NormalizeConfig has no stats path field (expected 'stats_path' or 'stats_json')."
        )

    norm2 = replace(norm, **kwargs)
    return replace(cfg, normalize=norm2)


def _with_valid_pixel_mask(cfg: PreprocessConfig, mode: str) -> PreprocessConfig:
    if hasattr(cfg, "valid_pixel_mask"):
        return replace(cfg, valid_pixel_mask=mode)
    return cfg


def _find_mean_std_by_band(
    stats: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract mean_by_band/std_by_band from a flat stats.json structure."""
    if not isinstance(stats, dict):
        raise TypeError("stats must be a dict")

    mean = stats.get("mean_by_band")
    std = stats.get("std_by_band")

    if not isinstance(mean, dict) or not mean:
        raise ValueError("stats.json missing or invalid 'mean_by_band'")
    if not isinstance(std, dict) or not std:
        raise ValueError("stats.json missing or invalid 'std_by_band'")

    if not all(
        isinstance(k, str) and isinstance(v, (int, float)) for k, v in mean.items()
    ):
        raise ValueError("'mean_by_band' must be a dict[str, float]")
    if not all(
        isinstance(k, str) and isinstance(v, (int, float)) for k, v in std.items()
    ):
        raise ValueError("'std_by_band' must be a dict[str, float]")

    return (
        {k: float(v) for k, v in mean.items()},
        {k: float(v) for k, v in std.items()},
    )


@pytest.mark.integration
def test_run_preprocess_end_to_end_creates_x_y_angles_meta_and_step2_index(
    tmp_path: Path,
) -> None:
    index_json = _find_step1_index_json()
    band = _choose_available_l1c_band(index_json)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_cfg(
        index_json=index_json, out_dir=out_dir, run_id="pytest_e2e", band=band
    )

    run_preprocess(cfg)
    # We do not assume the exact result type beyond "it should not crash".

    # Step-2 index.json must exist and contain at least 1 record.
    step2_index_path = _find_step2_index_json(out_dir)
    step2_index = _read_json(step2_index_path)
    rec0 = _extract_first_sample_record(step2_index)

    # Prefer record.paths if present.
    paths = rec0.get("paths")
    assert isinstance(paths, dict), (
        "Expected Step-2 index sample record to contain dict field 'paths'."
    )

    # Required outputs: x, y, meta, angles (angles key name may vary, but file must exist).
    x_path = _join_if_relative(out_dir, paths.get("x"))
    y_path = _join_if_relative(out_dir, paths.get("y"))
    meta_path = _join_if_relative(out_dir, paths.get("meta"))

    assert x_path.exists(), f"x.tif missing at {x_path}"
    assert y_path.exists(), f"y.tif missing at {y_path}"
    assert meta_path.exists(), f"meta.json missing at {meta_path}"

    # angles: either explicit key 'angles' or any path ending with angles.tif
    angles_path = None
    if "angles" in paths:
        angles_path = _join_if_relative(out_dir, paths["angles"])
    else:
        for v in paths.values():
            if isinstance(v, str) and v.endswith("angles.tif"):
                angles_path = _join_if_relative(out_dir, v)
                break

    assert angles_path is not None, (
        "angles.tif not referenced in Step-2 index record.paths"
    )
    assert angles_path.exists(), f"angles.tif missing at {angles_path}"

    # Validate meta.json includes basic structure and references the local filenames.
    meta = _read_json(meta_path)
    assert isinstance(meta, dict)
    assert meta.get("paths"), "meta.json missing 'paths' field"
    assert meta["paths"].get("x") in ("x.tif", Path(x_path).name)
    assert meta["paths"].get("y") in ("y.tif", Path(y_path).name)
    assert meta["paths"].get("meta") in ("meta.json", Path(meta_path).name)
    assert meta["paths"].get("angles") in ("angles.tif", Path(angles_path).name)

    # Validate x/y rasters are readable.
    with rasterio.open(x_path) as ds:
        assert ds.count >= 1
        assert ds.width > 0 and ds.height > 0

    with rasterio.open(y_path) as ds:
        assert ds.count == 1
        assert ds.width > 0 and ds.height > 0

    # Validate angles raster: coarse grid and expected channel count for sin/cos per-band when configured that way.
    with rasterio.open(angles_path) as ds:
        assert ds.width <= 64 and ds.height <= 64, (
            "angles.tif is expected to be on a coarse grid"
        )
        assert ds.width > 0 and ds.height > 0

        # If your config exports sin/cos and per_band with B=1: 4 + 4*1 = 8 channels.
        # If you later change angle encoding/layout, adjust this expectation accordingly.
        assert ds.count == 8, (
            f"Unexpected angles.tif band count: {ds.count} (expected 8 for B=1 per_band sin/cos)"
        )


@pytest.mark.integration
def test_run_preprocess_normalize_compute_only_writes_stats_json(
    tmp_path: Path,
) -> None:
    """compute_only should at least produce stats.json for the dataset."""
    index_json = _find_step1_index_json()
    band = _choose_available_l1c_band(index_json)

    out_dir = tmp_path / "out_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_cfg(
        index_json=index_json, out_dir=out_dir, run_id="pytest_norm_compute", band=band
    )

    stats_path = out_dir / "meta" / "step2" / "normalize" / "stats.json"
    cfg = _with_normalize(cfg, mode="compute_only", stats_path=stats_path)

    run_preprocess(cfg)

    assert stats_path.exists(), f"Expected stats.json at {stats_path}"
    stats = _read_json(stats_path)
    assert isinstance(stats, dict)

    mean_by_band, std_by_band = _find_mean_std_by_band(stats)
    assert mean_by_band
    assert std_by_band


@pytest.mark.integration
def test_run_preprocess_normalize_apply_with_stats_exports_normalized_x(
    tmp_path: Path,
) -> None:
    """apply_with_stats should use precomputed stats.json and export normalized x.tif."""
    index_json = _find_step1_index_json()
    band = _choose_available_l1c_band(index_json)

    # 1) First pass: compute stats.
    stats_out = tmp_path / "stats_out"
    stats_out.mkdir(parents=True, exist_ok=True)

    cfg1 = _make_cfg(
        index_json=index_json, out_dir=stats_out, run_id="pytest_norm_stats", band=band
    )
    stats_path = stats_out / "meta" / "step2" / "normalize" / "stats.json"
    cfg1 = _with_normalize(cfg1, mode="compute_only", stats_path=stats_path)

    run_preprocess(cfg1)
    assert stats_path.exists(), f"Expected stats.json at {stats_path}"

    # 2) Second pass: apply stats and export samples.
    out_dir = tmp_path / "out_apply"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg2 = _make_cfg(
        index_json=index_json, out_dir=out_dir, run_id="pytest_norm_apply", band=band
    )
    cfg2 = _with_normalize(cfg2, mode="apply_with_stats", stats_path=stats_path)
    cfg2 = _with_valid_pixel_mask(cfg2, "all_in_one")

    run_preprocess(cfg2)

    step2_index_path = _find_step2_index_json(out_dir)
    step2_index = _read_json(step2_index_path)
    rec0 = _extract_first_sample_record(step2_index)

    paths = rec0.get("paths")
    assert isinstance(paths, dict)

    x_path = _join_if_relative(out_dir, paths.get("x"))
    meta_path = _join_if_relative(out_dir, paths.get("meta"))

    assert x_path.exists(), f"x.tif missing at {x_path}"
    assert meta_path.exists(), f"meta.json missing at {meta_path}"

    meta = _read_json(meta_path)
    x_info = (
        meta.get("assets", {}).get("x", {})
        if isinstance(meta.get("assets"), dict)
        else {}
    )
    band_names = x_info.get("band_names") if isinstance(x_info, dict) else None

    # Determine data and mask band indices.
    data_idx = 1
    mask_idx = None
    if isinstance(band_names, list) and band_names:
        if band in band_names:
            data_idx = int(band_names.index(band)) + 1
        # Prefer the aggregated mask 'valid' if present.
        if "valid" in band_names:
            mask_idx = int(band_names.index("valid")) + 1
        else:
            # Fallback to the first mask-like band.
            for i, bn in enumerate(band_names):
                if isinstance(bn, str) and bn.startswith("valid"):
                    mask_idx = i + 1
                    break

    with rasterio.open(x_path) as ds:
        if mask_idx is None:
            # Last band fallback.
            mask_idx = ds.count

        data = ds.read(data_idx).astype(np.float32, copy=False)
        mask = ds.read(mask_idx).astype(np.float32, copy=False)

    valid = mask > 0.5
    assert int(valid.sum()) > 0, "No valid pixels found in the exported validity mask."

    vals = data[valid]
    assert np.isfinite(vals).all(), "Normalized data contains NaN/Inf on valid pixels."

    m = float(vals.mean())
    s = float(vals.std())

    # Heuristic check: after normalization, valid pixels should be roughly zero-mean and unit-std.
    assert abs(m) < 0.2, f"Unexpected normalized mean: {m}"
    assert 0.7 < s < 1.3, f"Unexpected normalized std: {s}"

    # Invalid pixels should be near zero.
    invalid = ~valid
    if int(invalid.sum()) > 0:
        max_abs = float(np.max(np.abs(data[invalid])))
        assert max_abs < 1e-3, (
            f"Expected invalid pixels to be ~0 after fill, got max_abs={max_abs}"
        )
