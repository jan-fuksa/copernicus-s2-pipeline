from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np

from .cfg import NormalizeConfig


_STATS_SCHEMA = "s2pipe.normalize.stats.v1"


@dataclass
class HistogramAccumulator:
    band_names: List[str]
    hist_range: Tuple[float, float]
    hist_bin_width: float
    num_bins: int
    counts: np.ndarray  # (C, num_bins) uint64
    underflow_count: np.ndarray  # (C,) uint64 (diagnostics)
    overflow_count: np.ndarray  # (C,) uint64 (diagnostics)
    scenes_processed: int = 0
    scenes_skipped: int = 0
    rng: np.random.Generator = field(
        repr=False, default_factory=lambda: np.random.default_rng(0)
    )


def hist_init(band_names: List[str], cfg: NormalizeConfig) -> HistogramAccumulator:
    """Initialize histogram accumulator for dataset-scale normalization stats."""
    low, high = float(cfg.hist_range[0]), float(cfg.hist_range[1])
    if not (high > low):
        raise ValueError(f"Invalid hist_range: {cfg.hist_range!r}")

    w = float(cfg.hist_bin_width)
    if not (w > 0.0):
        raise ValueError(f"Invalid hist_bin_width: {cfg.hist_bin_width!r}")

    span = high - low
    num_bins = int(round(span / w))
    if not np.isclose(num_bins * w, span, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"hist_range width must be divisible by hist_bin_width: range={cfg.hist_range} bin_width={w}"
        )
    if num_bins <= 0:
        raise ValueError(f"Invalid num_bins computed: {num_bins}")

    c = len(band_names)
    counts = np.zeros((c, num_bins), dtype=np.uint64)
    under = np.zeros((c,), dtype=np.uint64)
    over = np.zeros((c,), dtype=np.uint64)

    rng = np.random.default_rng(int(cfg.seed))

    return HistogramAccumulator(
        band_names=list(band_names),
        hist_range=(low, high),
        hist_bin_width=w,
        num_bins=num_bins,
        counts=counts,
        underflow_count=under,
        overflow_count=over,
        scenes_processed=0,
        scenes_skipped=0,
        rng=rng,
    )


def _subsample_values(
    rng: np.random.Generator, vals: np.ndarray, max_n: int
) -> np.ndarray:
    """Subsample values without replacement (best-effort)."""
    n = int(vals.size)
    if n <= max_n:
        return vals
    idx = rng.choice(n, size=int(max_n), replace=False)
    return vals[idx]


def hist_update(
    acc: HistogramAccumulator,
    bands: np.ndarray,
    valid_masks: np.ndarray,
    *,
    max_pixels_per_scene: int | None,
) -> None:
    """Update histograms from a single scene/tile.

    Inputs:
      - bands: (C,H,W) float32/float64
      - valid_masks: (C,H,W) uint8/bool (non-zero => valid)
    """
    if bands.ndim != 3:
        raise ValueError(f"bands must be (C,H,W), got shape={bands.shape}")
    if valid_masks.shape != bands.shape:
        raise ValueError(
            f"valid_masks must match bands shape, got {valid_masks.shape} vs {bands.shape}"
        )

    c, _, _ = bands.shape
    if c != len(acc.band_names):
        raise ValueError(
            f"Band count mismatch: bands has C={c}, accumulator has {len(acc.band_names)}"
        )

    low, high = acc.hist_range
    w = acc.hist_bin_width
    nb = acc.num_bins

    for i in range(c):
        m = valid_masks[i]
        v = bands[i]

        vals = v[m != 0].astype(np.float64, copy=False)
        if vals.size == 0:
            continue

        if max_pixels_per_scene is not None:
            vals = _subsample_values(acc.rng, vals, int(max_pixels_per_scene))

        # Map values to bin indices
        idx = np.floor((vals - low) / w).astype(np.int64, copy=False)

        under = idx < 0
        over = idx >= nb

        if under.any():
            acc.underflow_count[i] += np.uint64(int(under.sum()))
        if over.any():
            acc.overflow_count[i] += np.uint64(int(over.sum()))

        idx = np.clip(idx, 0, nb - 1)

        # bincount expects non-negative ints
        bc = np.bincount(idx, minlength=nb).astype(np.uint64, copy=False)
        acc.counts[i] += bc

    acc.scenes_processed += 1


def _percentile_bin_index(counts_1d: np.ndarray, p: float) -> int:
    """Return the histogram bin index corresponding to percentile p (0..100)."""
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"Percentile must be within [0,100], got {p}")

    total = int(counts_1d.sum())
    if total <= 0:
        raise ValueError("Cannot compute percentiles from an empty histogram.")

    # Target rank in [1..total]
    target = int(np.ceil((p / 100.0) * total))
    target = max(1, min(target, total))

    cdf = np.cumsum(counts_1d, dtype=np.uint64)
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, counts_1d.size - 1))
    return idx


def _bin_center(low: float, w: float, idx: int) -> float:
    return float(low + (idx + 0.5) * w)


def stats_finalize_from_hist(
    acc: HistogramAccumulator, cfg: NormalizeConfig
) -> Dict[str, Any]:
    """Finalize dataset stats from histograms (percentiles + moments after clipping)."""
    low, high = acc.hist_range
    w = acc.hist_bin_width
    band_names = list(acc.band_names)

    clip = cfg.clip_percentiles
    if clip is None:
        p_lo, p_hi = 0.0, 100.0
    else:
        p_lo, p_hi = float(clip[0]), float(clip[1])
        if not (0.0 <= p_lo < p_hi <= 100.0):
            raise ValueError(f"Invalid clip_percentiles: {clip!r}")

    clip_low_by_band: Dict[str, float] = {}
    clip_high_by_band: Dict[str, float] = {}
    mean_by_band: Dict[str, float] = {}
    std_by_band: Dict[str, float] = {}

    for i, b in enumerate(band_names):
        h = acc.counts[i]
        total = int(h.sum())
        if total <= 0:
            raise ValueError(f"Empty histogram for band={b!r}; cannot finalize stats.")

        i_lo = _percentile_bin_index(h, p_lo)
        i_hi = _percentile_bin_index(h, p_hi)
        if i_hi < i_lo:
            i_lo, i_hi = i_hi, i_lo

        clip_low = _bin_center(low, w, i_lo)
        clip_high = _bin_center(low, w, i_hi)

        clip_low_by_band[b] = float(clip_low)
        clip_high_by_band[b] = float(clip_high)

        # Moments over the clipped interval [i_lo, i_hi]
        sel = slice(i_lo, i_hi + 1)
        counts_sel = h[sel].astype(np.float64, copy=False)

        n = float(counts_sel.sum())
        if n <= 0.0:
            raise ValueError(
                f"No samples in clipped interval for band={b!r}; cannot finalize stats."
            )

        idxs = np.arange(i_lo, i_hi + 1, dtype=np.float64)
        centers = low + (idxs + 0.5) * w

        mean = float(np.sum(counts_sel * centers) / n)
        var = float(np.sum(counts_sel * (centers - mean) ** 2) / n)
        std = float(np.sqrt(max(var, 0.0)))
        std = float(max(std, 1e-6))

        mean_by_band[b] = mean
        std_by_band[b] = std

    stats: Dict[str, Any] = {
        "schema": _STATS_SCHEMA,
        "bands": band_names,
        "hist_range": [float(low), float(high)],
        "hist_bin_width": float(w),
        "clip_percentiles": [float(p_lo), float(p_hi)]
        if cfg.clip_percentiles is not None
        else None,
        "moments_after_clipping": True,
        "scenes_processed": int(acc.scenes_processed),
        "scenes_skipped": int(acc.scenes_skipped),
        "clip_low_by_band": clip_low_by_band,
        "clip_high_by_band": clip_high_by_band,
        "mean_by_band": mean_by_band,
        "std_by_band": std_by_band,
        "underflow_count_by_band": {
            band_names[i]: int(acc.underflow_count[i]) for i in range(len(band_names))
        },
        "overflow_count_by_band": {
            band_names[i]: int(acc.overflow_count[i]) for i in range(len(band_names))
        },
    }
    return stats


def stats_save(
    stats: Dict[str, Any],
    path: Path,
    *,
    save_histograms: bool = False,
    acc: HistogramAccumulator | None = None,
) -> None:
    """Save stats JSON. Optionally save histogram arrays to a .npz next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    if save_histograms:
        if acc is None:
            raise ValueError(
                "save_histograms=True requires providing the histogram accumulator (acc)."
            )
        npz_path = path.with_name("histogram.npz")
        np.savez_compressed(
            npz_path,
            schema=_STATS_SCHEMA,
            bands=np.array(acc.band_names, dtype=object),
            hist_range=np.array(acc.hist_range, dtype=np.float64),
            hist_bin_width=np.array([acc.hist_bin_width], dtype=np.float64),
            counts=acc.counts,
            underflow_count=acc.underflow_count,
            overflow_count=acc.overflow_count,
            scenes_processed=np.array([acc.scenes_processed], dtype=np.int64),
            scenes_skipped=np.array([acc.scenes_skipped], dtype=np.int64),
        )


def stats_load(path: Path) -> Dict[str, Any]:
    """Load stats JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if not isinstance(stats, dict):
        raise ValueError("Invalid stats file: root must be an object.")
    return stats


def validate_stats(
    stats: Dict[str, Any],
    band_names: List[str],
    cfg: NormalizeConfig,
    *,
    to_toa_reflectance: bool,
) -> None:
    """Validate loaded stats against current configuration."""
    if stats.get("schema") != _STATS_SCHEMA:
        raise ValueError(f"Unsupported stats schema: {stats.get('schema')!r}")

    if list(stats.get("bands", [])) != list(band_names):
        raise ValueError("Stats bands do not match current band order.")

    hr = stats.get("hist_range")
    if not (isinstance(hr, list) and len(hr) == 2):
        raise ValueError("Stats hist_range is missing or invalid.")
    if tuple(map(float, hr)) != (float(cfg.hist_range[0]), float(cfg.hist_range[1])):
        raise ValueError("Stats hist_range does not match current configuration.")

    bw = float(stats.get("hist_bin_width"))
    if bw != float(cfg.hist_bin_width):
        raise ValueError("Stats hist_bin_width does not match current configuration.")

    if not bool(stats.get("moments_after_clipping", False)):
        raise ValueError("Stats must have moments_after_clipping=true.")

    # Optional compatibility check (recommended)
    if "to_toa_reflectance" in stats:
        if bool(stats["to_toa_reflectance"]) != bool(to_toa_reflectance):
            raise ValueError(
                "Stats to_toa_reflectance does not match current preprocessing configuration."
            )

    # clip_percentiles can be None
    if cfg.clip_percentiles is None:
        if stats.get("clip_percentiles") is not None:
            raise ValueError(
                "Stats were computed with clipping but current config disables clipping."
            )
    else:
        sp = stats.get("clip_percentiles")
        if not (isinstance(sp, list) and len(sp) == 2):
            raise ValueError("Stats clip_percentiles missing or invalid.")
        if (float(sp[0]), float(sp[1])) != (
            float(cfg.clip_percentiles[0]),
            float(cfg.clip_percentiles[1]),
        ):
            raise ValueError(
                "Stats clip_percentiles does not match current configuration."
            )


def apply_stats_to_bands(
    bands: np.ndarray,
    valid_masks: np.ndarray,
    band_names: List[str],
    stats: Dict[str, Any],
) -> np.ndarray:
    """Apply percentile clipping and per-band normalization to bands.

    This operates only on valid pixels (mask != 0) and enforces 0.0 on invalid pixels.
    """
    if bands.ndim != 3:
        raise ValueError(f"bands must be (C,H,W), got shape={bands.shape}")
    if valid_masks.shape != bands.shape:
        raise ValueError(
            f"valid_masks must match bands shape, got {valid_masks.shape} vs {bands.shape}"
        )
    if bands.shape[0] != len(band_names):
        raise ValueError("Band count mismatch between bands array and band_names.")

    clip_low = stats.get("clip_low_by_band", {})
    clip_high = stats.get("clip_high_by_band", {})
    mean_by_band = stats.get("mean_by_band", {})
    std_by_band = stats.get("std_by_band", {})

    out = bands.astype(np.float32, copy=True)

    for i, b in enumerate(band_names):
        valid = valid_masks[i] != 0
        if not np.any(valid):
            out[i, :, :] = 0.0
            continue

        x = out[i]
        lo = float(clip_low[b])
        hi = float(clip_high[b])
        mu = float(mean_by_band[b])
        sd = float(std_by_band[b])

        xv = x[valid]
        xv = np.clip(xv, lo, hi)
        xv = (xv - mu) / max(sd, 1e-6)

        x[valid] = xv
        x[~valid] = 0.0

        if not np.all(np.isfinite(x[valid])):
            raise ValueError(f"Non-finite values after normalization for band={b!r}.")

    return out
