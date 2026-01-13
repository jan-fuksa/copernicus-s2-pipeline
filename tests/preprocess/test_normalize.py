from pathlib import Path

import numpy as np
import pytest

from s2pipe.preprocess.cfg import NormalizeConfig
from s2pipe.preprocess.normalize import (
    hist_init,
    hist_update,
    stats_finalize_from_hist,
    stats_save,
    stats_load,
    validate_stats,
    apply_stats_to_bands,
)


def _cfg(**overrides) -> NormalizeConfig:
    # Assumes NormalizeConfig has these fields per our agreed design.
    base = dict(
        mode="compute_only",
        stats_path=None,
        max_pixels_per_scene=None,
        hist_range=(-0.2, 2.0),
        hist_bin_width=1e-4,
        clip_percentiles=(1.0, 99.0),
        seed=123,
        save_histograms=False,
    )
    base.update(overrides)
    return NormalizeConfig(**base)


def test_hist_init_num_bins_exact() -> None:
    cfg = _cfg(hist_range=(-0.2, 2.0), hist_bin_width=1e-4)
    acc = hist_init(["B02", "B03"], cfg)
    # (2.0 - (-0.2)) / 1e-4 = 2.2 / 1e-4 = 22000
    assert acc.num_bins == 22000
    assert acc.counts.shape == (2, 22000)
    assert acc.counts.dtype == np.uint64


def test_hist_init_invalid_divisibility_raises() -> None:
    # 1.0 / 0.3 is not an integer => should raise
    cfg = _cfg(hist_range=(0.0, 1.0), hist_bin_width=0.3)
    with pytest.raises(ValueError, match="divisible"):
        _ = hist_init(["B02"], cfg)


def test_hist_update_counts_and_under_overflow_clamped() -> None:
    cfg = _cfg(
        hist_range=(0.0, 10.0), hist_bin_width=1.0, clip_percentiles=None, seed=0
    )
    acc = hist_init(["B02"], cfg)

    # Values: one underflow, one in-range (bin 2), one overflow
    bands = np.array([[[-1.0, 2.2, 99.0]]], dtype=np.float32)  # (1,1,3)
    masks = np.array([[[1, 1, 1]]], dtype=np.uint8)

    hist_update(acc, bands, masks, max_pixels_per_scene=None)

    assert acc.tiles_processed == 1
    assert int(acc.underflow_count[0]) == 1
    assert int(acc.overflow_count[0]) == 1

    # Underflow clamped to bin 0, overflow clamped to last bin
    assert int(acc.counts[0, 0]) == 1
    assert int(acc.counts[0, 2]) == 1  # floor((2.2-0)/1)=2
    assert int(acc.counts[0, -1]) == 1
    assert int(acc.counts[0].sum()) == 3


def test_hist_update_subsampling_is_deterministic() -> None:
    cfg = _cfg(
        hist_range=(0.0, 1.0), hist_bin_width=0.01, clip_percentiles=None, seed=42
    )
    bands = np.linspace(0.0, 0.999, 1000, dtype=np.float32).reshape(1, 20, 50)
    masks = np.ones_like(bands, dtype=np.uint8)

    acc1 = hist_init(["B02"], cfg)
    acc2 = hist_init(["B02"], cfg)

    hist_update(acc1, bands, masks, max_pixels_per_scene=100)
    hist_update(acc2, bands, masks, max_pixels_per_scene=100)

    assert acc1.tiles_processed == 1
    assert acc2.tiles_processed == 1
    assert np.array_equal(acc1.counts, acc2.counts)
    assert int(acc1.counts.sum()) == 100
    assert int(acc2.counts.sum()) == 100


def test_stats_finalize_percentiles_and_moments_after_clipping() -> None:
    cfg = _cfg(
        hist_range=(0.0, 10.0),
        hist_bin_width=1.0,
        clip_percentiles=(10.0, 90.0),
        seed=0,
    )
    acc = hist_init(["B02"], cfg)

    # Construct a simple bi-modal distribution:
    # 50 samples in bin 0 center=0.5, 50 samples in bin 9 center=9.5
    acc.counts[0, 0] = 50
    acc.counts[0, 9] = 50
    acc.tiles_processed = 7
    acc.scenes_skipped = 2

    stats = stats_finalize_from_hist(acc, cfg)

    assert stats["schema"] == "s2pipe.normalize.stats.v1"
    assert stats["moments_after_clipping"] is True
    assert stats["tiles_processed"] == 7
    assert stats["scenes_skipped"] == 2

    # Percentiles (ceil rank):
    # 10% -> target=ceil(0.1*100)=10 -> still in bin 0
    # 90% -> target=ceil(0.9*100)=90 -> in bin 9
    assert stats["clip_low_by_band"]["B02"] == pytest.approx(0.5)
    assert stats["clip_high_by_band"]["B02"] == pytest.approx(9.5)

    # Mean/std over bins 0..9 inclusive with only bins 0 and 9 non-zero:
    # mean = (50*0.5 + 50*9.5) / 100 = 5.0
    # std  = 4.5
    assert stats["mean_by_band"]["B02"] == pytest.approx(5.0)
    assert stats["std_by_band"]["B02"] == pytest.approx(4.5)


def test_apply_stats_to_bands_only_valid_pixels() -> None:
    band_names = ["B02"]
    bands = np.array([[[0.0, 0.2, 10.0], [0.1, 0.0, 0.3]]], dtype=np.float32)  # (1,2,3)
    masks = np.array([[[0, 1, 1], [1, 0, 1]]], dtype=np.uint8)

    stats = {
        "schema": "s2pipe.normalize.stats.v1",
        "bands": band_names,
        "hist_range": [0.0, 1.0],
        "hist_bin_width": 0.01,
        "clip_percentiles": [1.0, 99.0],
        "moments_after_clipping": True,
        "clip_low_by_band": {"B02": 0.1},
        "clip_high_by_band": {"B02": 0.3},
        "mean_by_band": {"B02": 0.2},
        "std_by_band": {"B02": 0.1},
    }

    out = apply_stats_to_bands(bands, masks, band_names, stats)

    # invalid pixels must be forced to 0
    assert out[0, 0, 0] == 0.0
    assert out[0, 1, 1] == 0.0

    # valid pixels are clipped to [0.1, 0.3] then normalized: (x-0.2)/0.1
    # (0.2 -> 0.0), (10.0 -> clipped 0.3 -> 1.0), (0.1 -> -1.0), (0.3 -> 1.0)
    assert out[0, 0, 1] == pytest.approx(0.0)
    assert out[0, 0, 2] == pytest.approx(1.0)
    assert out[0, 1, 0] == pytest.approx(-1.0)
    assert out[0, 1, 2] == pytest.approx(1.0)


def test_stats_save_load_and_optional_npz(tmp_path: Path) -> None:
    cfg = _cfg(hist_range=(0.0, 1.0), hist_bin_width=0.1, clip_percentiles=None, seed=0)
    acc = hist_init(["B02"], cfg)
    acc.counts[0, 0] = 3
    acc.tiles_processed = 1
    acc.scenes_skipped = 4

    stats = stats_finalize_from_hist(acc, cfg)
    # Add this field if you include it in stats (recommended check in validate_stats)
    stats["to_toa_reflectance"] = True

    stats_path = tmp_path / "stats.json"
    stats_save(stats, stats_path, save_histograms=True, acc=acc)

    assert stats_path.exists()
    assert stats_path.with_name("histogram.npz").exists()

    loaded = stats_load(stats_path)
    assert loaded["schema"] == stats["schema"]
    assert loaded["bands"] == stats["bands"]
    assert loaded["tiles_processed"] == 1
    assert loaded["scenes_skipped"] == 4


def test_validate_stats_mismatch_raises() -> None:
    cfg = _cfg(hist_range=(0.0, 1.0), hist_bin_width=0.1, clip_percentiles=None, seed=0)

    stats = {
        "schema": "s2pipe.normalize.stats.v1",
        "bands": ["B03"],  # mismatch
        "hist_range": [0.0, 1.0],
        "hist_bin_width": 0.1,
        "clip_percentiles": None,
        "moments_after_clipping": True,
        "to_toa_reflectance": True,
        "clip_low_by_band": {"B03": 0.0},
        "clip_high_by_band": {"B03": 1.0},
        "mean_by_band": {"B03": 0.5},
        "std_by_band": {"B03": 0.2},
    }

    with pytest.raises(ValueError, match="bands"):
        validate_stats(stats, ["B02"], cfg, to_toa_reflectance=True)
