from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


@dataclass(frozen=True)
class AngleFeatureConfig:
    include_sun: bool = True
    include_view: bool = True
    encode_sin_cos: bool = True
    view_mode: Literal["per_band", "single"] = "per_band"
    view_bands: Sequence[str] = ()

    # How to aggregate per-detector view angle grids into a single grid.
    # "nanmean" expects non-footprint cells to be NaN and averages only finite values.
    # "mosaic" will later use MSK_DETFOO; for now it is not implemented.
    detector_aggregate: Literal["nanmean", "mosaic"] = "nanmean"


@dataclass(frozen=True)
class LabelConfig:
    # Expected value domain of the *source* label raster (used for LUT mapping).
    # For SCL this is uint8 -> 256.
    src_value_range: int = 256

    ignore_index: int = 255
    resample: Literal["nearest"] = "nearest"
    mapping: dict[int, int] | None = None


@dataclass(frozen=True)
class NormalizeConfig:
    mode: Literal["none", "compute_only", "apply_with_stats"] = "compute_only"
    stats_path: Path | None = None
    max_pixels_per_scene: int | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    # Input (Step 1)
    index_json: Path  # Step-1 meta/manifest/index.json

    # Output
    out_dir: Path  # output root (typically same root as Step 1)

    # Optional / defaults
    max_pairs: int | None = None
    run_id: str | None = None

    # Target grid
    target_res_m: int = 10
    target_crs: str | None = None  # if None, derive from tile / reference raster

    # Features
    l1c_bands: Sequence[str] = ()  # e.g. ("B02","B03","B04","B08","B11","B12")
    angles: AngleFeatureConfig = field(default_factory=AngleFeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)

    # Execution
    num_workers: int = 0
