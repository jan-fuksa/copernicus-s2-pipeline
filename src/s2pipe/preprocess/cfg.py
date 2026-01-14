from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


ResampleMethod = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
]


@dataclass(frozen=True)
class AngleAssetConfig:
    """Configuration for exporting Sentinel-2 angles as a separate Step-2 asset.

    The angles are NOT appended into x.tif. If enabled, they are exported as a separate
    GeoTIFF (typically on the native coarse angle grid, ~23x23) named by `output_name`
    (default: angles.tif) inside the processed sample directory.

    Notes on view angles:
    - Sentinel-2 provides viewing incidence angle grids per band and per detector.
      For each band, most detectors are NaN outside their footprint. We aggregate
      detectors into a single per-band grid (default: NaN-aware mean).
    - `view_mode` controls whether to export per-band view angles or a single aggregated
      set across selected bands.
    - `view_bands` (if non-empty) defines which bands to export (and their order).
    """

    enabled: bool = False

    include_sun: bool = True
    include_view: bool = True

    encode: Literal["sin_cos", "deg"] = "sin_cos"

    view_mode: Literal["per_band", "single"] = "per_band"
    view_bands: Sequence[str] = ()

    detector_aggregate: Literal["nanmean"] = "nanmean"

    output_name: str = "angles.tif"


@dataclass(frozen=True)
class LabelConfig:
    ignore_index: int = 255
    resample: Literal["nearest"] = "nearest"
    mapping: dict[int, int] | None = None

    # Performance hint for mapping: values in SCL are within 0..255.
    # If later you need wider ranges, increase this and mapping dtype will widen.
    src_value_range: int = 256


@dataclass(frozen=True)
class NormalizeConfig:
    # Controls normalization behavior.
    mode: Literal["none", "compute_only", "apply_with_stats"] = "none"

    # Path to JSON stats. Required for "apply_with_stats". Optional for "compute_only"
    # (a reasonable default can be derived from output_dir).
    stats_path: Path | None = None

    # Percentile clipping applied before computing mean/std and before normalization.
    # If None, clipping is disabled.
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0)

    # Histogram settings used to estimate percentiles and moments at dataset scale.
    hist_range: tuple[float, float] = (-0.2, 2.0)
    hist_bin_width: float = 1e-4

    # Optional per-scene subsampling for faster experimental runs.
    # If None, all valid pixels are used. Otherwise, up to this many valid pixels
    # are sampled PER BAND per scene.
    max_pixels_per_scene: int | None = None
    seed: int = 0

    # Optional diagnostics.
    save_histograms: bool = False


@dataclass(frozen=True)
class PreprocessConfig:
    # Input (Step 1)
    index_json: Path  # Step-1 meta/manifest/index.json

    # Output
    out_dir: Path  # output root (typically same root as Step 1)

    max_scenes: int | None = None
    run_id: str | None = None

    # Target grid: always derived EXACTLY from a reference raster via grid_from_reference_raster().
    # - "scl_20m" uses the L2A SCL raster grid.
    # - otherwise use a band name, e.g. "B02".
    target_grid_ref: str = "scl_20m"

    # Features
    l1c_bands: Sequence[str] = ()  # e.g. ("B02","B03","B04","B08","B11","B12")
    to_toa_reflectance: bool = True
    upsample_method: ResampleMethod = "bilinear"
    downsample_method: ResampleMethod = "average"
    valid_pixel_mask: Literal["single", "per_band"] = "single"

    angles: AngleAssetConfig = field(default_factory=AngleAssetConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)

    # Execution
    num_workers: int = 0
