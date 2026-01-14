from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class QueryConfig:
    tile_id: str
    date_from_utc: str
    date_to_utc: str

    product_type_l1c: str = "S2MSI1C"
    product_type_l2a: str = "S2MSI2A"

    # Cloud filter (percent). Implemented via two any(...) clauses for ge/le.
    cloud_min: Optional[float] = None
    cloud_max: Optional[float] = None

    # Coverage filter computed from GeoFootprint (applied to L1C by default in pipeline).
    min_coverage_ratio: float = 0.0

    # IMPORTANT: CDSE OData $top has a max of 1000.
    top: int = 1000
    orderby: str = "ContentDate/Start asc"

    # Expand Attributes when you want to *read* cloudCover value (not only filter by it).
    include_attributes_in_hits: bool = True

    # Coverage area model for "tile area" (defaults to 110km x 110km).
    tile_area_m2: float = 1.21e10


@dataclass(frozen=True)
class SelectionConfig:
    # L1C
    l1c_bands: Sequence[str] = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )
    l1c_tile_metadata: bool = True
    l1c_product_metadata: bool = True

    # L2A
    l2a_scl_20m: bool = True
    l2a_aot_20m: bool = False
    l2a_wvp_20m: bool = False
    l2a_tile_metadata: bool = True


@dataclass(frozen=True)
class NodesIndexConfig:
    skip_dir_names: frozenset[str] = frozenset(
        {"HTML", "rep_info", "DATASTRIP", "AUX_DATA", "QI_DATA"}
    )
    skip_prefixes: Tuple[Tuple[str, ...], ...] = ()
    max_dirs_to_visit: int = 50_000
    enable_cache: bool = True


@dataclass(frozen=True)
class DownloadConfig:
    out_dir: Path
    overwrite: bool = False
    dry_run: bool = True

    # Output layout (contract)
    raw_dirname: str = "raw"
    meta_dirname: str = "meta"
    tmp_dirname: str = "tmp"

    chunk_size_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class RunControlConfig:
    max_scenes: Optional[int] = None


@dataclass(frozen=True)
class ManifestConfig:
    manifest_version: str = "1.0"

    # Per-run outputs (written under meta/step1/runs/<RUN_ID>/)
    write_json: bool = True
    json_name: str = "manifest.json"

    export_table: bool = False
    table_csv_name: str = "manifest_table.csv"
    table_xlsx_name: str = "manifest_table.xlsx"

    runs_dir: str = "runs"  # subdir under <out_dir>/meta/step1/
    index_name: str = "index.json"  # aggregated across runs (deduped)

    # Optional:
    store_geofootprint: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    query: QueryConfig
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    nodes_index: NodesIndexConfig = field(default_factory=NodesIndexConfig)
    download: DownloadConfig = field(
        default_factory=lambda: DownloadConfig(out_dir=Path("./out"))
    )
    control: RunControlConfig = field(default_factory=RunControlConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)


def validate(cfg: PipelineConfig) -> None:
    if not (1 <= cfg.query.top <= 1000):
        raise ValueError(f"QueryConfig.top must be in [1, 1000], got {cfg.query.top}")
    if cfg.query.cloud_min is not None and not (0.0 <= cfg.query.cloud_min <= 100.0):
        raise ValueError(f"cloud_min must be in [0,100], got {cfg.query.cloud_min}")
    if cfg.query.cloud_max is not None and not (0.0 <= cfg.query.cloud_max <= 100.0):
        raise ValueError(f"cloud_max must be in [0,100], got {cfg.query.cloud_max}")
    if cfg.query.cloud_min is not None and cfg.query.cloud_max is not None:
        if cfg.query.cloud_min > cfg.query.cloud_max:
            raise ValueError("cloud_min cannot be greater than cloud_max")
    if cfg.query.min_coverage_ratio < 0.0:
        raise ValueError("min_coverage_ratio must be >= 0")
    if cfg.query.tile_area_m2 <= 0.0:
        raise ValueError("tile_area_m2 must be > 0")
