# s2pipe — Sentinel-2 CDSE pipeline

**Current version:** ![GitHub Tag](https://img.shields.io/github/v/tag/jan-fuksa/copernicus-s2-pipeline?sort=semver)

This repository contains a multi-stage pipeline:

- Step 1: Download
- Step 2: Preprocess to a training-ready format
- Step 3: Patch extraction and sharding (not implemented)

## What is s2pipe?

`s2pipe` is a pragmatic, reproducible pipeline for turning Copernicus Sentinel-2 data from the Copernicus Data Space Ecosystem (CDSE) into a machine-learning friendly dataset.

It focuses on the "last mile" problems that tend to turn into fragile, one-off scripts:
- selecting consistent inputs (bands, labels, angles) per tile/time,
- resampling everything onto a single target grid,
- building validity masks (nodata handling is explicit and reproducible),
- optional conversion from L1C DN to TOA reflectance,
- optional dataset-wide normalization computed in a scalable two-pass workflow.

The pipeline is manifest-driven: Step 1 produces an `index.json` that fully describes where each asset is stored and which roles it plays; later steps use the manifest as the single source of truth.

Step 2 can run **with or without labels**. When labels are disabled, `y.tif` is not produced and the Step-2 index stores `y_path=null` for each scene. This enables self-supervised workflows and inference preprocessing using the same code path.

## Why does this project exist?

Most Sentinel-2 workflows start as notebooks and grow into a collection of scripts that are difficult to reproduce, debug, or scale. `s2pipe` is designed to make preprocessing deterministic and inspectable:
- explicit configuration (YAML/Python),
- deterministic resampling and masking rules,
- separation of concerns (download vs preprocessing vs dataset shaping),
- outputs that can be consumed directly by training pipelines.


## Key features

- Manifest-driven, schema-versioned indexes (no hidden heuristics).
- Assets referenced by **hierarchical roles** (e.g. `l1c.band.B02`, `l2a.scl_20m`), making dependencies explicit.
- Explicit target grid selection (`target_grid_ref`) with **no fallbacks**: if an input is missing, the run fails early.
- Deterministic resampling with explicit nodata handling (`0` in original L1C bands) and validity masks computed **after** resampling.
- Optional conversion from L1C DN to TOA reflectance using product metadata.
- Scalable two-pass, dataset-wide normalization via histogram accumulation.
- Optional sun/view angles export aligned to the target grid.
- Labels are optional and backend-defined (`labels.enabled`, `labels.backend`).

## Use cases

- Self-supervised pretraining (no labels).
- Patch-level classification/regression.
- Cloud/quality filtering pipelines using weak labels.
- Consistent preprocessing for inference (same preprocessing branch as training).
- Multi-temporal and change detection datasets (Step 2 produces aligned tiles; Step 3 is planned for patch-level time stacks).

## Pipeline overview

### Step 1 — Download

Step 1 queries CDSE, selects assets for each **scene** (MGRS tile + sensing time), downloads them, and writes a Step-1 manifest (`index.json`). The manifest is the single source of truth for later stages.

Typical roles include:
- L1C bands: `l1c.band.<BAND>` (e.g. `l1c.band.B02`, `l1c.band.B8A`)
- L1C metadata: `l1c.product_metadata`, `l1c.tile_metadata`
- Optional L2A layers: `l2a.scl_20m` (baseline label backend), `l2a.aot_20m`, `l2a.wvp_20m`, `l2a.tile_metadata`

### Step 2 — Preprocess

Step 2 reads the Step-1 manifest, resolves the required roles from your config, and produces training-ready rasters on a single target grid.

Per scene it can produce:
- `x.tif`: float32 band stack on the target grid **plus appended validity mask channel(s)**
- `angles.tif` (optional): sun/view angles aligned to the target grid
- `y.tif` (optional): label raster produced only when `labels.enabled=true`
- `meta.json`: scene metadata and processing details

It also maintains a global Step-2 index (schema v2) that records `x_path`, `y_path` (nullable), and `meta_path` for each scene.

### Step 3 — Patch extraction and sharding (planned)

Step 3 (planned) will read the Step-2 index, extract patches (stride/overlap), filter them by quality (valid coverage, ignore fraction, cloud criteria), and write shards (e.g. WebDataset `.tar`). An optional patch database can support balancing and re-sharding.


## Quick start

1) Run **Download** to create a Step-1 manifest (`index.json`).
2) Run **Preprocess**:
   - optionally compute dataset-wide normalization stats (Pass 1),
   - then apply normalization and write normalized tiles (Pass 2).

If you do not need normalization, set `normalize.mode="none"` and run preprocess once.

## Install (editable)

```bash
pip install -e .
```

### System dependencies

Sentinel-2 L1C/L2A assets are commonly stored as JPEG2000 (`.jp2`). Make sure your GDAL / raster stack supports JPEG2000 decoding (e.g. via JP2OpenJPEG). If decoding fails, verify your GDAL installation and rasterio bindings.

## Run (Python)

### Download
```python
from pathlib import Path
from s2pipe.download.cfg import (
    DownloadConfig,
    ManifestConfig,
    NodesIndexConfig,
    QueryConfig,
    RunControlConfig,
    SelectionConfig,
    PipelineConfig,
)
from s2pipe.download.auth import prompt_auth
from s2pipe.download.pipeline import run_download

cfg = PipelineConfig(
    query=QueryConfig(
        tile_id="33UWQ",
        date_from_utc="2025-12-01T00:00:00.000Z",
        date_to_utc="2025-12-15T00:00:00.000Z",
        cloud_min=10.0,
        cloud_max=80.0,
        min_coverage_ratio=0.8,
        top=50,
    ),
    selection=SelectionConfig(),
    nodes_index=NodesIndexConfig(),
    download=DownloadConfig(out_dir=Path("./out"), dry_run=True),
    control=RunControlConfig(max_pairs=3),
    manifest=ManifestConfig(),
)

auth = prompt_auth()
result = run_download(cfg, auth=auth)
print("Pairs:", len(result.pairs))
print("Manifest:", result.manifest_path)
```
### Preprocess
The preprocess step builds training-ready tiles (band stacks, labels, optional angles) aligned to a single target grid.

Two-pass normalization is recommended for large datasets:
- **Pass 1**: compute dataset-wide stats (`normalize.mode="compute_only"`)
- **Pass 2**: apply normalization and write normalized tiles (`normalize.mode="apply_with_stats"`)

#### Pass 1 — compute stats
```python
from pathlib import Path
from s2pipe.preprocess.cfg import (
    PreprocessConfig,
    AngleAssetConfig,
    NormalizeConfig,
    LabelConfig,
)
from s2pipe.preprocess.run import run_preprocess

cfg = PreprocessConfig(
    index_json=Path("./out/meta/step1/index.json"),
    out_dir=Path("./out"),
    target_grid_ref="scl_20m",
    l1c_bands=("B02", "B03", "B04", "B08", "B11", "B12"),
    to_toa_reflectance=True,
    upsample_method="bilinear",
    downsample_method="average",
    valid_pixel_mask="single",
    angles=AngleAssetConfig(enabled=False),
    labels=LabelConfig(
        enabled=True,
        backend="scl",
    ),
    normalize=NormalizeConfig(
        mode="compute_only",
        stats_path=Path("./out/normalize/stats.json"),
        clip_percentiles=(1.0, 99.0),
        hist_range=(-0.2, 2.0),
        hist_bin_width=1e-4,
        max_pixels_per_scene=None,
        seed=0,
        save_histograms=True,  # writes histogram.npz next to stats.json
    ),
)

run_preprocess(cfg)
```

#### Pass 2 — apply stats and write normalized tiles
```python
from pathlib import Path
from s2pipe.preprocess.cfg import (
    PreprocessConfig,
    AngleAssetConfig,
    NormalizeConfig,
    LabelConfig,
)
from s2pipe.preprocess.run import run_preprocess

cfg = PreprocessConfig(
    index_json=Path("./out/meta/step1/index.json"),
    out_dir=Path("./out_norm"),
    target_grid_ref="scl_20m",
    l1c_bands=("B02", "B03", "B04", "B08", "B11", "B12"),
    to_toa_reflectance=True,
    upsample_method="bilinear",
    downsample_method="average",
    valid_pixel_mask="single",
    angles=AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="single",
    ),
    labels=LabelConfig(
        enabled=True,
        backend="scl",
    ),
    normalize=NormalizeConfig(
        mode="apply_with_stats",
        stats_path=Path("./out/normalize/stats.json"),
        clip_percentiles=(1.0, 99.0),
        hist_range=(-0.2, 2.0),
        hist_bin_width=1e-4,
        max_pixels_per_scene=None,
        seed=0,
        save_histograms=False,
    ),
)

run_preprocess(cfg)
```





#### No-label mode

To run Step 2 without labels, disable labels explicitly and choose a target grid that is available from downloaded assets (e.g. a band such as `B02`). In this mode `y.tif` is not produced and `y_path` is written as `null`.

```python
from pathlib import Path
from s2pipe.preprocess.cfg import PreprocessConfig, AngleAssetConfig, NormalizeConfig, LabelConfig
from s2pipe.preprocess.run import run_preprocess

cfg = PreprocessConfig(
    index_json=Path("./out/meta/step1/index.json"),
    out_dir=Path("./out_nolabel"),
    target_grid_ref="B02",
    l1c_bands=("B02", "B03", "B04", "B08", "B11", "B12"),
    to_toa_reflectance=True,
    upsample_method="bilinear",
    downsample_method="average",
    valid_pixel_mask="single",
    angles=AngleAssetConfig(enabled=True),
    labels=LabelConfig(enabled=False),
    normalize=NormalizeConfig(mode="none"),
)

run_preprocess(cfg)
```

## Run (CLI)

### Download
```bash
s2pipe download --config examples/configs/download.yaml
```

### Preprocess
```bash
# Pass 1 — compute stats
s2pipe preprocess --config examples/configs/preprocess_stats.yaml

# Pass 2 — apply stats and write normalized tiles
s2pipe preprocess --config examples/configs/preprocess_apply.yaml
```



## Configuration guide (download)

Step 1 (**download**) queries CDSE for Sentinel-2 products, selects the required assets, downloads them, and writes a Step-1 manifest (`index.json`) that fully describes what was downloaded and where it is stored.

The Python API mirrors the configuration structure:

- `query`: what to search for (time range, tile ID, cloud filters, etc.)
- `selection`: which assets to download (bands, metadata, optional L2A layers)
- `nodes_index`: indexing/listing behavior and caching (optional)
- `download`: output paths and download behavior (dry-run, overwrite, chunk size, etc.)
- `control`: run limits (e.g. max number of scenes)
- `manifest`: how the manifest and optional tables are written

### `query`
Controls the product search window and filters.

- `tile_id`: Sentinel-2 MGRS tile (e.g. `"33UWQ"`).
- `date_from_utc`, `date_to_utc`: time range in UTC (`"YYYY-MM-DDTHH:MM:SS.mmmZ"`).
- `cloud_min`, `cloud_max`: cloud filter bounds in percents (if supported by the endpoint).
- `min_coverage_ratio`: filter out incomplete tile coverage.
- `top`: maximum number of products to return per query.

### `selection`
Controls which assets are required for each scene. At minimum you typically select:

- `l1c_bands`: band list to download (used later by Step 2 to build `x.tif`).
- `l1c_tile_metadata`: required if you plan to export angles.
- `l1c_product_metadata`: required if you plan to convert to TOA reflectance.

Optional L2A layers:

- `l2a_scl_20m`: baseline label backend (SCL) used when `labels.backend="scl"`.
- `l2a_aot_20m`, `l2a_wvp_20m`: optional atmospheric products (available for quality metrics and future backends).
- `l2a_tile_metadata`: tile metadata for L2A.

The output Step-1 manifest stores assets under hierarchical roles such as `l1c.band.B02` and `l2a.scl_20m`. Step 2 will fail early if required roles are missing (no fallbacks).

### `download`
Controls where and how files are saved.

- `out_dir`: output root directory (required).
- `dry_run`: if `true`, resolve the plan and write manifests without downloading the files.
- `overwrite`: whether to overwrite existing files (if enabled).
- streaming settings such as chunk size (advanced).

### `control`
Run limits and safety controls.

- `max_scenes`: limit how many (L1C,L2A) scenes are processed in one run.

### `manifest`
Controls what metadata files are written for later steps.

- `index_name`: the name of Step-1 manifest consumed by preprocess.
- optional tables (CSV/XLSX) for inspection/debugging (if enabled).

## Configuration guide (preprocess)

### Step 2 outputs (per scene)

- `x.tif`: float32 band stack on the target grid with validity mask channel(s) appended.
- `angles.tif` (optional): angles aligned to the target grid.
- `y.tif` (optional): label raster produced only when `labels.enabled=true`.
- `meta.json`: metadata including `labels.{enabled,backend,stats,backend_stats}`.

The global Step-2 index stores `x_path`, `y_path` (nullable), and `meta_path` for each processed scene.


### Core inputs and output
- `index_json`: Path to the Step-1 manifest (`.../step1/index.json`). This is the only required input for Step 2.
- `out_dir`: Output directory for processed tiles and artifacts.
- `target_grid_ref`: **Mandatory.** Which raster defines the target grid (e.g. `"scl_20m"` or a band name such as `"B02"`). The referenced asset must be present in the Step-1 manifest; s2pipe does not guess or fall back to another grid.
- `l1c_bands`: List/tuple of band names to stack into `x.tif` (e.g. `"B02", "B03", "B04", "B08, "B11", "B12"`).

### Radiometry
- `to_toa_reflectance` (bool): If `true`, converts L1C DN values to TOA reflectance per band using `QUANTIFICATION_VALUE` and `RADIO_ADD_OFFSET` from product metadata.

### Resampling
- `upsample_method`: Resampling method used when the source grid has lower resolution than the target grid (e.g. `"bilinear"`).
- `downsample_method`: Resampling method used when the source grid has higher resolution than the target grid (e.g. `"average"`).
- Supported methods: `"nearest"`, `"bilinear"`, `"cubic"`, `"average"`, `"mode"`, `"max"`, `"min"`, `"med"`, `"q1"`, `"q3"`, `"sum"`.

### Valid pixel masks
- `valid_pixel_mask`:
  - `"single"`: appends one aggregated validity mask channel named `valid` (logical AND across all bands),
  - `"per_band"`: appends one mask per band (e.g. `valid_B02`, `valid_B03`, ...).
Validity masks are computed after resampling. Nodata is defined as `0` in the original L1C bands.

### Normalization (two-pass workflow)
Normalization is dataset-wide and uses histogram accumulation to scale to large datasets.

- `normalize.mode`:
  - `"none"`: no normalization,
  - `"compute_only"`: Pass 1 – compute stats and write `stats.json` (and optionally `histogram.npz`),
  - `"apply_with_stats"`: Pass 2 – apply stats to produce normalized tiles.
- `normalize.stats_path`: Where to write/read `stats.json` (used by both passes).
- `normalize.clip_percentiles`: Percentile clipping (e.g. `(1.0, 99.0)`) applied before computing mean/std and before normalization. Use `null` to disable clipping.
- `normalize.hist_range`, `normalize.hist_bin_width`: Histogram domain and resolution.
- `normalize.max_pixels_per_scene`: Optional per-scene subsampling for faster experiments.
- `normalize.seed`: RNG seed for deterministic subsampling.
- `normalize.save_histograms` (bool): If `true`, also writes `histogram.npz` next to `stats.json`.

### Angles (optional)
- `angles.enabled`: If `true`, exports an angles raster (sun + view) aligned to the target grid.
- `angles.encode`: Encoding of angles (`"sin_cos"`, `"deg"`).
- `angles.view_mode`: How view angles are represented (`"per_band"`, `"single"`).

### Labels
- `labels.enabled` (bool): If `false`, labels are disabled and `y.tif` is not produced (`y_path=null` in the Step-2 index).
- `labels.backend`: Label backend identifier. Currently `"scl"` is supported as the baseline backend.
- Other `labels.*` fields control label raster generation (resampling, ignore index, mapping).


## Notes

### Authentication

Step 1 downloads data from CDSE and requires valid credentials. The Python example uses `prompt_auth()`, which is suitable for local runs. For unattended runs (CI / servers), prefer non-interactive credential injection (environment variables or a secrets manager) and avoid committing credentials into this repository.



## Disclaimer

This project is an independent, open-source client for Copernicus Data Space Ecosystem (CDSE) APIs
and is not affiliated with or endorsed by ESA, the European Commission, or CDSE service providers.
Users are responsible for complying with CDSE [Terms & Conditions](https://dataspace.copernicus.eu/terms-and-conditions)
and applicable policies.

### Data licensing and attribution (Copernicus Sentinel)

Copernicus Sentinel data are provided on a free, full and open basis;
use is governed by the [Legal notice on the use of Copernicus Sentinel Data and Service Information](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).

When you communicate to the public or distribute Copernicus Sentinel data, include the following source notice:

* `Copernicus Sentinel data [Year]`

If you publish results that include modified/adapted Sentinel data, include:

* `Contains modified Copernicus Sentinel data [Year]`

### Note on CDSE portal content (web pages, images, documents)

This repository does not include CDSE portal content (web texts, images, documents).
Such portal materials are intended for non-commercial use
and are subject to additional restrictions on redistribution/derivative works;
do not commit downloaded portal assets into this repository.
