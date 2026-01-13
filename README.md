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

## Why does this project exist?

Most Sentinel-2 workflows start as notebooks and grow into a collection of scripts that are difficult to reproduce, debug, or scale. `s2pipe` is designed to make preprocessing deterministic and inspectable:
- explicit configuration (YAML/Python),
- deterministic resampling and masking rules,
- separation of concerns (download vs preprocessing vs dataset shaping),
- outputs that can be consumed directly by training pipelines.

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
    labels=LabelConfig(),
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
        view_mode="per_band",
    ),
    labels=LabelConfig(),
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
- `selection`: which assets to download (bands, metadata, SCL, etc.)
- `nodes_index`: indexing/listing behavior and caching (optional)
- `download`: output paths and download behavior (dry-run, overwrite, chunk size, etc.)
- `control`: run limits (e.g. max number of pairs)
- `manifest`: how the manifest and optional tables are written

### `query`
Controls the product search window and filters.

- `tile_id`: Sentinel-2 MGRS tile (e.g. `"33UWQ"`).
- `date_from_utc`, `date_to_utc`: time range in UTC (`"YYYY-MM-DDTHH:MM:SS.mmmZ"`).
- `cloud_min`, `cloud_max`: cloud filter bounds in percents (if supported by the endpoint).
- `min_coverage_ratio`: filter out incomplete tile coverage.
- `top`: maximum number of products to return per query.

### `selection`
Controls *which* files are required for each product/pair.

- `l1c_bands`: list/tuple of L1C bands to download (e.g. `"B02"`, `"B03"`, `"B04"`, `"B08"`, `"B11"`, `"B12"`).
- `l1c_product_metadata` (bool): if `true` download L1C product metadata (required for TOA reflectance in preprocess).
- `l1c_tile_metadata` (bool): if `true` download L1C tile metadata (required if you later export angles).
- `l2a_scl_20m` (bool): if `true` download L2A SCL at 20 m (commonly used as the target grid reference and as labels).

### `download`
Controls where and how files are saved.

- `out_dir`: output root directory (required).
- `dry_run`: if `true`, resolve the plan and write manifests without downloading the files.
- `overwrite`: whether to overwrite existing files (if enabled).
- streaming settings such as chunk size (advanced).

### `control`
Run limits and safety controls.

- `max_pairs`: limit how many (L1C,L2A) pairs are processed in one run.

### `manifest`
Controls what metadata files are written for later steps.

- `index_name`: the name of Step-1 manifest consumed by preprocess.
- optional tables (CSV/XLSX) for inspection/debugging (if enabled).



## Configuration guide (preprocess)

### Core inputs and output
- `index_json`: Path to the Step-1 manifest (`.../step1/index.json`). This is the only required input for Step 2.
- `out_dir`: Output directory for processed tiles and artifacts.
- `target_grid_ref`: Which raster defines the target grid (commonly `"scl_20m"` or band name, e.g. `"B02"`).
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
- `labels.*`: Controls label raster generation (resampling, ignore index, mapping). Defaults are suitable for SCL-based workflows.


## Notes

### Authentication

Step 1 downloads data from CDSE and requires valid credentials. The Python example uses `prompt_auth()`, which is suitable for local runs. For unattended runs (CI / servers), prefer non-interactive credential injection (environment variables or a secrets manager) and avoid committing credentials into this repository.

### Output layout

The exact directory structure is configurable, but the key artifact is the Step-1 manifest. In the examples above, preprocess reads:

- `./out/meta/step1/index.json`

Use the printed `result.manifest_path` as the single source of truth for downstream preprocessing.

### Reproducibility

`s2pipe` is manifest-driven: Step 1 writes a complete inventory of assets and their roles; Step 2 consumes only the manifest. This makes runs repeatable and makes it easier to debug data issues (missing assets, mismatched tiles, etc.) without relying on implicit filesystem conventions.

### Troubleshooting

- **JPEG2000 decode errors (`.jp2`)**: ensure GDAL and rasterio support JP2OpenJPEG.
- **Missing TOA metadata**: TOA reflectance in preprocess requires L1C *product metadata* to be downloaded in Step 1.
- **Nodata handling**: L1C DN value `0` is treated as nodata; validity masks are computed after resampling.


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
