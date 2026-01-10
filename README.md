# s2pipe — Sentinel-2 CDSE pipeline

This repository contains a multi-stage pipeline:

- Step 1: Download (implemented)
- Step 2: Preprocess to a training-ready format (implemented)
- Step 3: Patch extraction and sharding (not implemented)

## Install (editable)

```bash
pip install -e .
```

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
    l1c_bands=("B01",),
    angles=AngleAssetConfig(
        enabled=True,
        include_sun=True,
        include_view=True,
        encode="sin_cos",
        view_mode="per_band",
    ),
    labels=LabelConfig(),
    normalize=NormalizeConfig(mode="none"),
)

run_preprocess(cfg)
```


## Run (CLI)

### Download
```bash
s2pipe download --config examples/configs/download.yaml --dry-run
```
### Preprocess
```bash
s2pipe preprocess --config examples/configs/preprocess.yaml
```

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
