# s2pipe — Sentinel-2 CDSE pipeline

This repository contains **Step 1 (download)** of a multi-stage pipeline:

- Step 1: Download (implemented)
- Step 2: Preprocess to a training-ready format (skeleton only)
- Step 3: Patch extraction and sharding (skeleton only)

## Install (editable)

```bash
pip install -e .
```

## Run (Python)

```python
from pathlib import Path
from s2pipe.cfg import DownloadConfig, ManifestConfig, NodesIndexConfig, QueryConfig, RunControlConfig, SelectionConfig, PipelineConfig
from s2pipe.cdse.auth import prompt_auth
from s2pipe.pipeline import run_download

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

## Run (CLI)

```bash
s2pipe download --config examples/configs/download.yaml --dry-run
```
