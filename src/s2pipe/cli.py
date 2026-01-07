from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from s2pipe.download.cfg import (
    PipelineConfig,
    QueryConfig,
    SelectionConfig,
    NodesIndexConfig,
    DownloadConfig,
    RunControlConfig,
    ManifestConfig,
    validate,
)
from .download.auth import prompt_auth
from s2pipe.download.pipeline import run_download


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_from_dict(
    d: dict[str, Any],
    *,
    out_dir_override: str | None = None,
    dry_run_override: bool | None = None,
) -> PipelineConfig:
    q = d.get("query", {}) or {}
    sel = d.get("selection", {}) or {}
    nidx = d.get("nodes_index", {}) or {}
    dl = d.get("download", {}) or {}
    ctrl = d.get("control", {}) or {}
    man = d.get("manifest", {}) or {}

    query = QueryConfig(
        tile_id=str(q["tile_id"]),
        date_from_utc=str(q["date_from_utc"]),
        date_to_utc=str(q["date_to_utc"]),
        product_type_l1c=str(q.get("product_type_l1c", "S2MSI1C")),
        product_type_l2a=str(q.get("product_type_l2a", "S2MSI2A")),
        cloud_min=q.get("cloud_min", None),
        cloud_max=q.get("cloud_max", None),
        min_coverage_ratio=float(q.get("min_coverage_ratio", 0.0)),
        top=int(q.get("top", 1000)),
        orderby=str(q.get("orderby", "ContentDate/Start asc")),
        include_attributes_in_hits=bool(q.get("include_attributes_in_hits", True)),
        tile_area_m2=float(q.get("tile_area_m2", 1.21e10)),
    )

    selection = SelectionConfig(
        l1c_bands=tuple(sel.get("l1c_bands", SelectionConfig().l1c_bands)),
        l1c_tile_metadata=bool(sel.get("l1c_tile_metadata", True)),
        l2a_scl_20m=bool(sel.get("l2a_scl_20m", True)),
        l2a_aot_20m=bool(sel.get("l2a_aot_20m", False)),
        l2a_wvp_20m=bool(sel.get("l2a_wvp_20m", False)),
        l2a_tile_metadata=bool(sel.get("l2a_tile_metadata", True)),
    )

    nodes_index = NodesIndexConfig(
        skip_dir_names=frozenset(
            nidx.get("skip_dir_names", list(NodesIndexConfig().skip_dir_names))
        ),
        skip_prefixes=tuple(tuple(x) for x in nidx.get("skip_prefixes", ())),
        max_dirs_to_visit=int(nidx.get("max_dirs_to_visit", 50_000)),
        enable_cache=bool(nidx.get("enable_cache", True)),
    )

    out_dir = (
        Path(out_dir_override) if out_dir_override else Path(dl.get("out_dir", "./out"))
    )
    dry_run = (
        bool(dry_run_override)
        if dry_run_override is not None
        else bool(dl.get("dry_run", True))
    )

    download = DownloadConfig(
        out_dir=out_dir,
        overwrite=bool(dl.get("overwrite", False)),
        dry_run=dry_run,
        chunk_size_bytes=int(dl.get("chunk_size_bytes", 8 * 1024 * 1024)),
    )

    control = RunControlConfig(
        max_pairs=ctrl.get("max_pairs", None),
    )

    manifest = ManifestConfig(
        manifest_version=str(man.get("manifest_version", "1.0")),
        write_json=bool(man.get("write_json", True)),
        json_name=str(man.get("json_name", "manifest.json")),
        export_table=bool(man.get("export_table", True)),
        table_csv_name=str(man.get("table_csv_name", "manifest_table.csv")),
        table_xlsx_name=str(man.get("table_xlsx_name", "manifest_table.xlsx")),
        store_geofootprint=bool(man.get("store_geofootprint", True)),
    )

    cfg = PipelineConfig(
        query=query,
        selection=selection,
        nodes_index=nodes_index,
        download=download,
        control=control,
        manifest=manifest,
    )
    validate(cfg)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(prog="s2pipe")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser(
        "download", help="Run Step 1: download Sentinel-2 products + write manifest."
    )
    d.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    d.add_argument("--out", type=str, default=None, help="Override output directory.")
    d.add_argument("--dry-run", action="store_true", help="Plan only; do not download.")
    d.add_argument(
        "--no-dry-run",
        dest="dry_run_off",
        action="store_true",
        help="Actually download.",
    )
    args = p.parse_args()

    if args.cmd == "download":
        cfg_dict = _load_yaml(Path(args.config))
        dry_override = True if args.dry_run else (False if args.dry_run_off else None)
        cfg = _cfg_from_dict(
            cfg_dict, out_dir_override=args.out, dry_run_override=dry_override
        )

        auth = prompt_auth()
        res = run_download(cfg, auth=auth)
        print(f"Pairs: {len(res.pairs)}")
        if res.manifest_path:
            print(f"Manifest: {res.manifest_path}")
        if res.table_csv_path:
            print(f"Table CSV: {res.table_csv_path}")
        if res.table_xlsx_path:
            print(f"Table XLSX: {res.table_xlsx_path}")


if __name__ == "__main__":
    main()
