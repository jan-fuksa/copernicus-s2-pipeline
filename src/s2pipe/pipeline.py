from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import json
import re

from .cfg import PipelineConfig, validate
from .cdse.auth import CDSEAuth, get_access_token
from .cdse.http import CDSEHttpClient, TokenManager
from .cdse.odata import build_filter_basic, search_products_odata, parse_safe_name, choose_best_product_by_name, ProductHit
from .cdse.nodes import index_product_nodes
from .cdse.select import select_assets_l1c, select_assets_l2a
from .cdse.download import download_node, pair_dir
from .io.paths import make_paths, ensure_dirs
from .io.manifest import (
    FileItem, LevelFiles, PairEntry, PairKey, ProductInfo,
    new_manifest, write_manifest_json,
    run_id_now, update_download_index,
)
from .io.export import pairs_to_dataframe, export_table


@dataclass(frozen=True)
class RunResult:
    pairs: list[PairEntry]
    manifest_path: Optional[Path] = None
    table_csv_path: Optional[Path] = None
    table_xlsx_path: Optional[Path] = None


def _group_best(products: list[ProductHit]) -> dict[tuple[str, Any], ProductHit]:
    groups: dict[tuple[str, Any], list[ProductHit]] = {}
    for p in products:
        ps = parse_safe_name(p.name)
        key = (ps.tile_id, ps.sensing_start)
        groups.setdefault(key, []).append(p)
    best: dict[tuple[str, Any], ProductHit] = {}
    for k, lst in groups.items():
        best[k] = choose_best_product_by_name(lst)
    return best


def _pair_l1c_l2a(l1c: list[ProductHit], l2a: list[ProductHit]) -> list[tuple[str, Any, ProductHit, ProductHit]]:
    l1c_best = _group_best(l1c)
    l2a_best = _group_best(l2a)
    out: list[tuple[str, Any, ProductHit, ProductHit]] = []
    for (tile, sensing), p1 in l1c_best.items():
        p2 = l2a_best.get((tile, sensing))
        if p2 is not None:
            out.append((tile, sensing, p1, p2))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def run_download(cfg: PipelineConfig, *, auth: CDSEAuth) -> RunResult:
    validate(cfg)

    token_mgr = TokenManager(auth=auth, access_token=get_access_token(auth))
    client = CDSEHttpClient(token_mgr=token_mgr)

    paths = make_paths(cfg.download.out_dir)
    ensure_dirs(paths)

    # Search (coverage filter applied to L1C only; L2A min_coverage_ratio=0.0)
    flt_l1c = build_filter_basic(
        tile_id=cfg.query.tile_id,
        date_from_utc=cfg.query.date_from_utc,
        date_to_utc=cfg.query.date_to_utc,
        product_type=cfg.query.product_type_l1c,
        min_cloud_pctg=cfg.query.cloud_min,
        max_cloud_pctg=cfg.query.cloud_max,
    )
    flt_l2a = build_filter_basic(
        tile_id=cfg.query.tile_id,
        date_from_utc=cfg.query.date_from_utc,
        date_to_utc=cfg.query.date_to_utc,
        product_type=cfg.query.product_type_l2a,
        min_cloud_pctg=cfg.query.cloud_min,
        max_cloud_pctg=cfg.query.cloud_max,
    )

    l1c_hits = search_products_odata(
        client,
        flt_l1c,
        top=cfg.query.top,
        orderby=cfg.query.orderby,
        min_coverage_ratio=cfg.query.min_coverage_ratio,
        tile_area_m2=cfg.query.tile_area_m2,
        include_attributes=cfg.query.include_attributes_in_hits,
    )
    l2a_hits = search_products_odata(
        client,
        flt_l2a,
        top=cfg.query.top,
        orderby=cfg.query.orderby,
        min_coverage_ratio=0.0,
        tile_area_m2=cfg.query.tile_area_m2,
        include_attributes=False,
    )

    pairs_raw = _pair_l1c_l2a(l1c_hits, l2a_hits)
    if cfg.control.max_pairs is not None:
        pairs_raw = pairs_raw[: cfg.control.max_pairs]

    pair_entries: list[PairEntry] = []

    for tile, sensing_dt, l1c, l2a in pairs_raw:
        sensing_iso = sensing_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        sensing_compact = sensing_dt.strftime("%Y%m%dT%H%M%S")

        # Index nodes
        idx_l1c = index_product_nodes(
            client,
            l1c.cdse_id,
            skip_dir_names=set(cfg.nodes_index.skip_dir_names),
            skip_prefixes=cfg.nodes_index.skip_prefixes,
            max_dirs_to_visit=cfg.nodes_index.max_dirs_to_visit,
            enable_cache=cfg.nodes_index.enable_cache,
        )
        idx_l2a = index_product_nodes(
            client,
            l2a.cdse_id,
            skip_dir_names=set(cfg.nodes_index.skip_dir_names),
            skip_prefixes=cfg.nodes_index.skip_prefixes,
            max_dirs_to_visit=cfg.nodes_index.max_dirs_to_visit,
            enable_cache=cfg.nodes_index.enable_cache,
        )

        # Select nodes
        nodes_l1c = select_assets_l1c(idx_l1c, cfg.selection)
        nodes_l2a = select_assets_l2a(idx_l2a, cfg.selection)

        # Download paths
        l1c_root = pair_dir(paths.raw_l1c, tile_id=tile, sensing_compact=sensing_compact)
        l2a_root = pair_dir(paths.raw_l2a, tile_id=tile, sensing_compact=sensing_compact)

        file_items_l1c: list[FileItem] = []
        file_items_l2a: list[FileItem] = []

        if not cfg.download.dry_run:
            # Ensure directories exist before writing provenance files and downloading assets
            l1c_root.parent.mkdir(parents=True, exist_ok=True)  # .../sensing=...Z/
            l2a_root.parent.mkdir(parents=True, exist_ok=True)
            l1c_root.mkdir(parents=True, exist_ok=True)  # .../sensing=...Z/files/
            l2a_root.mkdir(parents=True, exist_ok=True)

            # Optional: keep basic provenance
            (l1c_root.parent / "product_id.txt").write_text(l1c.cdse_id, encoding="utf-8")
            (l1c_root.parent / "product_name.txt").write_text(l1c.name, encoding="utf-8")
            (l2a_root.parent / "product_id.txt").write_text(l2a.cdse_id, encoding="utf-8")
            (l2a_root.parent / "product_name.txt").write_text(l2a.name, encoding="utf-8")

        for n in nodes_l1c:
            fn = n.parts[-1]
            dst = l1c_root / fn
            role = "tile_metadata" if fn == "MTD_TL.xml" else ("band" if fn.lower().endswith(".jp2") else "file")
            band = None
            if role == "band":
                # try to parse band suffix like "_B02.jp2"
                m = re.search(r"_B(\d\d|8A)\.jp2$", fn)
                if m:
                    band = "B" + m.group(1)
            if cfg.download.dry_run:
                # still record intended path
                file_items_l1c.append(
                    FileItem(
                        role=role,
                        path=str(dst.relative_to(paths.root)),
                        band=band,
                        planned=True
                    )
                )
            else:
                download_node(
                    client,
                    product_id=l1c.cdse_id,
                    node=n,
                    dst=dst,
                    overwrite=cfg.download.overwrite,
                    chunk_size=cfg.download.chunk_size_bytes,
                )
                file_items_l1c.append(
                    FileItem(
                        role=role,
                        path=str(dst.relative_to(paths.root)),
                        band=band,
                        planned=False,
                        present=True
                    )
                )

        for n in nodes_l2a:
            fn = n.parts[-1]
            dst = l2a_root / fn
            role = "tile_metadata" if fn == "MTD_TL.xml" else ("scl_20m" if fn.endswith("_SCL_20m.jp2") else "file")
            if cfg.download.dry_run:
                file_items_l2a.append(
                    FileItem(
                        role=role,
                        path=str(dst.relative_to(paths.root)),
                        planned=True
                    )
                )
            else:
                download_node(
                    client,
                    product_id=l2a.cdse_id,
                    node=n,
                    dst=dst,
                    overwrite=cfg.download.overwrite,
                    chunk_size=cfg.download.chunk_size_bytes,
                )
                file_items_l2a.append(
                    FileItem(
                        role=role,
                        path=str(dst.relative_to(paths.root)),
                        planned=False,
                        present=True
                    )
                )

        ps1 = parse_safe_name(l1c.name)
        ps2 = parse_safe_name(l2a.name)

        l1c_info = ProductInfo(
            product_id=l1c.cdse_id,
            product_name=l1c.name,
            baseline=ps1.baseline,
            rel_orbit=ps1.rel_orbit,
            cloud_cover=l1c.cloud_cover,
            geofootprint=l1c.geofootprint if cfg.manifest.store_geofootprint else None,
            coverage_ratio=l1c.coverage_ratio,
        )
        l2a_info = ProductInfo(
            product_id=l2a.cdse_id,
            product_name=l2a.name,
            baseline=ps2.baseline,
            rel_orbit=ps2.rel_orbit,
        )

        pair_entries.append(PairEntry(
            key=PairKey(tile_id=tile, sensing_start_utc=sensing_iso),
            l1c=l1c_info,
            l2a=l2a_info,
            files_l1c=LevelFiles(root_dir=str(l1c_root.relative_to(paths.root)), items=file_items_l1c),
            files_l2a=LevelFiles(root_dir=str(l2a_root.relative_to(paths.root)), items=file_items_l2a),
        ))

    manifest_path = None
    table_csv_path = None
    table_xlsx_path = None

    if cfg.manifest.write_json:
        q = {
            "tile_id": cfg.query.tile_id if cfg.query.tile_id.startswith("T") else f"T{cfg.query.tile_id}",
            "date_from_utc": cfg.query.date_from_utc,
            "date_to_utc": cfg.query.date_to_utc,
            "cloud_min": cfg.query.cloud_min,
            "cloud_max": cfg.query.cloud_max,
            "min_coverage_ratio": cfg.query.min_coverage_ratio,
            "top": cfg.query.top,
        }
        m = new_manifest(
            manifest_version=cfg.manifest.manifest_version,
            query=q,
            dry_run=cfg.download.dry_run,
            out_dir=str(paths.root),
            layout="raw/<LEVEL>/tile=<TILE>/sensing=<SENSING>Z/files/<FILENAME>",
            pairs=pair_entries,
        )
        # write per-run outputs under meta/manifest/runs/<RUN_ID>/
        run_id = run_id_now()
        runs_dirname = cfg.manifest.runs_dir
        run_dir = paths.manifest_dir / runs_dirname / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / cfg.manifest.json_name
        write_manifest_json(m, manifest_path)

        # Load pairs (dict form) from the run manifest
        pairs_payload = json.loads(manifest_path.read_text(encoding="utf-8")).get("pairs", [])

        # update aggregated index.json.
        if not cfg.download.dry_run:
            index_name = cfg.manifest.index_name
            index_path = paths.manifest_dir / index_name
            update_download_index(
                index_path=index_path,
                manifest_version=cfg.manifest.manifest_version,
                out_dir=str(paths.root),
                layout="raw/<LEVEL>/tile=<TILE>/sensing=<SENSING>Z/files/<FILENAME>",
                new_pairs=pairs_payload,
            )

    if cfg.manifest.export_table and manifest_path is not None:
        # Export per-run 2D table next to the run manifest
        df = pairs_to_dataframe(pairs_payload)
        table_csv_path = manifest_path.parent / cfg.manifest.table_csv_name
        table_xlsx_path = manifest_path.parent / cfg.manifest.table_xlsx_name
        export_table(df, csv_path=str(table_csv_path), xlsx_path=str(table_xlsx_path))

    return RunResult(
        pairs=pair_entries,
        manifest_path=manifest_path,
        table_csv_path=table_csv_path,
        table_xlsx_path=table_xlsx_path,
    )
