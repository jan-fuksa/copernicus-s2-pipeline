from __future__ import annotations

import argparse
from dataclasses import MISSING
from pathlib import Path
from typing import Any, TypeVar

import yaml

from s2pipe.download.cfg import (
    DownloadConfig,
    ManifestConfig,
    NodesIndexConfig,
    PipelineConfig,
    QueryConfig,
    RunControlConfig,
    SelectionConfig,
    validate,
)
from s2pipe.download.pipeline import run_download
from .download.auth import prompt_auth

from s2pipe.preprocess.cfg import (
    AngleAssetConfig,
    LabelConfig,
    NormalizeConfig,
    PreprocessConfig,
)
from s2pipe.preprocess.run import run_preprocess


T = TypeVar("T")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _field_default_or_missing(cls: type, field_name: str) -> Any:
    """
    Return a dataclass field default (or the result of default_factory),
    or dataclasses.MISSING if the field has no default.
    """
    f = cls.__dataclass_fields__[field_name]  # type: ignore[attr-defined]
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return f.default_factory()
    return MISSING


def _require(mapping: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise KeyError(f'Missing required key "{ctx}.{key}" in YAML config.')
    return mapping[key]


def _get(mapping: dict[str, Any], key: str, default: Any) -> Any:
    if default is MISSING:
        return mapping[key]  # raises KeyError if missing
    return mapping.get(key, default)


def _download_cfg_from_dict(
    d: dict[str, Any],
    *,
    out_dir_override: str | None = None,
    dry_run_override: bool | None = None,
) -> PipelineConfig:
    """
    Required YAML shape for Step 1 (download):

      download:
        query: ...
        selection: ...
        nodes_index: ...
        download: ...
        control: ...
        manifest: ...
    """
    if "download" not in d or not isinstance(d["download"], dict):
        raise KeyError('Missing required top-level key "download" in YAML config.')

    root: dict[str, Any] = d["download"]

    q = root.get("query", {}) or {}
    sel = root.get("selection", {}) or {}
    nidx = root.get("nodes_index", {}) or {}
    dl = root.get("download", {}) or {}
    ctrl = root.get("control", {}) or {}
    man = root.get("manifest", {}) or {}

    # QueryConfig: required keys must exist in YAML
    query = QueryConfig(
        tile_id=str(_require(q, "tile_id", "download.query")),
        date_from_utc=str(_require(q, "date_from_utc", "download.query")),
        date_to_utc=str(_require(q, "date_to_utc", "download.query")),
        product_type_l1c=str(
            _get(
                q,
                "product_type_l1c",
                _field_default_or_missing(QueryConfig, "product_type_l1c"),
            )
        ),
        product_type_l2a=str(
            _get(
                q,
                "product_type_l2a",
                _field_default_or_missing(QueryConfig, "product_type_l2a"),
            )
        ),
        cloud_min=_get(
            q, "cloud_min", _field_default_or_missing(QueryConfig, "cloud_min")
        ),
        cloud_max=_get(
            q, "cloud_max", _field_default_or_missing(QueryConfig, "cloud_max")
        ),
        min_coverage_ratio=float(
            _get(
                q,
                "min_coverage_ratio",
                _field_default_or_missing(QueryConfig, "min_coverage_ratio"),
            )
        ),
        top=int(_get(q, "top", _field_default_or_missing(QueryConfig, "top"))),
        orderby=str(
            _get(q, "orderby", _field_default_or_missing(QueryConfig, "orderby"))
        ),
        include_attributes_in_hits=bool(
            _get(
                q,
                "include_attributes_in_hits",
                _field_default_or_missing(QueryConfig, "include_attributes_in_hits"),
            )
        ),
        tile_area_m2=float(
            _get(
                q,
                "tile_area_m2",
                _field_default_or_missing(QueryConfig, "tile_area_m2"),
            )
        ),
    )

    selection = SelectionConfig(
        l1c_bands=tuple(
            _get(
                sel,
                "l1c_bands",
                _field_default_or_missing(SelectionConfig, "l1c_bands"),
            )
        ),
        l1c_tile_metadata=bool(
            _get(
                sel,
                "l1c_tile_metadata",
                _field_default_or_missing(SelectionConfig, "l1c_tile_metadata"),
            )
        ),
        l2a_scl_20m=bool(
            _get(
                sel,
                "l2a_scl_20m",
                _field_default_or_missing(SelectionConfig, "l2a_scl_20m"),
            )
        ),
        l2a_aot_20m=bool(
            _get(
                sel,
                "l2a_aot_20m",
                _field_default_or_missing(SelectionConfig, "l2a_aot_20m"),
            )
        ),
        l2a_wvp_20m=bool(
            _get(
                sel,
                "l2a_wvp_20m",
                _field_default_or_missing(SelectionConfig, "l2a_wvp_20m"),
            )
        ),
        l2a_tile_metadata=bool(
            _get(
                sel,
                "l2a_tile_metadata",
                _field_default_or_missing(SelectionConfig, "l2a_tile_metadata"),
            )
        ),
    )

    nodes_index = NodesIndexConfig(
        skip_dir_names=frozenset(
            _get(
                nidx,
                "skip_dir_names",
                _field_default_or_missing(NodesIndexConfig, "skip_dir_names"),
            )
        ),
        skip_prefixes=tuple(
            tuple(x)
            for x in _get(
                nidx,
                "skip_prefixes",
                _field_default_or_missing(NodesIndexConfig, "skip_prefixes"),
            )
        ),
        max_dirs_to_visit=int(
            _get(
                nidx,
                "max_dirs_to_visit",
                _field_default_or_missing(NodesIndexConfig, "max_dirs_to_visit"),
            )
        ),
        enable_cache=bool(
            _get(
                nidx,
                "enable_cache",
                _field_default_or_missing(NodesIndexConfig, "enable_cache"),
            )
        ),
    )

    # DownloadConfig: out_dir has NO default in the dataclass -> must be provided (unless overridden by CLI).
    if out_dir_override is not None:
        out_dir = Path(out_dir_override)
    else:
        if "out_dir" not in dl:
            raise KeyError(
                'Missing required key "download.download.out_dir" in YAML config.'
            )
        out_dir = Path(str(dl["out_dir"]))

    if dry_run_override is None:
        dry_run = bool(
            _get(dl, "dry_run", _field_default_or_missing(DownloadConfig, "dry_run"))
        )
    else:
        dry_run = bool(dry_run_override)

    download = DownloadConfig(
        out_dir=out_dir,
        overwrite=bool(
            _get(
                dl, "overwrite", _field_default_or_missing(DownloadConfig, "overwrite")
            )
        ),
        dry_run=dry_run,
        raw_dirname=str(
            _get(
                dl,
                "raw_dirname",
                _field_default_or_missing(DownloadConfig, "raw_dirname"),
            )
        ),
        meta_dirname=str(
            _get(
                dl,
                "meta_dirname",
                _field_default_or_missing(DownloadConfig, "meta_dirname"),
            )
        ),
        tmp_dirname=str(
            _get(
                dl,
                "tmp_dirname",
                _field_default_or_missing(DownloadConfig, "tmp_dirname"),
            )
        ),
        chunk_size_bytes=int(
            _get(
                dl,
                "chunk_size_bytes",
                _field_default_or_missing(DownloadConfig, "chunk_size_bytes"),
            )
        ),
    )

    control = RunControlConfig(
        max_scenes=_get(
            ctrl,
            "max_scenes",
            _field_default_or_missing(RunControlConfig, "max_scenes"),
        ),
    )

    manifest = ManifestConfig(
        manifest_version=str(
            _get(
                man,
                "manifest_version",
                _field_default_or_missing(ManifestConfig, "manifest_version"),
            )
        ),
        write_json=bool(
            _get(
                man,
                "write_json",
                _field_default_or_missing(ManifestConfig, "write_json"),
            )
        ),
        json_name=str(
            _get(
                man, "json_name", _field_default_or_missing(ManifestConfig, "json_name")
            )
        ),
        export_table=bool(
            _get(
                man,
                "export_table",
                _field_default_or_missing(ManifestConfig, "export_table"),
            )
        ),
        table_csv_name=str(
            _get(
                man,
                "table_csv_name",
                _field_default_or_missing(ManifestConfig, "table_csv_name"),
            )
        ),
        table_xlsx_name=str(
            _get(
                man,
                "table_xlsx_name",
                _field_default_or_missing(ManifestConfig, "table_xlsx_name"),
            )
        ),
        runs_dir=str(
            _get(man, "runs_dir", _field_default_or_missing(ManifestConfig, "runs_dir"))
        ),
        index_name=str(
            _get(
                man,
                "index_name",
                _field_default_or_missing(ManifestConfig, "index_name"),
            )
        ),
        store_geofootprint=bool(
            _get(
                man,
                "store_geofootprint",
                _field_default_or_missing(ManifestConfig, "store_geofootprint"),
            )
        ),
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


def _preprocess_cfg_from_dict(
    d: dict[str, Any],
    *,
    out_dir_override: str | None = None,
    max_scenes_override: int | None = None,
    run_id_override: str | None = None,
    num_workers_override: int | None = None,
    # CLI overrides (optional)
    to_toa_reflectance_override: bool | None = None,
    upsample_method_override: str | None = None,
    downsample_method_override: str | None = None,
    valid_pixel_mask_override: str | None = None,
    normalize_mode_override: str | None = None,
    normalize_stats_path_override: str | None = None,
    save_histograms_override: bool | None = None,
    max_pixels_per_scene_override: int | None = None,
    clip_percentiles_override: tuple[float, float] | None = None,
    no_clip_override: bool = False,
    hist_range_override: tuple[float, float] | None = None,
    hist_bin_width_override: float | None = None,
    norm_seed_override: int | None = None,
) -> PreprocessConfig:
    """
    Required YAML shape for Step 2 (preprocess):

      preprocess:
        index_json: ...
        out_dir: ...
        ...
    """
    if "preprocess" not in d or not isinstance(d["preprocess"], dict):
        raise KeyError('Missing required top-level key "preprocess" in YAML config.')

    pd: dict[str, Any] = d["preprocess"]

    # Required (no defaults expected): index_json, out_dir
    index_json = Path(str(_require(pd, "index_json", "preprocess")))
    if out_dir_override is not None:
        out_dir = Path(out_dir_override)
    else:
        out_dir = Path(str(_require(pd, "out_dir", "preprocess")))

    angles_d = pd.get("angles", {}) or {}
    labels_d = pd.get("labels", {}) or {}
    norm_d = pd.get("normalize", {}) or {}

    # Preprocess feature toggles and methods
    to_toa_reflectance = (
        bool(to_toa_reflectance_override)
        if to_toa_reflectance_override is not None
        else bool(
            pd.get(
                "to_toa_reflectance",
                _field_default_or_missing(PreprocessConfig, "to_toa_reflectance"),
            )
        )
    )

    upsample_method = (
        str(upsample_method_override)
        if upsample_method_override is not None
        else str(
            pd.get(
                "upsample_method",
                _field_default_or_missing(PreprocessConfig, "upsample_method"),
            )
        )
    )

    downsample_method = (
        str(downsample_method_override)
        if downsample_method_override is not None
        else str(
            pd.get(
                "downsample_method",
                _field_default_or_missing(PreprocessConfig, "downsample_method"),
            )
        )
    )

    valid_pixel_mask = (
        str(valid_pixel_mask_override)
        if valid_pixel_mask_override is not None
        else str(
            pd.get(
                "valid_pixel_mask",
                _field_default_or_missing(PreprocessConfig, "valid_pixel_mask"),
            )
        )
    )

    # Default instances (from PreprocessConfig defaults/factories). If those fields have no defaults,
    # _field_default_or_missing will return MISSING and we will require explicit YAML keys below.
    angles_default = _field_default_or_missing(PreprocessConfig, "angles")
    labels_default = _field_default_or_missing(PreprocessConfig, "labels")
    norm_default = _field_default_or_missing(PreprocessConfig, "normalize")

    if angles_default is MISSING:
        raise KeyError(
            "PreprocessConfig.angles has no default; please provide preprocess.angles in YAML."
        )
    if labels_default is MISSING:
        raise KeyError(
            "PreprocessConfig.labels has no default; please provide preprocess.labels in YAML."
        )
    if norm_default is MISSING:
        raise KeyError(
            "PreprocessConfig.normalize has no default; please provide preprocess.normalize in YAML."
        )

    assert isinstance(angles_default, AngleAssetConfig)
    assert isinstance(labels_default, LabelConfig)
    assert isinstance(norm_default, NormalizeConfig)

    mapping_raw = labels_d.get("mapping", labels_default.mapping)
    if mapping_raw is None:
        mapping = None
    else:
        if not isinstance(mapping_raw, dict):
            raise TypeError("labels.mapping must be a dict (e.g. {0: 0, 1: 1, ...}).")
        mapping = {int(k): int(v) for k, v in mapping_raw.items()}

    angles_cfg = AngleAssetConfig(
        enabled=bool(angles_d.get("enabled", angles_default.enabled)),
        include_sun=bool(angles_d.get("include_sun", angles_default.include_sun)),
        include_view=bool(angles_d.get("include_view", angles_default.include_view)),
        encode=str(angles_d.get("encode", angles_default.encode)),
        view_mode=str(angles_d.get("view_mode", angles_default.view_mode)),
        view_bands=tuple(angles_d.get("view_bands", angles_default.view_bands)),
        detector_aggregate=str(
            angles_d.get("detector_aggregate", angles_default.detector_aggregate)
        ),
        output_name=str(angles_d.get("output_name", angles_default.output_name)),
    )

    labels_cfg = LabelConfig(
        ignore_index=int(labels_d.get("ignore_index", labels_default.ignore_index)),
        resample=str(labels_d.get("resample", labels_default.resample)),
        mapping=mapping,
        src_value_range=int(
            labels_d.get("src_value_range", labels_default.src_value_range)
        ),
    )

    # NormalizeConfig
    stats_path_raw = norm_d.get("stats_path", norm_default.stats_path)
    if normalize_stats_path_override is not None:
        stats_path_raw = normalize_stats_path_override

    max_pixels_raw = norm_d.get(
        "max_pixels_per_scene", norm_default.max_pixels_per_scene
    )
    if max_pixels_per_scene_override is not None:
        max_pixels_raw = max_pixels_per_scene_override

    # Percentile clipping
    if no_clip_override:
        clip_percentiles = None
    elif clip_percentiles_override is not None:
        clip_percentiles = (
            float(clip_percentiles_override[0]),
            float(clip_percentiles_override[1]),
        )
    else:
        clip_percentiles_raw = norm_d.get(
            "clip_percentiles", getattr(norm_default, "clip_percentiles", None)
        )
        if clip_percentiles_raw is None:
            clip_percentiles = None
        else:
            if (
                not isinstance(clip_percentiles_raw, (list, tuple))
                or len(clip_percentiles_raw) != 2
            ):
                raise TypeError(
                    "normalize.clip_percentiles must be a 2-item list/tuple (e.g. [1.0, 99.0]) or null."
                )
            clip_percentiles = (
                float(clip_percentiles_raw[0]),
                float(clip_percentiles_raw[1]),
            )

    # Histogram settings
    if hist_range_override is not None:
        hist_range = (float(hist_range_override[0]), float(hist_range_override[1]))
    else:
        hist_range_raw = norm_d.get(
            "hist_range", getattr(norm_default, "hist_range", (-0.2, 2.0))
        )
        if not isinstance(hist_range_raw, (list, tuple)) or len(hist_range_raw) != 2:
            raise TypeError(
                "normalize.hist_range must be a 2-item list/tuple (e.g. [-0.2, 2.0])."
            )
        hist_range = (float(hist_range_raw[0]), float(hist_range_raw[1]))

    if hist_bin_width_override is not None:
        hist_bin_width = float(hist_bin_width_override)
    else:
        hist_bin_width = float(
            norm_d.get("hist_bin_width", getattr(norm_default, "hist_bin_width", 1e-4))
        )

    if norm_seed_override is not None:
        seed = int(norm_seed_override)
    else:
        seed = int(norm_d.get("seed", getattr(norm_default, "seed", 0)))

    if save_histograms_override is not None:
        save_histograms = bool(save_histograms_override)
    else:
        save_histograms = bool(
            norm_d.get(
                "save_histograms", getattr(norm_default, "save_histograms", False)
            )
        )

    mode_val = str(norm_d.get("mode", norm_default.mode))
    if normalize_mode_override is not None:
        mode_val = str(normalize_mode_override)

    normalize_cfg = NormalizeConfig(
        mode=mode_val,
        stats_path=Path(str(stats_path_raw)) if stats_path_raw is not None else None,
        clip_percentiles=clip_percentiles,
        hist_range=hist_range,
        hist_bin_width=hist_bin_width,
        max_pixels_per_scene=int(max_pixels_raw)
        if max_pixels_raw is not None
        else None,
        seed=seed,
        save_histograms=save_histograms,
    )

    max_scenes = (
        int(max_scenes_override)
        if max_scenes_override is not None
        else pd.get(
            "max_scenes", _field_default_or_missing(PreprocessConfig, "max_scenes")
        )
    )
    run_id = (
        str(run_id_override)
        if run_id_override is not None
        else pd.get("run_id", _field_default_or_missing(PreprocessConfig, "run_id"))
    )
    num_workers = (
        int(num_workers_override)
        if num_workers_override is not None
        else int(
            pd.get(
                "num_workers",
                _field_default_or_missing(PreprocessConfig, "num_workers"),
            )
        )
    )

    cfg = PreprocessConfig(
        index_json=index_json,
        out_dir=out_dir,
        max_scenes=(int(max_scenes) if max_scenes is not None else None),
        run_id=(str(run_id) if run_id is not None else None),
        target_grid_ref=str(
            pd.get(
                "target_grid_ref",
                _field_default_or_missing(PreprocessConfig, "target_grid_ref"),
            )
        ),
        l1c_bands=tuple(
            pd.get(
                "l1c_bands", _field_default_or_missing(PreprocessConfig, "l1c_bands")
            )
        ),
        to_toa_reflectance=to_toa_reflectance,
        upsample_method=upsample_method,
        downsample_method=downsample_method,
        valid_pixel_mask=valid_pixel_mask,
        angles=angles_cfg,
        labels=labels_cfg,
        normalize=normalize_cfg,
        num_workers=num_workers,
    )
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(prog="s2pipe")
    sub = p.add_subparsers(dest="cmd", required=True)

    _RESAMPLE_METHOD_CHOICES = [
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
    _VALID_PIXEL_MASK_CHOICES = ["single", "per_band"]
    _NORMALIZE_MODE_CHOICES = ["none", "compute_only", "apply_with_stats"]

    d = sub.add_parser(
        "download", help="Run Step 1: download Sentinel-2 products and write manifests."
    )
    d.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    d.add_argument("--out", type=str, default=None, help="Override output directory.")
    d.add_argument("--dry-run", action="store_true", help="Plan only; do not download.")
    d.add_argument(
        "--no-dry-run",
        dest="dry_run_off",
        action="store_true",
        help="Perform actual download.",
    )

    pp = sub.add_parser(
        "preprocess",
        help="Run Step 2: preprocess Step-1 outputs into a training-ready format.",
    )
    pp.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    pp.add_argument("--out", type=str, default=None, help="Override output directory.")
    pp.add_argument(
        "--max-scenes", type=int, default=None, help="Limit scenes processed."
    )
    pp.add_argument("--run-id", type=str, default=None, help="Override run_id.")
    pp.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of workers (reserved; pipeline is currently sequential).",
    )

    # High-leverage preprocess overrides
    toa = pp.add_mutually_exclusive_group()
    toa.add_argument(
        "--toa",
        dest="to_toa_reflectance",
        action="store_const",
        const=True,
        default=None,
        help="Convert L1C DN to TOA reflectance using product metadata.",
    )
    toa.add_argument(
        "--no-toa",
        dest="to_toa_reflectance",
        action="store_const",
        const=False,
        default=None,
        help="Do not convert L1C DN to TOA reflectance.",
    )

    pp.add_argument(
        "--upsample-method",
        type=str,
        default=None,
        choices=_RESAMPLE_METHOD_CHOICES,
        help="Resampling method used when upsampling to the target grid.",
    )
    pp.add_argument(
        "--downsample-method",
        type=str,
        default=None,
        choices=_RESAMPLE_METHOD_CHOICES,
        help="Resampling method used when downsampling to the target grid.",
    )
    pp.add_argument(
        "--valid-pixel-mask",
        type=str,
        default=None,
        choices=_VALID_PIXEL_MASK_CHOICES,
        help="How valid pixel masks are stored in x.tif (single or per_band).",
    )

    pp.add_argument(
        "--normalize-mode",
        type=str,
        default=None,
        choices=_NORMALIZE_MODE_CHOICES,
        help="Normalization mode: none | compute_only | apply_with_stats.",
    )
    pp.add_argument(
        "--normalize-stats-path",
        type=str,
        default=None,
        help="Path to normalization stats.json used for compute/apply modes.",
    )

    sh = pp.add_mutually_exclusive_group()
    sh.add_argument(
        "--save-histograms",
        dest="save_histograms",
        action="store_const",
        const=True,
        default=None,
        help="Save histogram.npz next to stats.json when computing stats.",
    )
    sh.add_argument(
        "--no-save-histograms",
        dest="save_histograms",
        action="store_const",
        const=False,
        default=None,
        help="Do not save histogram.npz when computing stats.",
    )

    pp.add_argument(
        "--max-pixels-per-scene",
        type=int,
        default=None,
        help="Limit the number of valid pixels sampled per scene when computing stats.",
    )

    clip = pp.add_mutually_exclusive_group()
    clip.add_argument(
        "--clip-percentiles",
        nargs=2,
        type=float,
        default=None,
        metavar=("P_LO", "P_HI"),
        help="Percentile clipping applied before moments, e.g. --clip-percentiles 1.0 99.0.",
    )
    clip.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable percentile clipping during normalization.",
    )

    pp.add_argument(
        "--hist-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Histogram range used to compute dataset statistics, e.g. --hist-range -0.2 2.0.",
    )
    pp.add_argument(
        "--hist-bin-width",
        type=float,
        default=None,
        help="Histogram bin width used to compute dataset statistics (e.g. 0.0001).",
    )
    pp.add_argument(
        "--norm-seed",
        type=int,
        default=None,
        help="Random seed used for subsampling when computing stats.",
    )

    args = p.parse_args()

    if args.cmd == "download":
        cfg_dict = _load_yaml(Path(args.config))
        dry_override = True if args.dry_run else (False if args.dry_run_off else None)
        cfg = _download_cfg_from_dict(
            cfg_dict, out_dir_override=args.out, dry_run_override=dry_override
        )

        auth = prompt_auth()
        res = run_download(cfg, auth=auth)
        print(f"Scenes: {len(res.scenes)}")
        if res.manifest_path:
            print(f"Manifest: {res.manifest_path}")
        if res.table_csv_path:
            print(f"Table CSV: {res.table_csv_path}")
        if res.table_xlsx_path:
            print(f"Table XLSX: {res.table_xlsx_path}")

    elif args.cmd == "preprocess":
        cfg_dict = _load_yaml(Path(args.config))
        cfg = _preprocess_cfg_from_dict(
            cfg_dict,
            out_dir_override=args.out,
            max_scenes_override=args.max_scenes,
            run_id_override=args.run_id,
            num_workers_override=args.num_workers,
            to_toa_reflectance_override=args.to_toa_reflectance,
            upsample_method_override=args.upsample_method,
            downsample_method_override=args.downsample_method,
            valid_pixel_mask_override=args.valid_pixel_mask,
            normalize_mode_override=args.normalize_mode,
            normalize_stats_path_override=args.normalize_stats_path,
            save_histograms_override=args.save_histograms,
            max_pixels_per_scene_override=args.max_pixels_per_scene,
            clip_percentiles_override=(
                (float(args.clip_percentiles[0]), float(args.clip_percentiles[1]))
                if args.clip_percentiles is not None
                else None
            ),
            no_clip_override=bool(args.no_clip),
            hist_range_override=(
                (float(args.hist_range[0]), float(args.hist_range[1]))
                if args.hist_range is not None
                else None
            ),
            hist_bin_width_override=(
                float(args.hist_bin_width) if args.hist_bin_width is not None else None
            ),
            norm_seed_override=(
                int(args.norm_seed) if args.norm_seed is not None else None
            ),
        )
        res = run_preprocess(cfg)
        print(f"Run ID: {res.run_id}")
        print(f"Processed: {res.processed_count}")
        print(f"Failed: {res.failed_count}")
        print(f"Skipped: {res.skipped_count}")
        print(f"Run manifest: {res.run_manifest_path}")
        print(f"Step2 index: {res.step2_index_path}")


if __name__ == "__main__":
    main()
