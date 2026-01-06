from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from .http import CDSEHttpClient


CATALOGUE_ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"


@dataclass(frozen=True)
class ProductHit:
    cdse_id: str
    name: str
    start: str
    end: str
    geofootprint: Optional[dict] = None
    coverage_ratio: float = 0.0
    cloud_cover: Optional[float] = None


@dataclass(frozen=True)
class ParsedSafeName:
    mission: str
    level: str
    sensing_start: datetime
    baseline: str
    rel_orbit: str
    tile_id: str
    discriminator: datetime


_SAFE_RE = re.compile(
    r"^(S2[ABC])_(MSIL1C|MSIL2A)_(\d{8}T\d{6})_(N\d{4})_(R\d{3})_(T\d{2}[A-Z]{3})_(\d{8}T\d{6})\.SAFE$"
)


def _parse_dt_utc(s: str) -> datetime:
    dt = datetime.strptime(s, "%Y%m%dT%H%M%S")
    return dt.replace(tzinfo=timezone.utc)


def parse_safe_name(name: str) -> ParsedSafeName:
    m = _SAFE_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized SAFE name format: {name}")
    mission, level, sensing, baseline, rel_orbit, tile, discr = m.groups()
    return ParsedSafeName(
        mission=mission,
        level=level,
        sensing_start=_parse_dt_utc(sensing),
        baseline=baseline,
        rel_orbit=rel_orbit,
        tile_id=tile,
        discriminator=_parse_dt_utc(discr),
    )


def baseline_to_int(baseline: str) -> int:
    return int(baseline[1:])


def choose_best_product_by_name(products: Sequence[ProductHit]) -> ProductHit:
    parsed = [(p, parse_safe_name(p.name)) for p in products]
    parsed.sort(key=lambda x: (baseline_to_int(x[1].baseline), x[1].discriminator), reverse=True)
    return parsed[0][0]


def normalize_tile_id(tile_id: str) -> str:
    return tile_id if tile_id.startswith("T") else f"T{tile_id}"


def build_filter_basic(
    *,
    tile_id: str,
    date_from_utc: str,
    date_to_utc: str,
    product_type: str,
    min_cloud_pctg: Optional[float],
    max_cloud_pctg: Optional[float],
) -> str:
    tile = normalize_tile_id(tile_id)
    flt = (
        "Collection/Name eq 'SENTINEL-2' "
        "and Attributes/OData.CSC.StringAttribute/any(att:"
        "att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq "
        f"'{product_type}') "
        f"and contains(Name, '_{tile}_') "
        f"and ContentDate/Start ge {date_from_utc} "
        f"and ContentDate/Start lt {date_to_utc}"
    )

    # Cloud range filter as two separate any(...) clauses (works around 400 on combined predicates).
    if max_cloud_pctg is not None:
        flt += (
            " and Attributes/OData.CSC.DoubleAttribute/any(att:"
            "att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le "
            f"{float(max_cloud_pctg):.2f})"
        )
    if min_cloud_pctg is not None:
        flt += (
            " and Attributes/OData.CSC.DoubleAttribute/any(att:"
            "att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value ge "
            f"{float(min_cloud_pctg):.2f})"
        )

    return flt


SAFE_FILTER = "()'/$=,: "


def _build_odata_products_url(
    filter_expr: str,
    top: int,
    orderby: str,
    select: str | None,
    expand: str | None = None,
) -> str:
    url = (
        f"{CATALOGUE_ODATA_BASE}/Products"
        f"?$filter={quote(filter_expr, safe=SAFE_FILTER)}"
        f"&$top={top}"
        f"&$orderby={quote(orderby, safe=' ,/')}"
    )
    if select:
        url += f"&$select={quote(select, safe=',/')}"
    if expand:
        url += f"&$expand={quote(expand, safe=',()/')}"
    return url


def _coerce_geofootprint(obj: Any) -> Optional[dict]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_cloud_cover(attributes_obj: Any) -> Optional[float]:
    if attributes_obj is None:
        return None

    # Variant A: Attributes is a list of attribute objects
    if isinstance(attributes_obj, list):
        for a in attributes_obj:
            if not isinstance(a, dict):
                continue
            if str(a.get("Name", "")).lower() != "cloudcover":
                continue
            if "Value" in a:
                v = _to_float_or_none(a.get("Value"))
                if v is not None:
                    return v
            da = a.get("OData.CSC.DoubleAttribute")
            if isinstance(da, dict):
                v = _to_float_or_none(da.get("Value"))
                if v is not None:
                    return v
            v = _to_float_or_none(a.get("OData.CSC.DoubleAttribute/Value"))
            if v is not None:
                return v

    # Variant B: Attributes is a dict grouping by attribute type
    if isinstance(attributes_obj, dict):
        for key in ("OData.CSC.DoubleAttribute", "DoubleAttribute"):
            group = attributes_obj.get(key)
            if isinstance(group, list):
                for a in group:
                    if not isinstance(a, dict):
                        continue
                    if str(a.get("Name", "")).lower() != "cloudcover":
                        continue
                    v = _to_float_or_none(a.get("Value"))
                    if v is not None:
                        return v

    return None


def _utm_epsg_from_mgrs_tile(tile_id: str) -> int:
    if tile_id.startswith("T"):
        tile_id = tile_id[1:]
    zone = int(tile_id[:2])
    band = tile_id[2].upper()
    north = (band >= "N")
    return (32600 + zone) if north else (32700 + zone)


def coverage_ratio_from_geofootprint(geofootprint: Optional[dict], tile_id: str, tile_area_m2: float) -> float:
    if not geofootprint:
        return 0.0

    geom_wgs84 = shape(geofootprint)
    epsg = _utm_epsg_from_mgrs_tile(tile_id)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    geom_utm = shp_transform(transformer.transform, geom_wgs84)
    area_m2 = geom_utm.area
    return float(area_m2 / tile_area_m2)


def search_products_odata(
    client: CDSEHttpClient,
    filter_expr: str,
    *,
    top: int,
    orderby: str,
    min_coverage_ratio: float,
    tile_area_m2: float,
    include_attributes: bool,
    select: str = "Id,Name,ContentDate/Start,ContentDate/End,GeoFootprint",
) -> list[ProductHit]:
    # When expanding Attributes, avoid $select (CDSE may omit Attributes otherwise).
    effective_select = None if include_attributes else select

    url = _build_odata_products_url(
        filter_expr=filter_expr,
        top=top,
        orderby=orderby,
        select=effective_select,
        expand=("Attributes" if include_attributes else None),
    )

    out: list[ProductHit] = []
    while True:
        js = client.get_json(url)
        for it in js.get("value", []):
            name = str(it["Name"])
            geo = _coerce_geofootprint(it.get("GeoFootprint"))
            cloud = _extract_cloud_cover(it.get("Attributes"))

            try:
                ps = parse_safe_name(name)
                cov = coverage_ratio_from_geofootprint(geo, ps.tile_id, tile_area_m2=tile_area_m2)
            except Exception:
                cov = 0.0

            if cov < min_coverage_ratio:
                continue

            out.append(ProductHit(
                cdse_id=str(it["Id"]),
                name=name,
                start=str(it.get("ContentDate", {}).get("Start", it.get("ContentDate/Start", ""))),
                end=str(it.get("ContentDate", {}).get("End", it.get("ContentDate/End", ""))),
                geofootprint=geo,
                coverage_ratio=float(cov),
                cloud_cover=cloud,
            ))

        next_link = js.get("@odata.nextLink")
        if not next_link:
            break
        url = next_link

    return out
