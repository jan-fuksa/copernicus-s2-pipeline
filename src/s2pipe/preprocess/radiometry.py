from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import xml.etree.ElementTree as ET


# Sentinel-2 bandId -> band name mapping.
_BAND_ID_TO_NAME: dict[int, str] = {
    0: "B01",
    1: "B02",
    2: "B03",
    3: "B04",
    4: "B05",
    5: "B06",
    6: "B07",
    7: "B08",
    8: "B8A",
    9: "B09",
    10: "B10",
    11: "B11",
    12: "B12",
}

_EXPECTED_BANDS: tuple[str, ...] = tuple(
    _BAND_ID_TO_NAME[i] for i in sorted(_BAND_ID_TO_NAME.keys())
)


def _local_tag(tag: str) -> str:
    """Return local (namespace-shaved) tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_by_local_tag(root: ET.Element, local: str) -> Iterable[ET.Element]:
    """Iterate over all elements with the given local tag name."""
    for el in root.iter():
        if _local_tag(el.tag) == local:
            yield el


def _get_attr(el: ET.Element, *names: str) -> str | None:
    """Get an attribute by trying multiple names, plus a suffix match fallback."""
    for n in names:
        if n in el.attrib:
            return el.attrib[n]

    # Suffix match fallback for unusual attribute naming.
    for k, v in el.attrib.items():
        lk = k.lower()
        for n in names:
            nn = n.lower()
            if lk == nn or lk.endswith(nn):
                return v
    return None


def _parse_float(text: str, *, what: str) -> float:
    try:
        return float(text.strip())
    except Exception as e:
        raise ValueError(f"Failed to parse {what} as float: {text!r}") from e


@dataclass(frozen=True)
class RadiometryParams:
    quantification_value: float
    radio_add_offset_by_band: Dict[str, float]

    def get_band_params(self, band: str) -> Tuple[float, float]:
        """Return (RADIO_ADD_OFFSET, QUANTIFICATION_VALUE) for a given band name."""
        if band not in self.radio_add_offset_by_band:
            raise KeyError(f"Missing RADIO_ADD_OFFSET for band={band!r}.")
        return (
            float(self.radio_add_offset_by_band[band]),
            float(self.quantification_value),
        )


def parse_l1c_radiometry(product_metadata_xml: Path) -> RadiometryParams:
    """Parse L1C radiometry parameters from product-level metadata XML.

    Requirements:
      - QUANTIFICATION_VALUE must be present globally.
      - RADIO_ADD_OFFSET must be present per band.
      - Exception: if RADIO_ADD_OFFSET is missing for all bands, offsets are set to 0.0
        for the full Sentinel-2 band set (B01..B12 including B8A).
    """
    xml_path = Path(product_metadata_xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"Product metadata XML not found: {xml_path}")

    root = ET.parse(xml_path).getroot()

    # Global quantification value (must exist, and must be consistent if repeated).
    q_values: list[float] = []
    for el in _iter_by_local_tag(root, "QUANTIFICATION_VALUE"):
        if el.text is None:
            continue
        q_values.append(_parse_float(el.text, what="QUANTIFICATION_VALUE"))

    if not q_values:
        raise ValueError("Missing required QUANTIFICATION_VALUE in product metadata.")

    q_unique = {float(v) for v in q_values}
    if len(q_unique) != 1:
        raise ValueError(
            f"Conflicting QUANTIFICATION_VALUE values found: {sorted(q_unique)}"
        )

    quant = float(next(iter(q_unique)))

    # Per-band offsets.
    offsets: dict[str, float] = {}
    offset_elems = list(_iter_by_local_tag(root, "RADIO_ADD_OFFSET"))

    if not offset_elems:
        # Allowed legacy case: offsets absent entirely -> default to zeros for all bands.
        return RadiometryParams(
            quantification_value=quant,
            radio_add_offset_by_band={b: 0.0 for b in _EXPECTED_BANDS},
        )

    for el in offset_elems:
        band_id_raw = _get_attr(el, "bandId", "band_id")
        if band_id_raw is None:
            raise ValueError(
                "RADIO_ADD_OFFSET element is missing required bandId attribute."
            )

        try:
            band_id = int(str(band_id_raw).strip())
        except Exception as e:
            raise ValueError(
                f"Invalid bandId value in RADIO_ADD_OFFSET: {band_id_raw!r}"
            ) from e

        if band_id not in _BAND_ID_TO_NAME:
            raise ValueError(f"Unsupported bandId={band_id} in RADIO_ADD_OFFSET.")

        if el.text is None:
            raise ValueError(f"Missing RADIO_ADD_OFFSET value for bandId={band_id}.")

        band_name = _BAND_ID_TO_NAME[band_id]
        value = _parse_float(el.text, what=f"RADIO_ADD_OFFSET[{band_name}]")

        if band_name in offsets and offsets[band_name] != value:
            raise ValueError(
                f"Conflicting RADIO_ADD_OFFSET values for band={band_name}: "
                f"{offsets[band_name]} vs {value}"
            )

        offsets[band_name] = value

    # Enforce per-band availability (no fallback logic).
    missing = sorted(set(_EXPECTED_BANDS) - set(offsets.keys()))
    extra = sorted(set(offsets.keys()) - set(_EXPECTED_BANDS))
    if missing or extra:
        msg_parts: list[str] = ["RADIO_ADD_OFFSET must be provided per band."]
        if missing:
            msg_parts.append(f"Missing bands: {missing}")
        if extra:
            msg_parts.append(f"Unexpected bands: {extra}")
        raise ValueError(" ".join(msg_parts))

    return RadiometryParams(
        quantification_value=quant,
        radio_add_offset_by_band={b: float(offsets[b]) for b in _EXPECTED_BANDS},
    )
