from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from s2pipe.preprocess.radiometry import RadiometryParams, parse_l1c_radiometry


# Ensure the repository root is importable (works for both packaged and flat layouts).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fixtures_single_tile_dir() -> Path:
    tests_dir = Path(__file__).resolve().parents[1]
    return tests_dir / "fixtures" / "single_tile"


def _find_mtd_smil1c_xml() -> Path:
    # User requested search root: tests/fixtures/single_tile/raw/L1C/
    search_root = (_fixtures_single_tile_dir() / "raw" / "L1C").resolve()

    matches = sorted(search_root.rglob("MTD_MSIL1C.xml"))
    if not matches:
        raise FileNotFoundError(f"MTD_SMIL1C.xml not found under: {search_root}")
    if len(matches) > 1:
        # Keep deterministic behavior if multiple fixtures are present.
        return matches[0]
    return matches[0]


@pytest.mark.integration
def test_parse_l1c_radiometry_from_fixture() -> None:
    xml_path = _find_mtd_smil1c_xml()
    params = parse_l1c_radiometry(xml_path)

    assert isinstance(params, RadiometryParams)

    root = ET.parse(xml_path).getroot()

    def _local_tag(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    q_vals: list[float] = []
    for el in root.iter():
        if _local_tag(el.tag) == "QUANTIFICATION_VALUE" and el.text:
            q_vals.append(float(el.text.strip()))
    assert q_vals, "Fixture must contain QUANTIFICATION_VALUE."
    assert len(set(q_vals)) == 1, "Fixture must contain a single QUANTIFICATION_VALUE."
    expected_quant = float(q_vals[0])

    # Read offset for bandId=8 (B8A) from the XML and compare against the parser output.
    expected_b8a_offset: float | None = None
    for el in root.iter():
        if _local_tag(el.tag) != "RADIO_ADD_OFFSET":
            continue
        band_id_raw = el.attrib.get("bandId") or el.attrib.get("band_id")
        if band_id_raw is None:
            continue
        if int(str(band_id_raw).strip()) == 8 and el.text:
            expected_b8a_offset = float(el.text.strip())
            break

    assert params.quantification_value == pytest.approx(expected_quant)

    expected_bands = {
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
    }
    assert set(params.radio_add_offset_by_band.keys()) == expected_bands

    assert expected_b8a_offset is not None, (
        "Fixture must contain RADIO_ADD_OFFSET for bandId=8."
    )
    assert params.radio_add_offset_by_band["B8A"] == pytest.approx(expected_b8a_offset)

    offset, quant = params.get_band_params("B8A")
    assert offset == pytest.approx(expected_b8a_offset)
    assert quant == pytest.approx(expected_quant)
