import os
from pathlib import Path

import pytest

from s2pipe.preprocess.inputs import (
    load_download_index,
    iter_pairs,
    select_assets,
    DownloadIndex,
    IndexPair,
    SelectedAssets,
)


def _testdata_root() -> Path:
    env = os.environ.get("S2PIPE_TESTDATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "fixtures" / "single_tile").resolve()


@pytest.mark.integration
def test_load_iter_select_assets():
    root = _testdata_root()
    index_path = root / "meta" / "step1" / "index.json"
    if not index_path.exists():
        pytest.skip(f"Missing test data index.json at: {index_path}")

    assert index_path.is_file()
    d = load_download_index(index_path)
    assert isinstance(d, DownloadIndex)
    for item in iter_pairs(d):
        assert isinstance(item, IndexPair)
        a = select_assets(
            item,
            d,
            l1c_bands=[
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B09",
                "B10",
                "B11",
                "B12",
                "B8A",
            ],
            need_l1c_tile_metadata=True,
            need_l2a_tile_metadata=True,
            need_scl_20m=True,
            require_present=True,
        )
        assert isinstance(a, SelectedAssets)
