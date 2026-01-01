import pytest
from pathlib import Path

from s2pipe.preprocess.inputs import load_download_index, iter_pairs, select_assets, DownloadIndex, IndexPair


@pytest.fixture
def index_path() -> Path:
    # current working directory is tests
    return Path('fixtures/step1_single_tile/meta/manifest/index.json')


def test_load_iter_select_assets(index_path):
    assert index_path.is_file()
    d = load_download_index(index_path)
    assert isinstance(d, DownloadIndex)
    for item in iter_pairs(d):
        assert isinstance(item, IndexPair)
        a = select_assets(
            item, d,
            l1c_bands=[
                'B01', 'B02', 'B03', 'B04',
                'B05', 'B06', 'B07', 'B08',
                'B09', 'B10', 'B11', 'B12', 'B8A'
            ],
            need_l1c_tile_metadata=True,
            need_l2a_tile_metadata=True,
            need_scl_20m=True,
            require_present=True,
        )
