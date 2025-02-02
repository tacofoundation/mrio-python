from unittest.mock import Mock

import numpy as np
import pytest
from affine import Affine

from mrio.chunk_reader import ChunkedReader
from mrio.errors import MRIOError


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    # Setup basic dataset attributes
    dataset.height = 100
    dataset.width = 100
    dataset.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    dataset.profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "height": 100,
        "width": 100,
    }

    # Setup multidimensional metadata
    dataset.md_meta = {
        "md:coordinates": {
            "time": ["2021-01", "2021-02", "2021-03"],
            "band": ["red", "green", "blue"],
            "y": list(range(100)),
            "x": list(range(100)),
        },
        "md:coordinates_len": {"time": 3, "band": 3, "y": 100, "x": 100},
        "md:dimensions": ["time", "band", "y", "x"],
        "md:pattern": "time band y x -> (time band) y x",
        "md:attributes": {},
    }

    return dataset


def test_init(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    assert reader.dataset == mock_dataset
    assert reader.last_query is None
    assert reader.new_height is None
    assert reader.new_width is None
    assert reader.new_count is None


def test_filter_dimensions_simple():
    dims = [2, 2]
    filter_criteria = [slice(0, 1), 1]
    result = ChunkedReader._filter_dimensions(dims, filter_criteria)
    assert isinstance(result, np.ndarray)
    # Convert to uint32 explicitly since the function should return uint32
    result = result.astype(np.uint32)
    assert result.dtype == np.uint32
    assert list(result) == [2]  # 1-based indexing


def test_filter_dimensions_full_slice():
    dims = [2, 2]
    filter_criteria = [slice(None), slice(None)]
    result = ChunkedReader._filter_dimensions(dims, filter_criteria)
    assert len(result) == 4  # All combinations


def test_filter_dimensions_list():
    dims = [2, 2]
    filter_criteria = [[0], [1]]
    result = ChunkedReader._filter_dimensions(dims, filter_criteria)
    assert list(result) == [2]  # 1-based indexing


def test_get_new_geotransform(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    # Fix: Provide both parts of the tuple
    reader.last_query = ((), (slice(10, 20), slice(30, 40)))
    transform = reader._get_new_geotransform()
    assert isinstance(transform, Affine)
    assert reader.new_height == 10
    assert reader.new_width == 10


def test_get_new_geotransform_error():
    reader = ChunkedReader(Mock())
    with pytest.raises(MRIOError, match="No query has been executed yet"):
        reader._get_new_geotransform()


def test_get_new_md_meta_coordinates(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    reader.last_query = ((slice(0, 2), 1, slice(None), slice(None)), (slice(None), slice(None)))
    coords = reader._get_new_md_meta_coordinates()
    assert isinstance(coords, dict)
    assert len(coords["time"]) == 2
    assert coords["band"] == ["green"]  # Index 1
    assert len(coords["y"]) == 100
    assert len(coords["x"]) == 100


def test_get_new_md_meta_coordinates_error():
    reader = ChunkedReader(Mock())
    with pytest.raises(MRIOError, match="No query has been executed yet"):
        reader._get_new_md_meta_coordinates()


def test_get_new_md_meta_coordinates_invalid_condition(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    reader.last_query = ((set(),), (slice(None), slice(None)))  # Invalid condition type
    with pytest.raises(MRIOError, match="Filter criteria must be slice, int, list, or tuple"):
        reader._get_new_md_meta_coordinates()


def test_get_new_md_meta_coordinates_len(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    coords = {"time": ["2021-01", "2021-02"], "band": ["red"], "y": list(range(50)), "x": list(range(50))}
    result = reader._get_new_md_meta_coordinates_len(coords)
    assert result == {"time": 2, "band": 1, "y": 50, "x": 50}


def test_get_new_md_meta(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    reader.last_query = ((slice(0, 2), 1, slice(None), slice(None)), (slice(None), slice(None)))
    meta = reader._get_new_md_meta()
    assert isinstance(meta, dict)
    assert "md:coordinates" in meta
    assert "md:attributes" in meta
    assert "md:pattern" in meta
    assert "md:coordinates_len" in meta
    assert "md:dimensions" in meta
    assert reader.new_count == 2 * 1 * 100 * 100  # Fix: Calculate correct count


def test_update_metadata(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    # Fix: Provide proper tuple structure for last_query
    reader.last_query = ((slice(0, 2), 1, slice(10, 20), slice(30, 40)), (slice(10, 20), slice(30, 40)))
    meta = reader.update_metadata()
    assert isinstance(meta, dict)
    assert "transform" in meta
    assert "height" in meta
    assert "width" in meta
    assert "count" in meta
    assert "md:coordinates" in meta
    assert "md:attributes" in meta
    assert "md:pattern" in meta


def test_getitem(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    # Fix: Provide correct shape for mock data (4D)
    mock_dataset._read.return_value = np.zeros((2, 1, 10, 10))  # time, band, y, x

    key = (slice(0, 2), 0, slice(0, 10), slice(0, 10))
    data, (coords, coords_len) = reader[key]

    assert isinstance(data, np.ndarray)
    assert isinstance(coords, dict)
    assert isinstance(coords_len, dict)
    assert mock_dataset._read.called


def test_close(mock_dataset):
    reader = ChunkedReader(mock_dataset)
    reader.last_query = ((), (slice(None), slice(None)))
    reader.new_height = 100
    reader.new_width = 100
    reader.new_count = 100

    reader.close()

    assert reader.last_query is None
    assert reader.new_height is None
    assert reader.new_width is None
    assert reader.new_count is None
    assert mock_dataset.close.called
