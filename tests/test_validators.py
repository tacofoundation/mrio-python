"""Tests for the validators module.

This module provides comprehensive testing for mCOG and tCOG validation
functions, including metadata checking and error handling.
"""

"""Tests for the validators module with proper warning handling."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from mrio.validators import (
    _get_missing_attributes,
    _get_missing_fields,
    check_metadata,
    is_mcog,
    is_tcog,
)

# Configure warnings for the entire test module
pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")


@pytest.fixture
def valid_transform():
    """Create a valid geotransform that won't trigger warnings."""
    return Affine(
        100.0,  # pixel width
        0.0,  # row rotation
        453580.0,  # upper left x
        0.0,  # column rotation
        -100.0,  # pixel height (negative because north-up)
        5578700.0,  # upper left y
    )


@pytest.fixture
def valid_mcog(tmp_path, valid_transform):
    """Create a valid mCOG file for testing."""
    file_path = tmp_path / "valid_mgeo.tif"
    metadata = {"md:pattern": "band h w -> (band) h w", "md:coordinates": {"band": ["red", "green", "blue"]}}

    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 3,
        "dtype": "uint8",
        "transform": valid_transform,
        "crs": "EPSG:32618",  # UTM zone 18N
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    data = np.zeros((3, 10, 10), dtype=np.uint8)

    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**{"MD_METADATA": json.dumps(metadata)})

    return file_path


@pytest.fixture
def valid_tcog(tmp_path, valid_transform):
    """Create a valid tCOG file for testing."""
    file_path = tmp_path / "valid_tgeo.tif"
    metadata = {
        "md:pattern": "time band h w -> (time band) h w",
        "md:coordinates": {"time": ["2021", "2022"], "band": ["red", "green", "blue"]},
        "md:attributes": {"md:time_start": "2021-01-01", "md:id": "S2A_MSIL2A"},
    }

    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 6,
        "dtype": "uint8",
        "transform": valid_transform,
        "crs": "EPSG:32618",
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    data = np.zeros((6, 10, 10), dtype=np.uint8)

    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**{"MD_METADATA": json.dumps(metadata)})

    return file_path


@pytest.fixture
def invalid_metadata_file(tmp_path, valid_transform):
    """Create a file with invalid JSON metadata."""
    file_path = tmp_path / "invalid_metadata.tif"
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "uint8",
        "transform": valid_transform,
        "crs": "EPSG:32618",
    }

    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.uint8))
        dst.update_tags(**{"MD_METADATA": "{invalid json"})

    return file_path


def test_valid_mcog(valid_mcog):
    """Test validation of a valid mCOG file."""
    assert is_mcog(valid_mcog)
    assert check_metadata(valid_mcog, ["md:pattern", "md:coordinates"])


def test_valid_tcog(valid_tcog):
    """Test validation of a valid tCOG file."""
    assert is_tcog(valid_tcog)
    assert check_metadata(valid_tcog, ["md:pattern", "md:coordinates"], ["md:time_start", "md:id"])


def test_tcog_as_mcog(valid_tcog):
    """Test that a valid tCOG is also a valid mCOG."""
    assert is_mcog(valid_tcog)


def test_nonexistent_file():
    """Test handling of non-existent files."""
    with warnings.catch_warnings(record=True) as w:
        assert not is_mcog("nonexistent.tif", strict=False)
        assert len(w) == 1
        assert "Failed to open file" in str(w[0].message)

    with pytest.raises(ValueError, match="Failed to open file"):
        is_mcog("nonexistent.tif", strict=True)


def test_invalid_json_metadata(invalid_metadata_file):
    """Test handling of invalid JSON metadata."""
    with warnings.catch_warnings(record=True) as w:
        assert not is_mcog(invalid_metadata_file, strict=False)
        assert len(w) == 1
        assert "Invalid metadata JSON" in str(w[0].message)

    with pytest.raises(ValueError, match="Invalid metadata JSON"):
        is_mcog(invalid_metadata_file, strict=True)


def test_missing_metadata(tmp_path, valid_transform):
    """Test handling of files without metadata."""
    file_path = tmp_path / "no_metadata.tif"
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "uint8",
        "transform": valid_transform,
        "crs": "EPSG:32618",
    }

    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.uint8))

    with warnings.catch_warnings(record=True) as w:
        assert not is_mcog(file_path, strict=False)
        assert len(w) == 1
        assert "MD_METADATA not found" in str(w[0].message)


def test_get_missing_fields():
    """Test _get_missing_fields function."""
    metadata = {"field1": "value1", "field2": "value2"}
    required = ["field1", "field3"]
    assert _get_missing_fields(metadata, required) == {"field3"}


def test_get_missing_attributes():
    """Test _get_missing_attributes function."""
    metadata = {"md:attributes": {"attr1": "value1", "attr2": "value2"}}
    required = ["attr1", "attr3"]
    assert _get_missing_attributes(metadata, required) == {"attr3"}


def test_empty_attributes():
    """Test handling of missing attributes section."""
    metadata = {}
    required = ["attr1"]
    assert _get_missing_attributes(metadata, required) == {"attr1"}


@pytest.mark.parametrize(
    "path_type",
    [
        str,
        Path,
        lambda x: Path(x).resolve(),
    ],
)
def test_path_types(valid_mcog, path_type):
    """Test handling of different path types."""
    path = path_type(valid_mcog)
    assert is_mcog(path)


if __name__ == "__main__":
    pytest.main(["-v", "--cov=mrio.validators", "--cov-report=term-missing"])
