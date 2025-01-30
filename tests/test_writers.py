"""Tests for the MRIO Dataset Writer."""
import json
import warnings
from typing import Any, Dict

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds

import mrio
from mrio import DatasetWriter
from mrio.errors import ValidationError

# Filter rasterio warnings globally for tests
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Fixtures
# --------

@pytest.fixture
def sample_data() -> np.ndarray:
    """Create sample data for testing."""
    return np.random.randint(0, 255, size=(2, 3, 128, 128), dtype=np.uint8)

@pytest.fixture
def base_params() -> Dict[str, Any]:
    """Create base parameters for testing."""
    return {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": 128,
        "height": 128,
        "crs": "EPSG:4326",
        "transform": from_bounds(-76.2, 4.31, -76.1, 4.32, 128, 128),
        "md:pattern": "time band y x -> (time band) y x",
        "md:coordinates": {
            "time": ["2021", "2022"],
            "band": ["red", "green", "blue"]
        },
        "md:attributes": {
            "satellite": "Sentinel-2"
        }
    }

# Basic Write Tests
# ---------------

def test_basic_write(tmp_path, sample_data, base_params):
    """Test basic write functionality."""
    output_file = tmp_path / "test.tif"

    with DatasetWriter(output_file, **base_params) as writer:
        writer.write(sample_data)

    # Verify file was written correctly
    with rasterio.open(output_file) as src:
        assert src.count == 6  # 2 times * 3 bands
        assert src.width == base_params["width"]
        assert src.height == base_params["height"]
        read_data = src.read()
        np.testing.assert_array_equal(read_data.reshape(2, 3, 128, 128), sample_data)

# Parameter Inference Tests
# ----------------------

def test_parameter_inference(tmp_path, sample_data):
    """Test automatic parameter inference."""
    output_file = tmp_path / "inferred.tif"
    params = {
        "crs": "EPSG:4326",
        "transform": from_bounds(-76.2, 4.31, -76.1, 4.32, 128, 128),
        "md:pattern": "time band y x -> (time band) y x",
        "md:coordinates": {
            "time": ["2021", "2022"],
            "band": ["red", "green", "blue"]
        }
    }

    with DatasetWriter(output_file, **params) as writer:
        writer.write(sample_data)

    with rasterio.open(output_file) as src:
        assert src.width == 128
        assert src.height == 128
        assert src.count == 6  # 2 times * 3 bands
        assert src.dtypes[0] == 'uint8'

# Validation Tests
# -------------

def test_invalid_pattern(tmp_path, base_params):
    """Test invalid pattern format."""
    output_file = tmp_path / "invalid.tif"
    params = base_params.copy()
    params["md:pattern"] = "invalid pattern"

    try:
        writer = DatasetWriter(output_file, **params)
        # If no exception is raised during initialization
        print("No exception raised during initialization")

        # Try to write some data to trigger validation
        sample_data = np.random.randint(0, 255, size=(2, 3, 128, 128), dtype=np.uint8)
        writer.write(sample_data)

        # If write doesn't raise an exception
        pytest.fail("Expected ValidationError was not raised")
    except ValidationError as ve:
        # Check if the error message matches the expected pattern
        print(f"Caught ValidationError: {ve}")
        assert "Pattern must contain ' -> ' separator" in str(ve)
    except Exception as e:
        # Catch and print any other unexpected exceptions
        print(f"Unexpected exception: {type(e)} - {e}")
        pytest.fail(f"Unexpected exception: {e}")


def test_invalid_coordinates(tmp_path, base_params, sample_data):
    """Test invalid coordinates."""
    #import pathlib
    #tmp_path = pathlib.Path(".")
    output_file = tmp_path / "invalid.tif"
    params = base_params.copy()
    params["md:coordinates"] = {"wrong": "not a list"}

    with pytest.raises(ValidationError, match="All coordinate values must be lists"):
        with DatasetWriter(output_file, **params) as writer:
            writer.write(sample_data)

def test_missing_mandatory_fields(tmp_path, sample_data):
    """Test missing mandatory fields."""
    output_file = tmp_path / "missing.tif"

    with pytest.raises(Exception, match="Mandatory fields missing"):
        with DatasetWriter(output_file) as writer:
            writer.write(sample_data)

def test_invalid_dimensions(tmp_path, base_params):
    """Test data with invalid dimensions."""
    output_file = tmp_path / "invalid.tif"
    invalid_data = np.random.rand(10)  # 1D data

    with DatasetWriter(output_file, **base_params) as writer:
        with pytest.raises(ValueError, match="Data must have at least 2 dimensions"):
            writer._infer_parameters(invalid_data)

# Resource Management Tests
# ----------------------

def test_context_manager(tmp_path, base_params, sample_data):
    """Test context manager behavior."""
    output_file = tmp_path / "context.tif"

    with DatasetWriter(output_file, **base_params) as writer:
        writer.write(sample_data)
        assert writer._file is not None
        assert not writer._file.closed

def test_explicit_close(tmp_path, base_params):
    """Test explicit close method."""
    output_file = tmp_path / "close.tif"
    writer = DatasetWriter(output_file, **base_params)
    writer.write(np.random.randint(0, 255, size=(2, 3, 128, 128), dtype=np.uint8))
    writer.close()
    assert writer._file is None


# Metadata Tests
# ------------

def test_metadata_writing(tmp_path, sample_data, base_params):
    """Test metadata is written correctly."""
    output_file = tmp_path / "metadata.tif"

    with DatasetWriter(output_file, **base_params) as writer:
        writer.write(sample_data)

    with rasterio.open(output_file) as src:
        metadata = json.loads(src.tags()["MD_METADATA"])
        pattern_inv = " -> ".join(map(str.strip, base_params["md:pattern"].split("->")[::-1]))
        assert metadata["md:coordinates"] == base_params["md:coordinates"]
        assert metadata.get("md:pattern", "") == pattern_inv

def test_band_descriptions(tmp_path, sample_data, base_params):
    """Test band descriptions are set correctly."""
    output_file = tmp_path / "bands.tif"

    with DatasetWriter(output_file, **base_params) as writer:
        writer.write(sample_data)

    with mrio.open(output_file) as tensor:
        descriptions = tensor.descriptions
        assert len(descriptions) == 6  # 2 times * 3 bands
        assert all(descriptions)

if __name__ == "__main__":
    pytest.main(["-v", "--cov=mrio.writers", "--cov-report=term-missing"])
