"""
MRIO Test Suite

Comprehensive testing for Multi-dimensional Raster I/O operations.
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine

import mrio
from mrio import (
    open as mrio_open,
    DatasetReader,
    DatasetWriter,
    MRIOError
)

# Test Data Configuration
# ----------------------
@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample metadata configuration for MGeoTIFF."""
    return {
        "md:pattern": "time band h w -> (time band) h w",
        "md:coordinates": {
            "time": ["2019", "2020", "2021"],
            "band": ["B02", "B03", "B04", "B08"],
        },
        "md:attributes": {
            "satellite": "Sentinel-2",
            "resolution": 10,
            "processing_level": "L2A",
        }
    }

@pytest.fixture
def sample_profile() -> Dict[str, Any]:
    """Sample raster profile."""
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": 100,
        "height": 100,
        "count": 12,  # 3 times * 4 bands
        "crs": CRS.from_epsg(4326),
        "transform": Affine(0.1, 0, -180, 0, -0.1, 90),
        "tiled": True,
        "compress": "zstd",
        "interleave": "pixel",
    }

@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate sample 4D data array."""
    return np.random.rand(3, 4, 100, 100).astype(np.float32)

# Writer Tests
# -----------
class TestWriter:
    """Test DatasetWriter functionality."""

    def test_basic_write(self, tmp_path: Path, sample_data: np.ndarray, 
                        sample_profile: Dict[str, Any], sample_metadata: Dict[str, Any]):
        """Test basic writing functionality."""
        output_file = tmp_path / "test.tif"
        merged_profile = {**sample_profile, **sample_metadata}
        
        with mrio_open(output_file, "w", **merged_profile) as dst:
            dst.write(sample_data)
        
        # Verify file was created and is readable
        with rasterio.open(output_file) as src:
            assert src.count == sample_profile["count"]
            assert src.profile["crs"] == sample_profile["crs"]
            
            # Check metadata was written correctly
            metadata = json.loads(src.tags()["MD_METADATA"])
            assert metadata["md:pattern"] == sample_metadata["md:pattern"]
            assert len(metadata["md:coordinates"]["time"]) == 3
            assert len(metadata["md:coordinates"]["band"]) == 4

    def test_chunked_write(self, tmp_path: Path, sample_profile: Dict[str, Any],
                          sample_metadata: Dict[str, Any]):
        """Test writing data in chunks."""
        output_file = tmp_path / "chunked.tif"
        merged_profile = {**sample_profile, **sample_metadata}
        
        chunk_size = 25
        with mrio_open(output_file, "w", **merged_profile) as dst:
            for i in range(0, 100, chunk_size):
                chunk = np.random.rand(3, 4, chunk_size, 100).astype(np.float32)
                dst.write(chunk, window=((i, i+chunk_size), (0, 100)))

        # Verify chunks were written correctly
        with mrio_open(output_file) as src:
            full_data = src.read()
            assert full_data.shape == (3, 4, 100, 100)

    def test_invalid_data_shape(self, tmp_path: Path, sample_profile: Dict[str, Any],
                               sample_metadata: Dict[str, Any]):
        """Test error handling for invalid data shapes."""
        output_file = tmp_path / "invalid.tif"
        merged_profile = {**sample_profile, **sample_metadata}
        
        with pytest.raises(ValueError):
            with mrio_open(output_file, "w", **merged_profile) as dst:
                invalid_data = np.random.rand(2, 2, 100, 100)  # Wrong dimensions
                dst.write(invalid_data)

# Reader Tests
# -----------
class TestReader:
    """Test DatasetReader functionality."""

    @pytest.fixture
    def test_file(self, tmp_path: Path, sample_data: np.ndarray,
                  sample_profile: Dict[str, Any], sample_metadata: Dict[str, Any]) -> Path:
        """Create a test file for reading tests."""
        output_file = tmp_path / "test_read.tif"
        merged_profile = {**sample_profile, **sample_metadata}
        
        with mrio_open(output_file, "w", **merged_profile) as dst:
            dst.write(sample_data)
        
        return output_file

    def test_basic_read(self, test_file: Path):
        """Test basic reading functionality."""
        with mrio_open(test_file) as src:
            data = src.read()
            assert isinstance(data, xr.DataArray)
            assert data.dims == ("time", "band", "y", "x")
            assert data.shape == (3, 4, 100, 100)

    def test_partial_read(self, test_file: Path):
        """Test reading specific regions."""
        with mrio_open(test_file) as src:
            # Read specific time and bands
            subset = src[0:2, 1:3, 10:50, 20:60]
            assert subset.shape == (2, 2, 40, 40)
            
            # Read using boolean indexing
            mask = np.array([True, False, True, False])
            subset = src[:, mask, :, :]
            assert subset.shape == (3, 2, 100, 100)

    def test_lazy_loading(self, test_file: Path):
        """Test lazy loading behavior."""
        with mrio_open(test_file) as src:
            # Access metadata without loading data
            assert src.shape == (3, 4, 100, 100)
            assert src.crs
            assert src.transform
            
            # Ensure data isn't loaded until explicitly requested
            assert not hasattr(src, '_cached_data')
            _ = src.read()
            assert hasattr(src, '_cached_data')

    def test_coordinate_handling(self, test_file: Path):
        """Test coordinate system handling."""
        with mrio_open(test_file) as src:
            data = src.read()
            assert all(coord in data.coords for coord in ['time', 'band'])
            assert data.coords['time'].shape == (3,)
            assert data.coords['band'].shape == (4,)

# Slice Transformer Tests
# ---------------------
class TestSliceTransformer:
    """Test slice transformation logic."""

    @pytest.fixture
    def transformer(self):
        """Create a slice transformer instance."""
        return mrio.SliceTransformer(ndim=4)

    def test_basic_slicing(self, transformer):
        """Test basic slice operations."""
        # Integer indexing
        result = transformer.transform(1)
        assert result == (slice(1, 2), slice(None), slice(None), slice(None))
        
        # Slice indexing
        result = transformer.transform(slice(1, 3))
        assert result == (slice(1, 3), slice(None), slice(None), slice(None))
        
        # Mixed indexing
        result = transformer.transform((1, slice(None), slice(2, 4), 3))
        assert result == (slice(1, 2), slice(None), slice(2, 4), slice(3, 4))

    def test_advanced_indexing(self, transformer):
        """Test advanced indexing features."""
        # List indexing
        result = transformer.transform(([1, 3], 2))
        expected = ((slice(1, 2), slice(3, 4)), slice(2, 3), slice(None), slice(None))
        assert result == expected
        
        # Ellipsis
        result = transformer.transform((Ellipsis, 1))
        assert result == (slice(None), slice(None), slice(None), slice(1, 2))

    def test_error_handling(self, transformer):
        """Test error conditions."""
        with pytest.raises(ValueError):
            transformer.transform(([1, 2], 5))  # Invalid dimension
        
        with pytest.raises(TypeError):
            transformer.transform({"invalid": "type"})  # Invalid type

# Integration Tests
# ---------------
class TestIntegration:
    """Test complete read/write workflows."""

    def test_roundtrip(self, tmp_path: Path, sample_data: np.ndarray,
                      sample_profile: Dict[str, Any], sample_metadata: Dict[str, Any]):
        """Test writing and reading back data."""
        output_file = tmp_path / "roundtrip.tif"
        merged_profile = {**sample_profile, **sample_metadata}
        
        # Write data
        with mrio_open(output_file, "w", **merged_profile) as dst:
            dst.write(sample_data)
        
        # Read it back
        with mrio_open(output_file) as src:
            read_data = src.read()
            
            # Compare data
            np.testing.assert_array_almost_equal(sample_data, read_data.values)
            
            # Compare metadata
            assert src.coords["time"] == sample_metadata["md:coordinates"]["time"]
            assert src.coords["band"] == sample_metadata["md:coordinates"]["band"]

    def test_partial_updates(self, tmp_path: Path, sample_data: np.ndarray,
                           sample_profile: Dict[str, Any], sample_metadata: Dict[str, Any]):
        """Test partial data updates."""
        output_file = tmp_path / "updates.tif"
        merged_profile = {**sample_profile, **sample_metadata}
        
        # Write initial data
        with mrio_open(output_file, "w", **merged_profile) as dst:
            dst.write(sample_data)
        
        # Update specific regions
        update_data = np.random.rand(1, 2, 50, 50).astype(np.float32)
        with mrio_open(output_file, "r+") as dst:
            dst[0:1, 0:2, 0:50, 0:50] = update_data
        
        # Verify updates
        with mrio_open(output_file) as src:
            subset = src[0:1, 0:2, 0:50, 0:50]
            np.testing.assert_array_almost_equal(update_data, subset.values)

if __name__ == "__main__":
    pytest.main(["-v"])