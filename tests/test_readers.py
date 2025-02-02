import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

import mrio


@pytest.fixture
def sample_datacube():
    """Create a realistic sample datacube matching the demo example"""
    # Create sample data
    data = np.random.randint(0, 255, size=(2, 10, 4, 64, 64), dtype=np.uint8)  # Smaller size for testing
    time = list(pd.date_range("2021-01-01", periods=10, freq="5D").strftime("%Y%m%d"))
    bands = ["B01", "B02", "B03", "B04"]
    products = ["toa", "boa"]

    # Create xarray DataArray
    datacube = xr.DataArray(
        data=data,
        dims=["products", "time", "band", "lat", "lon"],
        coords={"products": products, "time": time, "band": bands},
    )

    return datacube


@pytest.fixture
def write_params():
    """Create parameters for writing the datacube"""
    time = list(pd.date_range("2021-01-01", periods=10, freq="5D").strftime("%Y%m%d"))
    bands = ["B01", "B02", "B03", "B04"]
    products = ["toa", "boa"]

    return {
        "crs": "EPSG:4326",
        "transform": Affine.from_gdal(-76.2, 0.001, 0, 4.32, 0, -0.001),
        "md:pattern": "products time band lat lon -> (products time band) lat lon",
        "md:coordinates": {"products": products, "time": time, "band": bands},
        "md:attributes": {"satellite": "Sentinel-2 L1C"},
    }


@pytest.fixture
def temp_dataset(tmp_path, sample_datacube, write_params):
    """Create a temporary dataset file using the real writer"""
    file_path = tmp_path / "test_datacube.tif"

    with mrio.open(file_path, mode="w", **write_params) as src:
        src.write(sample_datacube.values)

    return file_path


def test_full_workflow(temp_dataset):
    """Test the complete workflow from the demo example"""
    with mrio.open(temp_dataset, engine="numpy") as src:
        # Test various slicing patterns from the demo
        tensor_slice = src[:, 0:5, :2]

        # Verify the shape matches expected dimensions
        assert tensor_slice.shape == (2, 5, 2, 64, 64)
        assert tensor_slice.dtype == np.uint8

        # Test full read
        full_data = src.read()
        assert full_data.shape == (2, 10, 4, 64, 64)


def test_coordinate_handling(temp_dataset):
    """Test coordinate handling with real-world data"""
    with mrio.open(temp_dataset, engine="xarray") as src:
        # Test coordinate preservation
        assert all(dim in src.coords for dim in ["products", "time", "band"])
        assert len(src.coords["time"]) == 10
        assert len(src.coords["band"]) == 4

        # Test coordinate slicing
        subset = src[:, 0:5, :2]
        assert len(subset.coords["time"]) == 5
        assert len(subset.coords["band"]) == 2
        assert list(subset.coords["products"]) == ["toa", "boa"]


def test_large_data_handling(tmp_path):
    """Test handling of larger datasets"""
    # Create a larger dataset (but still manageable for testing)
    data = np.random.randint(0, 255, size=(2, 5, 4, 128, 128), dtype=np.uint8)
    time = list(pd.date_range("2021-01-01", periods=5, freq="5D").strftime("%Y%m%d"))
    bands = ["B01", "B02", "B03", "B04"]
    products = ["toa", "boa"]

    datacube = xr.DataArray(
        data=data,
        dims=["products", "time", "band", "lat", "lon"],
        coords={"products": products, "time": time, "band": bands},
    )

    file_path = tmp_path / "large_test.tif"
    params = {
        "crs": "EPSG:4326",
        "transform": Affine.from_gdal(-76.2, 0.001, 0, 4.32, 0, -0.001),
        "md:pattern": "products time band lat lon -> (products time band) lat lon",
        "md:coordinates": {"products": products, "time": time, "band": bands},
    }

    # Write and read the large dataset
    with mrio.open(file_path, mode="w", **params) as src:
        src.write(datacube.values)

    with mrio.open(file_path, engine="numpy") as src:
        # Test reading specific chunks
        chunk = src[0, :2, :2]
        assert chunk.shape == (1, 2, 2, 128, 128)

        # Test reading full data
        full_data = src.read()
        assert full_data.shape == (2, 5, 4, 128, 128)


def test_multidimensional_slicing(temp_dataset):
    """Test various multidimensional slicing patterns"""
    with mrio.open(temp_dataset, engine="numpy") as src:
        # Test different slicing patterns
        slice1 = src[0]  # Single product
        assert slice1.shape == (1, 10, 4, 64, 64)

        slice2 = src[:, :, 0]  # Single band
        assert slice2.shape == (2, 10, 1, 64, 64)

        slice3 = src[:, 0:5, :2]  # Complex slicing
        assert slice3.shape == (2, 5, 2, 64, 64)

        slice4 = src[1, 2:7, 1:3]  # Mixed slicing
        assert slice4.shape == (1, 5, 2, 64, 64)


def test_metadata_consistency(temp_dataset, write_params):
    """Test metadata consistency through write-read cycle"""
    with mrio.open(temp_dataset, engine="xarray") as src:
        # Verify metadata matches original parameters
        assert src.crs.to_string() == write_params["crs"]
        assert src.transform == write_params["transform"]

        # Verify coordinates
        for key, values in write_params["md:coordinates"].items():
            assert list(src.coords[key]) == values

        # Verify attributes
        assert src.attrs["satellite"] == "Sentinel-2 L1C"


def test_error_handling_large_slices(temp_dataset):
    """Test handling with invalid slice operations it
    must return the maximum possible data"""
    with mrio.open(temp_dataset, engine="numpy") as src:
        # Test out of bounds slicing
        slice1 = src[:, :200]  # Too many time steps
        slice2 = src[:, :, :100]  # Too many bands
        assert slice1.shape == (2, 10, 4, 64, 64)
        assert slice2.shape == (2, 10, 4, 64, 64)


def test_memory_efficient_reading(temp_dataset):
    """Test memory-efficient reading with chunks"""
    with mrio.open(temp_dataset, engine="numpy") as src:
        # Read data in chunks
        chunks = []
        for i in range(2):
            for j in range(0, 10, 2):
                chunk = src[i, j : j + 2, :]
                chunks.append(chunk)
                assert chunk.shape == (1, 2, 4, 64, 64)  # Note the leading 1 dimension

        # First concatenate along time axis (axis=1) within each product
        product_chunks = [
            np.concatenate(chunks[i : i + 5], axis=1)  # Concatenate time chunks
            for i in (0, 5)
        ]

        # Then concatenate along product axis (axis=0)
        reconstructed = np.concatenate(product_chunks, axis=0)
        assert reconstructed.shape == (2, 10, 4, 64, 64)


def test_xarray_integration(temp_dataset):
    """Test xarray integration features"""
    with mrio.open(temp_dataset, engine="xarray") as src:
        data = src.read()

        # Test xarray-specific features
        assert isinstance(data, xr.DataArray)
        assert list(data.dims) == ["products", "time", "band", "lat", "lon"]

        # Test xarray operations
        mean_by_time = data.mean(dim="time")
        assert mean_by_time.shape == (2, 4, 64, 64)

        # Test coordinate-based selection
        subset = data.sel(time=slice("20210101", "20210115"))
        assert isinstance(subset, xr.DataArray)


def test_str_representation(temp_dataset):
    """Test string representation of DatasetReader"""
    with mrio.open(temp_dataset, engine="xarray") as src:
        str_repr = str(src)

        # Check essential components
        assert "LazyDataArray" in str_repr
        assert "Coordinates:" in str_repr
        assert "products" in str_repr
        assert "time" in str_repr
        assert "band" in str_repr
        assert "Size:" in str_repr
        assert "Attributes:" in str_repr

        # Check coordinate values are present
        assert "toa" in str_repr
        assert "boa" in str_repr
        assert "B01" in str_repr

        # Check size information
        assert "MB" in str_repr or "KB" in str_repr or "GB" in str_repr

        # Check dimensions
        assert "64, 64" in str_repr or "y: 64, x: 64" in str_repr


def test_repr_with_long_coordinates(temp_dataset):
    """Test representation handling of long coordinate lists"""
    with mrio.open(temp_dataset, engine="xarray") as src:
        repr_str = repr(src)

        # Count the number of time coordinates shown
        time_coords_shown = len([line for line in repr_str.split("\n") if "20210" in line])

        # Should not show all 10 time coordinates
        assert time_coords_shown < 10
        assert "..." in repr_str  # Should indicate truncation


def test_repr_empty_attributes(tmp_path, sample_datacube, write_params):
    """Test representation with empty attributes"""
    # Modify params to remove attributes
    params = write_params.copy()
    params["md:attributes"] = {}

    file_path = tmp_path / "test_empty_attrs.tif"

    with mrio.open(file_path, mode="w", **params) as src:
        src.write(sample_datacube.values)

    with mrio.open(file_path, engine="xarray") as src:
        repr_str = repr(src)
        # Should still show Attributes section, but empty
        assert "Attributes:" in repr_str
        assert "Attributes:\n" in repr_str or "Attributes: {}" in repr_str
