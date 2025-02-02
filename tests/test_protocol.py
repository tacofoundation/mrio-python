"""Test suite for protocol.py module."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine

from mrio.protocol import DatasetReaderProtocol, DatasetWriterProtocol


class MockDatasetReader:
    """Mock implementation of DatasetReaderProtocol."""

    def __init__(self):
        self.file_path: Path = Path("test.tif")
        self.engine: str = "numpy"
        self.profile: dict[str, Any] = {"driver": "GTiff", "count": 1}
        self.meta: dict[str, Any] = {"description": "Test data"}
        self.md_meta: dict[str, Any] | None = {"field1": "value1"}
        self.coords: dict[str, Any] = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])}
        self.dims: list = ["band", "y", "x"]
        self.attrs: dict[str, Any] = {"attr1": "value1"}
        self.shape: tuple[int, ...] = (1, 100, 100)
        self.size: int = 10000
        self.width: int = 100
        self.height: int = 100
        self.crs: CRS = CRS.from_epsg(4326)
        self.transform: Affine = Affine.identity()
        self.count: int = 1
        self.bounds: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
        self.dtype: Any = np.float32
        self.nodata: float | None = -9999.0

    def read(self, *args: Any, **kwargs: Any) -> np.ndarray | xr.DataArray:
        """Read data from the dataset."""
        return np.zeros(self.shape, dtype=self.dtype)

    def _read(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Internal method for reading raw data."""
        return np.zeros(self.shape, dtype=self.dtype)

    def close(self) -> None:
        """Close the dataset and free resources."""
        pass

    def tags(self) -> dict[str, str]:
        """Get the dataset's tags/metadata."""
        return {"tag1": "value1"}

    def __getitem__(self, key: Any) -> np.ndarray | xr.DataArray:
        """Support array-like indexing."""
        if isinstance(key, int) and (key >= self.count):
            raise IndexError(f"Index {key} is out of bounds for axis 0 with size {self.count}")
        return np.zeros(self.shape, dtype=self.dtype)


class MockDatasetWriter:
    """Mock implementation of DatasetWriterProtocol."""

    def __init__(self):
        self.file_path: Path = Path("test_output.tif")
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}
        self.md_kwargs: Any = {"field1": "value1"}
        self.expected_dimensions: int = 3
        self.expected_shape: tuple[int, ...] = (1, 100, 100)

    def write(self, data: np.ndarray | xr.DataArray) -> None:
        """Write data to the dataset."""
        if not isinstance(data, (np.ndarray, xr.DataArray)):
            raise TypeError("Data must be a numpy array or xarray DataArray")

        # Convert xarray DataArray to numpy array if necessary
        if isinstance(data, xr.DataArray):
            data = data.values

        # Check dimensions
        if len(data.shape) != self.expected_dimensions:
            raise ValueError(f"Expected {self.expected_dimensions}D array, got {len(data.shape)}D array")

        # Check shape
        if data.shape != self.expected_shape:
            raise ValueError(f"Expected shape {self.expected_shape}, got {data.shape}")

    def _write_custom_data(self, data: np.ndarray | xr.DataArray) -> None:
        """Internal method for writing data with metadata."""
        self.write(data)

    def _rearrange_and_write(self, data: np.ndarray | xr.DataArray) -> None:
        """Rearrange and write data based on pattern."""
        self.write(data)

    def _generate_band_identifiers(self) -> list[str]:
        """Generate unique identifiers for bands."""
        return ["band1", "band2"]

    def _generate_metadata(self) -> dict[str, Any]:
        """Generate metadata dictionary."""
        return {"meta1": "value1"}

    def close(self) -> None:
        """Close the dataset and free resources."""
        pass

    def __setitem__(self, key: Any, value: np.ndarray | xr.DataArray) -> None:
        """Support array-like assignment."""
        self.write(value)


def test_protocol_compliance():
    """Test protocol compliance for both reader and writer."""
    reader = MockDatasetReader()
    writer = MockDatasetWriter()

    # Verify both implementations satisfy their respective protocols
    assert isinstance(reader, DatasetReaderProtocol)
    assert isinstance(writer, DatasetWriterProtocol)


def test_reader_attributes():
    """Test reader attributes match protocol requirements."""
    reader = MockDatasetReader()

    # Test required attributes
    assert isinstance(reader.file_path, Path)
    assert reader.engine in ["numpy", "xarray"]
    assert isinstance(reader.profile, dict)
    assert isinstance(reader.meta, dict)
    assert isinstance(reader.md_meta, (dict, type(None)))
    assert isinstance(reader.coords, dict)
    assert isinstance(reader.dims, list)
    assert isinstance(reader.attrs, dict)
    assert isinstance(reader.shape, tuple)
    assert isinstance(reader.size, int)
    assert isinstance(reader.width, int)
    assert isinstance(reader.height, int)
    assert isinstance(reader.crs, CRS)
    assert isinstance(reader.transform, Affine)
    assert isinstance(reader.count, int)
    assert isinstance(reader.bounds, tuple)
    assert len(reader.bounds) == 4
    assert all(isinstance(x, float) for x in reader.bounds)
    assert isinstance(reader.nodata, (float, type(None)))


def test_writer_attributes():
    """Test writer attributes match protocol requirements."""
    writer = MockDatasetWriter()

    assert isinstance(writer.file_path, Path)
    assert isinstance(writer.args, tuple)
    assert isinstance(writer.kwargs, dict)
    assert hasattr(writer, "md_kwargs")


def test_error_handling():
    """Test error handling scenarios."""
    reader = MockDatasetReader()
    writer = MockDatasetWriter()

    # Test invalid index
    with pytest.raises(IndexError):
        _ = reader[100]

    # Test wrong number of dimensions
    with pytest.raises(ValueError, match=r"Expected 3D array, got 2D array"):
        writer.write(np.zeros((10, 10)))

    # Test wrong shape
    with pytest.raises(ValueError, match=r"Expected shape \(1, 100, 100\), got \(1, 50, 50\)"):
        writer.write(np.zeros((1, 50, 50)))

    # Test invalid data type
    with pytest.raises(TypeError, match="Data must be a numpy array or xarray DataArray"):
        writer.write("invalid data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
