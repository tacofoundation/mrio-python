"""MRIO Dataset Protocols.

This module defines the protocols for reader and writer operations in the mrio
package. It specifies the required interfaces for dataset
implementations that handle multi-dimensional COG files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, Union, runtime_checkable

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from rasterio.crs import CRS
    from rasterio.transform import Affine

# Type aliases for better readability
Metadata = dict[str, Any]
Profile = dict[str, Any]
Coords = dict[str, Any]
DataArray = Union[np.ndarray, xr.DataArray]


@runtime_checkable
class DatasetReaderProtocol(Protocol):
    """Protocol defining the interface for MRIO dataset reading operations.

    This protocol specifies the required interface for reading multi-dimensional
    COG files with support for metadata, coordinates, and dimensions.
    """

    file_path: Path
    engine: Literal["numpy", "xarray"]
    profile: Profile
    meta: dict[str, Any]
    md_meta: dict[str, Any] | None  # MRIOFields type
    coords: Coords
    dims: list
    attrs: dict[str, Any]
    shape: tuple[int, ...]
    size: int

    # Geometric properties
    width: int
    height: int
    crs: CRS
    transform: Affine
    count: int
    bounds: tuple[float, float, float, float]

    # Data properties
    dtype: Any
    nodata: float | None

    def read(self, *args: Any, **kwargs: Any) -> NDArray | xr.DataArray:
        """Read data from the dataset.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Array or DataArray containing the dataset's contents

        """
        ...

    def _read(self, *args: Any, **kwargs: Any) -> NDArray:
        """Internal method for reading raw data.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            NumPy array containing the raw data

        """
        ...

    def close(self) -> None:
        """Close the dataset and free resources."""
        ...

    def tags(self) -> dict[str, str]:
        """Get the dataset's tags/metadata.

        Returns:
            Dictionary of tags and their values

        """
        ...

    def __getitem__(self, key: Any) -> NDArray | xr.DataArray:
        """Support array-like indexing.

        Args:
            key: Index or slice object

        Returns:
            Subset of the dataset

        """
        ...


@runtime_checkable
class DatasetWriterProtocol(Protocol):
    """Protocol defining the interface for MRIO dataset writing operations.

    This protocol specifies the required interface for writing multi-dimensional
    COG files with support for metadata, coordinates, and custom band descriptions.
    """

    file_path: Path
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    md_kwargs: Any  # MRIOFields type

    def write(self, data: DataArray) -> None:
        """Write data to the dataset.

        Args:
            data: Array-like data to write

        """
        ...

    def _write_custom_data(self, data: DataArray) -> None:
        """Internal method for writing data with metadata.

        Args:
            data: Array-like data to write

        """
        ...

    def _rearrange_and_write(self, data: DataArray) -> None:
        """Rearrange and write data based on pattern.

        Args:
            data: Array-like data to write

        """
        ...

    def _generate_band_identifiers(self) -> list[str]:
        """Generate unique identifiers for bands.

        Returns:
            List of band identifiers

        """
        ...

    def _generate_metadata(self) -> dict[str, Any]:
        """Generate metadata dictionary.

        Returns:
            Dictionary containing metadata

        """
        ...

    def close(self) -> None:
        """Close the dataset and free resources."""
        ...

    def __setitem__(self, key: Any, value: DataArray) -> None:
        """Support array-like assignment.

        Args:
            key: Index or slice object
            value: Data to write

        """
        ...
