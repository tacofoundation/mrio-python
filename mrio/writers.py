"""MRIO Dataset Writer Module with automatic parameter detection."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import rasterio as rio
from einops import rearrange
from typing_extensions import Self

from mrio import errors as mrio_errors
from mrio.fields import MRIOFields, WriteParams

if TYPE_CHECKING:
    from mrio.type_definitions import DataArray, MetadataDict, PathLike

# Constants
MD_PREFIX: str = "md:"
MD_METADATA_KEY: str = "MD_METADATA"


class DatasetWriter:
    """Writer for multi-dimensional COG files with metadata handling.

    Supports automatic inference of width, height, and dtype from input data.

    Example:
        >>> # Parameters will be inferred from data
        >>> params = {
        ...     'md:pattern': 'c h w -> (c) h w',
        ...     'md:coordinates': {'c': ['red', 'green', 'blue']}
        ... }
        >>> with DatasetWriter('output.tif', **params) as writer:
        ...     writer.write(data)

    """

    __slots__ = (
        "_file",
        "_initialized",
        "args",
        "file_path",
        "kwargs",
        "md_kwargs",
    )

    # Class variables
    DEFAULT_ENGINE: ClassVar[str] = "numpy"

    def __init__(self, file_path: PathLike, *args: Any, **kwargs: Any) -> None:
        """Initialize DatasetWriter with parameter auto-detection support.

        Args:
            file_path: Path to the output file
            *args: Additional positional arguments for rasterio
            **kwargs: Additional keyword arguments including metadata.
                     width, height, and dtype can be None for auto-detection

        Note:
            File is not opened until the first write operation to allow
            for parameter inference from data.

        """
        self.file_path = Path(file_path)
        self.args = args
        self.kwargs = kwargs
        self._file = None
        self.md_kwargs = None
        self._initialized = False

    def _infer_parameters(self, data: DataArray) -> dict[str, Any]:
        """Infer missing parameters from input data.

        Args:
            data: Input data array

        Returns:
            Dictionary of inferred parameters

        Note:
            Only infers parameters that are None in self.kwargs

        """
        data_type = type(data)
        if (
            data_type.__name__ == "DataArray"
            and data_type.__module__ == "xarray.core.dataarray"
        ):
            data = data.values

        if data.ndim < 2:
            msg = "Data must have at least 2 dimensions"
            raise ValueError(msg)

        inferred = {}
        if self.kwargs.get("height") is None:
            inferred["height"] = data.shape[-2]
        if self.kwargs.get("width") is None:
            inferred["width"] = data.shape[-1]
        if self.kwargs.get("dtype") is None:
            inferred["dtype"] = data.dtype

        return inferred

    def _initialize_write_mode(self, data: DataArray | None = None) -> None:
        """Initialize write mode and process metadata parameters.

        Args:
            data: Optional data array for parameter inference

        """
        # Initialize kwargs with WriteParams defaults
        kwargs = self.kwargs.copy()

        # Infer data parameters
        inferred = self._infer_parameters(data)
        kwargs.update(inferred)

        # Process write parameters
        kwargs = WriteParams(params=kwargs).to_dict()

        # Extract metadata parameters
        md_kwargs_dict = {
            k[len(MD_PREFIX) :]: v for k, v in kwargs.items() if k.startswith(MD_PREFIX)
        }

        # Remove processed metadata from kwargs
        for k in md_kwargs_dict:
            kwargs.pop(f"{MD_PREFIX}{k}")

        # Initialize metadata fields
        self.md_kwargs = MRIOFields(**md_kwargs_dict)

        # Calculate total number of bands
        kwargs["count"] = math.prod(
            len(coords) for coords in self.md_kwargs.coordinates.values()
        )

        self.kwargs = kwargs

        # Open the file if not already opened
        if self._file is None:
            try:
                self._file = rio.open(self.file_path, "w", *self.args, **self.kwargs)
            except Exception as e:
                raise mrio_errors.WritersFailedToOpenError(self.file_path, e) from e

        self._initialized = True

    def write(self, data: DataArray) -> None:
        """Write data to file with metadata handling and parameter inference.

        Args:
            data: Array data to write (numpy array or xarray.DataArray)

        Raises:
            ValueError: If data shape doesn't match metadata

        """
        if not self._initialized:
            self._initialize_write_mode(data)

        if isinstance(data, np.ndarray):
            self._write_custom_data(data)
        else:
            data_type = type(data)
            if (
                data_type.__name__ == "DataArray"
                and data_type.__module__ == "xarray.core.dataarray"
            ):
                self._write_custom_data(data.values)
            else:
                raise mrio_errors.WritersUnsupportedDataTypeError(data_type)

    def _write_custom_data(self, data: DataArray) -> None:
        """Write data with metadata handling.

        Args:
            data: Array data to write

        Note:
            Falls back to simple write if no pattern/coordinates specified

        """
        if not self.md_kwargs.pattern or not self.md_kwargs.coordinates:
            self._file.write(data)
            return

        self._rearrange_and_write(data)

    def _rearrange_and_write(self, data: DataArray) -> None:
        """Rearrange and write data according to pattern.

        Args:
            data: Array data to rearrange and write

        """
        # Filter coordinates to match pattern
        coordinates_keys = set(self.md_kwargs.coordinates)
        self.md_kwargs.coordinates = {
            k: self.md_kwargs.coordinates[k]
            for k in self.md_kwargs._before_arrow
            if k in coordinates_keys
        }

        # Rearrange data according to pattern
        image = rearrange(data, self.md_kwargs.pattern)

        # Generate band identifiers and metadata
        band_identifiers = self._generate_band_identifiers()
        md_metadata = self._generate_metadata()

        # Write data and metadata
        for i, (band_data, band_id) in enumerate(zip(image, band_identifiers), start=1):
            self._file.write(band_data, i)
            self._file.set_band_description(i, band_id)

        self._file.update_tags(MD_METADATA=json.dumps(md_metadata))

    def _generate_band_identifiers(self) -> list[str]:
        """Generate unique identifiers for each band.

        Returns:
            List of band identifiers based on coordinate combinations

        Example:
            >>> writer._generate_band_identifiers()
            ['red__2020', 'red__2021', 'green__2020', 'green__2021']

        """
        return [
            "__".join(map(str, combination))
            for combination in product(
                *[
                    self.md_kwargs.coordinates[band]
                    for band in self.md_kwargs._in_parentheses
                ]
            )
        ]

    def _generate_metadata(self) -> MetadataDict:
        """Generate complete metadata dictionary.

        Returns:
            Dictionary containing all metadata fields

        """
        return {
            "md:dimensions": self.md_kwargs._before_arrow,
            "md:coordinates": self.md_kwargs.coordinates,
            "md:coordinates_len": {
                band: len(coords)
                for band, coords in self.md_kwargs.coordinates.items()
                if band in self.md_kwargs._in_parentheses
            },
            "md:attributes": self.md_kwargs.attributes,
            "md:pattern": self.md_kwargs._inv_pattern,
        }

    def close(self) -> None:
        """Close the file and free resources."""
        if self._file is None:
            return  # Already closed

        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()
        self._file = None

    def __setitem__(self, key: Any, value: DataArray) -> None:
        """Support array-like assignment syntax.

        Args:
            key: Index or slice object
            value: Data to write

        """
        self._write_custom_data(value)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.close()

    def __repr__(self) -> str:
        """Detailed string representation."""
        if self._file is None:
            return f"<DatasetWriter name='{self.file_path}' mode='w'>"
        status = "closed" if self._file.closed else "open"
        return f"<{status} DatasetWriter name='{self.file_path}' mode='w'>"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return repr(self)
