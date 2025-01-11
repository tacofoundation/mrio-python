"""
Custom Dataset Reader for Multi-Dimensional GeoTIFFs
=====================================================

This module provides tools for reading and writing multi-dimensional 
GeoTIFF files, designed for efficient handling of large datasets. It 
supports advanced metadata handling, lazy loading, and customizable 
write parameters.

Features:
---------
- Efficient generation of compliant multi-dimensional GeoTIFF files.
- Metadata parsing and validation.

Usage:
------
    import mrio

    # Reading a dataset
    with mrio.open("path/to/input.tif") as reader:
        data = reader.read()

    # Writing a dataset
    write_params_minimal = {
        "driver": "GTiff",
        "crs": "EPSG:4326",
        "transform": mrio.transform.from_bounds(-76.2, 4.31, -76.1, 4.32, 128, 128),
        "md:pattern": "simulation time band lat lon -> (simulation band time) lat lon",
        "md:coordinates": {
            "time": ["2025-01-01", "2025-01-02"],
            "band": ["B01", "B02", "B03"],
            "simulation": ["A", "B"],
        },
    }
    with mrio.open("path/to/output.tif", mode="w", **write_params) as writer:
        writer.write(data)

Created with <3 at the Image Processing Lab, University of Valencia, Spain.
"""

import json
import math
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Set

import numpy as np
import rasterio as rio
from einops import rearrange

from mrio.datamodel import MRIOFields
from mrio.partial_read import PartialReadImage

# Constants
READ_MODE = "r"
WRITE_MODE = "w"
VALID_MODES = {READ_MODE, WRITE_MODE}
MD_PREFIX = "md:"
MD_METADATA_KEY = "MD_METADATA"
DEFAULT_WRITE_PARAMS = {
    "driver": "GTiff",  # Default to GeoTIFF format
    "dtype": "float32",  # Default data type for raster values
    "compress": "lzw",  # Default compression method
    "interleave": "band",  # Default interleaving (band or pixel)
    "tiled": True,  # Enable tiling for better performance
    "blockxsize": 256,  # Tile width
    "blockysize": 256,  # Tile height
    "nodata": None,  # NoData value
    "count": 1,  # AUTOMATIC: The package automatically sets the number of bands.
    "width": None,  # MANDATORY: Width of the raster in pixels.
    "height": None,  # MANDATORY: Height of the raster in pixels.
    "crs": None,  # MANDATORY: Coordinate Reference System to be set by the user.
    "transform": None,  # MANDATORY: Affine transform to be set by the user.
    "md:pattern": None,  # MANDATORY: Pattern for multi-dimensional data.
    "md:coordinates": None,  # MANDATORY: Coordinates for each dimension.
    "md:attributes": {},  # OPTIONAL: Additional attributes to include in the file.
}


class DatasetReader:
    """
    Custom DatasetReader for reading and writing files with multi-dimensional metadata.

    Attributes:
        file_path (Path): Path to the dataset file.
        mode (str): File mode ('r' for read, 'w' for write).
        args (tuple): Positional arguments for rasterio.
        kwargs (dict): Keyword arguments for rasterio.
        profile (dict): Raster profile (read mode only).
        meta (dict): File metadata (read mode only).
        width (int): Width of the raster in pixels (read mode only).
        height (int): Height of the raster in pixels (read mode only).
        crs (dict): Coordinate reference system (read mode only).
        transform (Affine): Affine transformation matrix (read mode only).
        count (int): Number of raster bands (read mode only).
        md_meta (Optional[dict]): Multi-dimensional metadata (read mode only).

    Methods:
        close(): Close the file if open.
        read(*args, **kwargs): Read data from the file. Only windows and indexes are supported.
        write(data): Write data to the file.
        tags(): Retrieve tags from the file.
        __enter__(): Enter runtime context.
        __exit__(): Exit runtime context and close the file.
    """

    __slots__ = (
        "file_path",
        "mode",
        "args",
        "kwargs",
        "_file",
        "md_kwargs",
        "profile",
        "meta",
        "width",
        "height",
        "crs",
        "transform",
        "count",
        "indexes",
        "window",
        "bounds",
        "res",
        "shape",
        "dtypes",
        "nodata",
        "md_meta",
    )

    def __init__(self, file_path: Path, mode: str, *args, **kwargs):
        """
        Initialize DatasetReader for reading/writing custom data formats.

        Args:
            file_path (Path): Path to the dataset file.
            mode (str): Mode ('r' for read, 'w' for write).
            *args: Additional arguments for rasterio.
            **kwargs: Additional keyword arguments for rasterio.
        """
        self.file_path = file_path
        self.mode = mode.lower()
        self.args = args
        self.kwargs = {**DEFAULT_WRITE_PARAMS, **kwargs}

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode. Use '{READ_MODE}' for read or '{WRITE_MODE}' for write."
            )

        # Initialize mode-specific settings
        if self.mode == WRITE_MODE:
            self._initialize_write_mode()

        # Open the file
        try:
            self._file = rio.open(self.file_path, self.mode, *self.args, **self.kwargs)
        except Exception as e:
            raise IOError(f"Failed to open file {self.file_path}: {e}")

        # Initialize attributes for read mode
        if self.mode == READ_MODE:
            self._initialize_read_mode()

    def _initialize_write_mode(self):
        """Validate and process metadata for write mode."""
        md_kwargs_dict = {
            k[len(MD_PREFIX) :]: v
            for k, v in self.kwargs.items()
            if k.startswith(MD_PREFIX)
        }
        for k in md_kwargs_dict:
            self.kwargs.pop(f"{MD_PREFIX}{k}")

        self.md_kwargs = MRIOFields(**md_kwargs_dict)
        self.kwargs["count"] = math.prod(map(len, self.md_kwargs.coordinates.values()))

    @lru_cache(maxsize=None)
    def _initialize_read_mode(self):
        """Load metadata and attributes for read mode."""
        self.profile = self._file.profile
        profile_get = self.profile.get
        self.meta = self._file.meta
        self.width = profile_get("width")
        self.height = profile_get("height")
        self.crs = profile_get("crs")
        self.transform = profile_get("transform")
        self.count = profile_get("count")

        # Batch attribute assignment
        for attr in ["indexes", "window", "bounds", "shape", "dtypes", "nodata"]:
            setattr(self, attr, getattr(self._file, attr))

        # Load the multi-dimensional metadata
        self.md_meta = self._get_md_metadata()

        # Create real shape
        self.shape = (
            *self.md_meta["md:coordinates_len"].values(),
            self.height,
            self.width,
        )

    def _obtain_dimensions_from_pattern(self, metadata: dict) -> dict:
        # Get dimensions from the pattern
        return metadata["md:pattern"].split("->")[1].split()

    def _obtain_coordinates_len(self, metadata: dict) -> dict:
        # Get the length of each coordinate
        return {
            band: len(metadata["md:coordinates"][band])
            for band in metadata["md:coordinates"]
            if band in metadata["md:dimensions"]
        }

    def _get_md_metadata(self) -> Optional[dict]:
        """Retrieve multi-dimensional metadata."""
        try:
            metadata = self._file.tags().get(MD_METADATA_KEY)
            if not metadata:
                return None

            metadata_dict = json.loads(metadata)

            # Check if "md:dimensions" exists in metadata
            if "md:dimensions" not in metadata_dict:
                metadata_dict["md:dimensions"] = self._obtain_dimensions_from_pattern(
                    metadata_dict
                )

            # Check if "md:coordinates_len" exists in metadata
            if "md:coordinates_len" not in metadata_dict:
                metadata_dict["md:coordinates_len"] = self._obtain_coordinates_len(
                    metadata
                )

            return metadata_dict

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in MD_METADATA: {metadata}") from e
        except Exception as e:
            raise RuntimeError(f"Error retrieving MD_METADATA: {e}") from e

    def _write_custom_data(self, data: Any):
        """Internal method to write custom data to the file."""
        if not self.md_kwargs.pattern or not self.md_kwargs.coordinates:
            self._file.write(data)
            return

        # sort md_kwargs.coordinates by self.md_kwargs._before_arrow
        coordinates_keys: Set[str] = set(self.md_kwargs.coordinates)
        self.md_kwargs.coordinates = {
            k: self.md_kwargs.coordinates[k]
            for k in self.md_kwargs._before_arrow
            if k in coordinates_keys
        }

        # Rearrange data according to the pattern
        image: np.ndarray = rearrange(data, self.md_kwargs.pattern)

        # Generate unique band identifiers: e.g. "dim1__dim2__dim3"
        band_unique_identifiers: List[str] = [
            "__".join(combination)
            for combination in product(
                *[
                    self.md_kwargs.coordinates[band]
                    for band in self.md_kwargs._in_parentheses
                ]
            )
        ]

        # Create GLOBAL METADATA for the file
        md_metadata = {
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

        # Write bands and set descriptions
        for i, (band_data, band_id) in enumerate(
            zip(image, band_unique_identifiers), start=1
        ):
            self._file.write(band_data, i)
            self._file.set_band_description(i, band_id)

        # Update file metadata
        self._file.update_tags(MD_METADATA=json.dumps(md_metadata))

    def _read(self, *args, **kwargs) -> Any:
        """Internal method to read data from the file."""
        return self._file.read(*args, **kwargs)

    def _read_custom_data(self, *args, **kwargs) -> Any:
        """Internal method to read custom data from the file."""

        # If no metadata is present, treat the file as a standard GeoTIFF
        if not self.md_meta:
            return self._read(*args, **kwargs)

        # Check for required metadata keys
        pattern = self.md_meta.get("md:pattern")
        coordinates_len = self.md_meta.get("md:coordinates_len")

        # Read the raw data and rearrange it
        raw_data = self._read(*args, **kwargs)
        return rearrange(raw_data, pattern, **coordinates_len)

    def close(self):
        """Ensure the file is closed."""
        if self._file and not self._file.closed:
            self._file.close()

    def read(self, *args, **kwargs) -> Any:
        """Read data from the custom file format."""
        if self.mode != "r":
            raise ValueError("File must be opened in read mode.")
        return self._read_custom_data(*args, **kwargs)

    def write(self, data: Any):
        """Write data to the custom file format."""
        if self.mode != "w":
            raise ValueError("File must be opened in write mode.")
        self._write_custom_data(data)

    def tags(self) -> dict:
        """Retrieve all tags from the file."""
        return self._file.tags()

    def get_slice(self) -> Any:
        """Retrieve a slice from the file."""
        return PartialReadImage(self)

    def __enter__(self):
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and close the file."""
        self.close()

    def __del__(self):
        """Ensure the file is closed when the object is deleted."""
        self.close()

    def __repr__(self):
        if self._file.closed:
            return f"<closed DatasetReader name='{self.file_path}' mode='{self.mode}'>"
        return f"<open DatasetReader name='{self.file_path}' mode='{self.mode}'>"

    def __str__(self):
        return repr(self)


def open(file_path: Path, mode: str = "r", *args, **kwargs) -> DatasetReader:
    """
    Open a DatasetReader object.

    Args:
        file_path (Path): Path to the dataset file.
        mode (str): Mode ('r' for read, 'w' for write).
        *args: Additional arguments for rasterio.
        **kwargs: Additional keyword arguments for rasterio.

    Returns:
        DatasetReader: An instance of DatasetReader.
    """

    return DatasetReader(file_path, mode, *args, **kwargs)
