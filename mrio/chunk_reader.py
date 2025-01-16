"""
ChunkedReader module for MRIO (Multi-Resolution I/O) package.

This module provides optimized chunked reading capabilities for multi-dimensional GeoTIFF files,
with support for partial reading, dimension filtering, and metadata handling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from einops import rearrange
from numpy.typing import NDArray
from rasterio.windows import Window

from mrio.errors import MRIOError
from mrio.protocol import DatasetReaderProtocol
from mrio.types import Coordinates, CoordinatesLen, DimensionFilter, FilterCondition, MetadataDict


class ChunkedReader:
    """
    Optimized implementation of multi-dimensional GeoTIFF reading with metadata handling.

    This class provides efficient partial reading capabilities for multi-dimensional
    GeoTIFF files, handling both spatial and non-spatial dimensions. It supports
    dimension filtering, window-based reading, and maintains correct geospatial metadata.

    Attributes:
        dataset: Source dataset implementing the DatasetProtocol interface
        last_query: Last executed query parameters for metadata updates
        new_height: Updated height after windowed reading
        new_width: Updated width after windowed reading
        new_count: Updated band count after dimension filtering

    Example:
        >>> with mrio.open("multidim.tif") as src:
        ...     data= reader[1:3, :, 0:100, 0:100]
    """

    def __init__(self, dataset: DatasetReaderProtocol) -> None:
        """
        Initialize the ChunkedReader.

        Args:
            dataset: Source dataset implementing the DatasetProtocol interface
        """
        self.dataset = dataset
        self.last_query: Optional[Tuple[DimensionFilter, Tuple[slice, slice]]] = None
        self.new_height: Optional[int] = None
        self.new_width: Optional[int] = None
        self.new_count: Optional[int] = None

    @staticmethod
    def _filter_dimensions(
        dims: Sequence[int], filter_criteria: Sequence[FilterCondition]
    ) -> NDArray[np.uint32]:
        """
        Filter dimensions using vectorized operations for optimal performance.

        Args:
            dims: Sequence of dimension sizes to filter
            filter_criteria: Sequence of filtering conditions for each dimension

        Returns:
            NDArray of filtered dimension indices (1-based)

        Example:
            >>> dims = [3, 4, 5]
            >>> criteria = [slice(1, 3), 2, [0, 2]]
            >>> ChunkedReader._filter_dimensions(dims, criteria)
            array([7, 15])  # Example output
        """
        data = np.indices(dims, dtype=np.uint32).reshape(len(dims), -1).T
        mask = np.ones(data.shape[0], dtype=bool)

        for dim, condition in enumerate(filter_criteria):
            if isinstance(condition, slice):
                if condition == slice(None):
                    continue
                start = condition.start or 0
                stop = condition.stop or dims[dim]
                mask &= (data[:, dim] >= start) & (data[:, dim] < stop)
            elif isinstance(condition, (int, list, tuple)):
                mask &= np.isin(data[:, dim], condition)

        return np.nonzero(mask)[0] + 1

    def _get_new_geotransform(self) -> Affine:
        """
        Calculate updated geotransform for the current window.

        Returns:
            Updated Affine transform object

        Raises:
            MRIOError: If no query has been executed yet
        """
        if self.last_query is None:
            raise MRIOError("No query has been executed yet")

        row_slice, col_slice = self.last_query[1]

        # Handle full slices
        row_slice = (
            row_slice if row_slice != slice(None) else slice(0, self.dataset.height)
        )
        col_slice = (
            col_slice if col_slice != slice(None) else slice(0, self.dataset.width)
        )

        window = Window.from_slices(row_slice, col_slice)

        # Update dimensions
        self.new_height = window.height
        self.new_width = window.width

        return rasterio.windows.transform(window, self.dataset.transform)

    def _get_new_md_meta_coordinates(self) -> Coordinates:
        """
        Update coordinates based on the current query conditions.

        Returns:
            Dictionary of updated coordinates for each dimension

        Raises:
            MRIOError: If filter criteria are invalid or no query has been executed
        """
        if self.last_query is None:
            raise MRIOError("No query has been executed yet")

        query_conditions = self.last_query[0]
        if not all(
            isinstance(cond, (slice, int, list, tuple)) for cond in query_conditions
        ):
            raise MRIOError("Filter criteria must be slice, int, list, or tuple")

        coords = self.dataset.md_meta["md:coordinates"]
        dims = self.dataset.md_meta["md:dimensions"]
        new_coords: Coordinates = {}

        for dim_index, cond in enumerate(query_conditions):
            dim_name = dims[dim_index]
            dim_coords = coords[dim_name]

            if isinstance(cond, (int, slice)):
                updated_coord = dim_coords[cond]
            elif isinstance(cond, (list, tuple)):
                updated_coord = [dim_coords[i] for i in cond]
            else:
                raise MRIOError(f"Unsupported condition type: {type(cond)}")

            new_coords[dim_name] = (
                [updated_coord]
                if isinstance(updated_coord, (str, int, bool))
                else updated_coord
            )

        return new_coords

    def _get_new_md_meta_coordinates_len(
        self, new_coords: Coordinates
    ) -> CoordinatesLen:
        """Calculate lengths of updated coordinates."""
        return {key: len(values) for key, values in new_coords.items()}

    def _get_new_md_meta(self) -> MetadataDict:
        """
        Generate updated metadata based on current query.

        Returns:
            Dictionary containing updated metadata
        """
        new_coords = self._get_new_md_meta_coordinates()
        dims = self.dataset.md_meta["md:dimensions"]
        new_coords_len = self._get_new_md_meta_coordinates_len(new_coords)

        self.new_count = math.prod(new_coords_len.values())

        return {
            "md:coordinates": new_coords,
            "md:attributes": self.dataset.md_meta["md:attributes"],
            "md:pattern": self.dataset.md_meta["md:pattern"],
            "md:coordinates_len": new_coords_len,
            "md:dimensions": dims,
        }

    def update_metadata(self) -> MetadataDict:
        """
        Generate complete metadata dictionary with validation.

        Returns:
            Updated metadata dictionary including profile and coordinates

        Raises:
            MRIOError: If no query has been executed yet
        """
        if self.last_query is None:
            raise MRIOError("No query has been executed yet")

        md_meta = self._get_new_md_meta()
        new_transform = self._get_new_geotransform()

        return {
            **self.dataset.profile,
            "md:coordinates": md_meta["md:coordinates"],
            "md:attributes": md_meta["md:attributes"],
            "md:pattern": md_meta["md:pattern"],
            "transform": new_transform,
            "height": self.new_height,
            "width": self.new_width,
            "count": self.new_count,
        }

    def __getitem__(
        self, key: DimensionFilter
    ) -> Tuple[NDArray[Any], Tuple[Coordinates, CoordinatesLen]]:
        """
        Perform optimized partial read operation.

        Args:
            key: Tuple of slice/index/list objects defining the selection

        Returns:
            Tuple containing:
                - NDArray of read data
                - Tuple of (coordinates, coordinate lengths) metadata

        Raises:
            MRIOError: If dimensions or filter criteria are invalid
        """
        # Store query for metadata updates
        self.last_query = (key[:-2], key[-2:])
        filter_criteria, (row_slice, col_slice) = self.last_query

        # Filter dimensions
        dims_len = tuple(self.dataset.md_meta["md:coordinates_len"].values())
        result = self._filter_dimensions(dims_len, filter_criteria)

        # Handle spatial slices
        row_slice = (
            row_slice if row_slice != slice(None) else slice(0, self.dataset.height)
        )
        col_slice = (
            col_slice if col_slice != slice(None) else slice(0, self.dataset.width)
        )
        window = Window.from_slices(row_slice, col_slice)

        # Read and rearrange data
        data_chunk = self.dataset._read(result.tolist(), window=window)
        new_coords = self._get_new_md_meta_coordinates()
        new_coords_len = self._get_new_md_meta_coordinates_len(new_coords)

        data = rearrange(
            data_chunk, self.dataset.md_meta["md:pattern"], **new_coords_len
        )

        return data, (new_coords, new_coords_len)

    def close(self) -> None:
        """Close the dataset and clean up resources."""
        self.dataset.close()
        self.last_query = None
        self.new_height = None
        self.new_width = None
        self.new_count = None
