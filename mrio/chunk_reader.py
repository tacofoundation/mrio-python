"""ChunkedReader module for MRIO (Multi-Resolution I/O) package.

This module provides optimized chunked reading capabilities for multi-dimensional COG files,
with support for partial reading, dimension filtering, and metadata handling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from einops import rearrange
from rasterio.windows import Window

from mrio.errors import MRIOError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from affine import Affine
    from numpy.typing import NDArray

    from mrio.protocol import DatasetReaderProtocol
    from mrio.type_definitions import (
        Coordinates,
        CoordinatesLen,
        DimensionFilter,
        FilterCondition,
        MetadataDict,
    )


class ChunkedReader:
    """Optimized implementation of multi-dimensional COG reading with metadata handling.

    This class provides efficient partial reading capabilities for multi-dimensional
    COG files, handling both spatial and non-spatial dimensions. It supports
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
        """Initialize the ChunkedReader.

        Args:
            dataset: Source dataset implementing the DatasetProtocol interface

        """
        self.dataset = dataset
        self.last_query: tuple[DimensionFilter, tuple[slice, slice]] | None = None
        self.new_height: int | None = None
        self.new_width: int | None = None
        self.new_count: int | None = None

    @staticmethod
    def _filter_dimensions(
        dims: Sequence[int], filter_criteria: Sequence[FilterCondition]
    ) -> NDArray[np.uint32]:
        """Filter dimensions using vectorized operations.

        Args:
            dims: Sequence of dimension sizes to filter
            filter_criteria: Sequence of filtering conditions for each dimension

        Returns:
            NDArray of filtered dimension indices (1-based) maintaining specified band order

        Example:
            >>> dims = [3, 4, 5]
            >>> criteria = [slice(1, 3), 2, [2, 0]]  # Different from [0, 2]
            >>> criteria = [slice(1, 3), 2, [0, 2]]  # Different from [0, 2]
            >>> ChunkedReader._filter_dimensions(dims, criteria)
            array([15, 7])  # Order matters!
        """
        data = np.indices(dims, dtype=np.uint32).reshape(len(dims), -1).T
        mask = np.ones(data.shape[0], dtype=bool)

        final_order = None

        for dim, condition in enumerate(filter_criteria):
            if isinstance(condition, slice):
                if condition == slice(None):
                    continue
                start = condition.start or 0
                stop = condition.stop or dims[dim]
                mask &= (data[:, dim] >= start) & (data[:, dim] < stop)
            elif isinstance(condition, (int, list, tuple)):
                if isinstance(condition, (list, tuple)):
                    dim_mask = np.isin(data[:, dim], condition)
                    mask &= dim_mask
                    # Store the values for final ordering
                    masked_data = data[mask]
                    if len(masked_data) > 0:
                        final_order = np.array(
                            [list(condition).index(v) for v in masked_data[:, dim]]
                        )
                else:
                    mask &= data[:, dim] == condition

        result = np.nonzero(mask)[0]

        if final_order is not None and len(result) > 0:
            result = result[np.argsort(final_order)]

        return result + 1

    def _get_new_geotransform(self) -> Affine:
        """Calculate updated geotransform for the current window.

        Returns:
            Updated Affine transform object

        Raises:
            MRIOError: If no query has been executed yet

        """
        if self.last_query is None:
            msg = "No query has been executed yet"
            raise MRIOError(msg)

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
        """Update coordinates based on the current query conditions.

        Returns:
            Dictionary of updated coordinates for each dimension

        Raises:
            MRIOError: If filter criteria are invalid or no query has been executed

        """
        if self.last_query is None:
            msg = "No query has been executed yet"
            raise MRIOError(msg)

        query_conditions = self.last_query[0]
        if not all(
            isinstance(cond, (slice, int, list, tuple)) for cond in query_conditions
        ):
            msg = "Filter criteria must be slice, int, list, or tuple"
            raise MRIOError(msg)

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
                msg = f"Unsupported condition type: {type(cond)}"
                raise MRIOError(msg)

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
        """Generate updated metadata based on current query.

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
        """Generate complete metadata dictionary with validation.

        Returns:
            Updated metadata dictionary including profile and coordinates

        Raises:
            MRIOError: If no query has been executed yet

        """
        if self.last_query is None:
            msg = "No query has been executed yet"
            raise MRIOError(msg)

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
    ) -> tuple[NDArray[Any], tuple[Coordinates, CoordinatesLen]]:
        """Execute a partial read operation with dimension filtering and spatial slicing.

        This method performs several key operations:
        1. Filters the data along specified dimensions (e.g., time, bands)
        2. Applies spatial windowing using row/column slices
        3. Rearranges the data according to the metadata pattern

        The operation sequence is:
        1. Split the key into dimension filters and spatial slices
        2. Reorder dimensions based on the pattern strategy
        3. Apply dimension filtering using vectorized operations
        4. Create a spatial window from row/column slices
        5. Read the data chunk and rearrange according to pattern

        Args:
            key (DimensionFilter): A tuple containing:
                - Dimension filters (e.g., for time, bands)
                - Spatial slices (row_slice, col_slice)
                Example: ((time_slice, band_slice), (row_slice, col_slice))

        Returns:
            tuple: Contains:
                - NDArray: The read and rearranged data
                - tuple: Updated coordinates metadata:
                    - Coordinates: New coordinate values
                    - CoordinatesLen: New coordinate lengths

        Raises:
            MRIOError: If dimension filters or spatial slices are invalid
            WindowError: If spatial window is invalid
        """

        # Split input key into dimension filters and spatial slices
        self.last_query = (key[:-2], key[-2:])
        filter_criteria, (row_slice, col_slice) = self.last_query

        # Get current dimension lengths from metadata
        dims_len = tuple(self.dataset.md_meta["md:coordinates_len"].values())

        # Reorder dimensions according to pattern strategy
        band_order = self.get_axis_order(self.dataset.md_meta["md:pattern"])
        filter_criteria = tuple(filter_criteria[i] for i in band_order)
        dims_len = tuple(dims_len[i] for i in band_order)

        # Apply dimension filtering
        result = self._filter_dimensions(dims_len, filter_criteria)

        # Create spatial window, defaulting to full extent if not specified
        row_slice = (
            row_slice if row_slice != slice(None) else slice(0, self.dataset.height)
        )
        col_slice = (
            col_slice if col_slice != slice(None) else slice(0, self.dataset.width)
        )
        window = Window.from_slices(row_slice, col_slice)

        # Read data chunk and update metadata
        data_chunk = self.dataset._read(result.tolist(), window=window)
        new_coords = self._get_new_md_meta_coordinates()
        new_coords_len = self._get_new_md_meta_coordinates_len(new_coords)

        # Rearrange data according to pattern
        rearranged_data = rearrange(
            data_chunk, self.dataset.md_meta["md:pattern"], **new_coords_len
        )

        return rearranged_data, (new_coords, new_coords_len)

    @staticmethod
    def get_axis_order(pattern):
        """
        Get axis order from dimension pattern string, mapping from parentheses to target

        Args:
            pattern (str): Pattern like "(band time) y x -> time band y x"

        Returns:
            list[int]: Indices showing new order of axes
        """
        # Get the order in parentheses (source)
        source = pattern.split("(")[1].split(")")[0].strip().split()

        # Get the target dimensions (after ->)
        target = pattern.split("->")[1].strip().split()

        # Get only the dimensions that were in parentheses
        target = target[: len(source)]

        # Create mapping from source to target order
        return [source.index(dim) for dim in target]

    def close(self) -> None:
        """Close the dataset and clean up resources."""
        self.dataset.close()
        self.last_query = None
        self.new_height = None
        self.new_width = None
        self.new_count = None
