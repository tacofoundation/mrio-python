"""ChunkedReader module for MRIO (Multi-dimensional Raster I/O).

This module provides chunked reading capabilities for mCOG files.
It supports partial reading, dimensions filtering, and metadata handling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from einops import rearrange
from rasterio.windows import Window

from mrio import errors

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
    """Optimized implementation of mCOG reading with metadata handling.

    This class provides partial reading capabilities for mCOG files, handling
    both spatial and non-spatial dimensions. It supports dimension
    filtering, window-based reading, and updated spatial metadata after
    each user query.

    Attributes:
        dataset: Source dataset implementing the DatasetProtocol interface
        last_query: Last executed user query (non-spatial filters, spatial slices)
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
    def _apply_dimension_filters(
        dims: Sequence[int], filter_criteria: Sequence[FilterCondition]
    ) -> NDArray[np.uint32]:
        """Filter dimensions using vectorized operations. MRIO support two
        types of filters: slices and lists/tuples. Integer and Ellipsis are
        converted to slices. Lists/tuples are tricky because they require
        stable sorting to maintain the original order. It is not the same to
        filter by [0, 2] than by [2, 0]. Therefore, to maintain the original
        order, we need to sort the result of the filtering when the user
        provides a list/tuple filter. It is only applied if at least one
        dimension has a list or tuple. If all filters are slices, the code
        skips the ordering step for efficiency.

        Args:
            dims: Sequence of dimension sizes to filter
            filter_criteria: Sequence of filtering conditions for each dimension

        Returns:
            NDArray of filtered dimension indices (1-based) maintaining specified band order

        Example:
            >>> dims = [3, 4, 5]
            >>> filter_criteria = [slice(1, 3), [0, 3], [2, 0]]  # Different from [0, 2]
            >>> filter_criteria = [slice(1, 3), [3, 0], [0, 2]]  # Different from [0, 2]
            >>> ChunkedReader._filter_dimensions(dims, filter_criteria)
            array([23, 43, 38, 58, 21, 41, 36, 56]) // array([36, 56, 21, 41, 38, 58, 23, 43])
        """

        # Build all grid coordinates for the given dims.
        data = np.indices(dims, dtype=np.uint32).reshape(len(dims), -1).T

        # Boolean mask to keep track of which entries survive.
        mask = np.ones(data.shape[0], dtype=bool)

        # Track each dimension with a list/tuple condition (for final ordering).
        # We store tuples of (dim_index, [conditionValues]).
        list_conditions = []

        for dim, condition in enumerate(filter_criteria):
            # If the condition is a slice
            if isinstance(condition, slice):
                if condition == slice(None):
                    continue
                start = condition.start or 0
                stop = condition.stop or dims[dim]
                mask &= (data[:, dim] >= start) & (data[:, dim] < stop)

            # If the condition is a integer or list/tuple
            elif isinstance(condition, (int, list, tuple)):

                # If the condition is a list/tuple
                if isinstance(condition, (list, tuple)):
                    # Keep only the entries whose coordinate is in 'condition':
                    mask &= np.isin(data[:, dim], condition)

                    # Record the list/tuple for ordering later:
                    list_conditions.append((dim, list(condition)))

                # If the condition is an integer
                else:
                    mask &= data[:, dim] == condition

        # Collect indices that survived the filter.
        result = np.nonzero(mask)[0]

        # If no dimension has a list/tuple filter, return immediately (no ordering needed).
        if not list_conditions:
            return result

        # Otherwise, apply stable sorts for each dimension with a list/tuple condition.
        masked_data = data[result]
        for dim_index, cond_list in list_conditions:
            # Create a map from value -> position in cond_list
            position_map = {val: i for i, val in enumerate(cond_list)}
            # Build array of sort-keys for stable sorting
            sort_keys = np.array([position_map[val] for val in masked_data[:, dim_index]])
            stable_order = np.argsort(sort_keys, kind="stable")

            # Reorder result and masked_data in sync
            result = result[stable_order]
            masked_data = masked_data[stable_order]

        # Convert to 1-based index before returning
        return result

    def _get_new_geotransform(self) -> Affine:
        """Calculate new geotransform based on the last query.

        Returns:
            Updated Affine transform object

        Raises:
            ChunkReaderNoQueryError: If no query has been executed yet

        """
        if self.last_query is None:
            raise errors.ChunkReaderNoQueryError()

        # Obtain spatial slices from the last query
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
        """Update md:coordinates based on the last query.

        Returns:
            Dictionary of updated coordinates for each dimension

        Raises:
            MRIOError: If filter criteria are invalid or no query has been executed

        """
        if self.last_query is None:
            raise errors.ChunkReaderNoQueryError()

        query_conditions = self.last_query[0]
        if not all(
            isinstance(cond, (slice, int, list, tuple)) for cond in query_conditions
        ):
            raise errors.ChunkReaderInvalidFilterError()

        # Get image coordinates and dimensions
        coords = self.dataset.md_meta["md:coordinates"]
        dims = self.dataset.md_meta["md:dimensions"]

        # Update coordinates based on query conditions
        new_coords: Coordinates = {}
        for dim_index, cond in enumerate(query_conditions):
            dim_name = dims[dim_index]
            dim_coords = coords[dim_name]

            if isinstance(cond, (int, slice)):
                updated_coord = dim_coords[cond]
            elif isinstance(cond, (list, tuple)):
                updated_coord = [dim_coords[i] for i in cond]
            else:
                raise errors.ChunkReaderInvalidConditionError(cond)

            new_coords[dim_name] = (
                [updated_coord]
                if isinstance(updated_coord, (str, int, bool))
                else updated_coord
            )

        return new_coords

    def _get_new_md_meta_coordinates_len(
        self, new_coords: Coordinates
    ) -> CoordinatesLen:
        """Calculate lengths of updated md:coordinates."""
        return {key: len(values) for key, values in new_coords.items()}

    def _get_new_md_meta(self) -> MetadataDict:
        """Generate updated md:coordinates and md:coordinates_len based on
        the last query.

        Returns:
            Dictionary containing updated metadata

        """
        # Update md:cordinates and md:coordinates_len
        new_coords = self._get_new_md_meta_coordinates()
        new_coords_len = self._get_new_md_meta_coordinates_len(new_coords)

        # Update band count based on the last query
        self.new_count = math.prod(new_coords_len.values())

        return {
            "md:coordinates": new_coords,
            "md:attributes": self.dataset.md_meta["md:attributes"],
            "md:pattern": self.dataset.md_meta["md:pattern"],
            "md:coordinates_len": new_coords_len,
            "md:dimensions": self.dataset.md_meta["md:dimensions"],
        }

    def __getitem__(
        self, key: DimensionFilter
    ) -> tuple[NDArray[Any], tuple[Coordinates, CoordinatesLen]]:
        """Retrieve a filtered subset of the mCOG dataset.

        Processes non-spatial dimension filters and spatial slices to efficiently read
        and restructure a subset of the dataset.

        Lifecycle:
        1. Non-spatial Filtering:
            - Reorders filters based on md:pattern
            - Applies categorical/range filters to non-spatial dimensions
        2. Spatial Windowing:
            - Converts pixel slices to rasterio Window
            - Handles blockZsize-based slicing
        3. Data Restructuring:
            - Reassembles blocks according to original pattern
            - Maintains MD_METADATA and geotransform consistency

        Args:
            key: Filter specification tuple containing:
                1) Non-spatial filters (band, time, simulation, etc. dimensions)
                2) Spatial slices (height, width dimensions)
                Example: (slice(0, 2), [3, 7], slice(100, 200), slice(300, 400))
                ------------- [Non-spatial] ------------ [Spatial] ------------

        Returns:
            tuple: Contains:
                - 1) filtered_data: Restructured array subset
                - 2) (new_coords, new_coords_len): Updated coordinate metadata:
                    * [NEW] md:coordinates: Filtered coordinate values
                    * [NEW] md:coordinates_len: Resulting dimension lengths
                    * [NEW] geotransform: Updated spatial transform

        Raises:
            InvalidDimensionError: For mismatched filter/dimension lengths
            WindowOutOfBoundsError: For invalid spatial coordinates

        Example:
            >>> reader = ChunkedReader(dataset)
            >>> data_subset, new_metadata = reader[(0, 1), slice(100, 200), slice(300, 400)]
        """

        # Parse input filters
        self.last_query = (key[:-2], key[-2:]) # (Non-spatial filters, spatial slices)
        dim_filters, (row_slice, col_slice) = self.last_query

        # Align filters with storage pattern (md:pattern)
        # This is importante because:
        # "time band y x -> (time band) y x"
        # IS NOT EQUAL to
        # "time band y x -> (band time) y x"
        pattern_order = self.get_axis_order(self.dataset.md_meta["md:pattern"])
        ordered_filters = tuple(dim_filters[i] for i in pattern_order)
        original_dims = tuple(self.dataset.md_meta["md:coordinates_len"].values())
        ordered_dims = tuple(original_dims[i] for i in pattern_order)

        # Apply dimensional subsetting to non-spatial dimensions
        filter_mask = self._apply_dimension_filters(ordered_dims, ordered_filters)

        # Convert spatial slices to rasterio Window
        height_window = self._scale_slice(
            row_slice if row_slice != slice(None) else slice(0, self.dataset.height),
            self.dataset.blockzsize
        )
        width_window = self._scale_slice(
            col_slice if col_slice != slice(None) else slice(0, self.dataset.width),
            self.dataset.blockzsize
        )
        read_window = Window.from_slices(height_window, width_window)

        # Calculate chunk indices considering blockZsize
        # We descompose each band index into: blockZsize*chunk_indices + sub_indices
        chunk_indices, sub_indices = self._decompose_indices(
            block_size=filter_mask,
            indices=self.dataset.blockzsize**2
        )

        # Read and restructure data
        raw_chunk = self.dataset._read(chunk_indices.tolist(), window=read_window)

        # Reshape blocks to match original pattern (from blockzsize n to blockzsize 1)
        block_reshaped = rearrange(
            raw_chunk,
            "c (h c1) (w c2) -> (c c1 c2) h w",
            c1=self.dataset.blockzsize,
            c2=self.dataset.blockzsize
        )

        # Filter blocks based on sub_indices
        filtered_blocks = block_reshaped[sub_indices]

        # Update md:coordinates and md:coordinates_len
        new_coords = self._get_new_md_meta_coordinates()
        new_dims = self._get_new_md_meta_coordinates_len(new_coords)
        new_geotransform = self._get_new_geotransform()

        # Restructure to final pattern
        final_array = rearrange(
            filtered_blocks,
            self.dataset.md_meta["md:pattern"],
            **new_dims
        )

        return final_array, (new_coords, new_dims, new_geotransform)


    def _decompose_indices(self, indices: NDArray[Any], block_size: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Decompose indices into band and intra-band positions.

        This method separates global band indices into:
        1. Block identifiers (which block contains the band index)
        2. Local positions (where within the block the index falls)

        Args:
            indices: Flat array of global indices to decompose
            block_size: Number of elements per block (BLOCKZSIZE)

        Returns:
            tuple: Contains:
                - unique_bands: Array of unique global band identifiers
                - local_positions: Corresponding positions within blocks

        Example:
            >>> indices = np.array([0, 1, 5, 6, 7])
            >>> blocks, positions = _decompose_indices(indices, block_size=4)
            >>> blocks
            array([0, 1])  # Blocks 0 and 1
            >>> positions
            array([0, 1, 1, 2, 3])  # Positions within blocks
        """
        # Calculate block identifiers and local positions
        block_ids = (indices // block_size) + 1 # GDAL is 1-based!

        # Calculate local positions within respect to the min block id
        local_positions = indices % block_size + (block_size * (block_ids - block_ids.min()))

        # Return unique blocks and corresponding positions
        return np.unique(block_ids), local_positions


    def _scale_slice(self, s, scalar):
        """
        Multiplies the start and stop of a slice object by a scalar.

        Parameters:
        s (slice): The slice object to be scaled.
        scalar (numeric): The value to multiply the slice by.

        Returns:
        slice: A new slice object with scaled values.
        """
        return slice(s.start * scalar, s.stop * scalar, s.step if s.step else None)


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
