from __future__ import annotations

import math
from typing import (Any, Dict, List, Optional, Protocol, Sequence, Tuple,
                    Union, runtime_checkable)

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from rasterio.windows import Window

# Type definitions
Slice = Union[slice, int, list, tuple]
Coordinate = Union[str, int, bool, Sequence[Any]]
DimensionFilter = List[Slice]


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol defining required dataset interface."""

    def close(self) -> None: ...
    def _read(self, bands: List[int], window: Optional[Window] = None) -> NDArray: ...

    @property
    def transform(self) -> Affine: ...

    @property
    def profile(self) -> Dict[str, Any]: ...

    @property
    def md_meta(self) -> Dict[str, Any]: ...


class PartialReadImage:
    """
    Optimized implementation of partial image reading with metadata handling.
    """

    def __init__(self, dataset: DatasetProtocol) -> None:
        """
        Initialize the PartialReadImage instance.

        Args:
            dataset: Source dataset implementing the required interface
            md_meta: Initial metadata dictionary
        """
        self.dataset = dataset
        self.last_query: Optional[Tuple[DimensionFilter, Tuple[slice, slice]]] = None
        self.new_height = None
        self.new_width = None
        self.new_count = None

    @staticmethod
    def _filter_dimensions(
        dims: Sequence[int], filter_criteria: DimensionFilter
    ) -> NDArray:
        """
        Optimized dimension filtering using vectorized operations.

        Args:
            dims: Sequence of dimension sizes
            filter_criteria: List of filtering conditions

        Returns:

        """
        data = np.indices(dims, dtype=np.uint32).reshape(len(dims), -1).T
        mask = np.ones(data.shape[0], dtype=bool)

        for dim, condition in enumerate(filter_criteria):
            if isinstance(condition, slice):
                if condition == slice(None):  # Handle the `:` case
                    continue  # If it's a full slice, keep all elements
                start = condition.start or 0
                stop = condition.stop or dims[dim]
                mask &= (data[:, dim] >= start) & (data[:, dim] < stop)
            elif isinstance(condition, (int, list, tuple)):
                mask &= np.isin(data[:, dim], condition)

        return np.nonzero(mask)[0] + 1

    def _get_new_geotransform(self) -> Affine:
        """
        Calculate new geotransform for the specified window.
        """
        row_slice, col_slice = self.last_query[1]

        # Replace slice(None) with the full range if necessary
        row_slice = (
            row_slice if row_slice != slice(None) else slice(0, self.dataset.height)
        )
        col_slice = (
            col_slice if col_slice != slice(None) else slice(0, self.dataset.width)
        )
        window = Window.from_slices(row_slice, col_slice)

        # Update the new height and width
        self.new_height = window.height
        self.new_width = window.width

        return rasterio.windows.transform(window, self.dataset.transform)

    def _get_new_md_meta(self) -> Dict[str, Any]:
        """
        Update metadata based on the provided key.
        """

        if not all(
            isinstance(cond, (slice, int, list, tuple)) for cond in self.last_query[0]
        ):
            raise ValueError("Filter criteria must be slice, int, list, or tuple.")

        # Get the initial coordinates and dimensions
        coords = self.dataset.md_meta["md:coordinates"]
        dims = self.dataset.md_meta["md:dimensions"]

        new_coords = {
            dims[dim]: (
                [coords[dims[dim]][cond]]
                if isinstance(coords[dims[dim]][cond], (str, int, bool))
                else coords[dims[dim]][cond]
            )
            for dim, cond in enumerate(self.last_query[0])
        }

        new_coords_len = {key: len(values) for key, values in new_coords.items()}
        self.new_count = math.prod(new_coords_len.values())

        return {
            "md:coordinates": new_coords,
            "md:attributes": self.dataset.md_meta["md:attributes"],
            "md:pattern": self.dataset.md_meta["md:pattern"],
            "md:coordinates_len": new_coords_len,
            "md:dimensions": dims,
        }

    def __getitem__(self, key: DimensionFilter) -> NDArray:
        """
        Perform optimized partial read operation.
        """
        if len(key) < 2:
            raise IndexError("Invalid key format")

        # Store the last query for metadata update
        self.last_query = (key[:-2], key[-2:])
        filter_criteria, (row_slice, col_slice) = self.last_query

        # Filter the dimensions based on the criteria
        dims_len = self.dataset.md_meta["md:coordinates_len"].values()

        # Get the filtered bands
        result = self._filter_dimensions(dims_len, filter_criteria)

        # Replace slice(None) with the full range if necessary
        row_slice = (
            row_slice if row_slice != slice(None) else slice(0, self.dataset.height)
        )
        col_slice = (
            col_slice if col_slice != slice(None) else slice(0, self.dataset.width)
        )
        window = Window.from_slices(row_slice, col_slice)

        return self.dataset._read(result.tolist(), window=window)

    def update_metadata(self) -> Dict[str, Any]:
        """
        Generate complete metadata dictionary with validation.
        """
        if self.last_query is None:
            raise ValueError("No query has been made yet")

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

    def close(self) -> None:
        """Close the dataset and clean up resources."""
        self.dataset.close()
        self.last_query = self.new_height = self.new_width = self.new_count = None
