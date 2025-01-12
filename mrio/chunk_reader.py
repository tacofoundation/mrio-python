import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import rasterio
from einops import rearrange
from affine import Affine
from numpy.typing import NDArray
from rasterio.windows import Window

from mrio.errors import MRIOError
from mrio.protocol import DatasetProtocol
from mrio.typing import DimensionFilter


class ChunkedReader:
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

    def _get_new_md_meta_coordinates(self) -> Dict[str, Any]:
        """
        Updates "md:coordinates" based on the provided query conditions.

        Returns:
            A dictionary of updated coordinates.
        Raises:
            MRIOError: If any filter criteria are of unsupported types.
        """
        # Validate query conditions
        query_conditions = self.last_query[0]
        if not all(isinstance(cond, (slice, int, list, tuple)) for cond in query_conditions):
            raise MRIOError("Filter criteria must be of type slice, int, list, or tuple.")
        
        # Retrieve initial metadata
        coords = self.dataset.md_meta["md:coordinates"]
        dims = self.dataset.md_meta["md:dimensions"]

        # Update coordinates
        new_coords = {}
        for dim_index, cond in enumerate(query_conditions):
            dim_name = dims[dim_index]
            dim_coords = coords[dim_name]

            # Handle each condition type
            if isinstance(cond, (int, slice)):  # Single index or slice
                updated_coord = dim_coords[cond]
            elif isinstance(cond, (list, tuple)):  # List or tuple of indices
                updated_coord = [dim_coords[c] for c in cond]
            else:
                raise MRIOError(f"Unsupported condition type: {type(cond)}")
            
            # Ensure single value is wrapped in a list
            if isinstance(updated_coord, (str, int, bool)):
                updated_coord = [updated_coord]
            
            new_coords[dim_name] = updated_coord

        return new_coords

    def _get_new_md_meta_coordinates_len(self, new_coords: Dict[str, Any]) -> Dict[str, Any]:
        return {key: len(values) for key, values in new_coords.items()}

    def _get_new_md_meta(self) -> Dict[str, Any]:
        """
        Update metadata based on the provided key.
        """

        # Get the new md:coordinates
        new_coords = self._get_new_md_meta_coordinates()
        dims = self.dataset.md_meta["md:dimensions"]
        
        # Update the md:coordinates_len and count
        new_coords_len = self._get_new_md_meta_coordinates_len(new_coords)

        # Set the new count
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

        # Parameters for the rearrange function
        data_chunk: NDArray = self.dataset._read(result.tolist(), window=window)
        coordinates_len: Dict[str, Any] = self._get_new_md_meta_coordinates_len(
            new_coords=self._get_new_md_meta_coordinates()
        )
                
        return rearrange(data_chunk, self.dataset.md_meta.get("md:pattern"), **coordinates_len)


    def update_metadata(self) -> Dict[str, Any]:
        """
        Generate complete metadata dictionary with validation.
        """
        if self.last_query is None:
            raise MRIOError("No query has been made yet")

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