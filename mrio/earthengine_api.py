from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, TypeVar, Union, Literal, Any

import numpy as np
import rasterio as rio

from mrio.protocol import DatasetReaderProtocol
from mrio.readers import DatasetReader
from mrio.writers import DatasetWriter
from mrio.env_options import MRIOConfig

T = TypeVar("T", bound="Collection")


@dataclass(frozen=True)
class ImageQuery:
    """
    Immutable query parameters for image operations.

    Attributes:
        filepath (str): Path to the image file
        bands (tuple[str, ...]): Tuple of band names to select
        date_range (tuple[datetime, datetime] | None): Start and end dates for filtering
        bounds (tuple[float, float, float, float] | None): Spatial bounds (min_x, min_y, max_x, max_y)
    """

    filepath: str
    bands: tuple[str, ...] = field(default_factory=tuple)
    date_range: tuple[datetime, datetime] | None = None
    bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate query parameters after initialization."""
        # if self.bounds:
        #    min_x, min_y, max_x, max_y = self.bounds
        #    if min_x >= max_x or min_y >= max_y:
        #        raise ValueError("Invalid bounds: min values must be less than max values")
        if self.date_range and self.date_range[0] > self.date_range[1]:
            raise ValueError("Invalid date range: start date must be before end date")

    def with_bands(self, bands: List[str]) -> ImageQuery:
        """Create new query with updated bands."""
        return ImageQuery(filepath=self.filepath, bands=tuple(bands), date_range=self.date_range, bounds=self.bounds)

    def with_dates(self, start_date: datetime, end_date: datetime) -> ImageQuery:
        """Create new query with updated date range."""
        return ImageQuery(
            filepath=self.filepath, bands=self.bands, date_range=(start_date, end_date), bounds=self.bounds
        )

    def with_bounds(self, bounds: tuple[float, float, float, float]) -> ImageQuery:
        """Create new query with updated bounds."""
        return ImageQuery(filepath=self.filepath, bands=self.bands, date_range=self.date_range, bounds=bounds)


class Collection:
    """
    A lazy-loading image collection with immutable query state and optimized operations.
    """

    def __init__(
        self,
        filepath: str,
        engine: str = "numpy",
        env_options: Union[Literal["mrio", "default"], dict[str, str]] = "mrio",
        **kwargs: Any,
    ) -> None:
        """Initialize the ImageCollection."""
        self._query = ImageQuery(filepath=filepath)
        self._engine = engine
        with MRIOConfig.get_env(env_options):
            self._dataset: DatasetReaderProtocol = DatasetReader(filepath, engine=engine, **kwargs)
        self._cache: dict = {}

    @property
    def dataset(self) -> DatasetReaderProtocol:
        """Lazy loading of dataset."""
        return self._dataset

    def select(self: T, bands: List[str]) -> T:
        """Select specific bands from the image."""
        if not set(bands).issubset(self.dataset.coords["band"]):
            available = ", ".join(self.dataset.coords["band"])
            raise ValueError(f"Invalid bands. Available bands: {available}")

        self._query = self._query.with_bands(bands)
        return self

    def FilterDate(self: T, start_date: str, end_date: str) -> T:
        """Filter by date range."""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        self._query = self._query.with_dates(start, end)
        return self

    def FilterBounds(self: T, min_x: float, min_y: float, max_x: float, max_y: float) -> T:
        """Filter by geographic bounds."""
        bounds = (min_x, min_y, max_x, max_y)
        self._query = self._query.with_bounds(bounds)
        return self

    @staticmethod
    @lru_cache(maxsize=128)
    def _check_continuous_indices(indices: tuple[int, ...]) -> Optional[slice]:
        """Check if indices are continuous and convert to slice if possible."""
        if not indices:
            return slice(None)

        if len(indices) == 1:
            return slice(indices[0], indices[0] + 1)

        if all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)):
            return slice(indices[0], indices[-1] + 1)

        return None

    def _get_selection_indices(
        self,
    ) -> tuple[Union[slice, List[int]], Union[slice, List[int]], tuple[slice, slice], Optional[rio.Affine]]:
        """Get all selection indices and transform in one pass."""
        band_idx = slice(None)
        date_idx = slice(None)
        bounds_idx = (slice(None), slice(None))
        transform = None

        # Process bands
        if self._query.bands:
            band_indices = tuple(self.dataset.coords["band"].index(b) for b in self._query.bands)
            band_idx = self._check_continuous_indices(band_indices) or list(band_indices)

        # Process dates
        if self._query.date_range:
            start, end = self._query.date_range
            dates = [datetime.fromisoformat(d) for d in self.dataset.attrs["md:time_start"]]
            date_indices = tuple(i for i, date in enumerate(dates) if start <= date <= end)
            if not date_indices:
                raise ValueError("No dates found within specified range")
            date_idx = self._check_continuous_indices(date_indices) or list(date_indices)

        # Process bounds
        if self._query.bounds:
            bounds_idx, transform = self._calculate_bounds()

        return band_idx, date_idx, bounds_idx, transform

    def _calculate_bounds(self) -> tuple[tuple[slice, slice], rio.Affine]:
        """Calculate spatial bounds and transform.

        Returns bounds aligned to the GeoTIFF's internal block structure. For any intersection
        with a block, returns the full block based on the file's actual block dimensions.

        Returns:
            tuple: ((y_slice, x_slice), affine_transform)
        """
        img_bounds = self.dataset.bounds
        query_bounds = self._query.bounds

        # IMPORTANT: Get the actual block size from the dataset profile
        if "blockxsize" not in self.dataset.profile or "blockysize" not in self.dataset.profile:
            raise ValueError("Dataset profile missing block size information")

        block_width = self.dataset.profile["blockxsize"]
        block_height = self.dataset.profile["blockysize"]

        # Get original transform and pixel sizes
        orig_transform = self.dataset.transform
        pixel_width = abs(orig_transform.a)
        pixel_height = abs(orig_transform.e)

        # Calculate initial pixel coordinates
        x_scale = (self.dataset.shape[-1]) / (img_bounds[2] - img_bounds[0])
        y_scale = (self.dataset.shape[-2]) / (img_bounds[3] - img_bounds[1])

        # Calculate initial pixel coordinates
        x_start = max(0, int((query_bounds[0] - img_bounds[0]) * x_scale))
        x_end = min(self.dataset.shape[-1], int(np.ceil((query_bounds[2] - img_bounds[0]) * x_scale)))
        y_start = max(0, int((img_bounds[3] - query_bounds[3]) * y_scale))
        y_end = min(self.dataset.shape[-2], int(np.ceil((img_bounds[3] - query_bounds[1]) * y_scale)))

        # Snap to block boundaries using actual block dimensions
        x_start = (x_start // block_width) * block_width
        y_start = (y_start // block_height) * block_height
        x_end = min(((x_end + block_width - 1) // block_width) * block_width, self.dataset.shape[-1])
        y_end = min(((y_end + block_height - 1) // block_height) * block_height, self.dataset.shape[-2])

        # Calculate new transform based on actual pixel edges
        new_transform = rio.Affine(
            pixel_width,
            0.0,
            img_bounds[0] + (x_start * pixel_width),
            0.0,
            -pixel_height,
            img_bounds[3] - (y_start * pixel_height),
        )

        return (slice(y_start, y_end), slice(x_start, x_end)), new_transform

    def getInfo(self) -> np.ndarray:
        """Execute the query and return the filtered image data."""
        band_idx, date_idx, (y_slice, x_slice), _ = self._get_selection_indices()
        print(date_idx, band_idx, y_slice, x_slice)
        tensor_slice = self.dataset[date_idx, band_idx, y_slice, x_slice]

        # After reading close the dataset
        self.dataset.close()

        return tensor_slice

    def save(self, output_path: str, **kwargs) -> str:
        """Save the filtered image data to a COG file."""
        band_idx, date_idx, bounds_idx, new_transform = self._get_selection_indices()
        y_slice, x_slice = bounds_idx

        # Get data
        data = self.dataset[date_idx, band_idx, y_slice, x_slice]

        # Prepare metadata
        md_coords = self._prepare_coordinates(band_idx, date_idx)
        md_attrs = self._prepare_attributes(date_idx)
        md_pattern = " -> ".join(map(str.strip, self.dataset.md_meta["md:pattern"].split("->")[::-1]))

        # Create final profile

        init_profile = {
            **self.dataset.profile,
            "transform": new_transform,
            "width": data.shape[-1],
            "height": data.shape[-2],
            "md:pattern": md_pattern,
            "md:coordinates": md_coords,
            "md:attributes": md_attrs,
            **kwargs,
        }

        # Create a copy to avoid modifying the original
        cog_profile = init_profile.copy()

        # Update driver
        cog_profile["driver"] = "COG"

        # Handle block size parameters
        if "blockxsize" in cog_profile:
            cog_profile["blocksize"] = cog_profile.pop("blockxsize")
            cog_profile.pop("blockysize", None)  # Remove if exists

        # Remove tiled parameter if exists
        cog_profile.pop("tiled", None)

        # Write data
        with DatasetWriter(output_path, **cog_profile) as dst:
            dst.write(data)

        # After writing close the dataset
        self.dataset.close()

        return output_path

    def _prepare_coordinates(self, band_idx: Union[slice, List[int]], date_idx: Union[slice, List[int]]) -> dict:
        """Prepare coordinate metadata."""
        coords = dict(self.dataset.coords)

        if self._query.bands:
            coords["band"] = (
                coords["band"][band_idx] if isinstance(band_idx, slice) else [coords["band"][i] for i in band_idx]
            )

        if self._query.date_range:
            coords["time"] = (
                coords["time"][date_idx] if isinstance(date_idx, slice) else [coords["time"][i] for i in date_idx]
            )

        return coords

    def _prepare_attributes(self, date_idx: Union[slice, List[int]]) -> dict:
        """Prepare attribute metadata."""
        attrs = dict(self.dataset.attrs)

        if not self._query.date_range:
            return attrs

        time_fields = ["md:time_start", "md:id", "md:time_end"]

        for field in time_fields:
            if field not in attrs:
                continue

            attrs[field] = (
                attrs[field][date_idx] if isinstance(date_idx, slice) else [attrs[field][i] for i in date_idx]
            )

        return attrs

    # Public utility methods
    def bandNames(self) -> List[str]:
        """Return available band names."""
        return list(self.dataset.coords["band"])

    def size(self) -> int:
        """Return the number of images."""
        return self.dataset.shape[0]

    def dateRange(self) -> tuple[datetime, datetime]:
        """Return the date range of the image collection."""
        dates = self.dataset.attrs["md:time_start"]
        return datetime.fromisoformat(dates[0]), datetime.fromisoformat(dates[-1])

    def bounds(self) -> tuple[float, float, float, float]:
        """Return the spatial bounds."""
        return self.dataset.bounds

    def __str__(self) -> str:
        return str(self.dataset)

    def __repr__(self) -> str:
        return repr(self.dataset)
