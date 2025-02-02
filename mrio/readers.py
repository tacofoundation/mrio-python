"""MRIO Dataset Reader Module.

Provides optimized reading capabilities for multi-dimensional COG files with
metadata handling and lazy loading support. Supports both numpy and xarray outputs.
"""

from __future__ import annotations

import json
import warnings
from importlib import import_module
from functools import lru_cache
from typing import Any, ClassVar, Literal, Union, List, Dict

import numpy as np
import rasterio as rio

from einops import rearrange
from numpy.typing import NDArray
from typing_extensions import Self
from mrio.types import PathLike

from mrio.chunk_reader import ChunkedReader
from mrio.slice_transformer import SliceTransformer

# Type aliases for better readability
MetadataDict = dict[str, Any]
Profile = dict[str, Any]
Coords = dict[str, list[Any]]


DataArray = Union[NDArray[Any], "xr.DataArray"]

# Constants
MD_METADATA_KEY: str = "MD_METADATA"


class DatasetReader:
    """Optimized reader for multi-dimensional COG files with metadata handling.

    This class provides efficient reading capabilities with metadata handling,
    coordinate management, and lazy loading support. It can output data as either
    numpy arrays or xarray DataArrays.

    Attributes:
        file_path: Path to the dataset file
        engine: Output format ('numpy' or 'xarray')
        profile: Dataset profile information
        md_meta: Multi-dimensional metadata
        coords: Coordinate values for each dimension
        dims: Dimension names
        attrs: Dataset attributes
        shape: Dataset shape including all dimensions
        size: Total size in bytes

    Example:
        >>> with DatasetReader("example.tif") as ds:
        ...     data = ds.read()  # Returns xarray.DataArray by default
        ...     subset = ds[1:3, :, :]  # Supports array-like indexing

    """

    __slots__ = (
        "_file",
        "args",
        "attrs",
        "block_shapes",
        "bounds",
        "chunks",
        "compression",
        "coords",
        "count",
        "crs",
        "descriptions",
        "dims",
        "driver",
        "dtype",
        "dtypes",
        "engine",
        "file_path",
        "gcps",
        "height",
        "indexes",
        "interleaving",
        "is_tiled",
        "kwargs",
        "mask_flag_enums",
        "md_meta",
        "meta",
        "mode",
        "ndim",
        "nodata",
        "nodatavals",
        "offsets",
        "options",
        "photometric",
        "profile",
        "res",
        "rpcs",
        "scales",
        "shape",
        "size",
        "subdatasets",
        "transform",
        "units",
        "width",
        "window",
    )

    # Class variables
    UNITS: ClassVar[list[str]] = ["B", "KB", "MB", "GB", "TB"]
    DEFAULT_ENGINE: ClassVar[str] = "numpy"

    def __init__(
        self,
        file_path: PathLike,
        engine: Literal["numpy", "xarray"] = DEFAULT_ENGINE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize DatasetReader with optimized metadata handling.

        Args:
            file_path: Path to the dataset file
            engine: Output format ('numpy' or 'xarray')
            *args: Additional positional arguments for rasterio
            **kwargs: Additional keyword arguments for rasterio

        Raises:
            IOError: If the file cannot be opened

        """
        self.file_path = file_path
        self.mode = "r"
        self.engine = engine
        self.args = args
        self.kwargs = kwargs

        try:
            self._file = rio.open(self.file_path, "r", *args, **kwargs)
        except Exception as e:
            msg = f"Failed to open {file_path}: {e}"
            raise OSError(msg)

        self._fast_initialize()

    @lru_cache(maxsize=128)
    def _fast_initialize(self) -> None:
        """Optimized initialization of attributes using direct assignment."""
        # Load profile and meta
        self.profile = self._file.profile
        self.meta = self._file.meta

        # Load geometry properties
        self._load_geometry_properties()

        # Load data properties
        self._load_data_properties()

        # Load metadata properties
        self._load_metadata_properties()

        # Calculate derived properties
        self._calculate_shape_and_size()

    def _load_geometry_properties(self) -> None:
        """Load geometric properties from profile."""
        self.width = self.profile["width"]
        self.height = self.profile["height"]
        self.crs = self.profile["crs"]
        self.transform = self.profile["transform"]
        self.count = self.profile["count"]
        self.window = self._file.window
        self.bounds = self._file.bounds
        self.res = self._file.res

    def _load_data_properties(self) -> None:
        """Load data-related properties."""
        self.block_shapes = self._file.block_shapes[0]
        self.indexes = self._file.indexes
        self.dtype = self._file.dtypes[0]
        self.dtypes = self._file.dtypes[0]
        self.nodata = self._file.nodata
        self.descriptions = self._file.descriptions
        self.compression = self._file.compression
        self.driver = self._file.driver
        self.gcps = self._file.gcps
        self.interleaving = self._file.interleaving
        self.is_tiled = self._file.is_tiled
        self.mask_flag_enums = self._file.mask_flag_enums
        self.nodatavals = self._file.nodatavals
        self.offsets = self._file.offsets
        self.options = self._file.options
        self.photometric = self._file.photometric
        self.rpcs = self._file.rpcs
        self.scales = self._file.scales
        self.subdatasets = self._file.subdatasets
        self.units = self._file.units

    def _load_metadata_properties(self) -> None:
        """Load and process metadata properties."""
        self.md_meta = self._fast_load_metadata()
        self.coords = self.md_meta.get("md:coordinates", {}) if self.md_meta else {}
        self.dims = self.md_meta.get("md:dimensions", []) if self.md_meta else []
        self.attrs = self.md_meta.get("md:attributes", {}) if self.md_meta else {}

    def _calculate_shape_and_size(self) -> None:
        """Calculate shape and size properties."""
        if self.md_meta:
            coord_lens = tuple(self.md_meta["md:coordinates_len"].values())
            self.shape = (*coord_lens, self.height, self.width)
        else:
            self.shape = (self.count, self.height, self.width)

        # Calculate size in bytes
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape) * np.dtype(self.dtype).itemsize

        # Add leading dimensions of size 1 to match ndim
        self.chunks = (1,) * (self.ndim - 2) + self.block_shapes

    @lru_cache(maxsize=1)
    def _fast_load_metadata(self) -> MetadataDict | None:
        """Load and cache metadata with optimizations.

        Returns:
            Optional metadata dictionary with dimensions and coordinates

        Warns:
            UserWarning: If metadata loading fails

        """
        try:
            metadata = self._file.tags().get(MD_METADATA_KEY)
            if not metadata:
                return None

            metadata_dict = json.loads(metadata)
            self._enhance_metadata(metadata_dict)
            return metadata_dict

        except Exception as e:
            warnings.warn(f"Metadata loading failed: {e}", stacklevel=2)
            return None

    def _enhance_metadata(self, metadata_dict: MetadataDict) -> None:
        """Enhance metadata with computed fields.

        Args:
            metadata_dict: Dictionary to enhance with additional metadata

        """
        if "md:dimensions" not in metadata_dict:
            metadata_dict["md:dimensions"] = metadata_dict["md:pattern"].split("->")[1].split()

        if "md:coordinates_len" not in metadata_dict:
            coords = metadata_dict["md:coordinates"]
            dims = metadata_dict["md:dimensions"]
            metadata_dict["md:coordinates_len"] = {band: len(coords[band]) for band in dims if band in coords}

    def _read(self, *args: Any, **kwargs: Any) -> DataArray:
        """Internal method to read raw data from the file.

        Returns:
            Array containing the read data

        """
        return self._file.read(*args, **kwargs)

    def read(self, *args: Any, **kwargs: Any) -> DataArray:
        """Read and process data with optional rearrangement.

        Returns:
            Data array with proper dimension arrangement

        Note:
            Returns xarray.DataArray when engine='xarray', numpy.ndarray otherwise

        """
        if not self.md_meta:
            return self._read(*args, **kwargs)

        raw_data = self._file.read(*args, **kwargs)
        data = rearrange(
            raw_data,
            self.md_meta["md:pattern"],
            **self.md_meta["md:coordinates_len"],
        )

        if self.engine == "xarray":
            self._create_xarray(
                data=data,
                dims=self.md_meta["md:dimensions"],
                coords=self.md_meta["md:coordinates"],
                attrs=self.md_meta.get("md:attributes"),
            )

        return data

    def __getitem__(self, key: Any) -> DataArray:
        """Support array-like indexing with metadata handling.

        Args:
            key: Index or slice object

        Returns:
            Subset of the dataset with updated metadata

        """
        new_key = SliceTransformer(ndim=self.ndim).transform(key)

        # Read data with new metadata
        data, (new_md_coord, new_md_coord_len) = ChunkedReader(self)[new_key]

        if self.md_meta and (self.engine == "xarray"):
            return self._create_xarray(
                data=data,
                dims=self.md_meta["md:dimensions"],
                coords=new_md_coord,
                attrs=self.md_meta.get("md:attributes"),
            )

        return data

    @lru_cache(maxsize=128)
    def tags(self) -> dict[str, str]:
        """Get cached dataset tags.

        Returns:
            Dictionary of dataset tags

        """
        return self._file.tags()

    def close(self) -> None:
        """Close the dataset and free resources."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.close()

    @staticmethod
    def _humanize_size(size_in_bytes: int) -> str:
        """Convert size in bytes to human-readable format.

        Args:
            size_in_bytes: Size to convert

        Returns:
            Formatted size string (e.g., "Size: 1.23 GB")

        """
        size = float(size_in_bytes)
        unit_index = 0

        while size >= 1024 and unit_index < len(DatasetReader.UNITS) - 1:
            size /= 1024
            unit_index += 1

        return f"Size: {size:.2f} {DatasetReader.UNITS[unit_index]}"

    def _repr_value(self, value: Any, max_items: int = 3) -> str:
        """
        Uniform representation of lists for both coordinates and attributes.
        """
        if not isinstance(value, (list, tuple)):
            return str(value)

        if len(value) <= max_items * 2:
            return ", ".join(map(str, value))

        start_vals = ", ".join(map(str, value[:max_items]))
        end_vals = ", ".join(map(str, value[-max_items:]))
        return f"{start_vals} ... {end_vals} (length: {len(value)})"

    def _repr_coordinates(self, max_items: int = 3) -> str:
        """Format coordinate representation with consistent style."""
        coords_repr = []
        for key, value in self.coords.items():
            # Calculate size
            total_bytes = sum(len(str(v)) for v in value)
            humanized_size = self._humanize_size(total_bytes)

            # Format the value representation
            value_repr = self._repr_value(value, max_items)
            coords_repr.append(f"  * {key} ({key}) <{humanized_size}> {value_repr}")

        return "\n".join(coords_repr)

    def _repr_attributes(self, max_items: int = 6) -> str:
        """Format attributes with consistent style."""
        attrs_repr = []

        for key, value in self.attrs.items():
            if isinstance(value, (list, tuple)):
                # Use same formatting as coordinates for lists
                value_repr = self._repr_value(value, 3)  # Use 3 items like coordinates
                attrs_repr.append(f"  {key}: [{value_repr}]")
            else:
                # For non-list values
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = f"{val_str[:97]}..."
                attrs_repr.append(f"  {key}: {val_str}")

            if len(attrs_repr) >= max_items:
                remaining = len(self.attrs) - max_items
                if remaining > 0:
                    attrs_repr.append(f"  ... (length: {remaining})")
                break

        return "\n".join(attrs_repr)

    def __repr__(self) -> str:
        """Uniform string representation."""
        coords_repr = self._repr_coordinates(max_items=3)
        attrs_repr = self._repr_attributes(max_items=6)
        size_str = self._humanize_size(self.size)

        extra_dims = [f"{key}: {len(value)}" for key, value in self.coords.items()]
        spatial_dims = [f"y: {self.height}", f"x: {self.width}"]
        all_dims = ", ".join(extra_dims + spatial_dims)

        return (
            f"<LazyDataArray ({all_dims})>\n"
            f"{size_str}\n"
            f"Coordinates:\n{coords_repr}\n"
            f"Dimensions without coordinates: y, x\n"
            f"Attributes:\n{attrs_repr}"
        )

    def __str__(self) -> str:
        """String representation matching __repr__."""
        return self.__repr__()

    def _create_xarray(
        self,
        data: NDArray[Any],
        dims: List[str],
        coords: Dict[str, Any],
        attrs: Dict[str, Any] | None = None,
    ) -> Any:
        """Create an xarray DataArray from input data and metadata.

        Args:
            data: The underlying array data
            dims: Names of the dimensions
            coords: Dictionary mapping dimension names to coordinate arrays
            attrs: Optional dictionary of attributes to add to DataArray

        Returns:
            xarray.DataArray: The created array with coordinates and attributes

        Raises:
            ImportError: If xarray is not installed
            ValueError: If input dimensions don't match data shape
        """
        try:
            xr = import_module("xarray")
            return xr.DataArray(
                data=data,
                dims=dims,
                coords=coords,
                attrs=attrs or {},  # Default to empty dict if None
                fastpath=False,
            )
        except ImportError:
            raise ImportError("xarray is required for this operation.\nInstall it via: pip install xarray")
