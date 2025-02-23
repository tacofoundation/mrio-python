"""MRIO Dataset Writer Module with metadata validation.

Provides writing of multidimensional COG with features:
- Automatic dimension detection from data array
- Einstein notation-based data rearrangement
- Non-spatial coordinate system support
- Metadata embedding in standard-compliant format
"""


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
    """Writer for multidimensional Cloud Optimized GeoTIFFs (COG) with semantic metadata.

    Key Features:
    - Automatic inference of spatial dimensions from data
    - Pattern-based data rearrangement using Einstein notation
    - Non-spatial coordinate system support with semantic band identifiers
    - Scalable tiling for large datasets
    - Metadata embedding in standard-compliant format

    Example:
        >>> # Create writer with metadata-driven parameters
        >>> params = {
        ...     'md:pattern': 'time band h w -> (time band) h w',  # Rearrangement pattern
        ...     'md:coordinates': {  # Coordinate system definition
        ...         'time': ['2020', '2021'],
        ...         'band': ['red', 'nir']
        ...     },
        ...     'dtype': 'uint16',  # Direct parameter
        ...     'blockxsize': 256,   # Tile sizing
        ...     'blockysize': 256
        ... }
        >>> with DatasetWriter('multispectral.tif', **params) as writer:
        ...     writer.write(numpy_data)

    Lifecycle:
    1. Initialization: Parameters are parsed, metadata separated
    2. First write: Array parameters (width, height, dtype) inferred from data if not provided
    3. Data writing: Automatic rearrangement and metadata embedding
    4. File closure: Final metadata written, resources cleaned up
    """

    __slots__ = (
        "_file",
        "_initialized",
        "args",
        "file_path",
        "kwargs",
        "md_kwargs",
    )

    # Class constants
    DEFAULT_ENGINE: ClassVar[str] = "numpy"

    def __init__(self, file_path: PathLike, *args: Any, **kwargs: Any) -> None:
        """Initialize a metadata-aware raster writer.

        Args:
            file_path: Output file path (any rasterio-supported format)
            *args: Additional positional arguments for rasterio.open()
            **kwargs: Keyword arguments including:
                - Standard rasterio parameters (width, height, dtype, etc.)
                - Metadata parameters prefixed with 'md:'
                - GDAL creation options (e.g., 'COMPRESS', 'TILED', 'BLOCKXSIZE')

        Metadata Parameters (md:*):
            pattern: Einstein rearrangement pattern (e.g., 'c h w -> (c) h w')
            coordinates: Non-spatial coordinates mapping (e.g., {'c': ['red', 'green']})
            attributes: Additional metadata attributes (e.g., {'author': 'Alice'})
            blockzsize: Z-dimension chunk size for 3D+ data

        Note:
            File handle creation is deferred until first write operation to enable
            parameter inference from initial data array.
        """
        self.file_path = Path(file_path)
        self.args = args
        self.kwargs = kwargs
        self._file = None  # Rasterio file handle
        self.md_kwargs = None # Processed metadata parameters
        self._initialized = False # State flag


    def _infer_parameters(self, data: DataArray) -> dict[str, Any]:
        """Infer missing spatial parameters from data array.

        Args:
            data: Input array (numpy.ndarray or xarray.DataArray)
                  Must have at least 2 spatial dimensions

        Returns:
            Dictionary with inferred parameters (height, width, dtype)

        Raises:
            ValueError: For insufficient dimensions or invalid data shape

        Process:
            1. Extract numpy array from xarray if needed
            2. Check minimum dimensionality
            3. Calculate missing parameters from last two dimensions (height, width)
            4. Apply blockzsize scaling if set
        """
        da_type = type(data)
        if (da_type.__name__ == "DataArray" and da_type.__module__ == "xarray.core.dataarray"):
            data = data.values

        if data.ndim < 2:
            msg = "Data must have at least 2 dimensions"
            raise ValueError(msg)

        inferred = {}
        if self.kwargs.get("height") is None:
            inferred["height"] = data.shape[-2] * self.kwargs.get("md:blockzsize", 1)
        if self.kwargs.get("width") is None:
            inferred["width"] = data.shape[-1] * self.kwargs.get("md:blockzsize", 1)
        if self.kwargs.get("dtype") is None:
            inferred["dtype"] = data.dtype

        return inferred

    def _initialize_write_mode(self, data: DataArray) -> None:
        """Finalize writer configuration and open output file.

        Args:
            data: Optional initial data array for parameter inference

        Process:
            1. Separate MRIO parameters from GDAL/rasterio parameters
            2. Infer missing array parameters (width, height, dtype) from data
            3. Calculate new band count considering md:blockzsize and md:coordinates length
            4. Update metadata parameters with new dimensions and geotransform
            5. Open rasterio file handle with final parameters
        """
        # Start with user parameters
        kwargs: dict[str, Any] = self.kwargs.copy()

        # Initialize processed parameters
        md_blockzsize: int = kwargs.get("md:blockzsize", 1)

        # Infer missing parameters from data if available
        inferred = self._infer_parameters(data)
        kwargs.update(inferred)

        # WriteParams is a dataclass that validates both rasterio and mrio parameters
        kwargs = WriteParams(params=kwargs).to_dict()

        # Extract metadata parameters (remove 'md:' prefix)
        md_kwargs_dict = {
            k[len(MD_PREFIX) :]: v for k, v in kwargs.items() if k.startswith(MD_PREFIX)
        }

        # Remove metadata params from main kwargs
        for k in md_kwargs_dict:
            kwargs.pop(f"{MD_PREFIX}{k}")

        # Validate mrio metadata parameters
        self.md_kwargs = MRIOFields(**md_kwargs_dict)

        # Calculate total number of bands in a 3D space
        coord_dims = [len(v) for v in self.md_kwargs.coordinates.values()]
        total_bands = math.prod(coord_dims) // (md_blockzsize ** 2)

        # Calculate new geotransform if blockzsize is set bigger than 1
        new_geo_transform = kwargs.get("transform") * rio.Affine.scale(1/md_blockzsize, 1/md_blockzsize)

        # Update the blocksize parameter
        new_blocksize = kwargs.get("blocksize", 128) * md_blockzsize


        # Validate & update write parameters
        kwargs.update(
            {
                "count": total_bands,
                "transform": new_geo_transform,
                "blocksize": new_blocksize,
            }
        )
        self.kwargs = kwargs

        # Open file handle in write mode
        print(self.kwargs)
        try:
            self._file = rio.open(self.file_path, "w", *self.args, **self.kwargs)
        except Exception as e:
            raise mrio_errors.WritersFailedToOpenError(self.file_path, e) from e

        self._initialized = True


    def _write_custom_data(self, data: DataArray) -> None:
        """Write data with metadata handling.

        Args:
            data: Array data to write

        Note:
            Falls back to simple write if no pattern/coordinates specified

        """
        # If it is a simple COG, write it directly
        if not self.md_kwargs.pattern or not self.md_kwargs.coordinates:
            self._file.write(data)
            return

        # Otherwise, rearrange and write with md:metadata
        self._rearrange_and_write(data)

    def _rearrange_and_write(self, data: DataArray) -> None:
        """Core metadata-aware writing routine.

        Args:
            data: Numpy-style array to write

        Process:
            1. Apply Einstein pattern rearrangement
            2. Handle scale factor for super-resolution writing
            3. Generate band metadata
            4. Write data with band descriptions
        """

        # Rearrange data dimensions according to pattern
        # It will create a 3D tensor with shape (bands, height, width)
        image = rearrange(tensor=data, pattern=self.md_kwargs.pattern)

        # Apply blockzsize scaling if set
        # It will still be a 3D tensor but with less bands, e.g.,
        # (bands, height, width) -> (bands//4, height*2, width*2)
        if self.md_kwargs.blockzsize > 1:
            image = rearrange(
                tensor=image,
                pattern="(c c1 c2) h w -> c (h c1) (w c2)",
                c1=self.md_kwargs.blockzsize,
                c2=self.md_kwargs.blockzsize
            )

        ## band_identifiers is the band BAND_NAME metadata storage in GDAL_METADATA ... to display in QGIS
        band_identifiers: list[str] = self._generate_band_identifiers()

        ## md_metadata are all the mrio metadata stored in the GDAL_METADATA.
        md_metadata: MetadataDict = self._generate_metadata()

        # Write data and metadata
        for i, (band_data, band_id) in enumerate(zip(image, band_identifiers), start=1):
            self._file.write(band_data, i)
            self._file.set_band_description(i, band_id)

        self._file.update_tags(MD_METADATA=json.dumps(md_metadata))

    def _generate_band_identifiers(self) -> list[str]:
        """Generate human-readable band identifiers from the md:coordinates.

        Returns:
            List of strings in format 'FROM(coords) TO(coords)' representing
            coordinate ranges for each band.

        Example:
            For coordinates {'time': [2020, 2021], 'band': ['red', 'nir']}
            Returns ['FROM(2020:red) TO(2020:red)', 'FROM(2020:nir) TO(2020:nir)', ...]
        """

        # Generate all coordinate combinations
        coord_axes = [
            self.md_kwargs.coordinates[dim]
            for dim in self.md_kwargs._in_parentheses  # Dimensions in pattern parentheses
        ]
        combinations = [":".join(map(str, combo)) for combo in product(*coord_axes)]

        # Group combinations by scale factor
        blockzsize: int = self.md_kwargs.blockzsize
        return [
            f"FROM({combinations[i]}) TO({combinations[i + blockzsize - 1]})"
            for i in range(0, len(combinations), blockzsize ** 2)
        ]


    def _generate_metadata(self) -> MetadataDict:
        """Clean up the final metadata dictionary for embedding into the mCOG.

        Returns:
            Dictionary containing:
            - Dimension order
            - Coordinate values
            - Coordinate lengths
            - Processing attributes
            - Rearrangement pattern
            - Blockzsize
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
            "md:blockzsize": self.md_kwargs.blockzsize,
        }


    def write(self, data: DataArray) -> None:
        """Write n-D data array to output file with metadata processing.

        Args:
            data: Input array (numpy.ndarray or xarray.DataArray)

        Raises:
            WritersUnsupportedDataTypeError: For unsupported array types
            ValueError: For data shape mismatch with metadata

        Process:
            1. Lazy initialization on first write
            2. Data type detection and conversion
            3. Metadata-aware writing
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
