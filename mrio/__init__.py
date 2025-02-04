"""MRIO (Multi-dimensional Raster I/O) is a Python package for reading and writing
multi-dimensional and standard COG files.

This package extends rasterio with multi-dimensional data support and provides a simple
interface for I/O operations. Additionally, for users who prefer working with xarray,
MRIO provides an xarray-like and Google Earth Engine-like interface
for reading multi-dimensional COG files.

Example:
    Basic usage for reading a file:
        >>> import mrio
        >>> with mrio.open("example.tif") as src:
        ...     data = src.read()

    Writing data to a file:
        >>> with mrio.open("output.tif", mode="w", **profile) as dst:
        ...     dst.write(data)

"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

# Core rasterio imports
from rasterio import band, crs, io, profiles, transform, windows
from rasterio._version import gdal_version, get_geos_version, get_proj_version
from rasterio.crs import CRS
from rasterio.env import Env
from rasterio.profiles import DefaultGTiffProfile, Profile
from rasterio.transform import Affine, from_bounds, from_gcps, from_origin
from rasterio.windows import Window

from mrio.env_options import MRIOConfig

from .earthengine_api import Collection

# Local imports
from .errors import MRIOError
from .readers import DatasetReader
from .temporal_utils import stack_temporal, unstack_temporal
from .type_definitions import DataArray, PathLike
from .validators import is_mcog, is_tcog
from .writers import DatasetWriter

__version__ = version("mrio")

# Stole this from rasterio
__gdal_version__ = gdal_version()
__geos_version__ = ".".join([str(version) for version in get_proj_version()])
__proj_version__ = ".".join([str(version) for version in get_geos_version()])


T = TypeVar("T", DatasetReader, DatasetWriter)


class Mode:
    """Constants for file operation modes.

    Attributes:
        READ: Read mode ('r')
        WRITE: Write mode ('w')
        VALID_MODES: Set of valid operation modes

    """

    READ: Literal["r"] = "r"
    WRITE: Literal["w"] = "w"
    VALID_MODES = frozenset({READ, WRITE})


@overload
def open(
    file_path: PathLike,
    mode: Literal["r"] = "r",
    engine: str = "numpy",
    **kwargs: Any,
) -> DatasetReader: ...


@overload
def open(
    file_path: PathLike,
    mode: Literal["w"],
    engine: str = "numpy",
    **kwargs: Any,
) -> DatasetWriter: ...


def open(
    file_path: PathLike,
    mode: str = Mode.READ,
    read_env_options: Literal["mrio", "default"] | dict[str, str] = "mrio",
    read_engine: str = "numpy",
    **kwargs: Any,
) -> DatasetReader | DatasetWriter:
    """Open a dataset for reading or writing with enhanced multi-dimensional support.

    This function provides a unified interface for opening both standard and
    multi-resolution COG files. It automatically detects the file type and
    returns the appropriate reader or writer.

    Args:
        file_path: Path to the dataset file. Can be string or Path object.
        mode: Operation mode, either 'r' for read or 'w' for write.
        read_engine: Backend engine to use for reading multi-dimensional data.
            Default is 'numpy'. Other option is 'xarray'.
        read_env_options: Configuration settings for the rasterio environment.
            By default, uses MRIO configuration settings. Check MRIOConfig for details.
            Other available options are 'default' and a custom dictionary.
        **kwargs: Additional keyword arguments passed to the reader/writer.

    Returns:
        Either a DatasetReader for read mode or DatasetWriter for write mode.

    Raises:
        MRIOError: If an invalid mode is provided.
        ValueError: If the file path is invalid or of wrong type.
        RasterioError: If there's an error opening the file.

    Examples:
        >>> with mrio.open("example.tif") as src:
        ...     data = src.read()
        ...     profile = src.profile

        >>> with mrio.open("output.tif", mode="w", **profile) as dst:
        ...     dst.write(data)
    """

    if mode not in Mode.VALID_MODES:
        msg = f"Invalid mode '{mode}'. Use '{Mode.READ}' for read or '{Mode.WRITE}' for write."
        raise MRIOError(msg)

    if mode == Mode.WRITE:
        return DatasetWriter(file_path, **kwargs)
    return read(file_path, engine=read_engine, env_options=read_env_options, **kwargs)


def read(
    file_path: PathLike,
    engine: str = "numpy",
    env_options: Literal["mrio", "default"] | dict[str, str] = "mrio",
    **kwargs: Any,
) -> DatasetReader:
    """Convenience function to read a dataset file.

    This is equivalent to calling open() with read mode.

    Args:
        file_path: Path to the dataset file.
        engine: Backend engine to use for data processing.
        env_options: Configuration settings for the rasterio environment.
            By default, uses MRIO configuration settings. Check MRIOConfig for details.
            Other available options are 'default' and a custom dictionary.
        **kwargs: Additional keyword arguments passed to the reader.

    Returns:
        DatasetReader instance configured for the input file.

    Examples:
        >>> reader = mrio.read("example.tif")
        >>> data = reader.read()

    """
    if not isinstance(file_path, (str, Path)):
        msg = "file_path must be a string or Path object"
        raise TypeError(msg)

    with MRIOConfig.get_env(env_options):
        return DatasetReader(file_path, engine=engine, **kwargs)


def write(file_path: PathLike, data: DataArray, **kwargs: Any) -> DatasetWriter:
    """Convenience function to write data to a dataset file.

    This is equivalent to calling open() with write mode and then writing the data.

    Args:
        file_path: Path to the output file.
        data: Data to write. Can be numpy array or xarray DataArray.
        **kwargs: Additional keyword arguments passed to the writer.

    Returns:
        DatasetWriter instance after writing the data.

    Raises:
        TypeError: If the writer is not a DatasetWriter instance.

    Examples:
        >>> data = np.random.rand(100, 100)
        >>> writer = mrio.write("output.tif", data, **profile)

    """
    writer = DatasetWriter(file_path, **kwargs)
    if not isinstance(writer, DatasetWriter):
        msg = "Expected DatasetWriter instance"
        raise TypeError(msg)

    writer.write(data)
    return writer


# Export public symbols
__all__ = [
    "CRS",
    "Affine",
    "Collection",
    "DefaultGTiffProfile",
    "Env",
    "Profile",
    "Window",
    "__version__",
    "band",
    "crs",
    "from_bounds",
    "from_gcps",
    "from_origin",
    "io",
    "is_mcog",
    "is_tcog",
    "open",
    "profiles",
    "read",
    "stack_temporal",
    "transform",
    "unstack_temporal",
    "windows",
    "write",
]
