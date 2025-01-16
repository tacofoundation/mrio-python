"""MRIO Types Module.

This module defines type aliases and custom types used throughout the MRIO package.
It provides type hints for JSON data, array operations, file paths, and dimension
handling to ensure type safety across the codebase.

Example:
    >>> from mrio.types import JSONValue, PathLike, DataArray
    >>> def save_metadata(path: PathLike, data: JSONValue) -> None:
    ...     pass

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Union

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr


# Core data types
# --------------
ArrayLike = TypeVar("ArrayLike")  # Generic array type
DataArray = Union["np.ndarray", "xr.DataArray"]  # Concrete array types

# JSON and metadata types
# ----------------------
JSONValue = Union[
    str,  # String values
    int,  # Integer values
    float,  # Floating point values
    bool,  # Boolean values
    list,  # List of JSON values
    dict,  # Dictionary of JSON values
    None,  # None/null values
]

MetadataDict = dict[str, Any]  # Generic metadata dictionary

# Path handling
# ------------
PathLike = Union[str, Path]  # File path types

# Coordinate and dimension types
# ----------------------------
Coordinates = dict[str, list[Any]]  # Coordinate values by dimension
CoordinatesLen = dict[str, int]  # Length of each coordinate dimension

# Slice and indexing types
# -----------------------
Slice = Union[
    slice,  # Python slice object
    int,  # Single index
    list,  # List of indices
    tuple,  # Tuple of indices
]

DimensionFilter = list[Slice]  # List of slice operations

# Transformer types
# ---------------
FilterCondition = Union[
    slice,  # Single slice
    int,  # Single integer
    list[int],  # List of integers
    tuple[int, ...],  # Tuple of integers
]

DimKey = Union[
    slice,  # Single slice
    int,  # Single integer
    tuple[Any, ...],  # Tuple of any values
    list[int],  # List of integers
    Any,  # Any other type
]

SliceTuple = tuple[
    Union[slice, tuple[slice, ...]],
    ...,  # Single slice  # Tuple of slices
]

IntList = list[int]  # List of integers
NestedKey = tuple[IntList, int]  # (list of integers, dimension) pair
