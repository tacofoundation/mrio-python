"""mrio Validators Module.

This module provides validation functions for multi-dimensional COG (mCOG) and
temporal COG (tCOG) files. It checks the presence and validity of required
metadata fields and attributes.

Example:
    >>> from mrio.validators import is_mcog, is_tcog
    >>> if is_mcog("example.tif"):
    ...     print("Valid mcog file")
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import rasterio as rio

from mrio import errors as mrio_errors

if TYPE_CHECKING:
    from mrio.type_definitions import MetadataDict, PathLike

# Constants
MD_METADATA_KEY = "MD_METADATA"
MCOG_REQUIRED_FIELDS: set[str] = {"md:pattern", "md:coordinates"}
TCOG_REQUIRED_FIELDS: set[str] = MCOG_REQUIRED_FIELDS
TCOG_REQUIRED_ATTRS: set[str] = {"md:time_start", "md:id"}


def check_metadata(
    path: PathLike,
    required_fields: list[str],
    required_attributes: list[str] | None = None,
    strict: bool = True,
) -> bool:
    """Validate COG metadata against required fields and attributes.

    Args:
        path: Path to the COG file
        required_fields: List of fields that must be present in top-level metadata
        required_attributes: Optional list of attributes required in md:attributes
        strict: If True, raises exceptions for validation failures; if False, returns False

    Returns:
        True if metadata is valid and contains all required fields/attributes

    Raises:
        ValueError: If metadata is invalid and strict=True
        RasterioError: If file cannot be opened and strict=True
        json.JSONDecodeError: If metadata JSON is invalid and strict=True

    Example:
        >>> check_metadata("data.tif", ["md:pattern"], ["time_start"])
        True
    """
    try:
        metadata = _load_metadata(path)
        if metadata is None:
            _handle_error("MD_METADATA not found", strict)
            return False

        # Validate required fields
        missing_fields = _get_missing_fields(metadata, required_fields)
        if missing_fields:
            _handle_error(f"Missing required fields: {missing_fields}", strict)
            return False

        # Validate required attributes if specified
        if required_attributes:
            missing_attrs = _get_missing_attributes(metadata, required_attributes)
            if missing_attrs:
                _handle_error(f"Missing required attributes: {missing_attrs}", strict)
                return False
        else:
            return True

    except Exception as e:
        _handle_error(f"Validation failed: {e!s}", strict)
        return False


def is_mcog(path: PathLike, strict: bool = False) -> bool:
    """Check if a file is a valid multi-dimensional COG (mCOG).

    mCOG files must contain md:pattern and md:coordinates in their metadata.

    Args:
        path: Path to the COG file
        strict: If True, raises exceptions for validation failures

    Returns:
        True if file is a valid mCOG

    Example:
        >>> is_mcog("multidim.tif")
        True

    """
    return check_metadata(path, list(MCOG_REQUIRED_FIELDS), strict=strict)


def is_tcog(path: PathLike, strict: bool = False) -> bool:
    """Check if a file is a valid temporal COG (tCOG).

    tCOG files must contain both mCOG fields and temporal attributes.

    Args:
        path: Path to the COG file
        strict: If True, raises exceptions for validation failures

    Returns:
        True if file is a valid tCOG

    Example:
        >>> is_tcog("temporal.tif")
        True

    """
    return check_metadata(
        path,
        list(TCOG_REQUIRED_FIELDS),
        list(TCOG_REQUIRED_ATTRS),
        strict=strict,
    )


def _load_metadata(path: PathLike) -> MetadataDict | None:
    """Load and parse metadata from a COG file."""
    try:
        with rio.open(path) as src:
            tags = src.tags()
            if MD_METADATA_KEY not in tags:
                return None
            return json.loads(tags[MD_METADATA_KEY])
    except rio.RasterioIOError as e:
        raise mrio_errors.ValidatorsFailedToOpenError(path, e) from e
    except json.JSONDecodeError as e:
        raise mrio_errors.ValidatorsInvalidMetadataError(e) from e


def _get_missing_fields(metadata: MetadataDict, required_fields: list[str]) -> set[str]:
    """Get set of missing required fields from metadata."""
    return set(required_fields) - set(metadata.keys())


def _get_missing_attributes(
    metadata: MetadataDict, required_attrs: list[str]
) -> set[str]:
    """Get set of missing required attributes from metadata."""
    attributes = metadata.get("md:attributes", {})
    return set(required_attrs) - set(attributes.keys())


def _handle_error(message: str, strict: bool) -> None:
    """Handle validation errors based on strict mode."""
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)
