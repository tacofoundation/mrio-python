"""
MRIO Validators Module

This module provides validation functions for Multi-dimensional GeoTIFF (MGeoTIFF) and
Temporal GeoTIFF (TGeoTIFF) files. It checks the presence and validity of required
metadata fields and attributes.

Example:
    >>> from mrio.validators import is_mgeotiff, is_tgeotiff
    >>> if is_mgeotiff("example.tif"):
    ...     print("Valid MGeoTIFF file")
"""

from __future__ import annotations

import json
import warnings
from typing import List, Optional, Set

import rasterio as rio
from rasterio.errors import RasterioError

from mrio.types import JSONValue, MetadataDict, PathLike

# Constants
MD_METADATA_KEY = "MD_METADATA"
MGEOTIFF_REQUIRED_FIELDS: Set[str] = {"md:pattern", "md:coordinates"}
TGEOTIFF_REQUIRED_FIELDS: Set[str] = MGEOTIFF_REQUIRED_FIELDS
TGEOTIFF_REQUIRED_ATTRS: Set[str] = {"md:time_start", "md:id"}


def check_metadata(
    path: PathLike,
    required_fields: List[str],
    required_attributes: Optional[List[str]] = None,
    strict: bool = True,
) -> bool:
    """
    Validate GeoTIFF metadata against required fields and attributes.

    Args:
        path: Path to the GeoTIFF file
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

        return True

    except Exception as e:
        _handle_error(f"Validation failed: {str(e)}", strict)
        return False


def is_mgeotiff(path: PathLike, strict: bool = False) -> bool:
    """
    Check if a file is a valid Multi-dimensional GeoTIFF (MGeoTIFF).

    MGeoTIFF files must contain md:pattern and md:coordinates in their metadata.

    Args:
        path: Path to the GeoTIFF file
        strict: If True, raises exceptions for validation failures

    Returns:
        True if file is a valid MGeoTIFF

    Example:
        >>> is_mgeotiff("multidim.tif")
        True
    """
    return check_metadata(path, list(MGEOTIFF_REQUIRED_FIELDS), strict=strict)


def is_tgeotiff(path: PathLike, strict: bool = False) -> bool:
    """
    Check if a file is a valid Temporal GeoTIFF (TGeoTIFF).

    TGeoTIFF files must contain both MGeoTIFF fields and temporal attributes.

    Args:
        path: Path to the GeoTIFF file
        strict: If True, raises exceptions for validation failures

    Returns:
        True if file is a valid TGeoTIFF

    Example:
        >>> is_tgeotiff("temporal.tif")
        True
    """
    return check_metadata(
        path,
        list(TGEOTIFF_REQUIRED_FIELDS),
        list(TGEOTIFF_REQUIRED_ATTRS),
        strict=strict,
    )


def _load_metadata(path: PathLike) -> Optional[MetadataDict]:
    """Load and parse metadata from a GeoTIFF file."""
    try:
        with rio.open(path) as src:
            tags = src.tags()
            if MD_METADATA_KEY not in tags:
                return None
            return json.loads(tags[MD_METADATA_KEY])
    except RasterioError as e:
        raise ValueError(f"Failed to open file: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid metadata JSON: {e}")


def _get_missing_fields(metadata: MetadataDict, required_fields: List[str]) -> Set[str]:
    """Get set of missing required fields from metadata."""
    return set(required_fields) - set(metadata.keys())


def _get_missing_attributes(
    metadata: MetadataDict, required_attrs: List[str]
) -> Set[str]:
    """Get set of missing required attributes from metadata."""
    attributes = metadata.get("md:attributes", {})
    return set(required_attrs) - set(attributes.keys())


def _handle_error(message: str, strict: bool) -> None:
    """Handle validation errors based on strict mode."""
    if strict:
        raise ValueError(message)
    warnings.warn(message)
