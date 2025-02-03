"""MRIO Metadata Module.

This module provides data structures for handling metadata in the MRIO (Multi-Resolution I/O) format.
It includes classes for coordinates, metadata fields, and write parameters with validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

from mrio import errors

if TYPE_CHECKING:
    from mrio.type_definitions import JSONValue


@dataclass
class Coordinates:
    """Immutable coordinate data structure for MRIO metadata.

    This class represents coordinate values for each dimension in a multi-dimensional array.
    The coordinate values are stored as lists and are immutable once created.

    Attributes:
        values: Dictionary mapping dimension names to their coordinate values

    Example:
        >>> coords = Coordinates({
        ...     'time': [1, 2, 3],
        ...     'bands': ['red', 'green', 'blue']
        ... })

    """

    values: dict[str, list[JSONValue]]

    def __post_init__(self) -> None:
        """Validate coordinates after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate that all coordinate values are lists.

        Raises:
            ValidationError: If any coordinate value is not a list

        """
        if not all(isinstance(v, list) for v in self.values.values()):
            raise errors.FieldsCoordinateListError()


@dataclass
class MRIOFields:
    """Metadata fields for MRIO format.

    This class handles the pattern specification and coordinates for multi-dimensional arrays.
    It validates the pattern format and ensures consistency between pattern variables and coordinates.

    Attributes:
        pattern: String specifying the dimension arrangement (e.g., "c h w -> (c) h w")
        coordinates: Coordinates object containing values for each dimension
        attributes: Optional dictionary of additional metadata attributes

    Example:
        >>> fields = MRIOFields(
        ...     pattern="time band h w -> (time band) h w",
        ...     coordinates=Coordinates({'time': [1, 2], 'band': ['red', 'green']}),
        ...     attributes={'scale': 0.01}
        ... )

    """

    pattern: str
    coordinates: Coordinates
    attributes: dict[str, JSONValue] | None = None

    # Class variable for pattern validation
    _PATTERN_REGEX: ClassVar[str] = r"^([\w\s]+)\s*->\s*\(([\w\s]+)\)\s+([\w\s]+)$"
    _pattern_match: Pattern[str] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Validate and parse the pattern after initialization."""
        self._validate_pattern()
        self._validate_coordinates()
        self._parse_pattern()

        # Wrap raw dictionary in Coordinates
        if isinstance(self.coordinates, dict):
            _ = Coordinates(self.coordinates)

        # ingle-pass coordinate validation
        mismatch = set(self.coordinates.keys()) ^ set(self._in_parentheses)
        if mismatch:
            raise errors.FieldsCoordinateMDPattern()

    def _validate_pattern(self) -> None:
        """Validate the pattern string format.

        Raises:
            ValidationError: If pattern format is invalid

        """
        if " -> " not in self.pattern:
            raise errors.FieldsPatternSeparatorError()

        self._pattern_match = re.match(self._PATTERN_REGEX, self.pattern)
        if not self._pattern_match:
            raise errors.FieldsPatternError()

    def _validate_coordinates(self) -> None:
        """Validate coordinates are not empty.

        Raises:
            ValidationError: If coordinates dictionary is empty

        """
        if not self.coordinates.values:
            raise errors.FieldsEmptyCoordinatesError()

    def _parse_pattern(self) -> None:
        """Parse and validate pattern components.

        Raises:
            ValueError: If variables before and after arrow don't match

        """
        if not self._pattern_match:
            raise errors.FieldsPatternParseError()

        self._before_arrow = self._pattern_match.group(1).split()
        self._in_parentheses = self._pattern_match.group(2).split()
        self._after_parentheses = self._pattern_match.group(3).split()
        self._inv_pattern = " -> ".join(map(str.strip, self.pattern.split("->")[::-1]))

        post_arrow_vars = set(self._in_parentheses + self._after_parentheses)
        if post_arrow_vars != set(self._before_arrow):
            raise errors.FieldsCoordinateMismatchError()


class WriteParamsDefaults(TypedDict, total=False):
    """Type definition for write parameter defaults."""

    driver: str
    dtype: str
    compress: str
    interleave: str
    tiled: bool
    blockxsize: int
    blockysize: int
    nodata: float | None
    count: int
    width: int | None
    height: int | None
    crs: Any
    transform: Any
    md_pattern: str | None
    md_coordinates: dict[str, list[JSONValue]] | None
    md_attributes: dict[str, JSONValue]


@dataclass
class WriteParams:
    """Configuration parameters for writing raster data in MRIO format.

    This class handles both optional and mandatory parameters for writing
    multi-dimensional raster data, with validation of required fields.

    Attributes:
        DEFAULTS: Default values for optional parameters
        params: User-provided parameter values
        merged_params: Combined default and user parameters (created after init)

    Example:
        >>> write_params = WriteParams({
        ...     'width': 1000,
        ...     'height': 800,
        ...     'crs': 'EPSG:4326',
        ...     'md:pattern': 'time band h w -> (time band) h w',
        ...     'md:coordinates': {'time': [1, 2], 'band': ['red', 'green']}
        ... })

    """

    DEFAULTS: ClassVar[WriteParamsDefaults] = {
        # Optional parameters with defaults
        "driver": "COG",
        "compress": "deflate",
        "interleave": "TILE",
        "bigtiff": "YES",
        "blocksize": 64,
        "nodata": None,
        "md:attributes": {},
        # Mandatory parameters (None indicates they must be provided)
        "crs": None,
        "transform": None,
        "md:pattern": None,
        "md:coordinates": None,
        # These parameters are estimated automatically
        "count": 1,
        "width": None,
        "height": None,
        "dtype": None,
    }

    params: dict[str, Any] = field(default_factory=dict)
    merged_params: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Merge and validate parameters after initialization."""
        self.merged_params = {**self.DEFAULTS, **self.params}
        self._validate_mandatory_fields()

    def _validate_mandatory_fields(self) -> None:
        """Validate that all mandatory fields are provided.

        Raises:
            ValidationError: If any mandatory field is missing

        """
        # Check for missing mandatory fields
        mandatory_fields = ["crs", "transform", "md:pattern", "md:coordinates"]
        missing_fields = [
            field for field in mandatory_fields if self.merged_params.get(field) is None
        ]

        if missing_fields:
            raise errors.FieldsMissingFieldsError(missing_fields)

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to dictionary format.

        Returns:
            Dictionary of all parameters with their values

        """
        return self.merged_params
