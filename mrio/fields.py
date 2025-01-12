import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from mrio.errors import ValidationError
from mrio.typing import JSONValue

import copy

@dataclass(frozen=True)
class Coordinates:
    """Immutable coordinate data structure."""

    values: Dict[str, List[JSONValue]]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not all(isinstance(v, list) for v in self.values.values()):
            raise ValidationError("All coordinate values must be lists")


@dataclass
class MRIOFields:
    """Metadata fields for MRIO format."""

    pattern: str
    coordinates: Coordinates
    attributes: Optional[Dict[str, JSONValue]] = None

    def __post_init__(self):
        self._validate_pattern()
        self._validate_coordinates()
        self._parse_pattern()

    def _validate_pattern(self):
        if " -> " not in self.pattern:
            raise ValidationError("Pattern must contain ' -> ' separator")

        pattern_regex = r"^([\w\s]+)\s*->\s*\(([\w\s]+)\)\s+([\w\s]+)$"
        self._pattern_match = re.match(pattern_regex, self.pattern)
        if not self._pattern_match:
            raise ValidationError("Invalid pattern format")

    def _validate_coordinates(self):
        if not self.coordinates.values:
            raise ValidationError("Coordinates cannot be empty")

    def _parse_pattern(self):
        # Extract components and store them for later validation
        self._before_arrow = self._pattern_match.group(1).split()
        self._in_parentheses = self._pattern_match.group(2).split()
        self._after_parentheses = self._pattern_match.group(3).split()
        self._inv_pattern = " -> ".join(map(str.strip, self.pattern.split("->")[::-1]))

        # Ensure variables after '->' match variables before it
        post_arrow_vars = self._in_parentheses + self._after_parentheses
        if set(post_arrow_vars) != set(self._before_arrow):
            raise ValueError("Variables after '->' must match variables before it.")


@dataclass
class WriteParams:
    """
    Represents configuration parameters for writing raster data.
    """
    DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        "driver": "GTiff",  # OPTIONAL: Default to GeoTIFF format
        "dtype": "float32",  # OPTIONAL: Default data type for raster values
        "compress": "lzw",  # OPTIONAL: Default compression method
        "interleave": "band",  # OPTIONAL: Default interleaving (band or pixel)
        "tiled": True,  # OPTIONAL: Enable tiling for better performance
        "blockxsize": 512,  # OPTIONAL: Tile width
        "blockysize": 512,  # OPTIONAL: Tile height
        "nodata": None,  # OPTIONAL: NoData value
        "count": 1,  # AUTOMATIC: The package automatically sets the number of bands.
        "width": None,  # MANDATORY: Width of the raster in pixels.
        "height": None,  # MANDATORY: Height of the raster in pixels.
        "crs": None,  # MANDATORY: Coordinate Reference System to be set by the user.
        "transform": None,  # MANDATORY: Affine transform to be set by the user.
        "md:pattern": None,  # MANDATORY: Pattern for multi-dimensional data.
        "md:coordinates": None,  # MANDATORY: Coordinates for each dimension.
        "md:attributes": {},  # OPTIONAL: Additional attributes to include in the file.
    })
    params: Dict[str, Any] = field(default_factory=dict)
    merged_params: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        """
        Merges default parameters with user-provided ones and validates mandatory fields.
        """
        # Make sure DEFAULTS is immutable across instances
        self.merged_params = {**copy.deepcopy(self.DEFAULTS), **self.params}
        self._validate_mandatory_fields()

    def _validate_mandatory_fields(self):
        """
        Validates that all mandatory fields are provided.
        Raises a ValidationError if any mandatory field is missing.
        """
        mandatory_fields = ["width", "height", "crs", "transform", "md:pattern", "md:coordinates"]
        missing_fields = [field for field in mandatory_fields if self.merged_params.get(field) is None]
        if missing_fields:
            raise ValidationError(f"Mandatory fields missing: {', '.join(missing_fields)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the merged parameters as a dictionary.
        """
        return self.merged_params