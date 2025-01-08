import re
from typing import Dict, Optional, Union

from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

# Allowed JSON-like value types
JSONValue = Union[str, int, float, bool, list, dict, None]


class MRIOFields(BaseModel):
    pattern: str
    coordinates: Dict[str, JSONValue]
    attributes: Optional[Dict[str, JSONValue]] = None

    # Internal fields for parsed pattern components
    _before_arrow: Optional[list[str]] = None
    _in_parentheses: Optional[list[str]] = None
    _after_parentheses: Optional[list[str]] = None
    _inv_pattern: Optional[str] = None

    @field_validator("pattern")
    def validate_pattern(cls, value: str) -> str:
        """
        Validates the pattern string and extracts its components.

        Args:
            value (str): The pattern string to validate.

        Returns:
            str: The validated pattern string.

        Raises:
            ValueError: If the pattern string does not match the required format.
        """
        # Regular expression to parse the format: "<vars> -> (<vars>) <vars>"
        re_pattern = re.compile(r"^([\w\s]+)\s*->\s*\(([\w\s]+)\)\s+([\w\s]+)$")
        match = re_pattern.match(value)

        if not match:
            raise ValueError(
                "pattern must match the format '<vars> -> (<vars>) <vars>'."
            )

        # Extract components and store them for later validation
        cls._before_arrow = match.group(1).split()
        cls._in_parentheses = match.group(2).split()
        cls._after_parentheses = match.group(3).split()
        cls._inv_pattern = " -> ".join(map(str.strip, value.split("->")[::-1]))

        # Ensure variables after '->' match variables before it
        post_arrow_vars = cls._in_parentheses + cls._after_parentheses
        if set(post_arrow_vars) != set(cls._before_arrow):
            raise ValueError("Variables after '->' must match variables before it.")

        return value

    @field_validator("coordinates")
    def validate_coordinates(
        cls, value: Optional[Dict[str, JSONValue]]
    ) -> Optional[Dict[str, JSONValue]]:
        """
        Validates the coordinates dictionary.

        Args:
            value (Optional[Dict[str, JSONValue]]): The coordinates dictionary to validate.

        Returns:
            Optional[Dict[str, JSONValue]]: The validated coordinates dictionary.

        Raises:
            ValueError: If nested dictionaries are found in the coordinates.
        """
        if any(isinstance(v, dict) for v in value.values()):
            raise ValueError("Nested dictionaries are not allowed in 'coordinates'.")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """
        Validates consistency between the pattern and coordinates fields.

        Returns:
            MRIOFields: The validated instance.

        Raises:
            ValueError: If the coordinates keys do not match the variables in parentheses.
        """
        if set(self.coordinates.keys()) != set(self._in_parentheses):
            raise ValueError(
                f"The keys in 'coordinates' {list(self.coordinates.keys())} "
                f"must match the variables in parentheses {self._in_parentheses}."
            )
        return self