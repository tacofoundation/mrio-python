from typing import Dict, Optional, Union
from typing_extensions import Self
from pydantic import BaseModel, field_validator, model_validator
import re

# Allowed JSON-like value types
JSONValue = Union[str, int, float, bool, list, dict, None]

class MRIOFields(BaseModel):
    strategy: str
    coordinates: Dict[str, JSONValue]
    attributes: Optional[Dict[str, JSONValue]] = None

    # Internal fields for parsed strategy components
    _before_arrow: Optional[list[str]] = None
    _in_parentheses: Optional[list[str]] = None
    _after_parentheses: Optional[list[str]] = None

    @field_validator("strategy")
    def validate_strategy(cls, value: str) -> str:
        """
        Validates the strategy string and extracts its components.

        Args:
            value (str): The strategy string to validate.

        Returns:
            str: The validated strategy string.

        Raises:
            ValueError: If the strategy string does not match the required format.
        """
        # Regular expression to parse the format: "<vars> -> (<vars>) <vars>"
        pattern = r"^([\w\s]+)\s*->\s*\(([\w\s]+)\)\s+([\w\s]+)$"
        match = re.match(pattern, value)

        if not match:
            raise ValueError("Strategy must match the format '<vars> -> (<vars>) <vars>'.")

        # Extract components and store them for later validation
        cls._before_arrow = match.group(1).split()
        cls._in_parentheses = match.group(2).split()
        cls._after_parentheses = match.group(3).split()

        # Ensure variables after '->' match variables before it
        post_arrow_vars = cls._in_parentheses + cls._after_parentheses
        if set(post_arrow_vars) != set(cls._before_arrow):
            raise ValueError("Variables after '->' must match variables before it.")

        return value

    @field_validator("coordinates")
    def validate_coordinates(cls, value: Optional[Dict[str, JSONValue]]) -> Optional[Dict[str, JSONValue]]:
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
        Validates consistency between the strategy and coordinates fields.

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

