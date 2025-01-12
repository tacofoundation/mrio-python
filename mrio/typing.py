from typing import List, TypeVar, Union

JSONValue = Union[str, int, float, bool, list, dict, None]
Slice = Union[slice, int, list, tuple]
ArrayLike = TypeVar("ArrayLike")
DimensionFilter = List[Slice]
