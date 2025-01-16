"""
MRIO Slice Transformer Module

This module provides functionality for transforming various types of array indices
into a standardized tuple of slices, supporting advanced indexing patterns for
multi-dimensional arrays.
"""

from __future__ import annotations

from typing import Any, cast

from mrio.types import DimKey, NestedKey, SliceTuple


class SliceTransformer:
    """
    Transform array indices into standardized slice tuples.

    This class handles various types of array indices and transforms them into
    a consistent format of slice tuples, supporting advanced indexing patterns
    like nested slices and ellipsis.

    Attributes:
        ndim: Number of dimensions in the target array

    Example:
        >>> transformer = SliceTransformer(ndim=4)
        >>> # Single integer index
        >>> transformer.transform(2)
        (slice(2, 3, None), slice(None), slice(None), slice(None))
        >>> # Nested slice with dimension
        >>> transformer.transform(([1, 3], 1))
        ((slice(1, 2), slice(3, 4)), slice(1, 2), slice(None), slice(None))
    """

    __slots__ = ("ndim",)

    def __init__(self, ndim: int) -> None:
        """
        Initialize the SliceTransformer.

        Args:
            ndim: Number of dimensions in the target array

        Raises:
            ValueError: If ndim is not a positive integer
        """
        if not isinstance(ndim, int) or ndim < 1:
            raise ValueError("ndim must be a positive integer")
        self.ndim = ndim

    def transform(self, key: DimKey, dim: int | None = None) -> SliceTuple:
        """
        Transform input key into a tuple of slices.

        This method handles various input types and converts them into a standardized
        tuple of slices that matches the specified number of dimensions.

        Args:
            key: Input key to transform. Can be:
                - slice: A single slice object
                - int: A single integer
                - List[int]: A list of integers (requires dim parameter)
                - Tuple[List[int], int]: Special case that creates a nested structure
                - Tuple[Any, ...]: A tuple of slices/integers/ellipsis
            dim: Optional dimension for list inputs (ignored for tuple inputs)

        Returns:
            Tuple of slices matching the number of dimensions

        Raises:
            ValueError: If dimension is invalid or for invalid input combinations
            TypeError: If key type is not supported

        Examples:
            >>> transformer = SliceTransformer(ndim=3)
            >>> # Single integer becomes a slice
            >>> transformer.transform(1)
            (slice(1, 2, None), slice(None), slice(None))
            >>> # List with dimension creates nested structure
            >>> transformer.transform([1, 2], dim=0)
            ((slice(1, 2), slice(2, 3)), slice(None), slice(None))
        """
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], list):
            return self._handle_nested_case(cast(NestedKey, key))

        if dim is not None:
            return self._handle_dim_case(key, dim)

        if isinstance(key, int):
            return self._handle_int_case(key)

        if isinstance(key, slice):
            return self._handle_slice_case(key)

        if isinstance(key, tuple):
            return self._handle_tuple_case(key)

        raise TypeError(f"Unsupported key type: {type(key)}")

    def _handle_nested_case(self, key: NestedKey) -> SliceTuple:
        """Handle the special case of (list, dim) tuple input."""
        lst, dim = key
        if not (0 <= dim < self.ndim):
            raise ValueError(f"Invalid dimension {dim} for ndim {self.ndim}")
        if not all(isinstance(item, int) for item in lst):
            raise ValueError(f"List must contain only integers. Got: {lst}")

        nested_slices = tuple(slice(item, item + 1) for item in lst)
        result = [nested_slices, slice(dim, dim + 1)]
        result.extend(slice(None) for _ in range(self.ndim - 2))
        return tuple(result)

    def _handle_dim_case(self, key: DimKey, dim: int) -> SliceTuple:
        """Handle the case where a dimension is specified."""
        if not (0 <= dim < self.ndim):
            raise ValueError(f"Invalid dimension {dim} for ndim {self.ndim}")
        if not isinstance(key, list) or not all(isinstance(item, int) for item in key):
            raise ValueError(
                f"When dim is specified, key must be a list of integers. Got: {key}"
            )

        result = [slice(None)] * self.ndim
        result[dim] = tuple(slice(item, item + 1) for item in key)
        return tuple(result)

    def _handle_int_case(self, key: int) -> SliceTuple:
        """Handle single integer input."""
        result = [slice(key, key + 1)]
        result.extend(slice(None) for _ in range(self.ndim - 1))
        return tuple(result)

    def _handle_slice_case(self, key: slice) -> SliceTuple:
        """Handle single slice input."""
        result = [key]
        result.extend(slice(None) for _ in range(self.ndim - 1))
        return tuple(result)

    def _handle_tuple_case(self, key: tuple[Any, ...]) -> SliceTuple:
        """Handle tuple input with possible ellipsis."""
        result = []
        found_ellipsis = False

        for item in key:
            if item is Ellipsis:
                if found_ellipsis:
                    raise ValueError("Multiple ellipsis found in key")
                found_ellipsis = True
                ellipsis_fill = self.ndim - len(key) + 1
                result.extend([slice(None)] * ellipsis_fill)
            elif isinstance(item, int):
                result.append(slice(item, item + 1))
            elif isinstance(item, slice):
                result.append(item)
            else:
                raise TypeError(f"Unsupported key type: {type(item)}")

        # Fill remaining dimensions with slice(None)
        if len(result) < self.ndim:
            result.extend([slice(None)] * (self.ndim - len(result)))

        return tuple(result[: self.ndim])
