from __future__ import annotations

from types import EllipsisType
from typing import Any

from mrio import errors
from mrio.type_definitions import SliceTuple


class SliceTransformer:
    """Universal slice transformer that handles all valid input patterns including Ellipsis.

    Features:
    - Converts integers to single-element slices (e.g., 5 → slice(5, 6))
    - Handles Ellipsis (...) to fill missing dimensions
    - Supports list indexing for non-spatial dimensions
    - Validates slice bounds and indexing rules

    Example:
        # Create a transformer for a 4-dimensional dataset (e.g., [Time, Bands, Height, Width])
        >>> transformer = SliceTransformer(ndim=4)

        # Example 1: Simple integer index for first dimension
        >>> key = 3
        >>> slices = transformer.transform(key)
        >>> print(slices)  # (slice(3, 4), slice(None), slice(None), slice(None))

        # Example 2: Slice in second dimension with Ellipsis
        >>> key = (slice(0, 5), ..., 100)
        >>> slices = transformer.transform(key)
        >>> print(slices)  # (slice(0, 5), slice(None), slice(None), slice(100, 101))

        # Example 3: List indices for non-spatial dimensions
        >>> key = ([1, 3, 5], 10, Ellipsis)
        >>> slices = transformer.transform(key)
        >>> print(slices)  # ([1, 3, 5], slice(10, 11), slice(None), slice(None))

        # Example 4: Ellipsis in middle of tuple
        >>> key = (2, ..., 150)
        >>> slices = transformer.transform(key)
        >>> print(slices)  # (slice(2, 3), slice(None), slice(None), slice(150, 151))

        # Example 5: Full explicit indexing
        >>> key = (slice(0, 10), [2, 4], 100, 200)
        >>> slices = transformer.transform(key)
        >>> print(slices)  # (slice(0, 10), [2, 4], slice(100, 101), slice(200, 201))
    """

    def __init__(self, ndim: int) -> None:
        """Initialize transformer for arrays with specified number of dimensions.

        Args:
            ndim: Number of dimensions (must be ≥1)

        Raises:
            ValueError: If ndim is not a positive integer
        """
        if not isinstance(ndim, int) or ndim < 1:
            raise ValueError("ndim must be a positive integer")
        self.ndim = ndim

    def _validate_slice(self, s: slice) -> None:
        """Validate slice start/stop values and ordering.

        Ensures:
            - start ≤ stop when both are specified
            - Handles None values appropriately (open-ended slices)

        Raises:
            ValueError: If start > stop
        """
        if s.start is not None and s.stop is not None and s.start > s.stop:
            raise ValueError(f"Invalid slice: start ({s.start}) cannot be greater than stop ({s.stop})")

    def _make_slice(
        self, val: int | slice | list[int] | EllipsisType, dim_idx: int
    ) -> slice | list[int]:
        """Convert input value to appropriate slice format for given dimension.

        Args:
            val: Input value to convert
            dim_idx: Current dimension index (0-based)

        Returns:
            Standardized slice object or list of indices

        Raises:
            TypeError: For invalid types or list in spatial dimensions
        """
        # Last two dimensions are considered spatial (e.g., height/width)
        is_spatial_dim = dim_idx >= self.ndim - 2

        if isinstance(val, slice):
            self._validate_slice(val)
            return val
        elif isinstance(val, int):
            # Convert scalar index to single-element slice (preserves dimension)
            return slice(val, val + 1)
        elif isinstance(val, list):
            if is_spatial_dim:
                raise TypeError("List indexing is only supported for non-spatial dimensions")
            return val
        elif val is Ellipsis:
            return val
        raise TypeError(f"Cannot convert type {type(val)} to slice")

    def transform(self, key: Any) -> SliceTuple:
        """Transform various index formats into standardized slice tuple.

        Handles:
        - Scalar indices
        - Slice objects
        - List indices (non-spatial only)
        - Ellipsis expansion
        - Mixed tuple patterns

        Args:
            key: Input indexing expression

        Returns:
            Tuple of slices/list-indices covering all dimensions

        Raises:
            IndexError: For too many dimensions or multiple Ellipsis
            TypeError: For unsupported input types
        """
        # Handle primitive indexing patterns
        if isinstance(key, (int, slice, list)):
            # Single-value indexing affects first dimension
            result = [slice(None)] * self.ndim
            result[0] = self._make_slice(key, 0)
            return tuple(result)

        # Handle full-array Ellipsis
        if key is Ellipsis:
            return tuple([slice(None)] * self.ndim)

        # Process tuple patterns (most complex case)
        if isinstance(key, tuple):
            ellipsis_count = key.count(Ellipsis)
            if ellipsis_count > 1:
                raise errors.SliceEllipsisError()

            if ellipsis_count == 0:
                # Direct mapping without Ellipsis expansion
                result = [slice(None)] * self.ndim
                for i, val in enumerate(key):
                    if i >= self.ndim:
                        raise errors.SliceEllipsisTooManyIndicesError(self.ndim)
                    result[i] = self._make_slice(val, i)
                return tuple(result)

            # Handle Ellipsis expansion
            ellipsis_pos = key.index(Ellipsis)
            num_explicit = len(key) - 1  # Exclude Ellipsis itself
            num_ellipsis_dims = self.ndim - num_explicit

            # Build result in three parts:
            result = []

            # 1. Elements before Ellipsis
            for dim_idx, val in enumerate(key[:ellipsis_pos]):
                result.append(self._make_slice(val, dim_idx))

            # 2. Fill Ellipsis dimensions with full slices
            result.extend([slice(None)] * num_ellipsis_dims)

            # 3. Elements after Ellipsis with adjusted dimension indices
            for offset, val in enumerate(key[ellipsis_pos+1:]):
                # Calculate absolute dimension index accounting for Ellipsis expansion
                abs_dim_idx = ellipsis_pos + num_ellipsis_dims + offset
                result.append(self._make_slice(val, abs_dim_idx))

            return tuple(result)

        raise errors.SliceUnsupportedTypeError(key)
