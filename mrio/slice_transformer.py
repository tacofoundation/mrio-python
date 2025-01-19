from __future__ import annotations
from typing import Any, List, Tuple, Union
from mrio.types import SliceTuple


class SliceTransformer:
    """Universal slice transformer that handles all valid input patterns including Ellipsis."""

    def __init__(self, ndim: int) -> None:
        if not isinstance(ndim, int) or ndim < 1:
            msg = "ndim must be a positive integer"
            raise ValueError(msg)
        self.ndim = ndim

    def _validate_slice(self, s: slice) -> None:
        """Validate slice start and stop values."""
        if s.start is not None and s.stop is not None:
            if s.start > s.stop:
                msg = f"Invalid slice: start ({s.start}) cannot be greater than stop ({s.stop})"
                raise ValueError(msg)

    def _make_slice(self, val: Union[int, slice, List[int], type(Ellipsis)], dim_idx: int) -> Union[slice, List[int]]:
        """Convert input value to appropriate slice format."""
        # Check if it's one of the last two dimensions
        is_last_two_dims = dim_idx >= self.ndim - 2
        
        if isinstance(val, slice):
            self._validate_slice(val)
            return val
        elif isinstance(val, int):
            return slice(val, val + 1)
        elif isinstance(val, list):
            if is_last_two_dims:
                msg = "List indexing is only supported for non-spatial dimensions"
                raise TypeError(msg)
            return val
        elif val is Ellipsis:
            return val
        msg = f"Cannot convert type {type(val)} to slice"
        raise TypeError(msg)

    def transform(self, key: Any) -> SliceTuple:
        """Transform any input pattern into a tuple of slices."""
        # Handle single values
        if isinstance(key, (int, slice, list)):
            result = [slice(None)] * self.ndim
            result[0] = self._make_slice(key, 0)
            return tuple(result)
            
        # Handle Ellipsis as a single value
        if key is Ellipsis:
            return tuple([slice(None)] * self.ndim)

        # Handle tuples with possible Ellipsis
        if isinstance(key, tuple):
            # Count non-Ellipsis items
            ellipsis_count = key.count(Ellipsis)
            if ellipsis_count > 1:
                msg = "Only one ellipsis (...) allowed in indexing expression"
                raise IndexError(msg)
                
            if ellipsis_count == 0:
                # Handle regular tuple without Ellipsis
                result = [slice(None)] * self.ndim
                for i, val in enumerate(key):
                    if i >= self.ndim:
                        break
                    result[i] = self._make_slice(val, i)
                return tuple(result)
            
            # Handle tuple with one Ellipsis
            ellipsis_idx = key.index(Ellipsis)
            
            # Calculate how many slice(None)s to insert for Ellipsis
            n_actual = len(key) - 1  # -1 for Ellipsis
            n_missing = self.ndim - n_actual
            
            # Construct result
            result = []
            
            # Add items before Ellipsis
            for i in range(ellipsis_idx):
                result.append(self._make_slice(key[i], i))
                
            # Add slice(None)s for Ellipsis
            result.extend([slice(None)] * n_missing)
            
            # Add items after Ellipsis
            for i in range(ellipsis_idx + 1, len(key)):
                # Calculate the correct dimension index after Ellipsis
                current_dim = i - 1 + n_missing
                result.append(self._make_slice(key[i], current_dim))
                
            return tuple(result)

        msg = f"Unsupported key type: {type(key)}"
        raise TypeError(msg)

#transformer = SliceTransformer(ndim=5)
#transformer.transform([1, 2])  # List only
## ((slice(1, 2), slice(2, 3)), slice(None), slice(None), slice(None))
#transformer.transform(1)  # Single int
## (slice(1, 2), slice(None), slice(None), slice(None))
#transformer.transform(slice(None))  # Single slice
##(slice(None), slice(None), slice(None), slice(None))
#transformer.transform((1, [1, 3], 1))  # 3-tuple
## (slice(1, 2), (slice(1, 2), slice(3, 4)), slice(1, 2), slice(None))
#transformer.transform((1, [1, 3], [1,2], 1))  # 4-tuple repeat
## (slice(1, 2), (slice(1, 2), slice(3, 4)), slice(1, 2), slice(1, 2))
#transformer.transform((1, 1, [1, 3], 1))  # 4-tuple positions
## (slice(1, 2), slice(1, 2), (slice(1, 2), slice(3, 4)), slice(None))
#transformer.transform((1, 1, [1, 3], slice(None), slice(0, 60)))  # With slice
#(slice(1, 2), slice(1, 2), (slice(1, 2), slice(3, 4)), slice(None))
#