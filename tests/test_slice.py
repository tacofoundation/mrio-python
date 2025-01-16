"""
Tests for the SliceTransformer class.

These tests cover all functionality, edge cases, and error conditions
to achieve 100% code coverage.
"""

from typing import Any, List, Tuple

import pytest

from mrio.slice_transformer import SliceTransformer

# Fixtures
# --------


@pytest.fixture
def transformer():
    """Create a basic transformer with 4 dimensions."""
    return SliceTransformer(ndim=4)


@pytest.fixture
def transformer_3d():
    """Create a transformer with 3 dimensions."""
    return SliceTransformer(ndim=3)


# Initialization Tests
# ------------------


def test_valid_initialization():
    """Test valid initialization cases."""
    t = SliceTransformer(ndim=1)
    assert t.ndim == 1

    t = SliceTransformer(ndim=100)
    assert t.ndim == 100


def test_invalid_initialization():
    """Test invalid initialization cases."""
    with pytest.raises(ValueError, match="ndim must be a positive integer"):
        SliceTransformer(ndim=0)

    with pytest.raises(ValueError, match="ndim must be a positive integer"):
        SliceTransformer(ndim=-1)

    with pytest.raises(ValueError, match="ndim must be a positive integer"):
        SliceTransformer(ndim=1.5)


# Basic Transformation Tests
# ------------------------


def test_integer_transform(transformer):
    """Test transforming single integer indices."""
    # Basic integer case
    assert transformer.transform(1) == (
        slice(1, 2),
        slice(None),
        slice(None),
        slice(None),
    )

    # Zero index
    assert transformer.transform(0) == (
        slice(0, 1),
        slice(None),
        slice(None),
        slice(None),
    )

    # Negative index
    assert transformer.transform(-1) == (
        slice(-1, 0),
        slice(None),
        slice(None),
        slice(None),
    )


def test_slice_transform(transformer):
    """Test transforming slice objects."""
    # Basic slice
    assert transformer.transform(slice(1, 3)) == (
        slice(1, 3),
        slice(None),
        slice(None),
        slice(None),
    )

    # Slice with step
    assert transformer.transform(slice(1, 10, 2)) == (
        slice(1, 10, 2),
        slice(None),
        slice(None),
        slice(None),
    )

    # Full slice
    assert transformer.transform(slice(None)) == (
        slice(None),
        slice(None),
        slice(None),
        slice(None),
    )

    # Negative indices in slice
    assert transformer.transform(slice(-3, -1)) == (
        slice(-3, -1),
        slice(None),
        slice(None),
        slice(None),
    )


# Advanced Transformation Tests
# ---------------------------


@pytest.mark.parametrize(
    "lst,dim,expected",
    [
        (
            [1, 3],
            0,
            ((slice(1, 2), slice(3, 4)), slice(None), slice(None), slice(None)),
        ),
        ([0], 1, (slice(None), (slice(0, 1),), slice(None), slice(None))),
        (
            [1, 2, 3],
            2,
            (
                slice(None),
                slice(None),
                (slice(1, 2), slice(2, 3), slice(3, 4)),
                slice(None),
            ),
        ),
    ],
)
def test_list_with_dimension(
    transformer, lst: List[int], dim: int, expected: Tuple[Any, ...]
):
    """Test transforming lists with specified dimensions."""
    assert transformer.transform(lst, dim=dim) == expected


def test_nested_case_transform(transformer):
    """Test transforming nested list-dimension pairs as tuples."""
    # Basic nested case
    result = transformer.transform(([1, 3], 1))
    expected = ((slice(1, 2), slice(3, 4)), slice(1, 2), slice(None), slice(None))
    assert result == expected

    # Single item list
    result = transformer.transform(([0], 0))
    expected = ((slice(0, 1),), slice(0, 1), slice(None), slice(None))
    assert result == expected


def test_tuple_transform(transformer):
    """Test transforming tuples with various combinations."""
    # Basic tuple
    assert transformer.transform((1, 2)) == (
        slice(1, 2),
        slice(2, 3),
        slice(None),
        slice(None),
    )

    # Mixed tuple
    assert transformer.transform((1, slice(1, 3), 2)) == (
        slice(1, 2),
        slice(1, 3),
        slice(2, 3),
        slice(None),
    )

    # Full specification
    assert transformer.transform((1, 2, 3, 4)) == (
        slice(1, 2),
        slice(2, 3),
        slice(3, 4),
        slice(4, 5),
    )


def test_ellipsis_handling(transformer):
    """Test handling of ellipsis in tuples."""
    # Basic ellipsis
    assert transformer.transform((1, Ellipsis, 2)) == (
        slice(1, 2),
        slice(None),
        slice(None),
        slice(2, 3),
    )

    # Ellipsis at start
    assert transformer.transform((Ellipsis, 1, 2)) == (
        slice(None),
        slice(None),
        slice(1, 2),
        slice(2, 3),
    )

    # Ellipsis at end
    assert transformer.transform((1, 2, Ellipsis)) == (
        slice(1, 2),
        slice(2, 3),
        slice(None),
        slice(None),
    )


# Error Cases
# ----------


def test_invalid_dimension_errors(transformer):
    """Test errors from invalid dimension specifications."""
    with pytest.raises(ValueError, match="Invalid dimension"):
        transformer.transform([1, 2], dim=4)

    with pytest.raises(ValueError, match="Invalid dimension"):
        transformer.transform([1, 2], dim=-1)

    with pytest.raises(ValueError, match="Invalid dimension"):
        transformer.transform(([1, 2], 5))


def test_invalid_list_errors(transformer):
    """Test errors from invalid list inputs."""
    with pytest.raises(ValueError, match="List must contain only integers"):
        transformer.transform(([1, "2"], 1))

    with pytest.raises(ValueError, match="must be a list of integers"):
        transformer.transform(["1", "2"], dim=1)


def test_multiple_ellipsis_error(transformer):
    """Test error from multiple ellipsis in tuple."""
    with pytest.raises(ValueError, match="Multiple ellipsis found in key"):
        transformer.transform((Ellipsis, 1, Ellipsis))


def test_unsupported_type_errors(transformer):
    """Test errors from unsupported input types."""
    with pytest.raises(TypeError, match="Unsupported key type"):
        transformer.transform({"key": "value"})

    with pytest.raises(TypeError, match="Unsupported key type"):
        transformer.transform((1, {"key": "value"}, 2))


# Edge Cases
# ---------


def test_single_dimension_transformer():
    """Test transformer with single dimension."""
    t = SliceTransformer(ndim=1)
    assert t.transform(0) == (slice(0, 1),)
    assert t.transform(slice(None)) == (slice(None),)
    assert t.transform((0,)) == (slice(0, 1),)


def test_large_dimension_transformer():
    """Test transformer with many dimensions."""
    t = SliceTransformer(ndim=10)
    result = t.transform(1)
    assert len(result) == 10
    assert result[0] == slice(1, 2)
    assert all(s == slice(None) for s in result[1:])


def test_dimension_overflow(transformer):
    """Test handling of too many dimensions in input."""
    # More dimensions than specified
    result = transformer.transform((1, 2, 3, 4, 5))
    assert len(result) == 4  # Should truncate to ndim
    assert result == (slice(1, 2), slice(2, 3), slice(3, 4), slice(4, 5))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov"])
