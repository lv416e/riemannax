"""Type system for RiemannAX library with JAX array validation.

This module provides type aliases and validation utilities for JAX arrays
used in Riemannian manifold operations, ensuring type safety and shape
consistency throughout the library.
"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

# Type aliases for common manifold objects
ManifoldPoint = Float[Array, "... dim"]
"""Type alias for points on a Riemannian manifold."""

TangentVector = Float[Array, "... dim"]
"""Type alias for tangent vectors on a Riemannian manifold."""

RiemannianMetric = Float[Array, "... dim dim"]
"""Type alias for Riemannian metric tensors."""


def validate_shape(array: Array, expected_shape: str) -> bool:
    """Validate that an array has the expected shape pattern.

    Args:
        array: JAX array to validate
        expected_shape: Expected shape pattern as string (e.g., "3", "2 2", "... 3")

    Returns:
        True if shape matches the pattern, False otherwise

    Examples:
        >>> arr = jnp.array([1.0, 2.0, 3.0])
        >>> validate_shape(arr, "3")
        True
        >>> validate_shape(arr, "2")
        False
        >>> arr2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> validate_shape(arr2d, "2 2")
        True
        >>> validate_shape(arr2d, "... 2")
        True
    """
    actual_shape = array.shape
    expected_parts = expected_shape.split()

    # Handle wildcard patterns
    if expected_parts[0] == "...":
        # Pattern like "... 3" matches any shape ending with 3
        if len(expected_parts) == 1:
            # Just "..." matches any shape
            return True
        expected_suffix = [int(dim) for dim in expected_parts[1:]]
        if len(actual_shape) < len(expected_suffix):
            return False
        return bool(actual_shape[-len(expected_suffix) :] == tuple(expected_suffix))

    # Exact shape matching
    try:
        expected_dims = tuple(int(dim) for dim in expected_parts)
        return bool(actual_shape == expected_dims)
    except ValueError:
        # Invalid shape specification
        return False


def validate_dtype(array: Array, expected_dtype: type | jnp.dtype[Any]) -> bool:
    """Validate that an array has the expected data type.

    Args:
        array: JAX array to validate
        expected_dtype: Expected dtype (e.g., jnp.float32, float, complex)

    Returns:
        True if dtype matches, False otherwise

    Examples:
        >>> arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> validate_dtype(arr, jnp.float32)
        True
        >>> validate_dtype(arr, float)
        True
        >>> validate_dtype(arr, jnp.int32)
        False
    """
    actual_dtype = array.dtype

    # Handle generic Python types
    if expected_dtype is float:
        return bool(jnp.issubdtype(actual_dtype, jnp.floating))
    elif expected_dtype is int:
        return bool(jnp.issubdtype(actual_dtype, jnp.integer))
    elif expected_dtype is complex:
        return bool(jnp.issubdtype(actual_dtype, jnp.complexfloating))

    # Handle exact JAX dtype matching
    return bool(actual_dtype == expected_dtype)


def validate_jaxtyping_annotation(array: Array, annotation: type) -> bool:
    """Validate that an array matches a jaxtyping annotation.

    Args:
        array: JAX array to validate
        annotation: Jaxtyping annotation (e.g., ManifoldPoint, TangentVector)

    Returns:
        True if array matches annotation, False otherwise

    Examples:
        >>> point = jnp.array([1.0, 2.0, 3.0])
        >>> validate_jaxtyping_annotation(point, ManifoldPoint)
        True
        >>> validate_jaxtyping_annotation(5.0, ManifoldPoint)
        False
    """
    # Basic array validation
    if not isinstance(array, jnp.ndarray):
        return False

    # Check if it's a floating point array (requirement for manifold types)
    # For now, basic validation - more sophisticated jaxtyping integration
    # can be added when jaxtyping provides better runtime validation
    return jnp.issubdtype(array.dtype, jnp.floating)


def ensure_array_dtype(array: Array, target_dtype: jnp.dtype) -> Array:
    """Ensure an array has the specified dtype, converting if necessary.

    Args:
        array: JAX array to convert
        target_dtype: Target dtype for the array

    Returns:
        Array with the specified dtype

    Examples:
        >>> arr = jnp.array([1, 2, 3])
        >>> result = ensure_array_dtype(arr, jnp.float32)
        >>> result.dtype
        dtype('float32')
    """
    if array.dtype == target_dtype:
        return array

    return array.astype(target_dtype)
