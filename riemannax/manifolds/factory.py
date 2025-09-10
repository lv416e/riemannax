"""Factory functions for creating manifold instances with default configurations."""

from .lorentz import Lorentz
from .poincare_ball import PoincareBall
from .se3 import SE3


def create_poincare_ball(
    dimension: int,
    curvature: float = -1.0,
    tolerance: float = 1e-6,
) -> PoincareBall:
    """Create a Poincaré Ball hyperbolic manifold with validated parameters.

    Factory function that creates a PoincareBall manifold instance with
    parameter validation and sensible defaults for hyperbolic optimization.

    Args:
        dimension: Dimension of the hyperbolic space (must be positive)
        curvature: Negative curvature parameter (must be negative)
        tolerance: Numerical tolerance for validation operations

    Returns:
        Configured PoincareBall manifold instance

    Raises:
        ValueError: If parameters are invalid (non-positive dimension,
                   non-negative curvature, or non-positive tolerance)

    Example:
        >>> manifold = create_poincare_ball(dimension=5, curvature=-2.0)
        >>> point = manifold.random_point(jax.random.PRNGKey(42))
    """
    # Parameter validation
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")

    if curvature >= 0:
        raise ValueError(f"Curvature must be negative for hyperbolic space, got {curvature}")

    if tolerance <= 0:
        raise ValueError(f"Tolerance must be positive, got {tolerance}")

    return PoincareBall(dimension=dimension, curvature=curvature, tolerance=tolerance)


def create_lorentz(
    dimension: int,
    atol: float = 1e-8,
) -> Lorentz:
    """Create a Lorentz/Hyperboloid manifold with validated parameters.

    Factory function that creates a Lorentz manifold instance with
    parameter validation and sensible defaults for hyperbolic optimization
    in the hyperboloid model.

    Args:
        dimension: Dimension of the hyperbolic space (must be positive)
        atol: Absolute tolerance for numerical computations

    Returns:
        Configured Lorentz manifold instance

    Raises:
        ValueError: If parameters are invalid (non-positive dimension
                   or non-positive tolerance)

    Example:
        >>> manifold = create_lorentz(dimension=3)
        >>> point = manifold.random_point(jax.random.PRNGKey(42))
    """
    # Parameter validation
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")

    if atol <= 0:
        raise ValueError(f"Absolute tolerance must be positive, got {atol}")

    return Lorentz(dimension=dimension, atol=atol)


def create_se3(atol: float = 1e-8) -> SE3:
    """Create an SE(3) Special Euclidean Group manifold with validated parameters.

    Factory function that creates an SE(3) manifold instance with
    parameter validation and sensible defaults for robotics and 3D
    transformation optimization.

    Args:
        atol: Absolute tolerance for numerical computations

    Returns:
        Configured SE(3) manifold instance

    Raises:
        ValueError: If tolerance parameter is invalid (non-positive)

    Example:
        >>> manifold = create_se3()
        >>> transform = manifold.random_point(jax.random.PRNGKey(42))
        >>> # transform is (qw, qx, qy, qz, tx, ty, tz) format
    """
    # Parameter validation
    if atol <= 0:
        raise ValueError(f"Absolute tolerance must be positive, got {atol}")

    return SE3(atol=atol)


# Configuration presets for common use cases
def create_poincare_ball_for_embeddings(
    dimension: int,
    tolerance: float = 1e-6,
) -> PoincareBall:
    """Create Poincaré Ball optimized for hierarchical embeddings.

    Preset configuration suitable for embedding hierarchical data
    such as word embeddings, knowledge graphs, or tree structures.
    Uses unit curvature (-1.0) which is standard for embeddings.

    Args:
        dimension: Embedding dimension
        tolerance: Numerical tolerance

    Returns:
        PoincareBall manifold configured for embeddings
    """
    return create_poincare_ball(
        dimension=dimension,
        curvature=-1.0,  # Standard unit curvature for embeddings
        tolerance=tolerance,
    )


def create_se3_for_robotics(atol: float = 1e-6) -> SE3:
    """Create SE(3) manifold optimized for robotics applications.

    Preset configuration suitable for robot trajectory optimization,
    pose estimation, and manipulation tasks. Uses tighter tolerance
    for better precision in physical applications.

    Args:
        atol: Numerical tolerance (tighter for robotics precision)

    Returns:
        SE(3) manifold configured for robotics
    """
    return create_se3(atol=atol)
