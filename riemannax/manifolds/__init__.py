"""Riemannian manifold implementations for optimization."""

from .base import DimensionError, Manifold, ManifoldError
from .factory import (
    create_lorentz,
    create_poincare_ball,
    create_poincare_ball_for_embeddings,
    create_se3,
    create_se3_for_robotics,
)
from .grassmann import Grassmann
from .lorentz import Lorentz
from .poincare_ball import PoincareBall
from .product import ProductManifold
from .se3 import SE3
from .so import SpecialOrthogonal
from .spd import SymmetricPositiveDefinite
from .sphere import Sphere
from .stiefel import Stiefel


def create_sphere(n: int = 2) -> Sphere:
    """Create a sphere manifold S^n with dimension validation.

    Factory function for creating Sphere manifolds with clear error messages
    and dimension validation.

    Args:
        n: The dimension of the sphere (default: 2 for S^2)

    Returns:
        Sphere: A sphere manifold instance

    Raises:
        ValueError: If dimension is not a positive integer
        TypeError: If n is not an integer

    Examples:
        >>> sphere = create_sphere(3)  # Creates S^3
        >>> sphere = create_sphere()   # Creates S^2 (default)
    """
    if not isinstance(n, int):
        raise TypeError(f"Sphere dimension must be an integer, got {type(n)}")
    if n <= 0:
        raise ValueError(f"Sphere dimension must be positive, got {n}")
    return Sphere(n=n)


def create_grassmann(p: int, n: int) -> Grassmann:
    """Create a Grassmann manifold Gr(p,n) with dimension validation.

    Factory function for creating Grassmann manifolds with clear error messages
    and dimension validation.

    Args:
        p: Number of basis vectors (must be positive and < n)
        n: Dimension of ambient space (must be positive and > p)

    Returns:
        Grassmann: A Grassmann manifold instance

    Raises:
        ValueError: If dimensions are invalid (p >= n or non-positive)
        TypeError: If p or n are not integers

    Examples:
        >>> grassmann = create_grassmann(2, 5)  # Gr(2,5)
        >>> grassmann = create_grassmann(3, 8)  # Gr(3,8)
    """
    if not isinstance(p, int) or not isinstance(n, int):
        raise TypeError("Grassmann dimensions p and n must be integers")
    if p <= 0:
        raise ValueError(f"Grassmann p dimension must be positive, got p={p}")
    if n <= 0:
        raise ValueError(f"Grassmann n dimension must be positive, got n={n}")
    if p >= n:
        raise ValueError(f"Grassmann requires p < n, got p={p}, n={n}")
    return Grassmann(p=p, n=n)


def create_stiefel(p: int, n: int) -> Stiefel:
    """Create a Stiefel manifold St(p,n) with dimension validation.

    Factory function for creating Stiefel manifolds with clear error messages
    and dimension validation.

    Args:
        p: Number of orthonormal vectors (must be positive and <= n)
        n: Dimension of ambient space (must be positive and >= p)

    Returns:
        Stiefel: A Stiefel manifold instance

    Raises:
        ValueError: If dimensions are invalid (p > n or non-positive)
        TypeError: If p or n are not integers

    Examples:
        >>> stiefel = create_stiefel(2, 5)  # St(2,5)
        >>> stiefel = create_stiefel(3, 3)  # St(3,3)
    """
    if not isinstance(p, int) or not isinstance(n, int):
        raise TypeError("Stiefel dimensions p and n must be integers")
    if p <= 0:
        raise ValueError(f"Stiefel p dimension must be positive, got p={p}")
    if n <= 0:
        raise ValueError(f"Stiefel n dimension must be positive, got n={n}")
    if p > n:
        raise ValueError(f"Stiefel requires p <= n, got p={p}, n={n}")
    return Stiefel(p=p, n=n)


def create_so(n: int) -> SpecialOrthogonal:
    """Create a Special Orthogonal group SO(n) with dimension validation.

    Factory function for creating SO manifolds with clear error messages
    and dimension validation.

    Args:
        n: Matrix dimension (must be >= 2)

    Returns:
        SpecialOrthogonal: A SO(n) manifold instance

    Raises:
        ValueError: If dimension is invalid (< 2)
        TypeError: If n is not an integer

    Examples:
        >>> so = create_so(3)   # SO(3) rotation matrices
        >>> so = create_so(4)   # SO(4) rotation matrices
    """
    if not isinstance(n, int):
        raise TypeError(f"SO dimension must be an integer, got {type(n)}")
    if n < 2:
        raise ValueError(f"SO manifold requires n >= 2, got n={n}")
    return SpecialOrthogonal(n=n)


def create_spd(n: int) -> SymmetricPositiveDefinite:
    """Create a Symmetric Positive Definite manifold SPD(n) with dimension validation.

    Factory function for creating SPD manifolds with clear error messages
    and dimension validation.

    Args:
        n: Matrix dimension (must be >= 2)

    Returns:
        SymmetricPositiveDefinite: An SPD(n) manifold instance

    Raises:
        ValueError: If dimension is invalid (< 2)
        TypeError: If n is not an integer

    Examples:
        >>> spd = create_spd(3)   # 3x3 symmetric positive definite matrices
        >>> spd = create_spd(5)   # 5x5 symmetric positive definite matrices
    """
    if not isinstance(n, int):
        raise TypeError(f"SPD dimension must be an integer, got {type(n)}")
    if n < 2:
        raise ValueError(f"SPD manifold requires n >= 2, got n={n}")
    return SymmetricPositiveDefinite(n=n)


__all__ = [
    "SE3",
    # Core classes and exceptions
    "DimensionError",
    # Manifold classes
    "Grassmann",
    "Lorentz",
    "Manifold",
    "ManifoldError",
    "PoincareBall",
    "ProductManifold",
    "SpecialOrthogonal",
    "Sphere",
    "Stiefel",
    "SymmetricPositiveDefinite",
    # Factory functions - existing
    "create_grassmann",
    # Factory functions - new hyperbolic and SE(3)
    "create_lorentz",
    "create_poincare_ball",
    "create_poincare_ball_for_embeddings",
    "create_se3",
    "create_se3_for_robotics",
    "create_so",
    "create_spd",
    "create_sphere",
    "create_stiefel",
]
