"""Numerical stability core infrastructure for hyperbolic and SE(3) manifolds."""

from typing import Literal

import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Float

from riemannax.manifolds.base import ManifoldError


class HyperbolicNumericalError(ManifoldError):
    """Raised when hyperbolic operations encounter numerical instability."""

    pass


class SE3SingularityError(ManifoldError):
    """Raised when SE(3) operations encounter singularities."""

    pass


class CurvatureBoundsError(ManifoldError):
    """Raised when curvature parameters exceed stable bounds."""

    pass


class NumericalStabilityManager:
    """Centralized numerical stability management for hyperbolic and SE(3) manifolds.

    Based on research findings that Poincaré Ball is stable for vectors <38 length
    and Lorentz model for vectors <19 length.
    """

    @staticmethod
    def validate_hyperbolic_vector(v: Float[Array, "..."], model: Literal["poincare", "lorentz"]) -> Array:
        """Validate vector length limits for hyperbolic models.

        Args:
            v: Vector to validate
            model: Hyperbolic model type ("poincare" or "lorentz")

        Returns:
            Validated vector

        Raises:
            HyperbolicNumericalError: If vector length exceeds model limits
        """
        norm = jnp.linalg.norm(v)

        if model == "poincare":
            max_norm = 38.0
        elif model == "lorentz":
            max_norm = 19.0
        else:
            raise ValueError(f"Unknown hyperbolic model: {model}")

        # Check if norm exceeds safety threshold
        if norm > max_norm:
            raise HyperbolicNumericalError(f"Vector norm {norm:.6f} exceeds {model} model limit {max_norm}")

        return v

    @staticmethod
    def safe_matrix_exponential(A: Array, method: str = "pade") -> Array:
        """Compute matrix exponential with numerical stability.

        Args:
            A: Matrix for exponential computation
            method: Method to use ("pade", "taylor", or other)

        Returns:
            Matrix exponential result

        Raises:
            SE3SingularityError: If method is invalid or computation fails
        """
        if method == "pade":
            # Use JAX's built-in matrix exponential (uses Padé approximation)
            return jax.scipy.linalg.expm(A)
        elif method == "taylor":
            # Simple Taylor approximation for small matrices
            return _taylor_matrix_exp(A)
        else:
            raise SE3SingularityError(f"Invalid matrix exponential method: {method}")


def _taylor_matrix_exp(A: Array, order: int = 10) -> Array:
    """Taylor series approximation of matrix exponential.

    Args:
        A: Input matrix
        order: Number of terms in Taylor series

    Returns:
        Matrix exponential approximation
    """
    # Taylor series: exp(A) = I + A + A²/2! + A³/3! + ...
    result = jnp.eye(A.shape[0], dtype=A.dtype)
    power = jnp.eye(A.shape[0], dtype=A.dtype)
    factorial = 1.0

    for i in range(1, order + 1):
        factorial *= i
        power = jnp.dot(power, A)
        result += power / factorial

    return result
