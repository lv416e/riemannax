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

        Supports both single vectors and batch operations.
        For batch inputs, all vectors must satisfy the constraints.

        Args:
            v: Vector(s) to validate - shape (..., dim)
            model: Hyperbolic model type ("poincare" or "lorentz")

        Returns:
            Validated vector(s)

        Raises:
            HyperbolicNumericalError: If any vector length exceeds model limits
        """
        # JAX-native norm computation (batch-compatible)
        norms = jnp.linalg.norm(v, axis=-1)

        if model == "poincare":
            model_max_norm = 38.0
        elif model == "lorentz":
            model_max_norm = 19.0
        else:
            raise ValueError(f"Unknown hyperbolic model: {model}")

        # JAX-native constraint checking
        constraint_violated = jnp.any(norms > model_max_norm)

        if constraint_violated:
            # Compute diagnostic values using JAX operations
            max_violating_norm = jnp.where(constraint_violated, jnp.max(norms), model_max_norm)

            raise HyperbolicNumericalError(
                f"Vector norm {max_violating_norm:.6f} exceeds {model} model "
                f"stability limit {model_max_norm}. Use vectors with ||v|| ≤ {model_max_norm} "
                f"for numerical stability in {model} manifold operations."
            )

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

    Uses JAX-native fori_loop for optimal JIT compilation and performance.

    Args:
        A: Input matrix
        order: Number of terms in Taylor series

    Returns:
        Matrix exponential approximation
    """
    # Taylor series: exp(A) = I + A + A²/2! + A³/3! + ...
    result = jnp.eye(A.shape[0], dtype=A.dtype)

    def body_fun(i, state):
        power, factorial, result = state
        factorial = factorial * i
        power = jnp.dot(power, A)
        result = result + power / factorial
        return (power, factorial, result)

    # Initialize state: (power matrix, factorial, result)
    init_state = (jnp.eye(A.shape[0], dtype=A.dtype), 1.0, result)

    # Use JAX-native fori_loop for optimal JIT performance
    _, _, result = jax.lax.fori_loop(1, order + 1, body_fun, init_state)

    return result
