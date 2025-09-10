"""Numerical stability core infrastructure for hyperbolic and SE(3) manifolds."""

from typing import Literal

import jax.numpy as jnp
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

        Supports both single vectors and batch operations with enhanced error diagnostics.
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

        # JAX-native constraint checking with enhanced diagnostics
        constraint_violated = jnp.any(norms > model_max_norm)

        if constraint_violated:
            # Enhanced error diagnostics using JAX-native operations
            max_violating_norm = jnp.where(constraint_violated, jnp.max(norms), model_max_norm)
            violation_margin = max_violating_norm - model_max_norm
            violating_count = jnp.sum(norms > model_max_norm)

            raise HyperbolicNumericalError(
                f"Vector norm {max_violating_norm:.6f} exceeds {model} model "
                f"stability limit {model_max_norm} by {violation_margin:.6f}. "
                f"Violating vectors: {violating_count} out of {norms.size}. "
                f"Use vectors with ||v|| ≤ {model_max_norm} "
                f"for numerical stability in {model} manifold operations."
            )

        return v
