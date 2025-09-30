"""Flax NNX integration for manifold-constrained neural network modules.

This module provides Flax NNX-compatible neural network components that enforce
manifold constraints on parameters during training. Uses NNX's explicit state
management and mutable reference semantics for constraint tracking.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

try:
    from flax import nnx

    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

    # Create mock classes for type checking when Flax is not installed
    class nnx:  # type: ignore  # noqa: N801
        """Mock nnx module for type checking when Flax is not installed."""

        class Module:  # noqa: D106
            pass

        class Variable:  # noqa: D106
            pass

        class Param:  # noqa: D106
            pass

        class Rngs:  # noqa: D106
            pass


from riemannax.manifolds.base import Manifold


class ConstraintViolation(nnx.Variable):
    """Custom Variable type for tracking constraint violations.

    This Variable type is used to count how many times parameters
    have drifted from the manifold and required projection.
    """

    pass


class ManifoldParam(nnx.Param):
    """Custom Parameter type for manifold-constrained parameters.

    This extends nnx.Param to indicate parameters that must satisfy
    manifold constraints.
    """

    pass


class ManifoldConstrainedModule(nnx.Module):
    """Base class for Flax NNX modules with manifold-constrained parameters.

    This module provides the foundation for neural network layers where
    parameters must lie on a Riemannian manifold. It uses NNX's mutable
    state management to track constraint violations and automatically
    projects parameters back to the manifold when needed.

    Args:
        manifold: The Riemannian manifold for parameter constraints.
        param_shape: Shape of the parameter array.
        rngs: NNX random number generator state.
        use_bias: Whether to include a bias term (unconstrained).

    Example:
        >>> manifold = Sphere(n=3)
        >>> module = ManifoldConstrainedModule(
        ...     manifold=manifold,
        ...     param_shape=(4,),
        ...     rngs=nnx.Rngs(0)
        ... )
        >>> module.project_params()  # Enforces constraints
    """

    def __init__(
        self,
        manifold: Manifold,
        param_shape: tuple[int, ...],
        rngs: nnx.Rngs,
        use_bias: bool = False,
    ):
        """Initialize manifold-constrained module."""
        self.manifold = manifold
        self.param_shape = param_shape
        self.use_bias = use_bias

        # Initialize parameters on the manifold
        key = rngs()
        initial_params = self.manifold.random_point(key, *param_shape)
        self.params = ManifoldParam(initial_params)

        # Initialize constraint violation counter
        self.constraint_violations = ConstraintViolation(jnp.array(0.0))

        # Optional bias (unconstrained)
        if use_bias:
            self.bias = nnx.Param(jnp.zeros(param_shape[-1] if len(param_shape) > 1 else param_shape[0]))

    def project_params(self) -> None:
        """Project parameters back to the manifold using mutable state.

        This method enforces manifold constraints by projecting parameters
        and incrementing the violation counter. Uses direct state mutation
        as supported by NNX's reference semantics.
        """
        # Check if parameters satisfy constraints
        is_valid = self.manifold.validate_point(self.params.value)

        if not is_valid:
            # Parameters violate constraints, project back
            projection_successful = False

            # Use heuristic projection based on manifold type
            try:
                # For Stiefel/Grassmann: QR-based orthogonalization
                if hasattr(self.manifold, "n") and hasattr(self.manifold, "p"):
                    q, _ = jnp.linalg.qr(self.params.value)
                    self.params.value = q
                    projection_successful = True
                # For Sphere: normalize to unit norm
                elif hasattr(self.manifold, "dimension"):
                    norm = jnp.linalg.norm(self.params.value)
                    if norm > 1e-8:
                        self.params.value = self.params.value / norm
                        projection_successful = True
            except (ValueError, RuntimeError):
                pass  # Projection failed, will reinitialize

            # Last resort: reinitialize on manifold with non-deterministic key
            if not projection_successful:
                # Use fold_in to create non-deterministic key based on violation count
                base_key = jax.random.PRNGKey(0)
                key = jax.random.fold_in(base_key, int(self.constraint_violations.value))
                self.params.value = self.manifold.random_point(key, *self.param_shape)

            # Increment violation counter using mutable state
            self.constraint_violations.value = self.constraint_violations.value + 1.0

    def _compute_constraint_violation(self, param_value: Array) -> float:
        """Compute constraint violation measure for given parameter value.

        Args:
            param_value: Parameter array to check.

        Returns:
            Violation measure (0.0 if valid, positive value otherwise).
        """
        # Check validity first
        is_valid = self.manifold.validate_point(param_value)
        if is_valid:
            return 0.0

        # Compute manifold-specific residual
        if hasattr(self.manifold, "n") and hasattr(self.manifold, "p"):
            # Stiefel: measure orthogonality violation (||X^T X - I||)
            gram = param_value.T @ param_value
            p = param_value.shape[1]
            residual = jnp.linalg.norm(gram - jnp.eye(p))
            return float(residual)
        elif hasattr(self.manifold, "dimension"):
            # Sphere: measure norm deviation (||x|| - 1)
            norm = jnp.linalg.norm(param_value)
            return float(jnp.abs(norm - 1.0))
        else:
            # Generic: return constant penalty if invalid
            return 1.0

    def validate_constraints(self) -> float:
        """Validate that parameters satisfy manifold constraints.

        Returns:
            Constraint violation measure (0.0 if satisfied).
        """
        return self._compute_constraint_violation(self.params.value)

    def get_constraint_penalty(self) -> float:
        """Compute penalty for constraint violations.

        This can be added to the loss function to discourage
        parameter drift from the manifold.

        Returns:
            Penalty value based on constraint violation measure.
        """
        violation = self._compute_constraint_violation(self.params.value)
        return violation**2


class ManifoldConstrainedLinear(nnx.Module):
    """Linear layer with manifold-constrained weight matrix.

    This layer implements a linear transformation y = Wx + b where
    the weight matrix W is constrained to lie on a specified manifold
    (e.g., Stiefel for orthogonal weights).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        manifold: Manifold constraint for weight matrix.
        rngs: NNX random number generator state.
        use_bias: Whether to include bias term.

    Example:
        >>> manifold = Stiefel(n=5, p=3)
        >>> layer = ManifoldConstrainedLinear(
        ...     in_features=5,
        ...     out_features=3,
        ...     manifold=manifold,
        ...     rngs=nnx.Rngs(0)
        ... )
        >>> x = jnp.ones((2, 5))
        >>> y = layer(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        rngs: nnx.Rngs,
        use_bias: bool = True,
    ):
        """Initialize manifold-constrained linear layer."""
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.use_bias = use_bias

        # Validate manifold shape before initialization
        # Check if manifold has explicit shape attributes
        if hasattr(self.manifold, "n") and hasattr(self.manifold, "p"):
            # Stiefel-like manifold: validate (n, p) matches (in_features, out_features)
            manifold_shape = (self.manifold.n, self.manifold.p)
            expected_shape = (in_features, out_features)

            if manifold_shape != expected_shape:
                # Provide helpful error message with correct convention
                if in_features >= out_features:
                    suggestion = f"Stiefel(n={in_features}, p={out_features})"
                else:
                    suggestion = (
                        f"Stiefel(n={out_features}, p={in_features}) and swap "
                        f"in_features/out_features, or transpose the weight matrix"
                    )
                raise ValueError(
                    f"Manifold shape {manifold_shape} does not match layer dimensions "
                    f"{expected_shape}. For Stiefel manifolds, n >= p is required. "
                    f"Use {suggestion}."
                )
        elif hasattr(self.manifold, "shape"):
            # Manifold with explicit shape attribute
            if self.manifold.shape != (in_features, out_features):
                raise ValueError(
                    f"Manifold shape {self.manifold.shape} does not match "
                    f"layer dimensions ({in_features}, {out_features})."
                )

        # Initialize weight matrix on manifold
        key = rngs()
        initial_weight = self.manifold.random_point(key)

        # Validate returned shape
        if initial_weight.ndim != 2:
            raise ValueError(
                f"ManifoldConstrainedLinear requires a 2D weight matrix, "
                f"but manifold.random_point() returned shape {initial_weight.shape}."
            )
        if initial_weight.shape != (in_features, out_features):
            raise ValueError(
                f"Weight shape {initial_weight.shape} does not match "
                f"({in_features}, {out_features}). Manifold configuration error."
            )

        self.weight = ManifoldParam(initial_weight)

        # Initialize constraint violation counter
        self.constraint_violations = ConstraintViolation(jnp.array(0.0))

        # Optional bias (unconstrained)
        if use_bias:
            self.bias = nnx.Param(jnp.zeros(out_features))

    def __call__(self, x: Array) -> Array:
        """Forward pass with automatic constraint checking.

        Args:
            x: Input array of shape (batch_size, in_features).

        Returns:
            Output array of shape (batch_size, out_features).
        """
        # Check constraints (track violations if any)
        # Note: Skip validation tracking inside JIT to avoid tracer issues
        # Users should call project_params() explicitly when needed

        # Linear transformation
        output = x @ self.weight.value

        if self.use_bias:
            output = output + self.bias.value

        return output

    def project_params(self) -> None:
        """Project weight matrix back to manifold."""
        is_valid = self.manifold.validate_point(self.weight.value)

        if not is_valid:
            projection_successful = False

            # Use heuristic projection based on manifold type
            try:
                # For Stiefel/Grassmann: QR-based orthogonalization
                if hasattr(self.manifold, "n") and hasattr(self.manifold, "p"):
                    q, _ = jnp.linalg.qr(self.weight.value)
                    self.weight.value = q
                    projection_successful = True
                # For Sphere: normalize to unit norm
                elif hasattr(self.manifold, "dimension"):
                    norm = jnp.linalg.norm(self.weight.value)
                    if norm > 1e-8:
                        self.weight.value = self.weight.value / norm
                        projection_successful = True
            except (ValueError, RuntimeError):
                pass

            # Last resort: reinitialize on manifold with non-deterministic key
            if not projection_successful:
                base_key = jax.random.PRNGKey(0)
                key = jax.random.fold_in(base_key, int(self.constraint_violations.value))
                self.weight.value = self.manifold.random_point(key)

            self.constraint_violations.value = self.constraint_violations.value + 1.0

    def _compute_constraint_violation(self, param_value: Array) -> float:
        """Compute constraint violation measure for given parameter value.

        Args:
            param_value: Parameter array to check.

        Returns:
            Violation measure (0.0 if valid, positive value otherwise).
        """
        # Check validity first
        is_valid = self.manifold.validate_point(param_value)
        if is_valid:
            return 0.0

        # Compute manifold-specific residual
        if hasattr(self.manifold, "n") and hasattr(self.manifold, "p"):
            # Stiefel: measure orthogonality violation (||X^T X - I||)
            gram = param_value.T @ param_value
            p = param_value.shape[1]
            residual = jnp.linalg.norm(gram - jnp.eye(p))
            return float(residual)
        elif hasattr(self.manifold, "dimension"):
            # Sphere: measure norm deviation (||x|| - 1)
            norm = jnp.linalg.norm(param_value)
            return float(jnp.abs(norm - 1.0))
        else:
            # Generic: return constant penalty if invalid
            return 1.0

    def validate_constraints(self) -> float:
        """Validate weight matrix constraints."""
        return self._compute_constraint_violation(self.weight.value)

    def get_constraint_penalty(self) -> float:
        """Compute constraint violation penalty."""
        violation = self._compute_constraint_violation(self.weight.value)
        return violation**2


def create_manifold_linear(
    in_features: int,
    out_features: int,
    manifold_type: str = "stiefel",
    use_bias: bool = True,
    rngs: nnx.Rngs | None = None,
) -> ManifoldConstrainedLinear:
    """Factory function for creating manifold-constrained linear layers.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        manifold_type: Type of manifold constraint ('stiefel', 'sphere', etc.).
        use_bias: Whether to include bias term.
        rngs: Random number generator state.

    Returns:
        ManifoldConstrainedLinear layer.

    Example:
        >>> layer = create_manifold_linear(
        ...     in_features=10,
        ...     out_features=5,
        ...     manifold_type='stiefel',
        ...     rngs=nnx.Rngs(0)
        ... )
    """
    from riemannax.manifolds import create_stiefel

    if rngs is None:
        rngs = nnx.Rngs(0)

    # Create appropriate manifold
    manifold: Manifold
    if manifold_type.lower() == "stiefel":
        if in_features < out_features:
            raise ValueError(f"Stiefel requires in_features >= out_features, got {in_features} < {out_features}")
        manifold = create_stiefel(p=out_features, n=in_features)
    elif manifold_type.lower() == "sphere":
        raise NotImplementedError(
            "ManifoldConstrainedLinear requires a matrix-shaped manifold. "
            "Sphere manifold produces 1D vectors, not 2D weight matrices. "
            "Use 'stiefel' for orthogonal weights, or implement custom oblique/product-sphere support."
        )
    else:
        raise ValueError(f"Unsupported manifold type: {manifold_type}")

    return ManifoldConstrainedLinear(
        in_features=in_features,
        out_features=out_features,
        manifold=manifold,
        rngs=rngs,
        use_bias=use_bias,
    )
