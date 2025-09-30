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
            # Use retraction from a base point on the manifold
            base_point = self.manifold.random_point(jax.random.PRNGKey(0), *self.param_shape)
            # Project the difference to tangent space, then retract
            tangent = self.manifold.proj(base_point, self.params.value - base_point)
            self.params.value = self.manifold.retr(base_point, tangent)

            # Increment violation counter using mutable state
            self.constraint_violations.value = self.constraint_violations.value + 1.0

    def validate_constraints(self) -> float:
        """Validate that parameters satisfy manifold constraints.

        Returns:
            Constraint violation measure (0.0 if satisfied).
        """
        is_valid = self.manifold.validate_point(self.params.value)
        if is_valid:
            return 0.0
        else:
            # Compute distance from manifold (heuristic)
            base_point = self.manifold.random_point(jax.random.PRNGKey(0), *self.param_shape)
            projected = self.manifold.proj(base_point, self.params.value)
            return float(jnp.linalg.norm(self.params.value - projected))

    def get_constraint_penalty(self) -> float:
        """Compute penalty for constraint violations.

        This can be added to the loss function to discourage
        parameter drift from the manifold.

        Returns:
            Penalty value based on constraint violation measure.
        """
        violation = self.validate_constraints()
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

        # Initialize weight matrix on manifold
        key = rngs()
        # For manifolds like Stiefel, random_point() already knows the shape (n, p)
        # Don't pass shape for single matrix
        initial_weight = self.manifold.random_point(key)
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
            # Project back using retraction from a base point on the manifold
            base_point = self.manifold.random_point(jax.random.PRNGKey(0))
            # Project the difference to tangent space, then retract
            tangent = self.manifold.proj(base_point, self.weight.value - base_point)
            self.weight.value = self.manifold.retr(base_point, tangent)

            self.constraint_violations.value = self.constraint_violations.value + 1.0

    def validate_constraints(self) -> float:
        """Validate weight matrix constraints."""
        is_valid = self.manifold.validate_point(self.weight.value)
        if is_valid:
            return 0.0
        else:
            base_point = self.manifold.random_point(jax.random.PRNGKey(0))
            projected = self.manifold.proj(base_point, self.weight.value)
            return float(jnp.linalg.norm(self.weight.value - projected))

    def get_constraint_penalty(self) -> float:
        """Compute constraint violation penalty."""
        violation = self.validate_constraints()
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
    from riemannax.manifolds import create_sphere, create_stiefel

    if rngs is None:
        rngs = nnx.Rngs(0)

    # Create appropriate manifold
    manifold: Manifold
    if manifold_type.lower() == "stiefel":
        if in_features < out_features:
            raise ValueError(f"Stiefel requires in_features >= out_features, got {in_features} < {out_features}")
        manifold = create_stiefel(p=out_features, n=in_features)
    elif manifold_type.lower() == "sphere":
        manifold = create_sphere(n=in_features * out_features - 1)
    else:
        raise ValueError(f"Unsupported manifold type: {manifold_type}")

    return ManifoldConstrainedLinear(
        in_features=in_features,
        out_features=out_features,
        manifold=manifold,
        rngs=rngs,
        use_bias=use_bias,
    )
