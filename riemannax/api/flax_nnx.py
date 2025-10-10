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
    class nnx:  # noqa: N801
        """Mock nnx module for type checking when Flax is not installed."""

        class Module:  # noqa: D106
            pass

        class Variable:  # noqa: D106
            pass

        class Param:  # noqa: D106
            pass

        class Rngs:  # noqa: D106
            pass


from riemannax.manifolds import Sphere, Stiefel, create_stiefel
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


class _ConstraintHandlerMixin:
    """Mixin providing common constraint validation and projection logic.

    This mixin centralizes manifold constraint handling methods used by both
    ManifoldConstrainedModule and ManifoldConstrainedLinear, eliminating
    code duplication.

    Required attributes in subclasses:
        - manifold: Manifold instance
        - constraint_violations: ConstraintViolation variable
        - _rngs: nnx.Rngs instance for reinitialization
        - _get_constrained_param(): method returning the parameter to constrain
        - _set_constrained_param(value): method to set the constrained parameter
        - _get_param_shape(): method returning parameter shape tuple
    """

    def _project_to_manifold(self, param_value: Array) -> Array:
        """Project a point onto the manifold.

        Prefer a manifold-provided projector when available; otherwise use small,
        well-known heuristics (e.g., Stiefel via QR, sphere-like via normalization).
        For supported manifolds, this function is JIT-safe. For unsupported manifolds,
        it raises ValueError to trigger reinitialization in project_params().

        Args:
            param_value: Point to project (may be off-manifold).

        Returns:
            Point on the manifold (may contain NaN/Inf if input is degenerate).

        Raises:
            ValueError: If manifold type is unsupported and no projection heuristic
                is available (e.g., custom manifolds without project_point method).

        Note:
            For supported manifolds (Stiefel, Sphere, or manifolds with project_point),
            this method is JIT-safe. Non-finite projections are detected and handled
            by project_params() which reinitializes parameters when necessary.
        """
        # Prioritize efficient built-in methods, then generic projector, then specific fallbacks
        # This order matches _compute_constraint_violation for consistency
        if isinstance(self.manifold, Stiefel) and param_value.ndim == 2:  # type: ignore[attr-defined]
            # Stiefel: orthogonalize via efficient reduced QR
            q, _ = jnp.linalg.qr(param_value, mode="reduced")
            projected = q
        elif callable(projector := getattr(self.manifold, "project_point", None)):  # type: ignore[attr-defined]
            # Use manifold-provided projector if available
            projected = projector(param_value)
        elif isinstance(self.manifold, Sphere):  # type: ignore[attr-defined]
            # Sphere: normalize to unit norm
            nrm = jnp.linalg.norm(param_value)
            projected = jnp.where(nrm > 0, param_value / nrm, param_value)
        else:
            # Unknown manifold: fail fast to trigger reinitialization
            # This prevents silent corruption of unsupported manifold geometries
            raise ValueError(
                f"No projection heuristic available for manifold {self.manifold.__class__.__name__}. "  # type: ignore[attr-defined]
                "Please implement project_point() method or add a dedicated projection fallback."
            )

        return projected

    def _compute_constraint_violation(self, param_value: Array) -> Array:
        """Compute constraint violation measure for given parameter value.

        This method is JIT-safe and can be used inside jax.jit-compiled functions,
        making it suitable for use in loss functions via get_constraint_penalty().

        Args:
            param_value: Parameter array to check.

        Returns:
            Violation measure (0.0 if valid, positive value otherwise) as JAX scalar.
        """
        # Check validity using manifold's validation
        is_valid = self.manifold.validate_point(param_value)  # type: ignore[attr-defined]

        # JIT-safe conditional using jax.lax.cond to avoid TracerBoolConversionError
        def _no_violation(v: Array) -> Array:
            return jnp.zeros((), dtype=v.dtype)

        def _compute_violation(v: Array) -> Array:
            # Prefer manifold-specific residuals for stability/JIT-safety
            # First check for Stiefel-specific residual
            if isinstance(self.manifold, Stiefel) and v.ndim == 2:  # type: ignore[attr-defined]
                gram = v.T @ v
                k = gram.shape[-1]
                return jnp.linalg.norm(gram - jnp.eye(k, dtype=v.dtype))

            # Then check for manifold-provided projector
            projector = getattr(self.manifold, "project_point", None)  # type: ignore[attr-defined]
            if callable(projector):
                projected = projector(v)
                return jnp.linalg.norm(v - projected)

            # Sphere-specific residual (only for actual Sphere instances)
            if isinstance(self.manifold, Sphere):  # type: ignore[attr-defined]
                # Sphere: | ||v|| - 1 |
                return jnp.abs(jnp.linalg.norm(v) - jnp.array(1.0, dtype=v.dtype))

            # Unknown manifold: return safe non-zero penalty
            # This prevents incorrect "0 violation" reports for unsupported manifolds
            return jnp.maximum(jnp.linalg.norm(v), jnp.array(1e-3, dtype=v.dtype))

        return jax.lax.cond(
            jnp.asarray(is_valid, dtype=bool),
            _no_violation,
            _compute_violation,
            param_value,
        )

    def project_params(self) -> None:
        """Project parameters back to the manifold using mutable state.

        This method enforces manifold constraints by projecting parameters
        and incrementing the violation counter. Uses direct state mutation
        as supported by NNX's reference semantics.

        Important:
            This method should be called **outside** of JIT-compiled functions.
            It performs mutable state updates (parameter modification and violation
            counter increment) which are not compatible with JAX's functional
            transformations. Call this method in your training loop after gradient
            updates, but before JIT-compiled forward/backward passes.

        Example:
            >>> # Correct usage: outside JIT
            >>> for epoch in range(num_epochs):
            ...     for batch in dataloader:
            ...         loss, grads = jitted_train_step(module, batch)
            ...         optimizer.update(grads)
            ...         module.project_params()  # Outside JIT
        """
        param_value = self._get_constrained_param()  # type: ignore[attr-defined]
        is_valid_result = self.manifold.validate_point(param_value)  # type: ignore[attr-defined]

        # Convert to Python bool explicitly using device_get (JAX best practice).
        # validate_point() returns bool in eager mode, but JAX array in JIT-traced contexts.
        # device_get ensures safe host transfer and works in both contexts.
        is_valid = bool(jax.device_get(jnp.asarray(is_valid_result, dtype=bool)))

        if not is_valid:
            # Parameters violate constraints, project back onto the manifold
            try:
                projected = self._project_to_manifold(param_value)

                # Check for NaN/Inf in projection result (only in eager execution, not JIT)
                # This catches degenerate cases like zero-norm parameters
                # Convert JAX array to Python bool using device_get (JAX best practice)
                is_finite = bool(jax.device_get(jnp.all(jnp.isfinite(projected))))
                projected_norm = float(jax.device_get(jnp.linalg.norm(projected)))
                is_zero_norm = projected_norm < 1e-8  # Threshold for numerical zero

                if not is_finite or is_zero_norm:
                    # Convert JAX array to Python float for f-string formatting
                    param_norm = float(jax.device_get(jnp.linalg.norm(param_value)))
                    error_type = "non-finite values (NaN/Inf)" if not is_finite else "near-zero norm vector"
                    raise ValueError(
                        f"Projection produced {error_type}. "
                        f"This typically indicates degenerate parameters. "
                        f"Original parameter norm: {param_norm:.2e}, "
                        f"projected norm: {projected_norm:.2e}. "
                        f"Consider reducing learning rate or checking for numerical instability."
                    )

                self._set_constrained_param(projected)  # type: ignore[attr-defined]
            except (ValueError, RuntimeError) as e:
                # Fallback: reinitialize on manifold and log a warning.
                import warnings

                key = self._rngs()  # type: ignore[attr-defined]
                projected = self.manifold.random_point(key)  # type: ignore[attr-defined]
                self._set_constrained_param(projected)  # type: ignore[attr-defined]
                warnings.warn(
                    f"Projection failed ({type(e).__name__}: {e}); parameter reinitialized on manifold.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Increment violation counter using mutable state
            self.constraint_violations.value = self.constraint_violations.value + 1.0  # type: ignore[attr-defined]

    def validate_constraints(self) -> Array:
        """Validate that parameters satisfy manifold constraints.

        Returns:
            Constraint violation measure (0.0 if satisfied).
        """
        param_value = self._get_constrained_param()  # type: ignore[attr-defined]
        return self._compute_constraint_violation(param_value)

    def get_constraint_penalty(self) -> Array:
        """Get regularization penalty for constraint violations.

        Returns:
            Penalty term (squared violation) to add to loss.
        """
        violation = self.validate_constraints()
        return jnp.square(violation)


class ManifoldConstrainedModule(_ConstraintHandlerMixin, nnx.Module):
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
        bias_shape: Optional explicit bias shape. If provided, this shape will be used
            directly for bias initialization. If None and use_bias=True, the bias shape
            is inferred from param_shape (last dimension for multi-D, first for 1D).

    Note:
        For unambiguous bias layout, especially with general manifold tensors, it is
        recommended to provide an explicit bias_shape rather than relying on inference.
        The fallback inference assumes the last dimension represents output/target space,
        which may not be appropriate for all manifold configurations.

    Example:
        >>> manifold = Sphere(n=2)  # 2-sphere (surface of a ball in 3D)
        >>> module = ManifoldConstrainedModule(
        ...     manifold=manifold,
        ...     param_shape=(3,),  # Sphere(n=2) produces shape (3,)
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
        bias_shape: tuple[int, ...] | None = None,
    ):
        """Initialize manifold-constrained module."""
        if not FLAX_AVAILABLE:
            raise ImportError(
                "Flax NNX is required for ManifoldConstrainedModule. Please install flax>=0.8 with nnx support."
            )
        self.manifold = manifold
        self.param_shape = param_shape
        self.use_bias = use_bias
        self._rngs = rngs  # Store for reinitialization

        # Initialize parameters on the manifold
        key = self._rngs()
        initial_params = self.manifold.random_point(key)

        # Validate param_shape matches manifold's intrinsic shape
        if initial_params.shape != param_shape:
            raise ValueError(
                f"param_shape={param_shape} does not match manifold's intrinsic shape "
                f"{initial_params.shape}. The manifold generates parameters with shape "
                f"{initial_params.shape}. Please use this exact shape for param_shape "
                f"to match the manifold's natural dimensionality."
            )

        self.params = ManifoldParam(initial_params)

        # Initialize constraint violation counter
        self.constraint_violations = ConstraintViolation(jnp.array(0.0))

        # Optional bias (unconstrained)
        if use_bias:
            if bias_shape is not None:
                # Use explicit bias shape - validate it's a valid shape
                if not all(dim > 0 for dim in bias_shape):
                    raise ValueError(
                        f"bias_shape={bias_shape} contains non-positive dimensions. "
                        "All dimensions must be positive integers."
                    )
                final_bias_shape = bias_shape
            else:
                # Fallback: infer bias shape from param_shape
                # This assumes last dimension = output/target space (standard for neural networks)
                inferred_size = param_shape[-1] if len(param_shape) > 1 else param_shape[0]

                # Validate inference makes sense
                if inferred_size <= 0:
                    raise ValueError(
                        f"Inferred bias size {inferred_size} is invalid for param_shape={param_shape}. "
                        "Cannot initialize bias with non-positive dimension. "
                        "Consider providing an explicit bias_shape parameter."
                    )

                final_bias_shape = (inferred_size,)

            self.bias = nnx.Param(jnp.zeros(final_bias_shape))

    # Implement mixin interface methods
    def _get_constrained_param(self) -> Array:
        """Get the constrained parameter value."""
        return self.params.value

    def _set_constrained_param(self, value: Array) -> None:
        """Set the constrained parameter value."""
        self.params.value = value

    def _get_param_shape(self) -> tuple[int, ...]:
        """Get the parameter shape."""
        return self.param_shape


class ManifoldConstrainedLinear(_ConstraintHandlerMixin, nnx.Module):
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
        if not FLAX_AVAILABLE:
            raise ImportError(
                "Flax NNX is required for ManifoldConstrainedLinear. Please install flax>=0.8 with nnx support."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.use_bias = use_bias
        self._rngs = rngs  # Store for reinitialization

        # Validate manifold shape before initialization
        if isinstance(self.manifold, Stiefel):
            # Stiefel manifold: validate (n, p) matches (in_features, out_features)
            manifold_shape = (self.manifold.n, self.manifold.p)
            expected_shape = (in_features, out_features)

            if manifold_shape != expected_shape:
                raise ValueError(
                    f"Manifold shape {manifold_shape} (from {self.manifold.__class__.__name__}) does not match "
                    f"layer weight dimensions (in_features={in_features}, out_features={out_features}). "
                    f"Please ensure the manifold is initialized to match the layer dimensions, "
                    f"e.g., Stiefel(n={in_features}, p={out_features})."
                )
        elif hasattr(self.manifold, "shape"):
            # Manifold with explicit shape attribute
            if self.manifold.shape != (in_features, out_features):
                raise ValueError(
                    f"Manifold shape {self.manifold.shape} does not match "
                    f"layer dimensions ({in_features}, {out_features})."
                )

        # Initialize weight matrix on manifold
        key = self._rngs()
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

        # Input shape guard (eager context)
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with last-dim={self.in_features}, got shape {tuple(x.shape)}")

        # Linear transformation
        output = x @ self.weight.value

        if self.use_bias:
            output = output + self.bias.value

        return output

    # Implement mixin interface methods
    def _get_constrained_param(self) -> Array:
        """Get the constrained parameter value."""
        return self.weight.value

    def _set_constrained_param(self, value: Array) -> None:
        """Set the constrained parameter value."""
        self.weight.value = value

    def _get_param_shape(self) -> tuple[int, ...]:
        """Get the parameter shape."""
        return (self.in_features, self.out_features)


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
    if not FLAX_AVAILABLE:
        raise ImportError(
            "Flax NNX is required to use create_manifold_linear(). Please install flax>=0.8 with nnx support."
        )

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
