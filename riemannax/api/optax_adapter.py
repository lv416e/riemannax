"""Optax integration adapter for Riemannian optimization.

This module provides adapters to make RiemannAX optimizers compatible with
Optax's GradientTransformation interface, enabling seamless integration with
the Optax ecosystem.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import optax
from jaxtyping import Array

from riemannax.manifolds.base import Manifold


class RiemannianOptaxState(NamedTuple):
    """State for Riemannian Optax adapter.

    Attributes:
        step_count: Number of optimization steps taken.
        manifold_state: Additional state for manifold-specific operations.
        adam_m: First moment estimate (for Adam-like optimizers).
        adam_v: Second moment estimate (for Adam-like optimizers).
    """

    step_count: int
    manifold_state: Any | None = None
    adam_m: Array | None = None
    adam_v: Array | None = None


class RiemannianOptaxAdapter:
    """Adapter to make Riemannian optimizers compatible with Optax interface.

    This adapter implements the Optax GradientTransformation protocol, allowing
    RiemannAX optimizers to be used seamlessly with Optax transformations like
    chain(), learning rate schedules, gradient clipping, etc.

    Args:
        manifold: The Riemannian manifold on which optimization occurs.
        learning_rate: Learning rate (can be a scalar or an Optax schedule).
        method: Optimization method ('sgd' or 'adam').
        b1: Exponential decay rate for first moment (Adam only).
        b2: Exponential decay rate for second moment (Adam only).
        eps: Small constant for numerical stability (Adam only).

    Example:
        >>> from riemannax.manifolds import Sphere
        >>> manifold = Sphere(n=2)  # 2-sphere S^2 embedded in R^3
        >>> adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)
        >>> params = jnp.array([1.0, 0.0, 0.0])
        >>> state = adapter.init(params)
        >>> grads = jnp.array([0.1, 0.2, 0.1])
        >>> updates, new_state = adapter.update(grads, state, params)
        >>> new_params = optax.apply_updates(params, updates)
    """

    def __init__(
        self,
        manifold: Manifold,
        learning_rate: float | Callable[[int], float] = 0.01,
        method: str = "sgd",
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Initialize the Riemannian Optax adapter."""
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.method = method.lower()
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        if self.method not in ["sgd", "adam"]:
            raise ValueError(f"Unsupported method: {method}. Use 'sgd' or 'adam'.")

    def init(self, params: Array) -> RiemannianOptaxState:
        """Initialize the optimizer state.

        Args:
            params: Initial parameters on the manifold.

        Returns:
            Initial optimizer state.
        """
        if self.method == "adam":
            return RiemannianOptaxState(
                step_count=0,
                adam_m=jnp.zeros_like(params),
                adam_v=jnp.zeros_like(params),
            )
        else:  # sgd
            return RiemannianOptaxState(step_count=0)

    def update(
        self,
        grads: Array,
        state: RiemannianOptaxState,
        params: Array | None = None,
        **extra_args: Any,
    ) -> tuple[Array, RiemannianOptaxState]:
        """Compute Riemannian gradient updates.

        Args:
            grads: Euclidean gradients.
            state: Current optimizer state.
            params: Current parameters on the manifold.
            **extra_args: Additional arguments (e.g., 'value' for some transformations).

        Returns:
            Tuple of (updates, new_state) where updates are in Optax-compatible format.
        """
        if params is None:
            raise ValueError("params must be provided for Riemannian updates")

        # Project gradients to tangent space
        riemannian_grads = self.manifold.proj(params, grads)

        # Get learning rate (handle schedules)
        step = state.step_count
        lr = self.learning_rate(step) if callable(self.learning_rate) else self.learning_rate

        if self.method == "adam":
            # Adam update with Riemannian gradients
            if state.adam_m is None or state.adam_v is None:
                raise ValueError(
                    "Adam state is not initialized correctly for method='adam'. "
                    "Please initialize the state using `adapter.init(params)` before the first update."
                )
            m = self.b1 * state.adam_m + (1 - self.b1) * riemannian_grads
            v = self.b2 * state.adam_v + (1 - self.b2) * (riemannian_grads**2)

            # Bias correction
            m_hat = m / (1 - self.b1 ** (step + 1))
            v_hat = v / (1 - self.b2 ** (step + 1))

            # Compute the Riemannian tangent step (standard Adam formula)
            tangent_step = -lr * m_hat / (jnp.sqrt(v_hat) + self.eps)
        else:  # sgd
            tangent_step = -lr * riemannian_grads

        # Retract back onto the manifold and compute the ambient-space update
        # This ensures that when optax.apply_updates(params, updates) computes
        # params + updates, the result is exactly new_params on the manifold
        new_params = self.manifold.retr(params, tangent_step)

        # Parallel transport momentum vectors for Adam (critical for correctness)
        if self.method == "adam":
            m_transported = self.manifold.transp(params, new_params, m)
            v_transported = self.manifold.transp(params, new_params, v)
            # Ensure transported values are in tangent space
            m_transported = self.manifold.proj(new_params, m_transported)
            v_transported = self.manifold.proj(new_params, v_transported)
            new_state = RiemannianOptaxState(
                step_count=step + 1,
                adam_m=m_transported,
                adam_v=v_transported,
            )
        else:  # sgd
            new_state = RiemannianOptaxState(step_count=step + 1)

        updates = new_params - params

        return updates, new_state

    def __call__(
        self, grads: Array, state: RiemannianOptaxState, params: Array | None = None
    ) -> tuple[Array, RiemannianOptaxState]:
        """Convenience method to call update."""
        return self.update(grads, state, params)


def create_riemannian_optimizer(
    manifold: Manifold,
    method: str = "sgd",
    learning_rate: float | Callable[[int], float] = 0.01,
    **kwargs: Any,
) -> RiemannianOptaxAdapter:
    """Create a Riemannian optimizer compatible with Optax.

    Args:
        manifold: The Riemannian manifold for optimization.
        method: Optimization method ('sgd' or 'adam').
        learning_rate: Learning rate (scalar or schedule).
        **kwargs: Additional optimizer parameters (b1, b2, eps for Adam).

    Returns:
        Optax-compatible Riemannian optimizer.

    Example:
        >>> from riemannax.manifolds import Sphere
        >>> manifold = Sphere(n=2)  # 2-sphere S^2 embedded in R^3
        >>> optimizer = create_riemannian_optimizer(
        ...     manifold=manifold,
        ...     method="adam",
        ...     learning_rate=0.001
        ... )
    """
    return RiemannianOptaxAdapter(
        manifold=manifold,
        learning_rate=learning_rate,
        method=method,
        **kwargs,
    )


def chain_with_optax(
    riemannian_opt: RiemannianOptaxAdapter,
    *optax_transforms: optax.GradientTransformation,
) -> optax.GradientTransformation:
    """Chain a Riemannian optimizer with Optax transformations.

    This function allows you to compose Riemannian optimization with standard
    Optax gradient transformations like clipping, weight decay, etc.

    IMPORTANT: The Riemannian optimizer must be the LAST transformation in the chain.
    This ensures that Euclidean gradient transformations (clipping, weight decay, etc.)
    are applied before projection to the tangent space and retraction onto the manifold.

    Args:
        riemannian_opt: Riemannian Optax adapter.
        *optax_transforms: Optax gradient transformations to chain.

    Returns:
        Chained optimizer combining Riemannian and Optax transformations.

    Example:
        >>> from riemannax.manifolds import Sphere
        >>> manifold = Sphere(n=2)  # 2-sphere S^2 embedded in R^3
        >>> riemannian_opt = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)
        >>> optimizer = chain_with_optax(
        ...     riemannian_opt,
        ...     optax.clip_by_global_norm(1.0)
        ... )
    """
    # Chain transformations: first apply optax transforms, then Riemannian
    return optax.chain(*optax_transforms, riemannian_opt)


def validate_optax_compatibility(
    manifold: Manifold,
    optax_transform: optax.GradientTransformation,
) -> tuple[bool, str | None]:
    """Validate compatibility between Optax transformation and manifold constraints.

    Args:
        manifold: The Riemannian manifold.
        optax_transform: Optax gradient transformation to validate.

    Returns:
        Tuple of (is_compatible, error_message). If compatible, error_message is None.

    Note:
        Most Optax transformations are compatible with manifold constraints as long
        as they operate on gradients/updates before the Riemannian projection.
    """
    # Most Optax transformations are compatible when properly ordered
    # Transformations that might be incompatible would be those that directly
    # modify parameters in ways that violate manifold constraints
    # For now, we assume compatibility but provide a hook for future validation
    return True, None
