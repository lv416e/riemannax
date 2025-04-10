"""Optimizer state types for Riemannian optimization.

This module contains the state classes used by Riemannian optimization algorithms.
These classes track the current state of optimization on Riemannian manifolds.
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
from jax import tree_util


@dataclasses.dataclass
class OptState:
    """Optimizer state for Riemannian optimization algorithms.

    This class represents the state of an optimizer during Riemannian optimization.
    It is registered as a JAX PyTree to enable its use in JAX's functional transformations.

    Attributes:
        x: Current point on the manifold.
    """

    x: jnp.ndarray  # Current point on the manifold

    def tree_flatten(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Flatten the OptState for JAX.

        Returns:
            A tuple containing a tuple of arrays (children) and auxiliary data (None).
        """
        children = (self.x,)
        aux_data: dict[str, Any] = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Any, ...]) -> "OptState":
        """Unflatten the OptState for JAX.

        Args:
            aux_data: Auxiliary data (empty dict).
            children: A tuple containing the arrays from tree_flatten.

        Returns:
            The reconstructed OptState.
        """
        return cls(x=children[0])


# Register the OptState class as a PyTree node
tree_util.register_pytree_node_class(OptState)
