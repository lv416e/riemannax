"""Optimization algorithms for Riemannian manifolds.

This package contains optimization algorithms designed specifically for Riemannian manifolds.
"""

from .adam import riemannian_adam
from .momentum import riemannian_momentum
from .sgd import riemannian_gradient_descent
from .state import OptState

__all__ = [
    "OptState",
    "riemannian_adam",
    "riemannian_gradient_descent",
    "riemannian_momentum",
]
