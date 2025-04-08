"""Optimization algorithms for Riemannian manifolds.

This package contains optimization algorithms designed specifically for Riemannian manifolds.
"""

from .sgd import riemannian_gradient_descent
from .state import OptState

__all__ = [
    "OptState",
    "riemannian_gradient_descent",
]
