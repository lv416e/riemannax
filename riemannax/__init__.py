"""JAX-native Riemannian manifold optimization framework.

This package implements optimization algorithms for Riemannian manifolds using JAX,
enabling GPU-accelerated gradient-based optimization on non-Euclidean domains.
"""

__version__ = "0.0.1"

from .manifolds import SpecialOrthogonal, Sphere
from .optimizers import riemannian_gradient_descent
from .problems import RiemannianProblem
from .solvers import OptimizeResult, minimize

__all__ = [
    "OptimizeResult",
    "RiemannianProblem",
    "SpecialOrthogonal",
    "Sphere",
    "minimize",
    "riemannian_gradient_descent",
]
