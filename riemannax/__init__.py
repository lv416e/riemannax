"""JAX-native Riemannian manifold optimization framework.

This package implements optimization algorithms for Riemannian manifolds using JAX,
enabling GPU-accelerated gradient-based optimization on non-Euclidean domains.
"""

__version__ = "0.0.2"

from .manifolds import Grassmann, SpecialOrthogonal, Sphere, Stiefel, SymmetricPositiveDefinite
from .optimizers import riemannian_adam, riemannian_gradient_descent, riemannian_momentum
from .problems import RiemannianProblem
from .solvers import OptimizeResult, minimize

__all__ = [
    "Grassmann",
    "OptimizeResult",
    "RiemannianProblem",
    "SpecialOrthogonal",
    "Sphere",
    "Stiefel",
    "SymmetricPositiveDefinite",
    "minimize",
    "riemannian_adam",
    "riemannian_gradient_descent",
    "riemannian_momentum",
]
