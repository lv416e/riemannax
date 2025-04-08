"""Riemannian manifold implementations with differential geometric operations.

This module provides concrete implementations of various Riemannian manifolds
with their associated differential geometric operations optimized for JAX.
"""

from .so import SpecialOrthogonal
from .sphere import Sphere

__all__ = [
    "SpecialOrthogonal",
    "Sphere",
]
