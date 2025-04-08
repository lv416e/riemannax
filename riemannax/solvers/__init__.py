"""Solver implementations for Riemannian optimization problems.

This module provides solvers for minimizing functions on Riemannian manifolds,
with various optimization algorithms and termination conditions.
"""

from .minimize import OptimizeResult, minimize

__all__ = ["OptimizeResult", "minimize"]
