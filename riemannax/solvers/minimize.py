"""Minimization solvers for Riemannian optimization problems.

This module provides implementations of solvers for minimizing functions
on Riemannian manifolds using various optimization algorithms.
"""

import dataclasses

import jax.numpy as jnp
from jax import lax

from ..optimizers import riemannian_gradient_descent


@dataclasses.dataclass
class OptimizeResult:
    """Optimization result with solution and diagnostic information.

    Attributes:
        x: Final point on the manifold.
        fun: Final function value.
        success: Whether the optimization was successful.
        niter: Number of iterations performed.
        message: Description of the termination reason.
    """

    x: jnp.ndarray  # Final point
    fun: float  # Final function value
    success: bool = True  # Whether the optimization was successful
    niter: int = 0  # Number of iterations performed
    message: str = "Optimization terminated successfully."  # Termination reason


def minimize(problem, x0, method="rsgd", options=None):
    """Minimize a function on a Riemannian manifold.

    Args:
        problem: RiemannianProblem instance defining the optimization problem.
        x0: Initial point on the manifold.
        method: Optimization method to use.
            - 'rsgd': Riemannian gradient descent
            - 'rcg': Riemannian conjugate gradient
            - 'lbfgs': Riemannian L-BFGS
        options: Dictionary of solver options, including:
            - max_iterations: Maximum number of iterations.
            - tolerance: Stopping tolerance for optimality.
            - learning_rate: Step size for the optimizer.
            - use_retraction: Whether to use retraction instead of exponential map.

    Returns:
        OptimizeResult containing the solution and diagnostic information.
    """
    if options is None:
        options = {}

    # Default options
    max_iterations = options.get("max_iterations", 100)
    learning_rate = options.get("learning_rate", 0.1)
    use_retraction = options.get("use_retraction", False)

    # Initialize the optimizer
    if method == "rsgd":
        init_fn, update_fn = riemannian_gradient_descent(learning_rate=learning_rate, use_retraction=use_retraction)
    else:
        raise ValueError(f"Unsupported optimization method: {method}")

    # Initialize optimizer state
    state = init_fn(x0)

    # Define the step function for the loop
    def step_fn(i, state_i):
        x_i = state_i.x
        grad_i = problem.grad(x_i)
        new_state = update_fn(grad_i, state_i, problem.manifold)
        return new_state

    # Run the optimizer
    final_state = lax.fori_loop(0, max_iterations, step_fn, state)

    # Compute final cost
    final_cost = problem.cost(final_state.x)

    # Return the optimization result
    return OptimizeResult(x=final_state.x, fun=final_cost, niter=max_iterations)
