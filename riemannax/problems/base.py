"""Base classes for Riemannian optimization problems.

This module defines the core interfaces for optimization problems on
Riemannian manifolds.
"""

import jax


class RiemannianProblem:
    """Base class for optimization problems on Riemannian manifolds.

    This class encapsulates the objective function and its gradient
    for optimization on a Riemannian manifold.
    """

    def __init__(self, manifold, cost_fn, grad_fn=None, euclidean_grad_fn=None):
        """Initialize optimization problem.

        Args:
            manifold: The manifold to optimize on.
            cost_fn: Objective function to minimize.
            grad_fn: Optional function returning the Riemannian gradient.
            euclidean_grad_fn: Optional function returning the Euclidean gradient,
                which will be projected to get the Riemannian gradient.
        """
        self.manifold = manifold
        self.cost_fn = cost_fn
        self.grad_fn = grad_fn
        self.euclidean_grad_fn = euclidean_grad_fn

    def cost(self, x):
        """Evaluate cost at point x.

        Args:
            x: Point on the manifold.

        Returns:
            Value of the cost function at x.
        """
        return self.cost_fn(x)

    def grad(self, x):
        """Compute Riemannian gradient at point x.

        The Riemannian gradient is computed using one of three approaches,
        in order of preference:
        1. Use the provided grad_fn if available.
        2. Use the euclidean_grad_fn and project if available.
        3. Compute the Euclidean gradient using JAX autodiff and project.

        Args:
            x: Point on the manifold.

        Returns:
            The Riemannian gradient at x.
        """
        if self.grad_fn is not None:
            # Use the provided Riemannian gradient function
            return self.grad_fn(x)
        elif self.euclidean_grad_fn is not None:
            # Use the provided Euclidean gradient function and project
            egrad = self.euclidean_grad_fn(x)
            return self.manifold.proj(x, egrad)
        else:
            # Use automatic differentiation to compute the Euclidean gradient
            egrad = jax.grad(self.cost_fn)(x)
            # Project onto the tangent space to get the Riemannian gradient
            return self.manifold.proj(x, egrad)
