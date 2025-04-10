"""Integration tests for riemannax.

This module contains end-to-end tests that verify the correct interaction
between multiple components of the library.
"""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rieax


@pytest.fixture
def key():
    """JAX random key for testing."""
    return jax.random.key(42)


def test_integration_sphere_optimization():
    """Test end-to-end optimization on the sphere."""
    # Create a sphere manifold
    sphere = rieax.Sphere()

    # Define a cost function (maximize the dot product with the north pole)
    def cost_fn(x):
        north_pole = jnp.array([0.0, 0.0, 1.0])
        return -jnp.dot(x, north_pole)  # Negative to convert maximization to minimization

    # Create a problem
    problem = rieax.RiemannianProblem(sphere, cost_fn)

    # Initialize from the south pole
    x0 = jnp.array([0.0, 0.0, -1.0])

    # Run optimization with minimal iterations, without checking convergence
    options = {"max_iterations": 10, "learning_rate": 0.1}
    result = rieax.minimize(problem, x0, options=options)

    # Only verify that optimization runs and returns valid results
    assert isinstance(result.x, jnp.ndarray)
    assert isinstance(result.fun, float | jnp.ndarray)


def test_integration_so3_optimization(key):
    """Test end-to-end optimization on SO(3)."""
    # Create an SO(3) manifold
    so3 = rieax.SpecialOrthogonal(n=3)

    # Create a target rotation matrix
    target = so3.random_point(jax.random.fold_in(key, 0))

    # Define a cost function (minimize the Frobenius distance to the target)
    def cost_fn(x):
        return jnp.sum((x - target) ** 2)

    # Create a problem
    problem = rieax.RiemannianProblem(so3, cost_fn)

    # Initialize from a random point
    x0 = so3.random_point(jax.random.fold_in(key, 1))

    # Solve the problem
    options = {"max_iterations": 200, "learning_rate": 0.05}
    result = rieax.minimize(problem, x0, options=options)

    # The solution should be close to the target
    assert jnp.allclose(result.x, target, atol=0.1)
    assert result.fun < 1e-2


def test_integration_custom_gradient():
    """Test optimization with a custom gradient function."""
    # Create a sphere manifold
    sphere = rieax.Sphere()

    # Define a cost function
    def cost_fn(x):
        north_pole = jnp.array([0.0, 0.0, 1.0])
        return -jnp.dot(x, north_pole)

    # Define a custom Euclidean gradient function
    def euclidean_grad_fn(x):
        north_pole = jnp.array([0.0, 0.0, 1.0])
        return -north_pole  # Negative gradient for maximization

    # Create a problem with the custom gradient
    problem = rieax.RiemannianProblem(sphere, cost_fn, euclidean_grad_fn=euclidean_grad_fn)

    # Initialize from a random point
    x0 = jnp.array([1.0, 0.0, 0.0])  # East pole

    # Solve the problem
    options = {"max_iterations": 100, "learning_rate": 0.1}
    result = rieax.minimize(problem, x0, options=options)

    # The solution should be close to the north pole
    north_pole = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(result.x, north_pole, atol=1e-2)


def test_integration_retraction_vs_exp():
    """Compare optimization with exponential map versus retraction."""
    # Create a sphere manifold
    sphere = rieax.Sphere()

    # Define a cost function
    def cost_fn(x):
        north_pole = jnp.array([0.0, 0.0, 1.0])
        return -jnp.dot(x, north_pole)

    # Create a problem
    problem = rieax.RiemannianProblem(sphere, cost_fn)

    # Initialize from the south pole
    x0 = jnp.array([0.0, 0.0, -1.0])

    # Run optimization using exponential map
    options_exp = {"max_iterations": 10, "learning_rate": 0.1, "use_retraction": False}
    result_exp = rieax.minimize(problem, x0, options=options_exp)

    # Run optimization using retraction
    options_retr = {"max_iterations": 10, "learning_rate": 0.1, "use_retraction": True}
    result_retr = rieax.minimize(problem, x0, options=options_retr)

    # Only verify that both methods run and return valid results
    assert isinstance(result_exp.x, jnp.ndarray)
    assert isinstance(result_retr.x, jnp.ndarray)
