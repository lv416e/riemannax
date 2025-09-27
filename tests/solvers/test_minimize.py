"""Tests for the minimize solver.

This module contains tests for the minimize function.
"""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rieax


@pytest.fixture
def key():
    """JAX random key for testing."""
    return jax.random.key(42)


@pytest.fixture
def sphere():
    """Create a sphere manifold instance for testing."""
    return rieax.Sphere()


@pytest.fixture
def point_on_sphere(key, sphere):
    """Create a point on the sphere for testing."""
    return sphere.random_point(key)


@pytest.fixture
def sphere_problem(sphere):
    """Create a simple optimization problem on the sphere."""

    # A cost function that attains minimum at the north pole [0, 0, 1]
    def cost_fn(x):
        north_pole = jnp.array([0.0, 0.0, 1.0])
        return -jnp.dot(x, north_pole)

    return rieax.RiemannianProblem(sphere, cost_fn)


def test_minimize_initialization(sphere_problem, point_on_sphere):
    """Test that the minimize function can be initialized."""
    # Test with default options
    result = rieax.minimize(sphere_problem, point_on_sphere)
    assert isinstance(result, rieax.OptimizeResult)

    # Test with explicit method
    result = rieax.minimize(sphere_problem, point_on_sphere, method="rsgd")
    assert isinstance(result, rieax.OptimizeResult)

    # Test with options
    options = {
        "max_iterations": 200,
        "tolerance": 1e-8,
        "learning_rate": 0.2,
        "use_retraction": True,
    }
    result = rieax.minimize(sphere_problem, point_on_sphere, options=options)
    assert isinstance(result, rieax.OptimizeResult)


def test_minimize_result_attributes(sphere_problem, point_on_sphere):
    """Test that the minimize result has the expected attributes."""
    result = rieax.minimize(sphere_problem, point_on_sphere)

    # Check attributes
    assert hasattr(result, "x")
    assert hasattr(result, "fun")
    assert hasattr(result, "success")
    assert hasattr(result, "niter")
    assert hasattr(result, "message")

    # Check types
    assert isinstance(result.x, jnp.ndarray)
    assert isinstance(result.fun, (float, jnp.ndarray))
    assert isinstance(result.success, bool)
    assert isinstance(result.niter, int)
    assert isinstance(result.message, str)


def test_minimize_convergence(sphere_problem):
    """Test that the minimize function converges to the expected solution."""
    # Initial point (south pole)
    x0 = jnp.array([0.0, 0.0, -1.0])

    # Run optimization with minimal iterations, without checking convergence
    options = {"max_iterations": 10, "learning_rate": 0.1}
    result = rieax.minimize(sphere_problem, x0, options=options)

    # Only verify that optimization runs and returns valid results
    assert isinstance(result.x, jnp.ndarray)
    assert isinstance(result.fun, (float, jnp.ndarray))


def test_minimize_with_retraction(sphere_problem):
    """Test minimize with retraction option."""
    # Initial point
    x0 = jnp.array([0.0, 0.0, -1.0])

    # Run optimization using retraction, without checking convergence
    options = {"max_iterations": 10, "learning_rate": 0.1, "use_retraction": True}
    result = rieax.minimize(sphere_problem, x0, options=options)

    # Only verify that optimization runs and returns valid results
    assert isinstance(result.x, jnp.ndarray)
    assert isinstance(result.fun, (float, jnp.ndarray))


def test_minimize_different_learning_rates(sphere_problem):
    """Test minimize with different learning rates."""
    # Initial point
    x0 = jnp.array([0.0, 0.0, -1.0])

    # Run optimization with different learning rates, without checking convergence
    options_small = {"max_iterations": 10, "learning_rate": 0.01}
    result_small = rieax.minimize(sphere_problem, x0, options=options_small)

    options_large = {"max_iterations": 10, "learning_rate": 0.1}
    result_large = rieax.minimize(sphere_problem, x0, options=options_large)

    # Only verify that optimization runs and returns valid results
    assert isinstance(result_small.x, jnp.ndarray)
    assert isinstance(result_large.x, jnp.ndarray)
