"""Tests for the Riemannian problem base class.

This module contains tests for the RiemannianProblem class.
"""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rx


@pytest.fixture
def key():
    """JAX random key for testing."""
    return jax.random.key(42)


@pytest.fixture
def sphere():
    """Create a sphere manifold instance for testing."""
    return rx.Sphere()


@pytest.fixture
def point_on_sphere(key, sphere):
    """Create a random point on the sphere for testing."""
    return sphere.random_point(jax.random.key(42))


def test_riemannian_problem_initialization(sphere):
    """Test initialization of the RiemannianProblem class."""

    # Simple cost function
    def cost_fn(x):
        return jnp.sum(x)

    # Initialize with just the cost function
    problem = rx.RiemannianProblem(sphere, cost_fn)
    assert problem.manifold is sphere
    assert problem.cost_fn is cost_fn
    assert problem.grad_fn is None
    assert problem.euclidean_grad_fn is None

    # Initialize with Riemannian gradient function
    def grad_fn(x):
        return sphere.proj(x, jnp.ones_like(x))

    problem = rx.RiemannianProblem(sphere, cost_fn, grad_fn=grad_fn)
    assert problem.grad_fn is grad_fn

    # Initialize with Euclidean gradient function
    def euclidean_grad_fn(x):
        return jnp.ones_like(x)

    problem = rx.RiemannianProblem(sphere, cost_fn, euclidean_grad_fn=euclidean_grad_fn)
    assert problem.euclidean_grad_fn is euclidean_grad_fn


def test_riemannian_problem_cost(sphere, point_on_sphere):
    """Test the cost method of the RiemannianProblem class."""

    # Create a cost function
    def cost_fn(x):
        return jnp.sum(x)

    # Create a problem
    problem = rx.RiemannianProblem(sphere, cost_fn)

    # Evaluate the cost
    cost = problem.cost(point_on_sphere)

    # Check that it matches the expected value
    expected_cost = jnp.sum(point_on_sphere)
    assert jnp.allclose(cost, expected_cost)


def test_riemannian_problem_grad_with_grad_fn(sphere, point_on_sphere):
    """Test the grad method when a Riemannian gradient function is provided."""

    # Create a cost function and gradient function
    def cost_fn(x):
        return jnp.sum(x)

    def grad_fn(x):
        return sphere.proj(x, jnp.ones_like(x))

    # Create a problem with the Riemannian gradient function
    problem = rx.RiemannianProblem(sphere, cost_fn, grad_fn=grad_fn)

    # Compute the gradient
    gradient = problem.grad(point_on_sphere)

    # Check that it matches the expected value
    expected_gradient = sphere.proj(point_on_sphere, jnp.ones_like(point_on_sphere))
    assert jnp.allclose(gradient, expected_gradient)


def test_riemannian_problem_grad_with_euclidean_grad_fn(sphere, point_on_sphere):
    """Test the grad method when an Euclidean gradient function is provided."""

    # Create a cost function and Euclidean gradient function
    def cost_fn(x):
        return jnp.sum(x)

    def euclidean_grad_fn(x):
        return jnp.ones_like(x)

    # Create a problem with the Euclidean gradient function
    problem = rx.RiemannianProblem(sphere, cost_fn, euclidean_grad_fn=euclidean_grad_fn)

    # Compute the gradient
    gradient = problem.grad(point_on_sphere)

    # Check that it matches the expected value
    expected_gradient = sphere.proj(point_on_sphere, jnp.ones_like(point_on_sphere))
    assert jnp.allclose(gradient, expected_gradient)


def test_riemannian_problem_grad_with_autodiff(sphere, point_on_sphere):
    """Test the grad method when using automatic differentiation."""

    # Create a cost function
    def cost_fn(x):
        return jnp.sum(x)

    # Create a problem without gradient functions
    problem = rx.RiemannianProblem(sphere, cost_fn)

    # Compute the gradient
    gradient = problem.grad(point_on_sphere)

    # Check that it matches the expected value
    expected_gradient = sphere.proj(point_on_sphere, jnp.ones_like(point_on_sphere))
    assert jnp.allclose(gradient, expected_gradient)
