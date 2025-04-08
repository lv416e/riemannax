"""Tests for the Riemannian gradient descent optimizer.

This module contains tests for the riemannian_gradient_descent optimizer.
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
    return sphere.random_point(key)


def test_sgd_initialization():
    """Test initialization of the Riemannian gradient descent optimizer."""
    # Default initialization
    _, _ = rx.riemannian_gradient_descent()

    # Explicit learning rate
    _, _ = rx.riemannian_gradient_descent(learning_rate=0.5)

    # Use retraction
    _, _ = rx.riemannian_gradient_descent(use_retraction=True)


def test_sgd_init_fn(point_on_sphere):
    """Test the init function of the optimizer."""
    init_fn, _ = rx.riemannian_gradient_descent()

    # Initialize state with a point
    state = init_fn(point_on_sphere)

    # Check that the state contains the point
    assert jnp.allclose(state.x, point_on_sphere)


def test_sgd_update_fn_with_exp(sphere, point_on_sphere):
    """Test the update function of the optimizer using exponential map."""
    learning_rate = 0.1
    _, update_fn = rx.riemannian_gradient_descent(learning_rate=learning_rate, use_retraction=False)

    # Create a gradient (tangent vector)
    gradient = jnp.array([0.1, 0.2, 0.0])
    gradient = sphere.proj(point_on_sphere, gradient)

    # Create state
    state = rx.optimizers.OptState(x=point_on_sphere)

    # Update state
    new_state = update_fn(gradient, state, sphere)

    # Check that the new point is on the sphere
    assert jnp.abs(jnp.linalg.norm(new_state.x) - 1.0) < 1e-6

    # Check that the update moves in the negative gradient direction
    expected_point = sphere.exp(point_on_sphere, -learning_rate * gradient)
    assert jnp.allclose(new_state.x, expected_point)


def test_sgd_update_fn_with_retr(sphere, point_on_sphere):
    """Test the update function of the optimizer using retraction."""
    learning_rate = 0.1
    _, update_fn = rx.riemannian_gradient_descent(learning_rate=learning_rate, use_retraction=True)

    # Create a gradient (tangent vector)
    gradient = jnp.array([0.1, 0.2, 0.0])
    gradient = sphere.proj(point_on_sphere, gradient)

    # Create state
    state = rx.optimizers.OptState(x=point_on_sphere)

    # Update state
    new_state = update_fn(gradient, state, sphere)

    # Check that the new point is on the sphere
    assert jnp.abs(jnp.linalg.norm(new_state.x) - 1.0) < 1e-6

    # Check that the update moves in the negative gradient direction
    expected_point = sphere.retr(point_on_sphere, -learning_rate * gradient)
    assert jnp.allclose(new_state.x, expected_point)
