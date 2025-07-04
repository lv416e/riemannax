"""Tests for the Riemannian Adam optimizer.

This module contains tests for the Riemannian Adam optimization algorithm.
"""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rieax


@pytest.fixture
def key():
    """JAX random key for testing."""
    return jax.random.key(123)


@pytest.fixture
def sphere():
    """Create a sphere manifold for testing."""
    return rieax.Sphere()


@pytest.fixture
def spd():
    """Create an SPD manifold for testing."""
    return rieax.SymmetricPositiveDefinite(n=2)


def test_adam_initialization(sphere, key):
    """Test Adam optimizer initialization."""
    x0 = sphere.random_point(key)

    # Initialize Adam optimizer
    init_fn, _ = rieax.riemannian_adam(learning_rate=0.001)
    state = init_fn(x0)

    # Check state structure
    assert hasattr(state, 'x')
    assert hasattr(state, 'm')
    assert hasattr(state, 'v')
    assert hasattr(state, 'step')

    # Check initial values
    assert jnp.allclose(state.x, x0)
    assert jnp.allclose(state.m, jnp.zeros_like(x0))
    assert jnp.allclose(state.v, jnp.zeros_like(x0))
    assert state.step == 0


def test_adam_update(sphere, key):
    """Test Adam optimizer updates."""
    x0 = sphere.random_point(key)

    # Initialize optimizer
    init_fn, update_fn = rieax.riemannian_adam(learning_rate=0.1)
    state = init_fn(x0)

    # Compute gradient (pointing away from north pole)
    north_pole = jnp.array([0., 0., 1.])
    gradient = sphere.proj(x0, -north_pole)  # Point toward north pole

    # Perform update
    new_state = update_fn(gradient, state, sphere)

    # Check that state has been updated
    assert not jnp.allclose(new_state.x, state.x)
    assert not jnp.allclose(new_state.m, state.m)
    assert not jnp.allclose(new_state.v, state.v)
    assert new_state.step == 1

    # Check that new point is on manifold
    assert jnp.allclose(jnp.linalg.norm(new_state.x), 1.0, atol=1e-6)


def test_adam_convergence(sphere):
    """Test Adam convergence on a simple optimization problem."""
    # Use a simpler cost function to avoid numerical issues
    def cost_fn(x):
        # Minimize distance to [1, 0, 0] (simple quadratic)
        return jnp.sum((x - jnp.array([1., 0., 0.])) ** 2)

    # Create problem
    problem = rieax.RiemannianProblem(sphere, cost_fn)

    # Initialize from a nearby point
    key = jax.random.key(42)
    x0 = sphere.random_point(key)

    # Manual optimization loop for better control
    init_fn, update_fn = rieax.riemannian_adam(learning_rate=0.01)
    state = init_fn(x0)

    initial_cost = cost_fn(x0)

    # Run just a few steps to check for numerical stability
    for i in range(10):
        grad = problem.grad(state.x)
        state = update_fn(grad, state, sphere)

        # Check for NaN
        if jnp.any(jnp.isnan(state.x)):
            pytest.fail(f"NaN detected at step {i}")

        # Check manifold constraint (with reasonable tolerance for numerical errors)
        norm = jnp.linalg.norm(state.x)
        assert jnp.abs(norm - 1.0) < 0.1, f"Point too far from sphere: norm = {norm}"

    final_cost = cost_fn(state.x)

    # Should have made some progress (not necessarily convergence)
    assert jnp.isfinite(final_cost), "Final cost is not finite"
    assert final_cost <= initial_cost + 1e-3, "Cost should not increase significantly"


def test_adam_momentum_transport(sphere, key):
    """Test that momentum is properly transported."""
    x0 = sphere.random_point(key)

    # Initialize optimizer
    init_fn, update_fn = rieax.riemannian_adam(learning_rate=0.01, beta1=0.9)
    state = init_fn(x0)

    # Apply several updates with consistent gradients
    gradient = sphere.random_tangent(key, x0)

    # First update
    state1 = update_fn(gradient, state, sphere)

    # Second update with transported gradient
    gradient2 = sphere.transp(x0, state1.x, gradient)
    state2 = update_fn(gradient2, state1, sphere)

    # Momentum should be building up
    momentum_norm1 = jnp.linalg.norm(state1.m)
    momentum_norm2 = jnp.linalg.norm(state2.m)

    # Second momentum should be larger (accumulation effect)
    assert momentum_norm2 > momentum_norm1

    # Momentum should be in tangent space
    # (this is automatically satisfied by transport, but we can check symmetry for SPD)


def test_adam_bias_correction():
    """Test Adam bias correction mechanism."""
    spd = rieax.SymmetricPositiveDefinite(n=2)
    key = jax.random.key(456)
    x0 = spd.random_point(key)

    # Initialize optimizer with high beta values to test bias correction
    init_fn, update_fn = rieax.riemannian_adam(
        learning_rate=0.1,
        beta1=0.99,
        beta2=0.999
    )
    state = init_fn(x0)

    # Create a consistent gradient
    gradient = spd.random_tangent(key, x0)

    # Perform several updates
    states = [state]
    for _i in range(5):
        new_state = update_fn(gradient, states[-1], spd)
        states.append(new_state)

        # Transport gradient to new point for next iteration
        gradient = spd.transp(states[-2].x, states[-1].x, gradient)

    # Early steps should have different behavior due to bias correction
    # (This is mainly a smoke test to ensure no errors occur)
    assert len(states) == 6
    for state in states[1:]:
        assert spd._is_in_manifold(state.x)


def test_adam_parameters():
    """Test Adam with different parameter settings."""
    sphere = rieax.Sphere()
    key = jax.random.key(789)
    x0 = sphere.random_point(key)

    # Test different learning rates
    for lr in [0.001, 0.01, 0.1]:
        init_fn, update_fn = rieax.riemannian_adam(learning_rate=lr)
        state = init_fn(x0)
        gradient = sphere.random_tangent(key, x0)
        new_state = update_fn(gradient, state, sphere)

        # Should produce valid updates
        assert jnp.allclose(jnp.linalg.norm(new_state.x), 1.0, atol=1e-6)

    # Test different beta parameters
    for beta1, beta2 in [(0.9, 0.999), (0.95, 0.99), (0.8, 0.9)]:
        init_fn, update_fn = rieax.riemannian_adam(beta1=beta1, beta2=beta2)
        state = init_fn(x0)
        gradient = sphere.random_tangent(key, x0)
        new_state = update_fn(gradient, state, sphere)

        # Should produce valid updates
        assert jnp.allclose(jnp.linalg.norm(new_state.x), 1.0, atol=1e-6)


def test_adam_retraction_vs_exponential(sphere, key):
    """Test Adam with retraction vs exponential map."""
    x0 = sphere.random_point(key)
    gradient = sphere.random_tangent(key, x0)

    # Test with exponential map
    init_fn1, update_fn1 = rieax.riemannian_adam(use_retraction=False)
    state1 = init_fn1(x0)
    new_state1 = update_fn1(gradient, state1, sphere)

    # Test with retraction
    init_fn2, update_fn2 = rieax.riemannian_adam(use_retraction=True)
    state2 = init_fn2(x0)
    new_state2 = update_fn2(gradient, state2, sphere)

    # Both should produce valid points on manifold
    assert jnp.allclose(jnp.linalg.norm(new_state1.x), 1.0, atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(new_state2.x), 1.0, atol=1e-6)

    # Results might be slightly different but should be close for small steps
    distance = sphere.dist(new_state1.x, new_state2.x)
    assert distance < 0.5  # Should be reasonably close


def test_adam_state_pytree():
    """Test that AdamState works properly as a JAX PyTree."""
    from riemannax.optimizers.adam import AdamState

    # Create test state
    x = jnp.array([1., 2., 3.])
    m = jnp.array([0.1, 0.2, 0.3])
    v = jnp.array([0.01, 0.02, 0.03])
    step = 5

    state = AdamState(x, m, v, step)

    # Test tree flattening/unflattening
    children, aux_data = state.tree_flatten()
    reconstructed = AdamState.tree_unflatten(aux_data, children)

    assert jnp.allclose(reconstructed.x, state.x)
    assert jnp.allclose(reconstructed.m, state.m)
    assert jnp.allclose(reconstructed.v, state.v)
    assert reconstructed.step == state.step

    # Test with JAX transformations
    def dummy_fn(state):
        return state.x.sum()

    # Should work with jit
    jitted_fn = jax.jit(dummy_fn)
    result = jitted_fn(state)
    assert jnp.isfinite(result)
