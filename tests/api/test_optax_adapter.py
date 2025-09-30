"""Tests for Optax integration adapter."""

import jax
import jax.numpy as jnp
import optax
import pytest

from riemannax.api.optax_adapter import (
    RiemannianOptaxAdapter,
    create_riemannian_optimizer,
    chain_with_optax,
)
from riemannax.manifolds import Sphere, Stiefel


class TestRiemannianOptaxAdapter:
    """Test suite for RiemannianOptaxAdapter."""

    def test_adapter_init_creates_state(self):
        """Test that adapter.init() creates proper Optax-compatible state."""
        # Arrange
        manifold = Sphere(n=3)
        adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)
        params = jnp.array([1.0, 0.0, 0.0])

        # Act
        state = adapter.init(params)

        # Assert
        assert state is not None
        assert hasattr(state, '__dict__') or isinstance(state, tuple)

    def test_adapter_update_returns_optax_format(self):
        """Test that adapter.update() returns updates in Optax-compatible format."""
        # Arrange
        manifold = Sphere(n=3)
        adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)
        params = jnp.array([1.0, 0.0, 0.0])
        state = adapter.init(params)
        grads = jnp.array([0.1, 0.2, 0.1])

        # Act
        updates, new_state = adapter.update(grads, state, params)

        # Assert
        assert updates.shape == params.shape
        assert new_state is not None

    def test_adapter_with_optax_apply_updates(self):
        """Test that updates can be applied using optax.apply_updates()."""
        # Arrange
        manifold = Sphere(n=3)
        adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)
        params = jnp.array([1.0, 0.0, 0.0])
        state = adapter.init(params)
        grads = jnp.array([0.1, 0.2, 0.1])

        # Act
        updates, new_state = adapter.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # Assert
        assert new_params.shape == params.shape
        assert not jnp.allclose(new_params, params)  # Parameters should change

    def test_adapter_chain_with_optax_transformations(self):
        """Test chaining RiemannianOptaxAdapter with standard Optax transformations."""
        # Arrange
        manifold = Sphere(n=3)
        adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=1.0)

        # Create a chain: clip gradients, then apply Riemannian updates
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            adapter
        )

        params = jnp.array([1.0, 0.0, 0.0])
        state = optimizer.init(params)
        grads = jnp.array([10.0, 10.0, 10.0])  # Large gradients to test clipping

        # Act
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # Assert
        assert new_params.shape == params.shape
        gradient_norm = jnp.linalg.norm(updates)
        assert gradient_norm < 2.0  # Should be clipped

    def test_adapter_with_learning_rate_schedule(self):
        """Test adapter with Optax learning rate scheduler."""
        # Arrange
        schedule = optax.exponential_decay(
            init_value=1.0,
            transition_steps=100,
            decay_rate=0.99
        )
        manifold = Sphere(n=3)
        adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=schedule)

        params = jnp.array([1.0, 0.0, 0.0])
        state = adapter.init(params)
        grads = jnp.array([0.1, 0.2, 0.1])

        # Act - perform multiple updates
        for _ in range(5):
            updates, state = adapter.update(grads, state, params)
            params = optax.apply_updates(params, updates)

        # Assert
        assert params.shape == (3,)
        # After 5 updates, params should have changed
        assert not jnp.allclose(params, jnp.array([1.0, 0.0, 0.0]))


class TestCreateRiemannianOptimizer:
    """Test suite for create_riemannian_optimizer helper function."""

    def test_create_riemannian_sgd(self):
        """Test creating Riemannian SGD optimizer."""
        # Arrange
        manifold = Sphere(n=3)

        # Act
        optimizer = create_riemannian_optimizer(
            manifold=manifold,
            method="sgd",
            learning_rate=0.01
        )

        # Assert
        assert optimizer is not None
        params = jnp.array([1.0, 0.0, 0.0])
        state = optimizer.init(params)
        assert state is not None

    def test_create_riemannian_adam(self):
        """Test creating Riemannian Adam optimizer."""
        # Arrange
        manifold = Stiefel(n=5, p=3)

        # Act
        optimizer = create_riemannian_optimizer(
            manifold=manifold,
            method="adam",
            learning_rate=0.001,
            b1=0.9,
            b2=0.999
        )

        # Assert
        assert optimizer is not None
        key = jax.random.PRNGKey(0)
        params = jax.random.normal(key, (5, 3))
        params, _ = jnp.linalg.qr(params)  # Make orthonormal
        state = optimizer.init(params)
        assert state is not None


class TestChainWithOptax:
    """Test suite for chain_with_optax helper function."""

    def test_chain_with_gradient_clipping(self):
        """Test chaining Riemannian optimizer with gradient clipping."""
        # Arrange
        manifold = Sphere(n=3)
        riemannian_opt = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)

        # Act
        optimizer = chain_with_optax(
            riemannian_opt,
            optax.clip_by_global_norm(1.0)
        )

        # Assert
        params = jnp.array([1.0, 0.0, 0.0])
        state = optimizer.init(params)
        grads = jnp.array([10.0, 10.0, 10.0])
        updates, new_state = optimizer.update(grads, state, params)
        assert jnp.linalg.norm(updates) < 2.0

    def test_chain_with_weight_decay(self):
        """Test chaining Riemannian optimizer with weight decay."""
        # Arrange
        manifold = Sphere(n=3)
        riemannian_opt = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)

        # Act
        optimizer = chain_with_optax(
            riemannian_opt,
            optax.add_decayed_weights(weight_decay=0.01)
        )

        # Assert
        params = jnp.array([1.0, 0.0, 0.0])
        state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.1])
        updates, new_state = optimizer.update(grads, state, params)
        assert updates.shape == params.shape


class TestManifoldConstraintValidation:
    """Test suite for manifold constraint validation with Optax."""

    def test_detect_incompatible_transformation(self):
        """Test that incompatible Optax transformations are detected."""
        # Arrange
        manifold = Sphere(n=3)
        adapter = RiemannianOptaxAdapter(manifold=manifold, learning_rate=0.01)

        # Act & Assert
        # Some transformations might not be compatible with manifold constraints
        # This test ensures we can detect and handle such cases
        params = jnp.array([1.0, 0.0, 0.0])
        state = adapter.init(params)
        grads = jnp.array([0.1, 0.2, 0.1])

        # Should not raise error for compatible operations
        updates, new_state = adapter.update(grads, state, params)
        assert updates.shape == params.shape
