"""Tests for Manifold-Constrained Parameters."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from riemannax.api.problems import ManifoldConstrainedParameter
from riemannax.manifolds import Stiefel, SymmetricPositiveDefinite


class TestManifoldParameterInterface:
    """Test ManifoldConstrainedParameter interface."""

    def test_initialization_with_stiefel(self):
        """Test initialization with Stiefel manifold."""
        manifold = Stiefel(n=5, p=3)
        initial_value = manifold.random_point(jax.random.PRNGKey(42))

        param = ManifoldConstrainedParameter(
            manifold=manifold, initial_value=initial_value
        )

        assert param.manifold == manifold
        assert param.value is not None
        assert param.value.shape == (5, 3)

    def test_initialization_with_spd(self):
        """Test initialization with SPD manifold."""
        manifold = SymmetricPositiveDefinite(n=4)
        initial_value = manifold.random_point(jax.random.PRNGKey(42))

        param = ManifoldConstrainedParameter(
            manifold=manifold, initial_value=initial_value
        )

        assert param.value.shape == (4, 4)


class TestManifoldParameterProjection:
    """Test projection functionality."""

    def test_project_returns_manifold_point(self):
        """Test that project returns a point on the manifold."""
        manifold = Stiefel(n=4, p=2)
        param = ManifoldConstrainedParameter(
            manifold=manifold,
            initial_value=manifold.random_point(jax.random.PRNGKey(42)),
        )

        # Perturb the value off manifold
        perturbed = param.value + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (4, 2))

        # Project back
        projected = param.project(perturbed)

        # Check orthonormality for Stiefel
        gram = projected.T @ projected
        assert jnp.allclose(gram, jnp.eye(2), atol=1e-5)

    def test_project_spd_ensures_positive_definite(self):
        """Test projection onto SPD manifold."""
        manifold = SymmetricPositiveDefinite(n=3)
        param = ManifoldConstrainedParameter(
            manifold=manifold,
            initial_value=manifold.random_point(jax.random.PRNGKey(42)),
        )

        # Create near-PD matrix (symmetric but maybe not PD)
        A = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
        symmetric = (A + A.T) / 2.0

        projected = param.project(symmetric)

        # Check SPD property
        eigenvalues = jnp.linalg.eigvalsh(projected)
        assert jnp.all(eigenvalues > 1e-9)  # Positive definite


class TestManifoldParameterGradient:
    """Test Riemannian gradient computation."""

    def test_riemannian_grad_from_euclidean(self):
        """Test conversion of Euclidean gradient to Riemannian gradient."""
        manifold = Stiefel(n=5, p=3)
        initial = manifold.random_point(jax.random.PRNGKey(42))

        param = ManifoldConstrainedParameter(manifold=manifold, initial_value=initial)

        # Euclidean gradient (random direction)
        euclidean_grad = jax.random.normal(jax.random.PRNGKey(1), (5, 3))

        # Convert to Riemannian gradient
        riemannian_grad = param.riemannian_gradient(param.value, euclidean_grad)

        # For Stiefel, check tangent space property: X^T G + G^T X = 0 (skew-symmetric)
        skew_check = param.value.T @ riemannian_grad + riemannian_grad.T @ param.value
        assert jnp.allclose(skew_check, jnp.zeros((3, 3)), atol=1e-5)


class TestManifoldParameterUpdate:
    """Test parameter update with manifold constraints."""

    def test_update_keeps_parameter_on_manifold(self):
        """Test that updates remain on manifold."""
        manifold = Stiefel(n=4, p=2)
        initial = manifold.random_point(jax.random.PRNGKey(42))

        param = ManifoldConstrainedParameter(manifold=manifold, initial_value=initial)

        # Simulate gradient update
        gradient = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
        learning_rate = 0.01

        updated_value = param.update(gradient, learning_rate)

        # Check still on Stiefel (orthonormal columns)
        gram = updated_value.T @ updated_value
        assert jnp.allclose(gram, jnp.eye(2), atol=1e-5)

    def test_update_with_spd_maintains_spd(self):
        """Test updates maintain SPD property."""
        manifold = SymmetricPositiveDefinite(n=3)
        initial = manifold.random_point(jax.random.PRNGKey(42))

        param = ManifoldConstrainedParameter(manifold=manifold, initial_value=initial)

        # Simulate symmetric gradient
        G = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
        gradient = (G + G.T) / 2.0
        learning_rate = 0.001

        updated_value = param.update(gradient, learning_rate)

        # Check SPD
        eigenvalues = jnp.linalg.eigvalsh(updated_value)
        assert jnp.all(eigenvalues > 1e-9)  # Positive definite


class TestManifoldParameterIntegration:
    """Test integration scenarios."""

    def test_multiple_update_steps(self):
        """Test multiple gradient descent steps."""
        manifold = Stiefel(n=5, p=3)
        param = ManifoldConstrainedParameter(
            manifold=manifold,
            initial_value=manifold.random_point(jax.random.PRNGKey(42)),
        )

        # Perform 10 update steps
        key = jax.random.PRNGKey(100)
        for i in range(10):
            gradient = jax.random.normal(jax.random.fold_in(key, i), (5, 3))
            param.value = param.update(gradient, learning_rate=0.01)

            # Check manifold constraint after each step
            gram = param.value.T @ param.value
            assert jnp.allclose(gram, jnp.eye(3), atol=1e-4)

    def test_value_getter_setter(self):
        """Test value property getter and setter."""
        manifold = Stiefel(n=4, p=2)
        initial = manifold.random_point(jax.random.PRNGKey(42))

        param = ManifoldConstrainedParameter(manifold=manifold, initial_value=initial)

        # Get value
        current_value = param.value
        assert current_value.shape == (4, 2)

        # Set new value (should be projected)
        new_value = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
        param.value = new_value

        # Check it was projected onto manifold
        gram = param.value.T @ param.value
        assert jnp.allclose(gram, jnp.eye(2), atol=1e-5)
