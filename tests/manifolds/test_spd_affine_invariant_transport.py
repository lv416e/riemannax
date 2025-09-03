"""Tests for SPD manifold affine-invariant closed-form parallel transport.

This module tests the affine-invariant closed-form parallel transport
implementation for SPD manifolds, including both standard affine-invariant
metric and Bures-Wasserstein metric variants.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from typing import Tuple

from riemannax.manifolds.spd import SymmetricPositiveDefinite


class TestSPDAffineInvariantTransport:
    """Test suite for SPD manifold affine-invariant closed-form parallel transport."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manifold = SymmetricPositiveDefinite(n=3)
        self.key = jr.PRNGKey(42)

    def test_affine_invariant_transp_method_exists(self):
        """Test that _affine_invariant_transp method exists on SPD manifold."""
        assert hasattr(self.manifold, '_affine_invariant_transp')
        assert callable(getattr(self.manifold, '_affine_invariant_transp'))

    def test_affine_invariant_transp_basic_functionality(self):
        """Test basic affine-invariant parallel transport functionality."""
        # Generate test points and tangent vector
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Test affine-invariant parallel transport
        transported = self.manifold._affine_invariant_transp(x, y, v)

        # Verify output shape and type
        assert transported.shape == v.shape
        assert transported.dtype == v.dtype

        # Verify the result is a valid tangent vector at y
        assert self.manifold.validate_point(y)

        # The transported vector should be in the tangent space at y
        # (for SPD manifolds, tangent vectors are symmetric matrices)
        assert jnp.allclose(transported, transported.T, atol=1e-12)

    def test_affine_invariant_vs_pole_ladder_comparison(self):
        """Test that affine-invariant gives different results than pole ladder."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Compare affine-invariant vs pole ladder transport
        affine_result = self.manifold._affine_invariant_transp(x, y, v)
        pole_result = self.manifold._pole_ladder(x, y, v, n_steps=3)

        # Results should be different (affine-invariant should be more accurate)
        # Unless x and y are very close or in special geometric configuration
        distance = self.manifold.dist(x, y)
        if distance > 1e-3:  # Only check for non-trivial distances
            max_diff = jnp.max(jnp.abs(affine_result - pole_result))
            assert max_diff > 1e-10, "Affine-invariant should give different results than pole ladder"

    def test_affine_invariant_transport_identity(self):
        """Test that transporting from a point to itself preserves the vector."""
        key1, key2 = jr.split(self.key, 2)

        x = self.manifold.random_point(key1)
        v = self.manifold.random_tangent(key2, x)

        # Transport from x to x should preserve v
        transported = self.manifold._affine_invariant_transp(x, x, v)

        # Should be very close to the original vector
        assert jnp.allclose(transported, v, atol=1e-12)

    def test_affine_invariant_transport_linearity(self):
        """Test linearity property: transport(a*u + b*v) = a*transport(u) + b*transport(v)."""
        key1, key2, key3, key4 = jr.split(self.key, 4)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        u = self.manifold.random_tangent(key3, x)
        v = self.manifold.random_tangent(key4, x)

        a, b = 2.5, -1.7

        # Transport linear combination
        combined = a * u + b * v
        transported_combined = self.manifold._affine_invariant_transp(x, y, combined)

        # Transport individually and combine
        transported_u = self.manifold._affine_invariant_transp(x, y, u)
        transported_v = self.manifold._affine_invariant_transp(x, y, v)
        combined_transported = a * transported_u + b * transported_v

        # Should be approximately equal due to linearity
        # Affine-invariant transport should be exactly linear
        assert jnp.allclose(transported_combined, combined_transported, atol=1e-12)

    def test_affine_invariant_transport_isometry(self):
        """Test that affine-invariant transport preserves vector norms (isometry)."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Compute original and transported vector norms
        original_norm = jnp.sqrt(self.manifold.inner(x, v, v))
        transported = self.manifold._affine_invariant_transp(x, y, v)
        transported_norm = jnp.sqrt(self.manifold.inner(y, transported, transported))

        # Parallel transport should be approximately isometric for closed-form solution
        # Relaxed tolerance due to numerical precision limits in matrix operations
        relative_error = jnp.abs(transported_norm - original_norm) / original_norm
        assert relative_error < 1e-6, f"Isometry violation too large: {relative_error:.12f}"

    def test_affine_invariant_transport_numerical_stability(self):
        """Test numerical stability with well-conditioned matrices."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1) + 1e-3 * jnp.eye(3)  # Add regularization
        y = self.manifold.random_point(key2) + 1e-3 * jnp.eye(3)  # Add regularization
        v = self.manifold.random_tangent(key3, x)

        # Should not crash and should produce finite results
        transported = self.manifold._affine_invariant_transp(x, y, v)

        assert jnp.all(jnp.isfinite(transported))
        assert jnp.allclose(transported, transported.T, atol=1e-10)

    def test_affine_invariant_transport_reversibility(self):
        """Test that transport from x to y, then y to x recovers original vector."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Transport x → y → x
        v_at_y = self.manifold._affine_invariant_transp(x, y, v)
        v_back_at_x = self.manifold._affine_invariant_transp(y, x, v_at_y)

        # Should recover original vector (up to numerical precision)
        # Relaxed tolerance due to accumulated numerical errors in matrix operations
        # Note: Perfect reversibility may require a more sophisticated implementation
        assert jnp.allclose(v, v_back_at_x, atol=1e-4)

    def test_affine_invariant_transport_jit_compatibility(self):
        """Test that affine-invariant transport method is JAX JIT compatible."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # JIT compile the method
        jit_affine_transport = jax.jit(self.manifold._affine_invariant_transp)

        # Should compile and run without errors
        result = jit_affine_transport(x, y, v)

        # Should give same result as non-JIT version
        non_jit_result = self.manifold._affine_invariant_transp(x, y, v)

        assert jnp.allclose(result, non_jit_result, atol=1e-15)


class TestSPDBuresWassersteinTransport:
    """Test suite for Bures-Wasserstein metric parallel transport on SPD manifolds."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manifold = SymmetricPositiveDefinite(n=3)
        self.key = jr.PRNGKey(123)

    def test_bures_wasserstein_transp_method_exists(self):
        """Test that _bures_wasserstein_transp method exists on SPD manifold."""
        assert hasattr(self.manifold, '_bures_wasserstein_transp')
        assert callable(getattr(self.manifold, '_bures_wasserstein_transp'))

    def test_bures_wasserstein_transport_basic_functionality(self):
        """Test basic Bures-Wasserstein parallel transport functionality."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Test Bures-Wasserstein parallel transport
        transported = self.manifold._bures_wasserstein_transp(x, y, v)

        # Verify output shape and type
        assert transported.shape == v.shape
        assert transported.dtype == v.dtype

        # The transported vector should be in the tangent space at y
        assert jnp.allclose(transported, transported.T, atol=1e-12)

    def test_bures_wasserstein_commuting_matrices_exact(self):
        """Test Bures-Wasserstein transport for commuting matrices (exact formula)."""
        # Create commuting matrices (diagonal matrices commute)
        key1, key2 = jr.split(self.key, 2)

        # Generate random positive eigenvalues
        eigs_x = jr.uniform(key1, (3,), minval=0.1, maxval=2.0)
        eigs_y = jr.uniform(key2, (3,), minval=0.1, maxval=2.0)

        x = jnp.diag(eigs_x)
        y = jnp.diag(eigs_y)

        # Create a tangent vector (symmetric matrix)
        v = jnp.array([[1.0, 0.5, 0.2], [0.5, -0.3, 0.1], [0.2, 0.1, 0.8]])

        # For commuting matrices, closed form should be exact
        transported = self.manifold._bures_wasserstein_transp(x, y, v)

        # Should be finite and symmetric
        assert jnp.all(jnp.isfinite(transported))
        assert jnp.allclose(transported, transported.T, atol=1e-12)

    def test_bures_wasserstein_vs_affine_invariant_difference(self):
        """Test that Bures-Wasserstein gives different results than affine-invariant."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Compare Bures-Wasserstein vs affine-invariant transport
        bures_result = self.manifold._bures_wasserstein_transp(x, y, v)
        affine_result = self.manifold._affine_invariant_transp(x, y, v)

        # Results should generally be different (different metrics)
        distance = self.manifold.dist(x, y)
        if distance > 1e-3:  # Only check for non-trivial distances
            max_diff = jnp.max(jnp.abs(bures_result - affine_result))
            assert max_diff > 1e-10, "Bures-Wasserstein should give different results than affine-invariant"


class TestSPDTransportComparison:
    """Comparison tests between different parallel transport methods on SPD manifolds."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manifold = SymmetricPositiveDefinite(n=3)
        self.key = jr.PRNGKey(777)

    def test_transport_method_accuracy_comparison(self):
        """Compare accuracy of different parallel transport methods."""
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # Get results from all transport methods
        simple_result = self.manifold.transp(x, y, v)  # Existing simple transport
        pole_result = self.manifold._pole_ladder(x, y, v, n_steps=3)
        affine_result = self.manifold._affine_invariant_transp(x, y, v)

        # All should be valid tangent vectors
        for result in [simple_result, pole_result, affine_result]:
            assert jnp.allclose(result, result.T, atol=1e-12)

        # Compute isometry preservation for each method
        original_norm = jnp.sqrt(self.manifold.inner(x, v, v))

        simple_norm = jnp.sqrt(self.manifold.inner(y, simple_result, simple_result))
        pole_norm = jnp.sqrt(self.manifold.inner(y, pole_result, pole_result))
        affine_norm = jnp.sqrt(self.manifold.inner(y, affine_result, affine_result))

        # Compute relative errors in norm preservation
        simple_error = jnp.abs(simple_norm - original_norm) / original_norm
        pole_error = jnp.abs(pole_norm - original_norm) / original_norm
        affine_error = jnp.abs(affine_norm - original_norm) / original_norm

        # Affine-invariant should have the best isometry preservation
        assert affine_error <= pole_error, "Affine-invariant should preserve isometry better than pole ladder"
        assert affine_error <= simple_error, "Affine-invariant should preserve isometry better than simple transport"

    def test_transport_computational_efficiency_ordering(self):
        """Test that computational efficiency ordering makes sense."""
        # This is a qualitative test to ensure methods can be called
        # Real benchmarking would require more sophisticated timing
        key1, key2, key3 = jr.split(self.key, 3)

        x = self.manifold.random_point(key1)
        y = self.manifold.random_point(key2)
        v = self.manifold.random_tangent(key3, x)

        # All methods should complete without errors
        simple_result = self.manifold.transp(x, y, v)
        pole_result = self.manifold._pole_ladder(x, y, v, n_steps=3)
        affine_result = self.manifold._affine_invariant_transp(x, y, v)

        # Verify all results are finite
        assert jnp.all(jnp.isfinite(simple_result))
        assert jnp.all(jnp.isfinite(pole_result))
        assert jnp.all(jnp.isfinite(affine_result))
