"""Tests for numerical stability layer.

This module tests the NumericalStabilityLayer class that provides:
- Condition number estimation
- Adaptive regularization for ill-conditioned matrices
- SPD property validation
- Numerical stability for extreme cases
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from riemannax.core.numerical_stability import NumericalStabilityLayer


class TestNumericalStabilityLayer:
    """Test suite for NumericalStabilityLayer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stability = NumericalStabilityLayer()

        # Well-conditioned SPD matrix (condition number ~1)
        self.well_conditioned = jnp.eye(3) * 2.0

        # Ill-conditioned SPD matrix
        eigenvals = jnp.array([1e-12, 1e-6, 1.0])
        Q = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.ill_conditioned = Q @ jnp.diag(eigenvals) @ Q.T

        # Non-SPD matrix (not symmetric)
        self.non_spd = jnp.array([[1.0, 2.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Nearly singular matrix (very high condition number)
        self.singular = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-15, 0.0], [0.0, 0.0, 1.0]])

    def test_estimate_condition_number_well_conditioned(self):
        """Test condition number estimation for well-conditioned matrix."""
        cond = self.stability.estimate_condition_number(self.well_conditioned)

        # Well-conditioned matrix should have condition number close to 1
        assert jnp.isfinite(cond)
        assert cond > 0.0
        assert cond < 10.0  # Should be small for well-conditioned

    def test_estimate_condition_number_ill_conditioned(self):
        """Test condition number estimation for ill-conditioned matrix."""
        cond = self.stability.estimate_condition_number(self.ill_conditioned)

        # Ill-conditioned matrix should have large condition number
        assert jnp.isfinite(cond)
        assert cond > 1e6  # Should be large for ill-conditioned

    def test_estimate_condition_number_singular(self):
        """Test condition number estimation for singular matrix."""
        cond = self.stability.estimate_condition_number(self.singular)

        # Nearly singular matrix should have very large condition number
        # But our implementation should handle this gracefully
        assert cond > 1e8  # Very large for nearly singular

    def test_validate_spd_properties_valid_spd(self):
        """Test SPD validation for valid SPD matrix."""
        is_spd = self.stability.validate_spd_properties(self.well_conditioned, tolerance=1e-8)
        assert is_spd is True

    def test_validate_spd_properties_non_symmetric(self):
        """Test SPD validation for non-symmetric matrix."""
        is_spd = self.stability.validate_spd_properties(self.non_spd, tolerance=1e-8)
        assert is_spd is False

    def test_validate_spd_properties_negative_eigenvalue(self):
        """Test SPD validation for matrix with negative eigenvalue."""
        # Matrix with negative eigenvalue
        eigenvals = jnp.array([-0.1, 0.5, 1.0])
        Q = jnp.eye(3)
        negative_matrix = Q @ jnp.diag(eigenvals) @ Q.T

        is_spd = self.stability.validate_spd_properties(negative_matrix, tolerance=1e-8)
        assert is_spd is False

    def test_regularize_matrix_no_regularization_needed(self):
        """Test regularization when matrix is well-conditioned."""
        regularized = self.stability.regularize_matrix(self.well_conditioned, condition_threshold=1e10)

        # Should be nearly identical to original
        np.testing.assert_allclose(regularized, self.well_conditioned, rtol=1e-10)

    def test_regularize_matrix_ill_conditioned(self):
        """Test regularization for ill-conditioned matrix."""
        regularized = self.stability.regularize_matrix(self.ill_conditioned, condition_threshold=1e6)

        # Regularized matrix should have better condition number
        orig_cond = self.stability.estimate_condition_number(self.ill_conditioned)
        reg_cond = self.stability.estimate_condition_number(regularized)

        assert reg_cond < orig_cond
        assert reg_cond < 1e10  # Should be reasonable

        # Should still be close to original matrix
        assert jnp.allclose(regularized, self.ill_conditioned, rtol=1e-1)

    def test_regularize_matrix_singular(self):
        """Test regularization for singular matrix."""
        regularized = self.stability.regularize_matrix(self.singular, condition_threshold=1e6)

        # Regularized matrix should be non-singular
        reg_cond = self.stability.estimate_condition_number(regularized)
        assert jnp.isfinite(reg_cond)
        assert reg_cond < 1e12

        # Regularized matrix should improve conditioning
        # (The exact form of regularization may vary)
        assert jnp.allclose(regularized, regularized.T, atol=1e-10)  # Should remain symmetric

    def test_adaptive_regularization_strength(self):
        """Test that regularization strength adapts to condition number."""
        # Create matrices with different condition numbers
        cond1 = 1e8
        cond2 = 1e12

        eigenvals1 = jnp.array([1.0 / cond1, 1.0, 1.0])
        eigenvals2 = jnp.array([1.0 / cond2, 1.0, 1.0])

        Q = jnp.eye(3)
        matrix1 = Q @ jnp.diag(eigenvals1) @ Q.T
        matrix2 = Q @ jnp.diag(eigenvals2) @ Q.T

        reg1 = self.stability.regularize_matrix(matrix1, condition_threshold=1e6)
        reg2 = self.stability.regularize_matrix(matrix2, condition_threshold=1e6)

        # Matrix with worse condition should get more regularization
        reg_strength1 = jnp.trace(reg1 - matrix1) / 3
        reg_strength2 = jnp.trace(reg2 - matrix2) / 3

        assert reg_strength2 > reg_strength1

    def test_jit_compilation(self):
        """Test that methods are JIT compilable."""
        # Test condition number estimation
        cond_jit = jax.jit(self.stability.estimate_condition_number)
        cond = cond_jit(self.well_conditioned)
        assert jnp.isfinite(cond)

        # Note: SPD validation cannot be JIT compiled due to assertions and bool return
        # Test regularization
        reg_jit = jax.jit(self.stability.regularize_matrix)
        regularized = reg_jit(self.ill_conditioned, condition_threshold=1e10)
        assert regularized.shape == self.ill_conditioned.shape

    def test_vmap_compatibility(self):
        """Test that methods work with vmap for batch processing."""
        # Create batch of matrices
        batch_matrices = jnp.stack([self.well_conditioned, self.well_conditioned * 0.5, self.well_conditioned * 2.0])

        # Test batch condition number estimation
        batch_cond = jax.vmap(self.stability.estimate_condition_number)(batch_matrices)
        assert batch_cond.shape == (3,)
        assert jnp.all(jnp.isfinite(batch_cond))

        # Test batch regularization (Note: may not trigger regularization for well-conditioned matrices)
        batch_reg = jax.vmap(self.stability.regularize_matrix, in_axes=(0, None))(batch_matrices, 1e10)
        assert batch_reg.shape == batch_matrices.shape

    def test_edge_cases_nan_inf(self):
        """Test handling of NaN and Inf values."""
        # Matrix with NaN
        nan_matrix = self.well_conditioned.at[0, 0].set(jnp.nan)

        cond = self.stability.estimate_condition_number(nan_matrix)
        # Should return large finite value for NaN input
        assert jnp.isfinite(cond) or cond == jnp.inf

        # Matrix with Inf
        inf_matrix = self.well_conditioned.at[0, 0].set(jnp.inf)

        cond = self.stability.estimate_condition_number(inf_matrix)
        # Should handle gracefully
        assert not jnp.isnan(cond)

    def test_tolerance_sensitivity(self):
        """Test sensitivity to different tolerance values."""
        # Matrix with small eigenvalue near tolerance
        small_eigenval = 1e-10
        eigenvals = jnp.array([small_eigenval, 1.0, 1.0])
        Q = jnp.eye(3)
        matrix = Q @ jnp.diag(eigenvals) @ Q.T

        # Test with strict tolerance
        is_spd_strict = self.stability.validate_spd_properties(matrix, tolerance=1e-8)

        # Test with loose tolerance
        is_spd_loose = self.stability.validate_spd_properties(matrix, tolerance=1e-12)

        # Strict tolerance should reject, loose should accept
        assert is_spd_strict is False
        assert is_spd_loose is True

    def test_symmetry_tolerance(self):
        """Test symmetry checking with numerical tolerance."""
        # Approximately symmetric matrix
        eps = 1e-14
        approx_symmetric = jnp.array([[1.0, 0.5, 0.0], [0.5 + eps, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Should pass symmetry check with reasonable tolerance
        is_spd = self.stability.validate_spd_properties(approx_symmetric, tolerance=1e-8)
        assert is_spd is True

        # But fail with very strict tolerance on asymmetry
        large_eps = 1e-6
        not_symmetric = jnp.array([[1.0, 0.5, 0.0], [0.5 + large_eps, 1.0, 0.0], [0.0, 0.0, 1.0]])

        is_spd = self.stability.validate_spd_properties(not_symmetric, tolerance=1e-8)
        assert is_spd is False

    def test_type_annotations(self):
        """Test that type annotations are correct."""
        import inspect

        # Check estimate_condition_number signature
        sig = inspect.signature(self.stability.estimate_condition_number)
        assert "x" in sig.parameters

        # Check regularize_matrix signature
        sig = inspect.signature(self.stability.regularize_matrix)
        assert "x" in sig.parameters
        assert "condition_threshold" in sig.parameters

        # Check validate_spd_properties signature
        sig = inspect.signature(self.stability.validate_spd_properties)
        assert "x" in sig.parameters
        assert "tolerance" in sig.parameters


class TestNumericalStabilityIntegration:
    """Integration tests for numerical stability functionality."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.stability = NumericalStabilityLayer()

    def test_regularization_workflow(self):
        """Test complete regularization workflow."""
        # Create problematic matrix
        eigenvals = jnp.array([1e-15, 1e-8, 1.0])
        Q = jnp.eye(3)
        problematic = Q @ jnp.diag(eigenvals) @ Q.T

        # Estimate initial condition
        orig_cond = self.stability.estimate_condition_number(problematic)

        # Check if regularization needed
        needs_reg = bool(orig_cond > 1e10)
        assert needs_reg is True

        # Apply regularization
        regularized = self.stability.regularize_matrix(problematic, condition_threshold=1e10)

        # Verify improvement
        new_cond = self.stability.estimate_condition_number(regularized)
        assert new_cond < orig_cond

        # Verify SPD properties maintained
        is_spd = self.stability.validate_spd_properties(regularized, tolerance=1e-12)
        assert is_spd is True

    def test_error_conditions(self):
        """Test error handling for invalid inputs."""
        # Wrong matrix dimensions
        with pytest.raises(AssertionError):
            self.stability.estimate_condition_number(jnp.array([1, 2, 3]))

        # Negative tolerance
        with pytest.raises(AssertionError):
            self.stability.validate_spd_properties(jnp.eye(3), tolerance=-1.0)

        # Test validation in convenience functions (JIT-safe methods don't validate)
        from riemannax.core.numerical_stability import regularize_matrix_func

        with pytest.raises(AssertionError):
            regularize_matrix_func(jnp.eye(3), condition_threshold=-1.0)
