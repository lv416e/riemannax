"""Tests for numerical stability core infrastructure."""

import jax.numpy as jnp
import numpy as np
import pytest

from riemannax.manifolds.numerical_stability import (
    CurvatureBoundsError,
    HyperbolicNumericalError,
    NumericalStabilityManager,
    SE3SingularityError,
)


class TestNumericalStabilityManager:
    """Test the NumericalStabilityManager class."""

    def test_validate_hyperbolic_vector_poincare_ball(self):
        """Test hyperbolic vector validation for Poincaré ball model."""
        manager = NumericalStabilityManager()

        # Valid vector for Poincaré ball (should not raise)
        valid_vector = jnp.array([1.0, 2.0, 3.0])  # norm < 38
        result = manager.validate_hyperbolic_vector(valid_vector, "poincare")
        assert result.shape == valid_vector.shape

        # Invalid vector for Poincaré ball (should raise)
        invalid_vector = jnp.ones(10) * 40.0  # norm > 38
        with pytest.raises(HyperbolicNumericalError):
            manager.validate_hyperbolic_vector(invalid_vector, "poincare")

    def test_validate_hyperbolic_vector_lorentz(self):
        """Test hyperbolic vector validation for Lorentz model."""
        manager = NumericalStabilityManager()

        # Valid vector for Lorentz model (should not raise)
        valid_vector = jnp.array([1.0, 2.0, 3.0])  # norm < 19
        result = manager.validate_hyperbolic_vector(valid_vector, "lorentz")
        assert result.shape == valid_vector.shape

        # Invalid vector for Lorentz model (should raise)
        invalid_vector = jnp.ones(10) * 25.0  # norm > 19
        with pytest.raises(HyperbolicNumericalError):
            manager.validate_hyperbolic_vector(invalid_vector, "lorentz")

    def test_safe_matrix_exponential_stable(self):
        """Test safe matrix exponential for well-conditioned matrices."""
        manager = NumericalStabilityManager()

        # Well-conditioned matrix
        A = jnp.array([[0.1, 0.2], [0.3, 0.1]])
        result = manager.safe_matrix_exponential(A, method="pade")

        # Should return finite values
        assert jnp.all(jnp.isfinite(result))
        # Result should be approximately identity + A for small matrices
        expected_approx = jnp.eye(2) + A
        np.testing.assert_allclose(result, expected_approx, atol=0.05)

    def test_safe_matrix_exponential_near_singular(self):
        """Test safe matrix exponential near singularities."""
        manager = NumericalStabilityManager()

        # Matrix near singularity
        A = jnp.array([[0.0, 1e-10], [0.0, 0.0]])
        result = manager.safe_matrix_exponential(A, method="taylor")

        # Should handle gracefully and return finite values
        assert jnp.all(jnp.isfinite(result))

    def test_safe_matrix_exponential_invalid_method(self):
        """Test safe matrix exponential with invalid method."""
        manager = NumericalStabilityManager()

        A = jnp.eye(2)
        with pytest.raises(SE3SingularityError):
            manager.safe_matrix_exponential(A, method="invalid_method")



class TestHyperbolicNumericalError:
    """Test HyperbolicNumericalError exception."""

    def test_error_creation(self):
        """Test that HyperbolicNumericalError can be created and raised."""
        with pytest.raises(HyperbolicNumericalError) as exc_info:
            raise HyperbolicNumericalError("Test hyperbolic error")

        assert "Test hyperbolic error" in str(exc_info.value)


class TestSE3SingularityError:
    """Test SE3SingularityError exception."""

    def test_error_creation(self):
        """Test that SE3SingularityError can be created and raised."""
        with pytest.raises(SE3SingularityError) as exc_info:
            raise SE3SingularityError("Test SE(3) singularity")

        assert "Test SE(3) singularity" in str(exc_info.value)


class TestCurvatureBoundsError:
    """Test CurvatureBoundsError exception."""

    def test_error_creation(self):
        """Test that CurvatureBoundsError can be created and raised."""
        with pytest.raises(CurvatureBoundsError) as exc_info:
            raise CurvatureBoundsError("Test curvature bounds error")

        assert "Test curvature bounds error" in str(exc_info.value)
