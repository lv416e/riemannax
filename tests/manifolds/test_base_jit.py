import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from riemannax.manifolds.base import Manifold


class ConcreteManifoldForTesting(Manifold):
    """Concrete manifold class for testing."""

    def __init__(self, dim: int = 3):
        super().__init__()
        self._dim = dim

    def proj(self, x: Array, v: Array) -> Array:
        """Projection implementation for testing."""
        return v  # Simple implementation

    def exp(self, x: Array, v: Array) -> Array:
        """Exponential map implementation for testing."""
        return x + v  # Simple implementation

    def log(self, x: Array, y: Array) -> Array:
        """Logarithmic map implementation for testing."""
        return y - x  # Simple implementation

    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport implementation for testing."""
        return v  # Simple implementation

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Inner product implementation for testing."""
        return jnp.sum(u * v)  # Simple implementation

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ambient_dimension(self) -> int:
        return self._dim


class TestCleanBaseManifold:
    """Test BaseManifold without JIT functionality (after JIT removal)."""

    def setup_method(self):
        """Initialize test setup."""
        self.manifold = ConcreteManifoldForTesting(dim=3)

    def test_base_manifold_no_jit_attributes(self):
        """Test that BaseManifold no longer has JIT-related attributes."""
        # Verify JIT-related attributes have been removed
        assert not hasattr(self.manifold, "_jit_compiled_methods")
        assert not hasattr(self.manifold, "_compile_core_methods")
        assert not hasattr(self.manifold, "_call_jit_method")
        assert not hasattr(self.manifold, "_safe_jit_wrapper")
        assert not hasattr(self.manifold, "_performance_tracking")

    def test_basic_manifold_operations_work(self):
        """Test that basic manifold operations work correctly without JIT in base."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])
        y = jnp.array([1.1, 2.2, 3.3])

        # All basic operations should work
        proj_result = self.manifold.proj(x, v)
        exp_result = self.manifold.exp(x, v)
        log_result = self.manifold.log(x, y)
        inner_result = self.manifold.inner(x, v, v)
        dist_result = self.manifold.dist(x, y)
        norm_result = self.manifold.norm(x, v)

        # Verify results have correct shapes and types
        assert proj_result.shape == v.shape
        assert exp_result.shape == x.shape
        assert log_result.shape == v.shape
        assert isinstance(inner_result.item(), float)
        assert isinstance(dist_result.item(), float)
        assert isinstance(norm_result.item(), float)

    def test_distance_calculation_correctness(self):
        """Test that distance calculation works correctly."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.1, 2.2, 3.3])

        # Distance calculation
        distance = self.manifold.dist(x, y)

        # Manual calculation for verification
        v = self.manifold.log(x, y)
        expected_distance = jnp.sqrt(self.manifold.inner(x, v, v))

        np.testing.assert_almost_equal(distance, expected_distance, decimal=8)

    def test_norm_calculation_correctness(self):
        """Test that norm calculation works correctly."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # Norm calculation
        norm_result = self.manifold.norm(x, v)
        expected_norm = jnp.sqrt(self.manifold.inner(x, v, v))

        np.testing.assert_almost_equal(norm_result, expected_norm, decimal=8)

    def test_batch_processing_works(self):
        """Test that batch processing works with vmap."""
        batch_size = 10
        x_batch = jnp.ones((batch_size, 3))
        v_batch = jnp.ones((batch_size, 3)) * 0.1

        # Batch processing using vmap
        results = jax.vmap(self.manifold.proj)(x_batch, v_batch)

        # Verify batch results
        assert results.shape == (batch_size, 3)

        # Verify individual result matches batch result
        individual_result = self.manifold.proj(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(results[0], individual_result)

    def test_error_handling_still_works(self):
        """Test that error handling works correctly without JIT in base."""

        class ErrorManifold(ConcreteManifoldForTesting):
            def proj(self, x: Array, v: Array) -> Array:
                if jnp.any(x < 0):
                    raise ValueError("Negative values not allowed")
                return v

        manifold = ErrorManifold(dim=3)

        # Valid input should work
        x_valid = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])
        result = manifold.proj(x_valid, v)
        np.testing.assert_array_almost_equal(result, v)

        # Invalid input should raise error
        x_invalid = jnp.array([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Negative values not allowed"):
            manifold.proj(x_invalid, v)

    def test_manifold_properties_work(self):
        """Test that manifold properties work correctly."""
        assert self.manifold.dimension == 3
        assert self.manifold.ambient_dimension == 3

    def test_validation_methods_work(self):
        """Test that validation methods work correctly."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # Test validate_tangent method
        is_valid = self.manifold.validate_tangent(x, v)
        assert isinstance(is_valid, (bool, np.bool_))

        # Test validate_point method if available (base class raises NotImplementedError)
        if hasattr(self.manifold, "validate_point"):
            try:
                is_valid_point = self.manifold.validate_point(x)
                assert isinstance(is_valid_point, (bool, np.bool_))
            except NotImplementedError:
                # Expected for base class - validate_point is not implemented
                pass
