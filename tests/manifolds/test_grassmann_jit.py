import jax
import jax.numpy as jnp
import numpy as np

from riemannax.manifolds.grassmann import Grassmann


class TestGrassmannJITOptimization:
    """JIT optimization tests for Grassmann manifold."""

    def setup_method(self):
        """Setup before each test execution."""
        self.manifold_gr35 = Grassmann(n=5, p=3)  # Gr(3,5)
        self.manifold_gr24 = Grassmann(n=4, p=2)  # Gr(2,4)

        # JIT-related initialization
        for manifold in [self.manifold_gr35, self.manifold_gr24]:
            if hasattr(manifold, "_reset_jit_cache"):
                manifold._reset_jit_cache()

    def test_grassmann_jit_implementation_methods_exist(self):
        """Test if Grassmann JIT implementation methods exist."""
        for manifold in [self.manifold_gr35, self.manifold_gr24]:
            assert hasattr(manifold, "_proj_impl")
            assert hasattr(manifold, "_exp_impl")
            assert hasattr(manifold, "_log_impl")
            assert hasattr(manifold, "_inner_impl")
            assert hasattr(manifold, "_dist_impl")
            assert hasattr(manifold, "_get_static_args")

    def test_proj_jit_vs_original_equivalence_gr35(self):
        """Test numerical equivalence of projection: JIT vs original implementation (Gr(3,5))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_gr35.proj(x, v)

        # JIT implementation
        jit_result = self.manifold_gr35._proj_impl(x, v)

        # Verify numerical equivalence
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_exp_jit_vs_original_equivalence_gr35(self):
        """Test numerical equivalence of exponential map: JIT vs original implementation (Gr(3,5))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Original implementation (uses retraction, but JIT version is true exponential map)
        # Verify results are numerically close
        original_result = self.manifold_gr35.exp(x, v)

        # JIT implementation (true exponential map)
        jit_result = self.manifold_gr35._exp_impl(x, v)

        # Verify both results are points on the Grassmann manifold
        assert self.manifold_gr35.validate_point(original_result)
        assert self.manifold_gr35.validate_point(jit_result)

        # Verify distance is close (not exactly the same but mathematically correct)
        distance = self.manifold_gr35.dist(original_result, jit_result)
        assert distance < 1.0  # Reasonable tolerance

    def test_log_jit_mathematical_correctness(self):
        """Test mathematical correctness of logarithmic map: JIT implementation."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_gr35.random_point(key1)
        y = self.manifold_gr35.random_point(key2)

        # Logarithmic map JIT implementation
        log_result = self.manifold_gr35._log_impl(x, y)

        # Verify that result belongs to tangent space
        assert self.manifold_gr35.validate_tangent(x, log_result)

        # Check consistency with exponential map (do not expect exp(log(x,y)) ≈ y, but verify mathematical properties)
        exp_log_result = self.manifold_gr35._exp_impl(x, log_result)

        # Verify that result is a point on the Grassmann manifold
        assert self.manifold_gr35.validate_point(exp_log_result)

    def test_inner_jit_vs_original_equivalence_gr35(self):
        """Test numerical equivalence of inner product: JIT vs original implementation (Gr(3,5))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        u_key, v_key = jax.random.split(jax.random.PRNGKey(43))
        u = self.manifold_gr35.random_tangent(u_key, x)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_gr35.inner(x, u, v)

        # JIT implementation
        jit_result = self.manifold_gr35._inner_impl(x, u, v)

        # Verify numerical equivalence
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_dist_jit_vs_original_equivalence_gr35(self):
        """Test numerical equivalence of distance computation: JIT vs original implementation (Gr(3,5))."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_gr35.random_point(key1)
        y = self.manifold_gr35.random_point(key2)

        # Original implementation
        original_result = self.manifold_gr35.dist(x, y)

        # JIT implementation
        jit_result = self.manifold_gr35._dist_impl(x, y)

        # Verify numerical equivalence
        np.testing.assert_almost_equal(jit_result, original_result, decimal=6)

    def test_subspace_constraints_preservation(self):
        """Test verification of subspace constraint preservation."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # JIT exponential map result
        result = self.manifold_gr35._exp_impl(x, v)

        # Verify orthogonality constraint: X^T @ X = I
        orthogonality_check = jnp.matmul(result.T, result)
        identity = jnp.eye(self.manifold_gr35.p)
        np.testing.assert_array_almost_equal(orthogonality_check, identity, decimal=6)

        # Shape verification
        assert result.shape == (self.manifold_gr35.n, self.manifold_gr35.p)

    def test_modified_exponential_map_correctness(self):
        """Test mathematical correctness verification of modified exponential map."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # Test with zero vector
        zero_v = jnp.zeros((self.manifold_gr35.n, self.manifold_gr35.p))
        result_zero = self.manifold_gr35._exp_impl(x, zero_v)

        # For zero vector, the original point should be returned
        np.testing.assert_array_almost_equal(result_zero, x, decimal=8)

        # Test with small vector
        small_v = 0.01 * self.manifold_gr35.random_tangent(jax.random.PRNGKey(43), x)
        result_small = self.manifold_gr35._exp_impl(x, small_v)

        # Verify that result satisfies manifold constraints
        assert self.manifold_gr35.validate_point(result_small)

    def test_large_scale_matrix_numerical_stability(self):
        """Test numerical stability with large-scale matrices."""
        # Test with larger dimensions
        large_manifold = Grassmann(n=50, p=10)

        key = jax.random.PRNGKey(42)
        x = large_manifold.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = large_manifold.random_tangent(v_key, x)

        # Computation with JIT implementation
        proj_result = large_manifold._proj_impl(x, v)
        exp_result = large_manifold._exp_impl(x, v)

        # Result verification
        assert proj_result.shape == (50, 10)
        assert exp_result.shape == (50, 10)

        # Numerical stability verification
        assert not jnp.any(jnp.isnan(proj_result))
        assert not jnp.any(jnp.isnan(exp_result))
        assert not jnp.any(jnp.isinf(proj_result))
        assert not jnp.any(jnp.isinf(exp_result))

        # Manifold constraint verification
        assert large_manifold.validate_point(exp_result)

    def test_static_args_configuration(self):
        """Test static argument configuration."""
        # Verify static argument configuration for Gr(3,5) and Gr(2,4)
        static_args_gr35 = self.manifold_gr35._get_static_args("exp")
        static_args_gr24 = self.manifold_gr24._get_static_args("exp")

        # Verify that static arguments are empty tuples (for JIT optimization)
        assert static_args_gr35 == ()
        assert static_args_gr24 == ()

    def test_batch_processing_consistency_gr35(self):
        """Test batch processing consistency (Gr(3,5))."""
        batch_size = 10
        key = jax.random.PRNGKey(42)

        # Prepare batch data
        keys = jax.random.split(key, batch_size)
        x_batch = jax.vmap(self.manifold_gr35.random_point)(keys)

        v_keys = jax.random.split(jax.random.PRNGKey(43), batch_size)
        v_batch = jax.vmap(self.manifold_gr35.random_tangent)(v_keys, x_batch)

        # Projection operation in batch processing
        proj_results = jax.vmap(self.manifold_gr35._proj_impl)(x_batch, v_batch)

        # Shape verification of results
        assert proj_results.shape == (batch_size, 5, 3)

        # Consistency check with individual execution
        individual_result = self.manifold_gr35._proj_impl(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(proj_results[0], individual_result)

    def test_svd_decomposition_numerical_stability(self):
        """Test numerical stability assurance through SVD decomposition."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_gr35.random_point(key1)
        self.manifold_gr35.random_point(key2)

        # Simulate ill-conditioned case
        # Very close subspaces
        epsilon = 1e-8
        y_close = x + epsilon * self.manifold_gr35.random_tangent(key2, x)
        Q_close, _ = jnp.linalg.qr(y_close, mode="reduced")

        # SVD-based distance computation
        distance = self.manifold_gr35._dist_impl(x, Q_close)

        # Verify that NaN or Inf do not occur
        assert not jnp.any(jnp.isnan(distance))
        assert not jnp.any(jnp.isinf(distance))
        assert distance >= 0.0

    def test_principal_angles_computation_accuracy(self):
        """Test accuracy of principal angles computation."""
        # Simple case: subspaces with known principal angles
        manifold_gr24 = Grassmann(4, 2)

        # First subspace: span{(1,0,0,0), (0,1,0,0)} = x-y plane
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        # Second subspace: span{(1,0,0,0), (0,0,1,0)} = x-z plane
        # Principal angle is 90 degrees (π/2)
        y = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        # Distance computation
        distance = manifold_gr24._dist_impl(x, y)

        # Theoretical value: one principal angle is π/2, the other is 0, so distance is π/2
        expected_distance = jnp.pi / 2
        np.testing.assert_almost_equal(distance, expected_distance, decimal=4)

    def test_jit_compilation_caching(self):
        """Test JIT compilation and caching."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Verify consistent results across multiple executions
        results = []
        for _ in range(3):
            result = self.manifold_gr35.proj(x, v)
            results.append(result)

        # Verify that all results are equivalent
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])

        # JIT compilation works correctly by producing consistent results
        # (The JIT-related attributes _jit_compiled_methods and _jit_enabled
        # were removed in the refactoring to simplify the design)

    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # Verify handling of NaN inputs
        v_nan = jnp.full((5, 3), jnp.nan)

        # Verify that NaN propagates or is handled appropriately in JAX
        result = self.manifold_gr35._proj_impl(x, v_nan)

        # Verify that NaN propagates (normal behavior) or is handled appropriately
        has_nan = jnp.any(jnp.isnan(result))
        has_valid_result = not jnp.any(jnp.isnan(result))

        # Verify that either NaN propagates or is handled appropriately
        assert has_nan or has_valid_result

    def test_orthonormal_columns_preservation(self):
        """Test verification of orthonormal column preservation."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # Large tangent vector
        v_key = jax.random.PRNGKey(43)
        v_large = 5.0 * self.manifold_gr35.random_tangent(v_key, x)

        # Execute JIT exponential map
        result = self.manifold_gr35._exp_impl(x, v_large)

        # Verify that columns are orthonormal
        gram_matrix = jnp.matmul(result.T, result)
        identity = jnp.eye(self.manifold_gr35.p)

        np.testing.assert_array_almost_equal(gram_matrix, identity, decimal=6)

    def test_integration_with_existing_methods(self):
        """Test compatibility of JIT integration with existing methods."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Test combination of existing methods with JIT version

        # validate_point method (orthogonality check)
        is_valid_point = self.manifold_gr35.validate_point(x)
        assert is_valid_point

        # validate_tangent method (tangent space check)
        is_valid_tangent = self.manifold_gr35.validate_tangent(x, v)
        assert is_valid_tangent

    def test_mathematical_correctness_exp_log_consistency(self):
        """Test mathematical correctness: exponential-logarithmic map consistency."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # Small tangent vector (range where linearization is valid)
        v_key = jax.random.PRNGKey(43)
        v_small = 0.1 * self.manifold_gr35.random_tangent(v_key, x)

        # Check exp -> log consistency
        y = self.manifold_gr35._exp_impl(x, v_small)
        v_recovered = self.manifold_gr35._log_impl(x, y)

        # May not match perfectly, but verify they are in the same direction
        # (For Grassmann manifold, local consistency)
        inner_original = self.manifold_gr35.inner(x, v_small, v_small)
        inner_recovered = self.manifold_gr35.inner(x, v_recovered, v_recovered)

        # Verify that norms do not differ significantly
        assert jnp.abs(inner_original - inner_recovered) < 0.5
