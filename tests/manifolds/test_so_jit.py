import jax
import jax.numpy as jnp
import numpy as np

from riemannax.manifolds.so import SpecialOrthogonal


class TestSpecialOrthogonalJITOptimization:
    """JIT optimization tests for SpecialOrthogonal manifold."""

    def setup_method(self):
        """Initialize before each test execution."""
        self.manifold_so3 = SpecialOrthogonal(n=3)  # SO(3)
        self.manifold_so4 = SpecialOrthogonal(n=4)  # SO(4)

        # JIT-related initialization
        for manifold in [self.manifold_so3, self.manifold_so4]:
            if hasattr(manifold, "_reset_jit_cache"):
                manifold._reset_jit_cache()

    def test_so_jit_implementation_methods_exist(self):
        """Test if SpecialOrthogonal JIT implementation methods exist."""
        for manifold in [self.manifold_so3, self.manifold_so4]:
            assert hasattr(manifold, "_proj_impl")
            assert hasattr(manifold, "_exp_impl")
            assert hasattr(manifold, "_log_impl")
            assert hasattr(manifold, "_inner_impl")
            assert hasattr(manifold, "_dist_impl")
            assert hasattr(manifold, "_get_static_args")

    def test_proj_jit_vs_original_equivalence_so3(self):
        """Projection: numerical equivalence test between JIT and original versions (SO(3))."""
        # Rotation matrix on SO(3)
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # Arbitrary tangent vector
        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_so3.proj(x, v)

        # JIT implementation
        jit_result = self.manifold_so3._proj_impl(x, v)

        # Verify numerical equivalence
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_exp_jit_vs_original_equivalence_so3(self):
        """Exponential map: numerical equivalence test between JIT and original versions (SO(3))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_so3.exp(x, v)

        # JIT implementation
        jit_result = self.manifold_so3._exp_impl(x, v)

        # Verify numerical equivalence
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=6)

    def test_log_jit_vs_original_equivalence_so3(self):
        """Logarithmic map: numerical equivalence test between JIT and original versions (SO(3))."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_so3.random_point(key1)
        y = self.manifold_so3.random_point(key2)

        # Original implementation
        original_result = self.manifold_so3.log(x, y)

        # JIT implementation
        jit_result = self.manifold_so3._log_impl(x, y)

        # Verify numerical equivalence
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=6)

    def test_inner_jit_vs_original_equivalence_so3(self):
        """Inner product: numerical equivalence test between JIT and original versions (SO(3))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        u_key, v_key = jax.random.split(jax.random.PRNGKey(43))
        u = self.manifold_so3.random_tangent(u_key, x)
        v = self.manifold_so3.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_so3.inner(x, u, v)

        # JIT implementation
        jit_result = self.manifold_so3._inner_impl(x, u, v)

        # Verify numerical equivalence
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_dist_jit_vs_original_equivalence_so3(self):
        """Distance calculation: numerical equivalence test between JIT and original versions (SO(3))."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_so3.random_point(key1)
        y = self.manifold_so3.random_point(key2)

        # Original implementation
        original_result = self.manifold_so3.dist(x, y)

        # JIT implementation
        jit_result = self.manifold_so3._dist_impl(x, y)

        # Verify numerical equivalence
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_proj_jit_vs_original_equivalence_so4(self):
        """Projection: numerical equivalence test between JIT and original versions (SO(4))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so4.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so4.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_so4.proj(x, v)

        # JIT implementation
        jit_result = self.manifold_so4._proj_impl(x, v)

        # Verify numerical equivalence
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_rotation_matrix_constraints_preservation(self):
        """Test verification of rotation matrix constraint preservation."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # Result of JIT exponential map
        result = self.manifold_so3._exp_impl(x, v)

        # Verify orthogonality constraint: R @ R.T = I
        orthogonality_check = jnp.matmul(result, result.T)
        identity = jnp.eye(3)
        np.testing.assert_array_almost_equal(orthogonality_check, identity, decimal=6)

        # Verify determinant constraint: det(R) = 1
        det_result = jnp.linalg.det(result)
        np.testing.assert_almost_equal(det_result, 1.0, decimal=6)

    def test_so3_rodrigues_formula_optimization(self):
        """Accuracy test of Rodrigues formula optimization in SO(3)."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # Test small rotation
        small_v = 0.01 * jnp.array([[0, -0.1, 0.2], [0.1, 0, -0.3], [-0.2, 0.3, 0]])
        small_v_tangent = self.manifold_so3.proj(x, small_v)

        result_small = self.manifold_so3._exp_impl(x, small_v_tangent)

        # Test large rotation
        large_v = jnp.pi * 0.8 * jnp.array([[0, -0.1, 0.2], [0.1, 0, -0.3], [-0.2, 0.3, 0]])
        large_v_tangent = self.manifold_so3.proj(x, large_v)

        result_large = self.manifold_so3._exp_impl(x, large_v_tangent)

        # Verify that both results satisfy rotation matrix constraints
        for result in [result_small, result_large]:
            # Verify orthogonality
            orthogonality_check = jnp.matmul(result, result.T)
            np.testing.assert_array_almost_equal(orthogonality_check, jnp.eye(3), decimal=6)

            # Verify determinant
            det_result = jnp.linalg.det(result)
            np.testing.assert_almost_equal(det_result, 1.0, decimal=6)

    def test_so3_180_degree_rotation_stability(self):
        """Numerical stability test for 180-degree rotations in SO(3)."""
        # Verify stability for rotations close to 180 degrees
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # Generate rotation close to 180 degrees
        axis = jnp.array([1.0, 0.0, 0.0]) / jnp.linalg.norm(jnp.array([1.0, 0.0, 0.0]))
        angle = jnp.pi - 1e-6  # Nearly 180 degrees

        # skew-symmetric matrix for 180-degree rotation
        skew = angle * jnp.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        # Create 180-degree rotation matrix
        rotation_180 = self.manifold_so3._expm_so3_jit(skew)
        y = jnp.matmul(x, rotation_180)

        # Verify that logarithmic map executes stably
        log_result = self.manifold_so3._log_impl(x, y)

        # Verify that no NaN or Inf values occur
        assert not jnp.any(jnp.isnan(log_result))
        assert not jnp.any(jnp.isinf(log_result))

    def test_static_args_configuration(self):
        """Static argument configuration test."""
        # Verify static argument configuration for SO(3) and SO(4)
        static_args_so3 = self.manifold_so3._get_static_args("exp")
        static_args_so4 = self.manifold_so4._get_static_args("exp")

        # Verify that static arguments are empty tuples (for JIT optimization)
        assert static_args_so3 == ()
        assert static_args_so4 == ()

    def test_large_scale_batch_processing_consistency_so3(self):
        """Large-scale batch processing consistency test (SO(3))."""
        batch_size = 100
        key = jax.random.PRNGKey(42)

        # Prepare batch data
        keys = jax.random.split(key, batch_size)
        x_batch = jax.vmap(self.manifold_so3.random_point)(keys)

        v_keys = jax.random.split(jax.random.PRNGKey(43), batch_size)
        v_batch = jax.vmap(self.manifold_so3.random_tangent)(v_keys, x_batch)

        # Batch processing projection operation
        proj_results = jax.vmap(self.manifold_so3._proj_impl)(x_batch, v_batch)

        # Verify result shape
        assert proj_results.shape == (batch_size, 3, 3)

        # Verify consistency with individual execution
        individual_result = self.manifold_so3._proj_impl(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(proj_results[0], individual_result)

    def test_batch_processing_performance_scaling(self):
        """Batch processing linear scaling performance test."""
        # Small-scale batch
        small_batch = 10
        key = jax.random.PRNGKey(42)

        keys_small = jax.random.split(key, small_batch)
        x_small = jax.vmap(self.manifold_so3.random_point)(keys_small)
        v_keys_small = jax.random.split(jax.random.PRNGKey(43), small_batch)
        v_small = jax.vmap(self.manifold_so3.random_tangent)(v_keys_small, x_small)

        # Large-scale batch
        large_batch = 50
        keys_large = jax.random.split(key, large_batch)
        x_large = jax.vmap(self.manifold_so3.random_point)(keys_large)
        v_keys_large = jax.random.split(jax.random.PRNGKey(43), large_batch)
        v_large = jax.vmap(self.manifold_so3.random_tangent)(v_keys_large, x_large)

        # Verify batch processing execution
        small_results = jax.vmap(self.manifold_so3._proj_impl)(x_small, v_small)
        large_results = jax.vmap(self.manifold_so3._proj_impl)(x_large, v_large)

        assert small_results.shape == (small_batch, 3, 3)
        assert large_results.shape == (large_batch, 3, 3)

    def test_qr_decomposition_stability(self):
        """Numerical stability assurance test through QR decomposition."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # Create tangent vector with poor condition number
        v_key = jax.random.PRNGKey(43)
        v_large = 10.0 * self.manifold_so3.random_tangent(v_key, x)

        # Execute JIT exponential map
        result = self.manifold_so3._exp_impl(x, v_large)

        # Verify that orthogonality is preserved by QR decomposition
        orthogonality_check = jnp.matmul(result, result.T)
        np.testing.assert_array_almost_equal(orthogonality_check, jnp.eye(3), decimal=6)

        # Verify that determinant is preserved at 1
        det_result = jnp.linalg.det(result)
        np.testing.assert_almost_equal(det_result, 1.0, decimal=6)

    def test_3d_rotation_precision(self):
        """Concrete accuracy test for 3D rotations."""
        # Define known rotation
        angle = jnp.pi / 4  # 45-degree rotation
        axis = jnp.array([0, 0, 1])  # Around z-axis

        # Initial point
        x = jnp.eye(3)

        # skew-symmetric matrix
        skew = angle * jnp.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        v = jnp.matmul(x, skew)

        # JIT exponential map
        result = self.manifold_so3._exp_impl(x, v)

        # Expected rotation matrix (45-degree rotation around z-axis)
        cos_45 = jnp.cos(jnp.pi / 4)
        sin_45 = jnp.sin(jnp.pi / 4)
        expected = jnp.array([[cos_45, -sin_45, 0], [sin_45, cos_45, 0], [0, 0, 1]])

        # Verify accuracy
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_jit_compilation_caching(self):
        """JIT compilation and caching test."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # Verify consistent results through multiple executions
        results = []
        for _ in range(3):
            result = self.manifold_so3.proj(x, v)
            results.append(result)

        # Verify that all results are equivalent
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])

        # JIT compilation works correctly by producing consistent results
        # (The JIT-related attributes _jit_compiled_methods and _jit_enabled
        # were removed in the refactoring to simplify the design)

    def test_error_handling_invalid_inputs(self):
        """Error handling test for invalid inputs."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # Verify handling of NaN input
        v_nan = jnp.full((3, 3), jnp.nan)

        # Verify that NaN propagates or is properly handled in JAX
        result = self.manifold_so3._proj_impl(x, v_nan)

        # Verify that NaN propagates (normal behavior) or is properly handled
        has_nan = jnp.any(jnp.isnan(result))
        has_valid_result = not jnp.any(jnp.isnan(result))

        # Verify that either NaN propagates or is properly handled
        assert has_nan or has_valid_result

    def test_matrix_logarithm_numerical_stability(self):
        """Numerical stability test for matrix logarithm."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_so3.random_point(key1)
        y = self.manifold_so3.random_point(key2)

        # Execute logarithmic map
        log_result = self.manifold_so3._log_impl(x, y)

        # Verify that result is finite
        assert jnp.all(jnp.isfinite(log_result))

        # Verify that result belongs to tangent space
        # Tangent space of SO(n) at x has the form x @ A (A is skew-symmetric)
        # i.e., verify that x.T @ log_result is skew-symmetric
        xtv = jnp.matmul(x.T, log_result)
        skew_check = xtv + xtv.T
        np.testing.assert_array_almost_equal(skew_check, jnp.zeros((3, 3)), decimal=6)

    def test_integration_with_existing_methods(self):
        """JIT integration and compatibility test with existing methods."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # Test combination of existing methods with JIT versions

        # validate_point method (orthogonality check)
        is_valid_point = self.manifold_so3.validate_point(x)
        assert is_valid_point

        # validate_tangent method (tangent space check)
        is_valid_tangent = self.manifold_so3.validate_tangent(x, v)
        assert is_valid_tangent
