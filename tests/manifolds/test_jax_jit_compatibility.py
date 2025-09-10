"""Tests for JAX JIT compatibility of data models."""

import jax
import jax.numpy as jnp

from riemannax.manifolds.data_models import HyperbolicPoint, ManifoldParameters, SE3Transform


class TestJAXJITCompatibility:
    """Test JAX JIT compilation compatibility."""

    def test_hyperbolic_point_norm_jit_compatibility(self):
        """Test that HyperbolicPoint.norm() is JIT-compatible."""

        @jax.jit
        def jitted_norm(coords):
            point = HyperbolicPoint(coords, model="poincare", validate=False)
            return point.norm()

        coords = jnp.array([0.3, 0.4])
        result = jitted_norm(coords)

        # Should return JAX array, not Python float
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()  # scalar array
        assert jnp.allclose(result, 0.5)  # sqrt(0.3^2 + 0.4^2) = 0.5

    def test_hyperbolic_point_distance_jit_compatibility(self):
        """Test that HyperbolicPoint.distance_to_origin() is JIT-compatible."""

        @jax.jit
        def jitted_distance(coords):
            point = HyperbolicPoint(coords, model="poincare", validate=False)
            return point.distance_to_origin()

        coords = jnp.array([0.3, 0.4])
        result = jitted_distance(coords)

        # Should return JAX array, not Python float
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()  # scalar array
        assert jnp.isfinite(result)

    def test_hyperbolic_point_lorentz_distance_jit_compatibility(self):
        """Test that Lorentz model distance_to_origin() is JIT-compatible."""

        @jax.jit
        def jitted_lorentz_distance(coords):
            point = HyperbolicPoint(coords, model="lorentz", validate=False)
            return point.distance_to_origin()

        # Valid Lorentz coordinates: x₀² - x₁² = 1
        spatial = jnp.array([0.5])
        x0 = jnp.sqrt(1.0 + jnp.sum(spatial**2))
        coords = jnp.concatenate([jnp.array([x0]), spatial])

        result = jitted_lorentz_distance(coords)

        # Should return JAX array, not Python float
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()  # scalar array
        assert jnp.isfinite(result)

    def test_se3_transform_operations_jit_compatibility(self):
        """Test that SE3Transform operations are JIT-compatible."""

        @jax.jit
        def jitted_se3_ops(rotation, translation):
            transform = SE3Transform(rotation, translation, validate=False)
            inverse = transform.inverse()
            composed = transform.compose(inverse)
            return composed.rotation, composed.translation

        rotation = jnp.eye(3)
        translation = jnp.array([1.0, 2.0, 3.0])

        result_rot, result_trans = jitted_se3_ops(rotation, translation)

        # Should compose to approximately identity
        assert jnp.allclose(result_rot, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(result_trans, jnp.zeros(3), atol=1e-6)

    def test_manifold_parameters_jit_compatibility(self):
        """Test that ManifoldParameters operations work in JIT context."""

        def create_and_modify_params():
            params = ManifoldParameters(tolerance=1e-6, max_iterations=100)
            modified = params.copy_with(tolerance=1e-8)
            return modified.tolerance, modified.max_iterations

        # Should work without JIT issues
        tolerance, max_iter = create_and_modify_params()

        assert tolerance == 1e-8
        assert max_iter == 100

    def test_no_tracer_conversion_errors(self):
        """Test that no TracerConversionError is raised in JIT context."""

        @jax.jit
        def complex_operations(coords1, coords2):
            point1 = HyperbolicPoint(coords1, model="poincare", validate=False)
            point2 = HyperbolicPoint(coords2, model="poincare", validate=False)

            norm1 = point1.norm()
            norm2 = point2.norm()
            dist1 = point1.distance_to_origin()
            dist2 = point2.distance_to_origin()

            return norm1 + norm2, dist1 + dist2

        coords1 = jnp.array([0.1, 0.2])
        coords2 = jnp.array([0.3, 0.4])

        # Should execute without TracerConversionError
        norm_sum, dist_sum = complex_operations(coords1, coords2)

        assert jnp.isfinite(norm_sum)
        assert jnp.isfinite(dist_sum)
        assert isinstance(norm_sum, jnp.ndarray)
        assert isinstance(dist_sum, jnp.ndarray)
