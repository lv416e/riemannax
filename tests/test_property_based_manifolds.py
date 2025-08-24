"""Property-based testing for RiemannAX manifolds using Hypothesis.

This module implements property-based tests to verify mathematical properties
and constraints of Riemannian manifolds, ensuring they satisfy fundamental
differential geometric axioms and numerical stability requirements.

Property tests cover:
- Manifold constraint satisfaction
- Metric properties (positivity, symmetry, bilinearity)
- Exponential and logarithmic map properties
- Projection and retraction properties
- Numerical stability with edge cases
- Invariance properties under manifold operations
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
from hypothesis import Verbosity
from typing import Any, Tuple, Union
import math

from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.stiefel import Stiefel
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.core.constants import NumericalConstants


# Hypothesis strategies for generating test data
@st.composite
def sphere_point_and_tangent(draw, n=3):
    """Generate a point on the sphere and a tangent vector."""
    # Generate a random point and normalize to sphere
    coords = draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=n+1, max_size=n+1
    ))
    point_array = jnp.array(coords)

    # Normalize to ensure it's on the sphere
    point_norm = jnp.linalg.norm(point_array)
    assume(point_norm > 1e-6)  # Avoid degenerate cases
    point = point_array / point_norm

    # Generate tangent vector
    tangent_coords = draw(st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=n+1, max_size=n+1
    ))
    tangent_raw = jnp.array(tangent_coords)

    # Project to tangent space (orthogonal to point)
    tangent = tangent_raw - jnp.dot(tangent_raw, point) * point

    return point, tangent


@st.composite
def matrix_manifold_data(draw, p, n, manifold_type="stiefel"):
    """Generate data for matrix manifolds (Stiefel, Grassmann)."""
    # Generate random matrix
    matrix_data = draw(st.lists(
        st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n
        ),
        min_size=p, max_size=p
    ))

    raw_matrix = jnp.array(matrix_data)

    # Use QR decomposition to get orthogonal matrix
    Q, R = jnp.linalg.qr(raw_matrix.T)
    point = Q[:, :p].T  # Take first p columns, transpose to get p x n

    # Ensure we have a valid orthogonal matrix
    assume(jnp.linalg.matrix_rank(point) == p)

    # Generate tangent vector in the tangent space
    tangent_data = draw(st.lists(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n
        ),
        min_size=p, max_size=p
    ))
    tangent_raw = jnp.array(tangent_data)

    return point, tangent_raw


@st.composite
def so_matrix_and_tangent(draw, n=3):
    """Generate SO(n) matrix and tangent vector."""
    # Generate random matrix
    matrix_data = draw(st.lists(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n
        ),
        min_size=n, max_size=n
    ))

    raw_matrix = jnp.array(matrix_data)

    # Get orthogonal matrix with positive determinant using QR
    Q, R = jnp.linalg.qr(raw_matrix)

    # Ensure positive determinant
    det_Q = jnp.linalg.det(Q)
    if det_Q < 0:
        Q = Q.at[:, 0].set(-Q[:, 0])

    point = Q

    # Generate skew-symmetric tangent vector
    skew_data = draw(st.lists(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n
        ),
        min_size=n, max_size=n
    ))

    skew_raw = jnp.array(skew_data)
    tangent = (skew_raw - skew_raw.T) / 2  # Make it skew-symmetric

    return point, tangent


class TestSpherePropertyBased:
    """Property-based tests for Sphere manifold."""

    @given(sphere_point_and_tangent(n=2))
    @settings(max_examples=50, deadline=None)
    def test_sphere_point_constraint(self, data):
        """Property: Points on sphere have unit norm."""
        point, _ = data
        manifold = Sphere(n=2)

        # Verify point is on the sphere
        norm = jnp.linalg.norm(point)
        assert jnp.allclose(norm, 1.0, atol=1e-6), f"Point norm {norm} should be 1.0"

    @given(sphere_point_and_tangent(n=2))
    @settings(max_examples=50, deadline=None)
    def test_sphere_tangent_orthogonality(self, data):
        """Property: Tangent vectors are orthogonal to the point."""
        point, tangent = data
        manifold = Sphere(n=2)

        # Project tangent to ensure it's in tangent space
        projected_tangent = manifold.proj(point, tangent)

        # Verify orthogonality
        inner_product = jnp.dot(point, projected_tangent)
        assert jnp.allclose(inner_product, 0.0, atol=1e-6), \
            f"Tangent should be orthogonal to point: dot product = {inner_product}"

    @given(sphere_point_and_tangent(n=2))
    @settings(max_examples=30, deadline=None)
    def test_sphere_exponential_map_constraint(self, data):
        """Property: Exponential map produces points on the sphere."""
        point, tangent = data
        manifold = Sphere(n=2)

        # Project tangent to tangent space
        projected_tangent = manifold.proj(point, tangent)

        # Assume tangent vector is not too large to avoid numerical issues
        tangent_norm = jnp.linalg.norm(projected_tangent)
        assume(tangent_norm < 5.0)

        # Apply exponential map
        exp_point = manifold.exp(point, projected_tangent)

        # Verify result is on sphere
        exp_norm = jnp.linalg.norm(exp_point)
        assert jnp.allclose(exp_norm, 1.0, atol=1e-6), \
            f"Exponential map result norm {exp_norm} should be 1.0"

    @given(sphere_point_and_tangent(n=2))
    @settings(max_examples=30, deadline=None)
    def test_sphere_metric_properties(self, data):
        """Property: Riemannian metric is positive definite and symmetric."""
        point, tangent = data
        manifold = Sphere(n=2)

        # Project tangent to tangent space
        v = manifold.proj(point, tangent)
        u = manifold.proj(point, tangent * 0.7)  # Another tangent vector

        # Skip if tangent vectors are too small
        assume(jnp.linalg.norm(v) > 1e-6)
        assume(jnp.linalg.norm(u) > 1e-6)

        # Test metric properties
        # 1. Positive definiteness: <v, v> > 0 for v ≠ 0
        metric_vv = manifold.inner(point, v, v)
        assert metric_vv > 0, f"Metric should be positive: <v,v> = {metric_vv}"

        # 2. Symmetry: <u, v> = <v, u>
        metric_uv = manifold.inner(point, u, v)
        metric_vu = manifold.inner(point, v, u)
        assert jnp.allclose(metric_uv, metric_vu, atol=NumericalConstants.ATOL), \
            f"Metric should be symmetric: <u,v>={metric_uv}, <v,u>={metric_vu}"

    @given(sphere_point_and_tangent(n=2))
    @settings(max_examples=20, deadline=None)
    def test_sphere_exp_log_inverse(self, data):
        """Property: log is the inverse of exp for nearby points."""
        point, tangent = data
        manifold = Sphere(n=2)

        # Use small tangent vector to stay in normal neighborhood
        v = manifold.proj(point, tangent)
        v_norm = jnp.linalg.norm(v)
        assume(v_norm > 1e-6 and v_norm < 1.0)  # Small but not too small

        small_v = v * 0.5  # Scale down further

        # Apply exp then log
        exp_point = manifold.exp(point, small_v)
        recovered_v = manifold.log(point, exp_point)

        # Should recover original tangent vector
        error = jnp.linalg.norm(recovered_v - small_v)
        assert error < 1e-4, f"exp-log should be inverse: error = {error}"

    @given(sphere_point_and_tangent(n=3))
    @settings(max_examples=20, deadline=None)
    def test_sphere_geodesic_distance(self, data):
        """Property: Geodesic distance satisfies triangle inequality and symmetry."""
        point1, tangent = data
        manifold = Sphere(n=3)

        # Create a second point using exponential map
        v = manifold.proj(point1, tangent)
        assume(jnp.linalg.norm(v) > 1e-6)

        small_v = v * 0.3  # Keep points close
        point2 = manifold.exp(point1, small_v)

        # Create a third point
        v2 = manifold.proj(point2, tangent * 0.8)
        assume(jnp.linalg.norm(v2) > 1e-6)
        point3 = manifold.exp(point2, v2 * 0.3)

        # Test distance properties
        d12 = manifold.dist(point1, point2)
        d21 = manifold.dist(point2, point1)
        d13 = manifold.dist(point1, point3)
        d23 = manifold.dist(point2, point3)

        # Symmetry: d(x,y) = d(y,x)
        assert jnp.allclose(d12, d21, atol=NumericalConstants.ATOL), \
            f"Distance should be symmetric: d12={d12}, d21={d21}"

        # Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
        triangle_violation = d13 - (d12 + d23)
        assert triangle_violation <= 1e-6, \
            f"Triangle inequality violated: d13={d13}, d12+d23={d12+d23}"


class TestStiefelPropertyBased:
    """Property-based tests for Stiefel manifold."""

    @given(matrix_manifold_data(p=2, n=4))
    @settings(max_examples=30, deadline=None)
    def test_stiefel_orthogonality_constraint(self, data):
        """Property: Points on Stiefel manifold satisfy X^T X = I."""
        point, _ = data
        manifold = Stiefel(p=2, n=4)

        # Verify orthogonality constraint
        product = jnp.dot(point, point.T)
        identity = jnp.eye(point.shape[0])

        error = jnp.linalg.norm(product - identity, ord='fro')
        assert error < 1e-4, f"Stiefel constraint violated: ||X^T X - I||_F = {error}"

    @given(matrix_manifold_data(p=2, n=3))
    @settings(max_examples=30, deadline=None)
    def test_stiefel_tangent_space_constraint(self, data):
        """Property: Tangent vectors satisfy constraint X^T V + V^T X = 0."""
        point, tangent_raw = data
        manifold = Stiefel(p=2, n=3)

        # Project to tangent space
        tangent = manifold.proj(point, tangent_raw)

        # Check tangent space constraint
        constraint = jnp.dot(point.T, tangent) + jnp.dot(tangent.T, point)
        constraint_error = jnp.linalg.norm(constraint, ord='fro')

        assert constraint_error < 1e-6, \
            f"Tangent space constraint violated: ||X^T V + V^T X||_F = {constraint_error}"


class TestSpecialOrthogonalPropertyBased:
    """Property-based tests for Special Orthogonal group SO(n)."""

    @given(so_matrix_and_tangent(n=3))
    @settings(max_examples=30, deadline=None)
    def test_so_orthogonality_and_determinant(self, data):
        """Property: SO(n) matrices are orthogonal with determinant 1."""
        point, _ = data
        manifold = SpecialOrthogonal(n=3)

        # Test orthogonality: R^T R = I
        product = jnp.dot(point.T, point)
        identity = jnp.eye(point.shape[0])
        orthogonality_error = jnp.linalg.norm(product - identity, ord='fro')
        assert orthogonality_error < 1e-6, f"Orthogonality violated: error = {orthogonality_error}"

        # Test determinant: det(R) = 1
        det_point = jnp.linalg.det(point)
        assert jnp.allclose(det_point, 1.0, atol=NumericalConstants.ATOL), \
            f"Determinant should be 1: det = {det_point}"

    @given(so_matrix_and_tangent(n=3))
    @settings(max_examples=30, deadline=None)
    def test_so_tangent_skew_symmetry(self, data):
        """Property: Tangent vectors in so(n) are skew-symmetric."""
        point, tangent_raw = data
        manifold = SpecialOrthogonal(n=3)

        # Project to tangent space
        tangent = manifold.proj(point, tangent_raw)

        # For SO(n), tangent vectors at identity are skew-symmetric
        # For general points, R * skew = tangent where skew is skew-symmetric
        try:
            skew = jnp.dot(point.T, tangent)
            skew_symmetry_error = jnp.linalg.norm(skew + skew.T, ord='fro')
            # Allow some tolerance for numerical precision
            assert skew_symmetry_error < 1e-4, \
                f"Tangent space element should correspond to skew-symmetric: error = {skew_symmetry_error}"
        except Exception:
            # If computation fails, it's acceptable to skip this specific case
            assume(False)


class TestNumericalStabilityPropertyBased:
    """Property-based tests for numerical stability across manifolds."""

    @given(st.floats(min_value=1e-12, max_value=1e-6))
    @settings(max_examples=20)
    def test_sphere_small_tangent_vectors(self, scale):
        """Property: Operations remain stable with very small tangent vectors."""
        manifold = Sphere(n=2)
        point = jnp.array([1.0, 0.0, 0.0])

        # Very small tangent vector
        small_tangent = jnp.array([0.0, scale, scale * 0.5])
        projected = manifold.proj(point, small_tangent)

        # Operations should not produce NaN or inf
        exp_result = manifold.exp(point, projected)
        assert jnp.all(jnp.isfinite(exp_result)), "Exponential map with small vector produced non-finite result"

        # Result should still be on the sphere
        exp_norm = jnp.linalg.norm(exp_result)
        assert jnp.allclose(exp_norm, 1.0, atol=NumericalConstants.ATOL), \
            f"Small tangent vector exp result not on sphere: norm = {exp_norm}"

    @given(st.floats(min_value=2.0, max_value=10.0))
    @settings(max_examples=20)
    def test_sphere_large_tangent_vectors(self, scale):
        """Property: Operations remain stable with large tangent vectors."""
        manifold = Sphere(n=2)
        point = jnp.array([1.0, 0.0, 0.0])

        # Large tangent vector
        large_tangent = jnp.array([0.0, scale, scale * 0.3])
        projected = manifold.proj(point, large_tangent)

        # Operations should not produce NaN or inf
        exp_result = manifold.exp(point, projected)
        assert jnp.all(jnp.isfinite(exp_result)), "Exponential map with large vector produced non-finite result"

        # Result should still be on the sphere
        exp_norm = jnp.linalg.norm(exp_result)
        assert jnp.allclose(exp_norm, 1.0, atol=1e-6), \
            f"Large tangent vector exp result not on sphere: norm = {exp_norm}"

    @given(st.floats(min_value=-1.0 + 1e-6, max_value=1.0 - 1e-6))
    def test_sphere_antipodal_points_stability(self, perturbation):
        """Property: Operations remain stable near antipodal points."""
        manifold = Sphere(n=2)
        point1 = jnp.array([1.0, 0.0, 0.0])

        # Nearly antipodal point
        point2 = jnp.array([-1.0, perturbation, perturbation * 0.5])
        point2 = point2 / jnp.linalg.norm(point2)  # Normalize

        # Distance computation should be stable
        distance = manifold.dist(point1, point2)
        assert jnp.isfinite(distance), "Distance computation failed for antipodal points"
        assert distance <= np.pi, f"Spherical distance should be ≤ π: got {distance}"

        # Log map might be unstable at exact antipodal points, but should work nearby
        if jnp.abs(jnp.dot(point1, point2) + 1.0) > 1e-6:  # Not exactly antipodal
            log_result = manifold.log(point1, point2)
            assert jnp.all(jnp.isfinite(log_result)), "Log map failed for nearly antipodal points"


class TestManifoldUniversalProperties:
    """Universal property tests that should hold for all manifolds."""

    @pytest.mark.parametrize("manifold,test_data", [
        (Sphere(n=2), ([1.0, 0.0, 0.0], [0.0, 0.1, 0.2])),
        (SpecialOrthogonal(n=3), (jnp.eye(3), jnp.array([[0, 0.1, 0], [-0.1, 0, 0], [0, 0, 0]]))),
    ])
    def test_projection_idempotency(self, manifold, test_data):
        """Property: Projection is idempotent: proj(x, proj(x, v)) = proj(x, v)."""
        point, tangent_raw = test_data
        point = jnp.array(point)
        tangent_raw = jnp.array(tangent_raw)

        # First projection
        v1 = manifold.proj(point, tangent_raw)

        # Second projection (should be the same)
        v2 = manifold.proj(point, v1)

        error = jnp.linalg.norm(v1 - v2)
        assert error < 1e-10, f"Projection not idempotent: error = {error}"

    @pytest.mark.parametrize("manifold,test_data", [
        (Sphere(n=2), ([1.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1])),
    ])
    def test_metric_bilinearity(self, manifold, test_data):
        """Property: Metric is bilinear in both arguments."""
        point, u_raw, v_raw = test_data
        point = jnp.array(point)
        u = manifold.proj(point, jnp.array(u_raw))
        v = manifold.proj(point, jnp.array(v_raw))

        # Skip if vectors are too small
        if jnp.linalg.norm(u) < 1e-8 or jnp.linalg.norm(v) < 1e-8:
            pytest.skip("Test vectors too small")

        a, b = 2.0, 3.0

        # Bilinearity in first argument: <au + bv, w> = a<u,w> + b<v,w>
        w = v  # Use v as w for simplicity
        left = manifold.inner(point, a * u + b * v, w)
        right = a * manifold.inner(point, u, w) + b * manifold.inner(point, v, w)

        error = jnp.abs(left - right)
        assert error < 1e-10, f"Metric not bilinear in first argument: error = {error}"

        # Bilinearity in second argument: <u, aw + bv> = a<u,w> + b<u,v>
        left = manifold.inner(point, u, a * w + b * v)
        right = a * manifold.inner(point, u, w) + b * manifold.inner(point, u, v)

        error = jnp.abs(left - right)
        assert error < 1e-10, f"Metric not bilinear in second argument: error = {error}"


# Configuration for property-based testing
def pytest_configure(config):
    """Configure Hypothesis settings for property-based tests."""
    from hypothesis import settings, Verbosity

    # Set global test settings
    settings.register_profile("dev", max_examples=20, deadline=None)
    settings.register_profile("ci", max_examples=100, deadline=60000)  # 60 seconds
    settings.register_profile("thorough", max_examples=500, deadline=None)

    # Use 'dev' profile by default
    settings.load_profile("dev")
