"""Test that retr(x, 0) works as a projector for off-manifold points.

This test demonstrates that CodeRabbit's claim "retr(x, 0) = x unchanged"
is mathematically correct in theory but empirically incorrect for our
implementation, which intentionally uses normalization/QR as projection.
"""

import jax.numpy as jnp
import pytest

from riemannax.manifolds import Sphere, Stiefel


class TestRetrAsProjector:
    """Test that retr(x, zero_tangent) projects off-manifold points."""

    def test_sphere_retr_projects_off_manifold_point(self):
        """Test that Sphere's retr(x, 0) projects off-manifold points."""
        # Arrange
        manifold = Sphere(n=2)  # S^2 embedded in R^3

        # Create an off-manifold point (not unit norm)
        off_manifold_point = jnp.array([3.0, 4.0, 0.0])
        assert not manifold.validate_point(off_manifold_point), "Point should be off-manifold"

        # Act: Use retr with zero tangent
        zero_tangent = jnp.zeros_like(off_manifold_point)
        projected = manifold.retr(off_manifold_point, zero_tangent)

        # Assert: Result should be on the manifold
        assert manifold.validate_point(projected), "Projected point should be on manifold"

        # Verify it's the normalized version
        expected = off_manifold_point / jnp.linalg.norm(off_manifold_point)
        assert jnp.allclose(projected, expected, atol=1e-6)

        # Verify norm is 1
        assert jnp.allclose(jnp.linalg.norm(projected), 1.0, atol=1e-6)

    def test_stiefel_retr_projects_off_manifold_point(self):
        """Test that Stiefel's retr(x, 0) projects off-manifold points."""
        # Arrange
        manifold = Stiefel(n=5, p=3)

        # Create an off-manifold point (not orthonormal)
        off_manifold_point = jnp.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        assert not manifold.validate_point(off_manifold_point), "Point should be off-manifold"

        # Act: Use retr with zero tangent
        zero_tangent = jnp.zeros_like(off_manifold_point)
        projected = manifold.retr(off_manifold_point, zero_tangent)

        # Assert: Result should be on the manifold
        assert manifold.validate_point(projected), "Projected point should be on manifold"

        # Verify orthonormality: Q^T Q = I
        gram = projected.T @ projected
        identity = jnp.eye(3)
        assert jnp.allclose(gram, identity, atol=1e-5)

    def test_sphere_retr_with_zero_is_not_identity(self):
        """Verify that retr(x, 0) ≠ x for off-manifold points (correcting a previous review claim)."""
        # Arrange
        manifold = Sphere(n=2)
        off_manifold_point = jnp.array([2.0, 2.0, 2.0])  # norm = 2√3 ≠ 1

        # Act
        zero_tangent = jnp.zeros_like(off_manifold_point)
        result = manifold.retr(off_manifold_point, zero_tangent)

        # Assert: retr(x, 0) ≠ x (projection occurs for off-manifold points)
        assert not jnp.allclose(result, off_manifold_point), (
            "retr(x, 0) should NOT equal x for off-manifold points"
        )

        # Assert: result is normalized version of x
        expected = off_manifold_point / jnp.linalg.norm(off_manifold_point)
        assert jnp.allclose(result, expected, atol=1e-6), (
            "retr(x, 0) should return normalized x"
        )

    def test_stiefel_retr_with_zero_is_not_identity(self):
        """Verify that retr(x, 0) ≠ x for off-manifold points (correcting a previous review claim)."""
        # Arrange
        manifold = Stiefel(n=4, p=2)
        off_manifold_point = jnp.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 0.5],
            [0.5, 1.0],
        ])  # Not orthonormal

        # Act
        zero_tangent = jnp.zeros_like(off_manifold_point)
        result = manifold.retr(off_manifold_point, zero_tangent)

        # Assert: retr(x, 0) ≠ x (QR projection occurs for off-manifold points)
        assert not jnp.allclose(result, off_manifold_point, atol=1e-3), (
            "retr(x, 0) should NOT equal x for off-manifold points"
        )

        # Assert: result is orthonormal (QR projection)
        gram = result.T @ result
        assert jnp.allclose(gram, jnp.eye(2), atol=1e-5), (
            "retr(x, 0) should return QR-orthogonalized x"
        )

    def test_constraint_violation_computation_works(self):
        """Test that _compute_constraint_violation correctly measures violations."""
        # This test validates our use of retr(x, 0) in flax_nnx.py
        pytest.importorskip("flax", reason="flax not installed")
        from riemannax.api.flax_nnx import ManifoldConstrainedModule
        from flax import nnx

        # Arrange
        manifold = Sphere(n=2)
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(3,),
            rngs=nnx.Rngs(0),
        )

        # Create off-manifold parameter
        off_manifold_param = jnp.array([2.0, 0.0, 0.0])  # norm = 2

        # Act
        violation = module._compute_constraint_violation(off_manifold_param)

        # Assert: Should measure non-zero violation
        assert float(violation) > 0.0, "Should detect violation"

        # Expected: distance from [2, 0, 0] to normalized [1, 0, 0]
        expected_violation = jnp.linalg.norm(jnp.array([2.0, 0.0, 0.0]) - jnp.array([1.0, 0.0, 0.0]))
        assert jnp.allclose(violation, expected_violation, atol=1e-6)
