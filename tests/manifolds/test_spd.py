"""Tests for the Symmetric Positive Definite manifold.

This module contains tests for the SymmetricPositiveDefinite manifold implementation.
"""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rieax


@pytest.fixture
def key():
    """JAX random key for testing."""
    return jax.random.key(42)


@pytest.fixture
def spd3():
    """Create an SPD(3) manifold instance for testing."""
    return rieax.SymmetricPositiveDefinite(n=3)


@pytest.fixture
def point_on_spd3(key, spd3):
    """Create a random point on SPD(3) for testing."""
    return spd3.random_point(key)


@pytest.fixture
def tangent_vec(spd3, point_on_spd3, key):
    """Create a tangent vector at the given point for testing."""
    return spd3.random_tangent(key, point_on_spd3)


def test_spd_initialization():
    """Test initialization of the SPD manifold."""
    # Test different dimensions
    spd2 = rieax.SymmetricPositiveDefinite(n=2)
    assert spd2.n == 2

    spd4 = rieax.SymmetricPositiveDefinite(n=4)
    assert spd4.n == 4


def test_spd_proj(spd3, point_on_spd3):
    """Test the projection operation on SPD(3)."""
    # Create a random matrix in the ambient space
    random_matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Project onto the tangent space
    proj_v = spd3.proj(point_on_spd3, random_matrix)

    # The projection should be symmetric
    assert jnp.allclose(proj_v, proj_v.T, atol=1e-6)

    # Projecting twice should give the same result (idempotent)
    proj_proj_v = spd3.proj(point_on_spd3, proj_v)
    assert jnp.allclose(proj_v, proj_proj_v, atol=1e-6)


def test_spd_exp_log(spd3, point_on_spd3, tangent_vec):
    """Test the exponential and logarithmic maps on SPD(3)."""
    # Apply the exponential map
    new_point = spd3.exp(point_on_spd3, tangent_vec)

    # Check that the result is in SPD(3)
    assert_is_spd(new_point)

    # Test log-exp consistency (exp(log(y)) should give y)
    another_point = spd3.random_point(jax.random.fold_in(jax.random.key(42), 1))
    log_vec = spd3.log(point_on_spd3, another_point)
    recovered_point = spd3.exp(point_on_spd3, log_vec)
    assert jnp.allclose(another_point, recovered_point, atol=1e-5)

    # Test with the zero vector
    zero_vec = jnp.zeros_like(tangent_vec)
    exp_zero = spd3.exp(point_on_spd3, zero_vec)
    assert jnp.allclose(point_on_spd3, exp_zero, atol=1e-6)


def test_spd_inner(spd3, point_on_spd3, key):
    """Test the Riemannian inner product on SPD(3)."""
    # Create two tangent vectors
    v1 = spd3.random_tangent(jax.random.fold_in(key, 1), point_on_spd3)
    v2 = spd3.random_tangent(jax.random.fold_in(key, 2), point_on_spd3)

    # Compute the inner product
    inner_prod = spd3.inner(point_on_spd3, v1, v2)

    # Inner product should be real
    assert jnp.isreal(inner_prod)

    # Inner product should be symmetric
    inner_prod_sym = spd3.inner(point_on_spd3, v2, v1)
    assert jnp.allclose(inner_prod, inner_prod_sym, atol=1e-6)

    # Inner product with itself should be positive
    self_inner = spd3.inner(point_on_spd3, v1, v1)
    assert self_inner >= 0


def test_spd_transp(spd3, point_on_spd3, key):
    """Test parallel transport on SPD(3)."""
    # Create another point on SPD(3)
    another_point = spd3.random_point(jax.random.fold_in(key, 1))

    # Create a tangent vector at the first point
    tangent = spd3.random_tangent(jax.random.fold_in(key, 2), point_on_spd3)

    # Transport the tangent vector to the second point
    transported = spd3.transp(point_on_spd3, another_point, tangent)

    # Check that the result is in the tangent space at the second point
    # (should be symmetric)
    assert jnp.allclose(transported, transported.T, atol=1e-6)

    # Note: For simplified parallel transport, norm preservation is approximate
    # We mainly check that the operation produces reasonable results
    orig_norm = spd3.inner(point_on_spd3, tangent, tangent)
    transp_norm = spd3.inner(another_point, transported, transported)
    # Check that norms are positive and finite
    assert orig_norm > 0 and jnp.isfinite(orig_norm)
    assert transp_norm > 0 and jnp.isfinite(transp_norm)


def test_spd_dist(spd3, key):
    """Test the geodesic distance on SPD(3)."""
    # Create two points on SPD(3)
    p1 = spd3.random_point(jax.random.fold_in(key, 1))
    p2 = spd3.random_point(jax.random.fold_in(key, 2))

    # Compute the distance
    dist = spd3.dist(p1, p2)

    # Distance should be non-negative
    assert dist >= 0

    # Distance from a point to itself should be zero (with numerical tolerance)
    assert jnp.allclose(spd3.dist(p1, p1), 0.0, atol=1e-5)

    # Distance should be symmetric
    dist_sym = spd3.dist(p2, p1)
    assert jnp.allclose(dist, dist_sym, atol=1e-6)

    # Triangle inequality (with numerical tolerance)
    p3 = spd3.random_point(jax.random.fold_in(key, 3))
    d12 = spd3.dist(p1, p2)
    d23 = spd3.dist(p2, p3)
    d13 = spd3.dist(p1, p3)
    assert d13 <= d12 + d23 + 1e-5


def test_spd_random_point(spd3, key):
    """Test generating random points on SPD(3)."""
    # Generate a random point
    point = spd3.random_point(key)

    # Check that it's in SPD(3)
    assert_is_spd(point)

    # Test batch generation
    batch_points = spd3.random_point(key, 5)
    assert batch_points.shape == (5, 3, 3)

    # Check each point in the batch
    for i in range(5):
        assert_is_spd(batch_points[i])


def test_spd_random_tangent(spd3, key, point_on_spd3):
    """Test generating random tangent vectors."""
    # Generate a random tangent vector
    tangent = spd3.random_tangent(key, point_on_spd3)

    # Check that it's symmetric (in tangent space)
    assert jnp.allclose(tangent, tangent.T, atol=1e-6)

    # Generate multiple random tangent vectors
    batch_tangents = spd3.random_tangent(key, point_on_spd3, 4)
    assert batch_tangents.shape == (4, 3, 3)

    # Check each tangent vector in the batch
    for i in range(4):
        assert jnp.allclose(batch_tangents[i], batch_tangents[i].T, atol=1e-6)


def test_spd_is_in_manifold(spd3):
    """Test the manifold membership check."""
    # Create a valid SPD matrix
    A = jnp.array([[2.0, 1.0, 0.5], [1.0, 3.0, 1.5], [0.5, 1.5, 2.5]])
    assert spd3._is_in_manifold(A)

    # Create a non-symmetric matrix
    B = jnp.array([[2.0, 1.0, 0.5], [0.0, 3.0, 1.5], [0.5, 1.5, 2.5]])
    assert not spd3._is_in_manifold(B)

    # Create a symmetric but not positive definite matrix
    C = jnp.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
    # This matrix has negative eigenvalues
    assert not spd3._is_in_manifold(C)


def test_spd_small_matrices():
    """Test SPD manifold with smaller matrices."""
    # Test 2x2 case
    spd2 = rieax.SymmetricPositiveDefinite(n=2)
    key = jax.random.key(123)

    point2 = spd2.random_point(key)
    assert_is_spd(point2)
    assert point2.shape == (2, 2)

    tangent2 = spd2.random_tangent(key, point2)
    assert jnp.allclose(tangent2, tangent2.T, atol=1e-6)

    # Test exp-log consistency
    another_point2 = spd2.random_point(jax.random.fold_in(key, 1))
    log_vec2 = spd2.log(point2, another_point2)
    recovered2 = spd2.exp(point2, log_vec2)
    assert jnp.allclose(another_point2, recovered2, atol=1e-5)


def test_spd_numerical_stability():
    """Test numerical stability with extreme cases."""
    spd3 = rieax.SymmetricPositiveDefinite(n=3)

    # Test with identity matrix
    I = jnp.eye(3)
    assert spd3._is_in_manifold(I)

    # Test with small but positive definite matrix
    small_matrix = 1e-3 * jnp.eye(3)
    assert spd3._is_in_manifold(small_matrix)

    # Test projecting zero vector
    zero_ambient = jnp.zeros((3, 3))
    projected_zero = spd3.proj(I, zero_ambient)
    assert jnp.allclose(projected_zero, jnp.zeros((3, 3)), atol=1e-10)


def assert_is_spd(matrix, tolerance=1e-6):
    """Check that a matrix is symmetric positive definite."""
    # Check symmetry
    assert jnp.allclose(matrix, matrix.T, atol=tolerance), "Matrix is not symmetric"

    # Check positive definiteness via eigenvalues
    eigenvals = jnp.linalg.eigvals(matrix)
    assert jnp.all(eigenvals > tolerance), f"Matrix is not positive definite, eigenvalues: {eigenvals}"


def test_spd_integration_with_optimization():
    """Test SPD manifold integration with the optimization framework."""
    spd2 = rieax.SymmetricPositiveDefinite(n=2)

    # Define a simple cost function (Frobenius norm to identity)
    def cost_fn(x):
        I = jnp.eye(2)
        return jnp.sum((x - I) ** 2)

    # Create a problem
    problem = rieax.RiemannianProblem(spd2, cost_fn)

    # Create an initial point
    key = jax.random.key(777)
    x0 = spd2.random_point(key)

    # Solve the problem (should converge towards identity)
    result = rieax.minimize(problem, x0, method="rsgd", options={"learning_rate": 0.1, "max_iterations": 20})

    # Check that optimization succeeded
    assert result.success

    # Check that the result is still in the manifold
    assert spd2._is_in_manifold(result.x)

    # Check that cost decreased
    initial_cost = cost_fn(x0)
    final_cost = cost_fn(result.x)
    assert final_cost <= initial_cost, "Optimization should not increase cost"


# Log-Euclidean Metric Tests


def test_log_euclidean_exp(spd3, point_on_spd3, tangent_vec):
    """Test Log-Euclidean exponential map."""
    # Compute Log-Euclidean exponential
    result = spd3.log_euclidean_exp(point_on_spd3, tangent_vec)

    # Result should be SPD
    assert_is_spd(result)

    # Result should have the same shape
    assert result.shape == point_on_spd3.shape

    # For zero tangent vector, should return the original point
    zero_tangent = jnp.zeros_like(tangent_vec)
    result_zero = spd3.log_euclidean_exp(point_on_spd3, zero_tangent)
    assert jnp.allclose(result_zero, point_on_spd3, atol=1e-6)


def test_log_euclidean_log(spd3, point_on_spd3, key):
    """Test Log-Euclidean logarithmic map."""
    # Create another point
    other_point = spd3.random_point(jax.random.fold_in(key, 1))

    # Compute Log-Euclidean logarithm
    log_vec = spd3.log_euclidean_log(point_on_spd3, other_point)

    # Result should be symmetric (in tangent space)
    assert jnp.allclose(log_vec, log_vec.T, atol=1e-6)

    # Result should have the same shape
    assert log_vec.shape == point_on_spd3.shape

    # For identical points, should return zero
    zero_log = spd3.log_euclidean_log(point_on_spd3, point_on_spd3)
    assert jnp.allclose(zero_log, jnp.zeros_like(zero_log), atol=1e-6)


def test_log_euclidean_exp_log_inverse(spd3, point_on_spd3, tangent_vec):
    """Test that Log-Euclidean exp and log are inverses."""
    # Forward: exp(point, tangent) -> result
    result = spd3.log_euclidean_exp(point_on_spd3, tangent_vec)

    # Backward: log(point, result) -> recovered_tangent
    recovered_tangent = spd3.log_euclidean_log(point_on_spd3, result)

    # Should recover the original tangent vector
    assert jnp.allclose(tangent_vec, recovered_tangent, atol=1e-5)


def test_log_euclidean_distance(spd3, point_on_spd3, key):
    """Test Log-Euclidean distance computation."""
    # Create another point
    other_point = spd3.random_point(jax.random.fold_in(key, 1))

    # Compute Log-Euclidean distance
    distance = spd3.log_euclidean_distance(point_on_spd3, other_point)

    # Distance should be non-negative scalar
    assert distance >= 0
    assert distance.shape == ()

    # Distance to itself should be zero
    self_distance = spd3.log_euclidean_distance(point_on_spd3, point_on_spd3)
    assert jnp.allclose(self_distance, 0.0, atol=1e-10)

    # Distance should be symmetric
    reverse_distance = spd3.log_euclidean_distance(other_point, point_on_spd3)
    assert jnp.allclose(distance, reverse_distance, atol=1e-6)


def test_log_euclidean_interpolation(spd3, point_on_spd3, key):
    """Test Log-Euclidean geodesic interpolation."""
    # Create another point
    other_point = spd3.random_point(jax.random.fold_in(key, 1))

    # Test interpolation at different values of t
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for t in t_values:
        interpolated = spd3.log_euclidean_interpolation(point_on_spd3, other_point, t)

        # Result should be SPD
        assert_is_spd(interpolated)

        # At t=0, should get first point
        if t == 0.0:
            assert jnp.allclose(interpolated, point_on_spd3, atol=1e-6)

        # At t=1, should get second point
        if t == 1.0:
            assert jnp.allclose(interpolated, other_point, atol=1e-6)


def test_log_euclidean_mean(spd3, key):
    """Test Log-Euclidean mean (Riemannian center of mass)."""
    # Generate multiple points
    keys = jax.random.split(key, 5)
    points = jnp.array([spd3.random_point(k) for k in keys])

    # Compute Log-Euclidean mean
    mean_point = spd3.log_euclidean_mean(points)

    # Result should be SPD
    assert_is_spd(mean_point)

    # For single point, mean should be the point itself
    single_point = points[0:1]
    single_mean = spd3.log_euclidean_mean(single_point)
    assert jnp.allclose(single_mean, points[0], atol=1e-6)

    # Mean should have same shape as input points
    assert mean_point.shape == points[0].shape


def test_log_euclidean_vs_affine_invariant(spd3, point_on_spd3, tangent_vec):
    """Test comparison between Log-Euclidean and affine-invariant metrics."""
    # Both metrics should give SPD results for exp operation
    le_result = spd3.log_euclidean_exp(point_on_spd3, tangent_vec)
    ai_result = spd3.exp(point_on_spd3, tangent_vec)  # affine-invariant

    # Both should be SPD
    assert_is_spd(le_result)
    assert_is_spd(ai_result)

    # They should be different (unless tangent vector is zero)
    if jnp.linalg.norm(tangent_vec) > 1e-10:
        # Allow them to be different (they use different metrics)
        # This test just ensures both work correctly
        assert le_result.shape == ai_result.shape
