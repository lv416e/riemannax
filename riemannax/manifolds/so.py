"""Implementation of the special orthogonal group SO(n) with its Riemannian geometry.

This module provides operations for optimization on the special orthogonal group,
which represents rotations in n-dimensional space. SO(n) consists of all n x n
orthogonal matrices with determinant 1.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from .base import Manifold


class SpecialOrthogonal(Manifold):
    """Special orthogonal group SO(n) with canonical Riemannian metric.

    The special orthogonal group SO(n) consists of n x n orthogonal matrices with
    determinant 1, representing rotations in n-dimensional Euclidean space.
    """

    def __init__(self, n=3):
        """Initialize SO(n) manifold.

        Args:
            n: Dimension of the rotation space (default: 3 for 3D rotations)
        """
        super().__init__()
        self.n = n

    def proj(self, x, v):
        """Project a matrix onto the tangent space of SO(n) at point x.

        The tangent space at x consists of matrices of the form x @ A where A is
        skew-symmetric (A = -A.T).

        Args:
            x: Point on SO(n) (orthogonal matrix with det=1).
            v: Matrix in the ambient space R^(n x n).

        Returns:
            The projection of v onto the tangent space at x.
        """
        # The tangent space of SO(n) at x consists of matrices of the form x @ A,
        # where A is skew-symmetric, i.e., A = -A.T

        # Compute x.T @ v
        xtv = jnp.matmul(x.T, v)

        # Extract the skew-symmetric part: 0.5(xtv - xtv.T)
        skew_part = 0.5 * (xtv - xtv.T)

        # Project back to the tangent space at x
        return jnp.matmul(x, skew_part)

    def exp(self, x, v):
        """Compute the exponential map on SO(n).

        The exponential map corresponds to the matrix exponential of the
        skew-symmetric matrix representing the tangent vector.

        Args:
            x: Point on SO(n) (orthogonal matrix with det=1).
            v: Tangent vector at x.

        Returns:
            The point on SO(n) reached by following the geodesic from x in direction v.
        """
        # For SO(n), the exponential map is: x @ expm(x.T @ v)
        # First, convert the tangent vector to a skew-symmetric matrix in the Lie algebra
        xtv = jnp.matmul(x.T, v)
        skew = 0.5 * (xtv - xtv.T)

        # Compute the matrix exponential of the skew-symmetric matrix
        # This is implemented using Rodrigues' formula for efficiency in the 3D case
        if self.n == 3:
            # For SO(3), we can use the Rodrigues' formula
            return x @ self._expm_so3(skew)
        else:
            # For general SO(n), use the matrix exponential
            return x @ self._expm(skew)

    def log(self, x, y):
        """Compute the logarithmic map on SO(n).

        For two points x and y on SO(n), this finds the tangent vector v at x
        such that following the geodesic in that direction reaches y.

        Args:
            x: Starting point on SO(n) (orthogonal matrix with det=1).
            y: Target point on SO(n) (orthogonal matrix with det=1).

        Returns:
            The tangent vector at x that points toward y along the geodesic.
        """
        # For SO(n), the logarithmic map is: x @ logm(x.T @ y)
        rel_rot = jnp.matmul(x.T, y)

        skew = self._logm_so3(rel_rot) if self.n == 3 else self._logm(rel_rot)

        return jnp.matmul(x, skew)

    def transp(self, x, y, v):
        """Parallel transport on SO(n) from x to y.

        For SO(n), parallel transport is given by conjugation.

        Args:
            x: Starting point on SO(n) (orthogonal matrix with det=1).
            y: Target point on SO(n) (orthogonal matrix with det=1).
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        # For SO(n), parallel transport is a conjugation
        # Compute the relative rotation
        rel_rot = jnp.matmul(y, x.T)

        # Apply the transport
        return jnp.matmul(rel_rot, v)

    def inner(self, x, u, v):
        """Compute the Riemannian inner product on SO(n).

        The canonical Riemannian metric on SO(n) is the Frobenius inner product
        of the corresponding matrices.

        Args:
            x: Point on SO(n) (orthogonal matrix with det=1).
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x in the Riemannian metric.
        """
        # The canonical Riemannian metric is the Frobenius inner product
        return jnp.sum(u * v)

    def dist(self, x, y):
        """Compute the geodesic distance between points on SO(n).

        The geodesic distance is the Frobenius norm of the matrix logarithm of x.T @ y.

        Args:
            x: First point on SO(n) (orthogonal matrix with det=1).
            y: Second point on SO(n) (orthogonal matrix with det=1).

        Returns:
            The geodesic distance between x and y.
        """
        # Compute the relative rotation
        rel_rot = jnp.matmul(x.T, y)

        if self.n == 3:
            # For SO(3), we can use the angle-axis representation
            return self._geodesic_distance_so3(rel_rot)
        else:
            # For general SO(n), compute the matrix logarithm
            skew = self._logm(rel_rot)
            return jnp.sqrt(jnp.sum(skew**2))

    def random_point(self, key, *shape):
        """Generate random point(s) on SO(n).

        Points are sampled uniformly from SO(n) using the QR decomposition of
        random normal matrices.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) on SO(n) with specified shape.
        """
        shape = (self.n, self.n) if not shape else (*tuple(shape), self.n, self.n)

        # Sample random matrices from normal distribution
        key, subkey = jr.split(key)
        random_matrices = jr.normal(subkey, shape)

        # Use the QR decomposition to get orthogonal matrices
        q, _ = jnp.linalg.qr(random_matrices)

        # Ensure determinant is 1 by flipping the sign of a column if necessary
        det_sign = jnp.sign(jnp.linalg.det(q))

        # Reshape det_sign to broadcast correctly
        if len(shape) > 2:
            reshape_dims = tuple([-1] + [1] * (len(shape) - 1))
            det_sign = det_sign.reshape(reshape_dims)

        # Multiply the last column by sign(det) to ensure determinant is 1
        q = q.at[..., :, -1].multiply(det_sign)

        return q

    def random_tangent(self, key, x, *shape):
        """Generate random tangent vector(s) at point x.

        Tangent vectors are generated as x @ A where A is a random skew-symmetric matrix.

        Args:
            key: JAX PRNG key.
            x: Point on SO(n) (orthogonal matrix with det=1).
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        shape = x.shape if not shape else (*tuple(shape), *x.shape)

        # Generate random skew-symmetric matrices
        key, subkey = jr.split(key)
        tril_indices = jnp.tril_indices(self.n, -1)

        # Determine how many random values we need
        num_random_values = self.n * (self.n - 1) // 2
        if len(shape) > 2:
            batch_size = jnp.prod(jnp.array(shape[:-2]))
            random_vals = jr.normal(subkey, (batch_size, num_random_values))
        else:
            random_vals = jr.normal(subkey, (num_random_values,))

        # Create skew-symmetric matrices
        def create_skew(vals):
            skew = jnp.zeros((self.n, self.n))
            skew = skew.at[tril_indices].set(vals)
            return skew - skew.T

        if len(shape) > 2:
            skews = jax.vmap(create_skew)(random_vals)
            skews = skews.reshape((*shape[:-2], self.n, self.n))
        else:
            skews = create_skew(random_vals)

        # Map to tangent space at x
        # For SO(n), the tangent space at x is {x @ A | A is skew-symmetric}
        return jnp.matmul(x, skews)

    def _expm_so3(self, skew):
        """Compute the matrix exponential for a skew-symmetric 3x3 matrix.

        Uses Rodrigues' rotation formula for efficiency.

        Args:
            skew: 3x3 skew-symmetric matrix.

        Returns:
            The matrix exponential exp(skew).
        """
        # Extract the rotation vector from the skew-symmetric matrix
        phi_1 = skew[2, 1]
        phi_2 = skew[0, 2]
        phi_3 = skew[1, 0]

        # Construct the rotation vector
        phi = jnp.array([phi_1, phi_2, phi_3])

        # Compute the angle of rotation
        angle = jnp.linalg.norm(phi)

        # Handle small angles to avoid numerical issues
        small_angle = angle < 1e-8

        def small_angle_case():
            # For small angles, use Taylor expansion
            return jnp.eye(3) + skew + 0.5 * jnp.matmul(skew, skew)

        def normal_case():
            # For normal angles, use Rodrigues' formula
            K = skew / angle
            return jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * jnp.matmul(K, K)

        return lax.cond(small_angle, small_angle_case, normal_case)

    def _expm(self, skew):
        """Compute the matrix exponential for a skew-symmetric matrix.

        Uses Padé approximation or eigendecomposition.

        Args:
            skew: Skew-symmetric matrix.

        Returns:
            The matrix exponential exp(skew).
        """
        # For now, use JAX's implementation via scipy
        # In the future, implement a more efficient version specific to skew-symmetric matrices
        from jax.scipy.linalg import expm

        return expm(skew)

    def _logm_so3(self, rot):
        """Compute the matrix logarithm for a 3x3 rotation matrix.

        Inverse of Rodrigues' formula for SO(3).

        Args:
            rot: 3x3 rotation matrix.

        Returns:
            The skew-symmetric matrix log(rot).
        """
        # Compute the trace
        trace = jnp.trace(rot)

        # Handle different cases based on the trace
        # If trace = 3, then rot = I (no rotation)
        # If trace = -1, then rot represents a 180-degree rotation

        # Clamp trace to valid range to handle numerical imprecision
        trace_clamped = jnp.clip(trace, -1.0, 3.0)

        # Compute the rotation angle using arccos((trace - 1)/2)
        cos_angle = (trace_clamped - 1.0) / 2.0
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))

        # Handle different cases
        def small_angle_case():
            # For small angles (rot ≈ I), use first-order approximation
            return 0.5 * (rot - rot.T)

        def normal_case():
            # For normal cases, use the full formula
            factor = angle / (2.0 * jnp.sin(angle))
            return factor * (rot - rot.T)

        def pi_angle_case():
            # For 180-degree rotations (trace = -1), special handling is needed
            # Find the rotation axis from the eigenvector with eigenvalue 1
            diag = jnp.diag(rot)
            max_idx = jnp.argmax(diag)

            if max_idx == 0:
                axis = jnp.array([1 + rot[0, 0], rot[0, 1], rot[0, 2]])
            elif max_idx == 1:
                axis = jnp.array([rot[1, 0], 1 + rot[1, 1], rot[1, 2]])
            else:
                axis = jnp.array([rot[2, 0], rot[2, 1], 1 + rot[2, 2]])

            axis = axis / jnp.linalg.norm(axis)

            # Construct the skew-symmetric matrix
            return jnp.pi * self._skew_from_vector(axis)

        # Choose the appropriate case based on the angle
        small_angle = angle < 1e-8
        pi_angle = jnp.abs(angle - jnp.pi) < 1e-8

        result = lax.cond(small_angle, small_angle_case, lambda: lax.cond(pi_angle, pi_angle_case, normal_case))

        return result

    def _logm(self, rot):
        """Compute the matrix logarithm for a rotation matrix.

        Args:
            rot: Rotation matrix in SO(n).

        Returns:
            The skew-symmetric matrix log(rot).
        """
        # For now, use JAX's implementation via scipy
        # In the future, implement a more efficient version specific to rotation matrices
        from jax.scipy.linalg import logm

        return logm(rot)

    def _geodesic_distance_so3(self, rel_rot):
        """Compute the geodesic distance for SO(3) using the rotation angle.

        Args:
            rel_rot: Relative rotation matrix between two points.

        Returns:
            The geodesic distance.
        """
        # The geodesic distance is the rotation angle
        trace = jnp.trace(rel_rot)
        trace_clamped = jnp.clip(trace, -1.0, 3.0)
        cos_angle = (trace_clamped - 1.0) / 2.0
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        return angle

    def _skew_from_vector(self, v):
        """Convert a 3D vector to a skew-symmetric matrix.

        Args:
            v: 3D vector.

        Returns:
            3x3 skew-symmetric matrix.
        """
        return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
