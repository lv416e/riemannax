"""SE(3) Special Euclidean Group manifold implementation.

This module implements the SE(3) manifold, which represents rigid transformations
in 3D space combining rotations and translations. Points are parameterized using
unit quaternions (qw, qx, qy, qz) for rotation and (x, y, z) for translation.

SE(3) = SO(3) ⋉ R³ represents the semidirect product of 3D rotations and translations.
The quaternion follows the wxyz convention where w is the scalar (real) part.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from riemannax.manifolds.base import Manifold, ManifoldPoint

# Numerical constants for Taylor expansions and stability thresholds
_SMALL_ANGLE_THRESHOLD = 1e-8
_TAYLOR_EPS = 1e-12  # Numerical stability threshold
_SINC_TAYLOR_C2 = 1.0 / 6.0      # -theta²/6
_SINC_TAYLOR_C4 = 1.0 / 120.0    # theta⁴/120
_ONE_MINUS_COS_C0 = 0.5          # 1/2
_ONE_MINUS_COS_C2 = 1.0 / 24.0   # -theta²/24
_ONE_MINUS_COS_C4 = 1.0 / 720.0  # theta⁴/720
_V_INVERSE_C0 = 1.0 / 12.0       # 1/12 for small angle V^(-1)
_V_INVERSE_C2 = 1.0 / 720.0      # -theta²/720 for small angle V^(-1)


class SE3(Manifold):
    """SE(3) Special Euclidean Group manifold.

    Represents rigid transformations in 3D space as the semidirect product
    SE(3) = SO(3) ⋉ R³. Points are parameterized using unit quaternions for
    rotation and 3D vectors for translation: (qw, qx, qy, qz, x, y, z).

    The quaternion follows the convention (w, x, y, z) where w is the real part.
    """

    def __init__(self, atol: float = 1e-8) -> None:
        """Initialize SE(3) manifold with quaternion + translation parameterization.

        Args:
            atol: Absolute tolerance for numerical validation and operations.
        """
        super().__init__()
        self.atol = atol

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of SE(3) manifold."""
        return 6  # 3 for rotation (SO(3)) + 3 for translation

    @property
    def ambient_dimension(self) -> int:
        """Dimension of ambient representation space."""
        return 7  # 4 for quaternion + 3 for translation

    def _quaternion_normalize(self, q: Array) -> Array:
        """Normalize quaternion with numerical stability.

        Ensures quaternion has unit norm while handling edge cases robustly.
        For zero quaternions, defaults to identity quaternion (1, 0, 0, 0).

        Args:
            q: Quaternion array of shape (..., 4) in (w, x, y, z) format.

        Returns:
            Normalized quaternion with unit norm.
        """
        norm = jnp.linalg.norm(q, axis=-1, keepdims=True)

        # Prevent division by zero with safe normalization threshold
        safe_norm = jnp.maximum(norm, 1e-12)
        normalized = q / safe_norm

        # Replace invalid results with identity quaternion
        is_degenerate = norm < 1e-12
        identity_q = jnp.array([1.0, 0.0, 0.0, 0.0])

        # Handle batch case efficiently
        if q.ndim > 1:
            identity_q = jnp.broadcast_to(identity_q, q.shape)

        result = jnp.where(is_degenerate, identity_q, normalized)
        return result

    def random_point(self, key: PRNGKeyArray, *shape: int) -> ManifoldPoint:
        """Generate random SE(3) transform(s).

        Generates uniformly distributed random rotations using quaternion representation
        and normally distributed translations.

        Args:
            key: JAX PRNG key for random generation.
            *shape: Additional shape dimensions for batch generation.

        Returns:
            Random SE(3) transform(s) with shape (*shape, 7) where each transform
            is parameterized as (qw, qx, qy, qz, x, y, z).
        """
        # Split key for quaternion and translation generation
        key_q, key_t = jax.random.split(key)

        if shape:
            # Batch generation
            batch_shape = shape

            # Generate random quaternions from normal distribution and normalize
            quaternions = jax.random.normal(key_q, (*batch_shape, 4))
            quaternions = self._quaternion_normalize(quaternions)

            # Generate random translations
            translations = jax.random.normal(key_t, (*batch_shape, 3))

            # Combine quaternion and translation efficiently
            points = jnp.concatenate([quaternions, translations], axis=-1)
        else:
            # Single point generation - more efficient path
            quaternion = jax.random.normal(key_q, (4,))
            quaternion = self._quaternion_normalize(quaternion)

            translation = jax.random.normal(key_t, (3,))

            points = jnp.concatenate([quaternion, translation])

        return points

    def validate_point(self, x: ManifoldPoint, atol: float | None = None) -> bool | Array:
        """Validate that x is a valid SE(3) transform.

        Checks that the quaternion part has unit norm, which is the primary
        constraint for SE(3) representations. Translation part is unconstrained.

        Args:
            x: Point to validate with shape (..., 7).
            atol: Absolute tolerance for validation. Uses self.atol if None.

        Returns:
            True if x is valid SE(3) transform, False otherwise.
            For batched input, returns boolean array.
        """
        if atol is None:
            atol = self.atol

        # Check shape constraint
        if x.shape[-1] != 7:
            return False

        # Extract quaternion part (first 4 components)
        quaternion = x[..., :4]

        # Validate quaternion normalization
        q_norm = jnp.linalg.norm(quaternion, axis=-1)
        is_normalized = jnp.abs(q_norm - 1.0) <= atol

        # Return result compatible with JAX transformations
        try:
            # Handle both scalar and batch cases
            return bool(jnp.all(is_normalized)) if quaternion.ndim > 1 else bool(is_normalized)
        except TypeError:
            # In JAX traced context, return array directly
            return jnp.all(is_normalized) if quaternion.ndim > 1 else is_normalized

    def _skew_symmetric(self, v: Array) -> Array:
        """Create skew-symmetric matrix from 3D vector.

        Converts 3D vector v = [v1, v2, v3] to skew-symmetric matrix:
        [[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]]

        Args:
            v: 3D vector of shape (..., 3).

        Returns:
            Skew-symmetric matrix of shape (..., 3, 3).
        """
        zeros = jnp.zeros_like(v[..., 0:1])
        return jnp.stack(
            [
                jnp.concatenate([zeros, -v[..., 2:3], v[..., 1:2]], axis=-1),
                jnp.concatenate([v[..., 2:3], zeros, -v[..., 0:1]], axis=-1),
                jnp.concatenate([-v[..., 1:2], v[..., 0:1], zeros], axis=-1),
            ],
            axis=-2,
        )

    def _quaternion_to_rotation_matrix(self, q: Array) -> Array:
        """Convert unit quaternion to rotation matrix.

        Args:
            q: Unit quaternion [qw, qx, qy, qz] of shape (..., 4).

        Returns:
            3x3 rotation matrix of shape (..., 3, 3).
        """
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        # Precompute frequently used terms
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz

        # Construct rotation matrix
        R11 = 1 - 2 * (yy + zz)
        R12 = 2 * (xy - wz)
        R13 = 2 * (xz + wy)
        R21 = 2 * (xy + wz)
        R22 = 1 - 2 * (xx + zz)
        R23 = 2 * (yz - wx)
        R31 = 2 * (xz - wy)
        R32 = 2 * (yz + wx)
        R33 = 1 - 2 * (xx + yy)

        return jnp.stack(
            [
                jnp.stack([R11, R12, R13], axis=-1),
                jnp.stack([R21, R22, R23], axis=-1),
                jnp.stack([R31, R32, R33], axis=-1),
            ],
            axis=-2,
        )

    def _rotation_matrix_to_quaternion(self, R: Array) -> Array:
        """Convert rotation matrix to unit quaternion.

        Uses Shepperd's method for numerical stability.

        Args:
            R: 3x3 rotation matrix of shape (..., 3, 3).

        Returns:
            Unit quaternion [qw, qx, qy, qz] of shape (..., 4).
        """
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

        # Case 1: trace > 0
        s1 = jnp.sqrt(trace + 1.0) * 2  # s1 = 4 * qw
        qw1 = 0.25 * s1
        qx1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
        qy1 = (R[..., 0, 2] - R[..., 2, 0]) / s1
        qz1 = (R[..., 1, 0] - R[..., 0, 1]) / s1
        q1 = jnp.stack([qw1, qx1, qy1, qz1], axis=-1)

        # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
        s2 = jnp.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) * 2
        qw2 = (R[..., 2, 1] - R[..., 1, 2]) / s2
        qx2 = 0.25 * s2
        qy2 = (R[..., 0, 1] + R[..., 1, 0]) / s2
        qz2 = (R[..., 0, 2] + R[..., 2, 0]) / s2
        q2 = jnp.stack([qw2, qx2, qy2, qz2], axis=-1)

        # Case 3: R[1,1] > R[2,2]
        s3 = jnp.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]) * 2
        qw3 = (R[..., 0, 2] - R[..., 2, 0]) / s3
        qx3 = (R[..., 0, 1] + R[..., 1, 0]) / s3
        qy3 = 0.25 * s3
        qz3 = (R[..., 1, 2] + R[..., 2, 1]) / s3
        q3 = jnp.stack([qw3, qx3, qy3, qz3], axis=-1)

        # Case 4: else
        s4 = jnp.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]) * 2
        qw4 = (R[..., 1, 0] - R[..., 0, 1]) / s4
        qx4 = (R[..., 0, 2] + R[..., 2, 0]) / s4
        qy4 = (R[..., 1, 2] + R[..., 2, 1]) / s4
        qz4 = 0.25 * s4
        q4 = jnp.stack([qw4, qx4, qy4, qz4], axis=-1)

        # Select appropriate case based on conditions
        cond1 = trace > 0
        cond2 = jnp.logical_and(~cond1, jnp.logical_and(R[..., 0, 0] > R[..., 1, 1], R[..., 0, 0] > R[..., 2, 2]))
        cond3 = jnp.logical_and(~cond1, jnp.logical_and(~cond2, R[..., 1, 1] > R[..., 2, 2]))

        q = jnp.where(cond1[..., None], q1, jnp.where(cond2[..., None], q2, jnp.where(cond3[..., None], q3, q4)))

        return self._quaternion_normalize(q)

    def _compute_rodrigues_coefficients(self, theta: Array) -> tuple[Array, Array]:
        """Compute Rodrigues formula coefficients with numerical stability.

        Computes sin(θ)/θ and (1-cos(θ))/θ² with Taylor expansions for small angles.

        Args:
            theta: Angle magnitudes of shape (..., 1).

        Returns:
            Tuple of (sin_over_theta, one_minus_cos_over_theta_sq) coefficients.
        """
        small_angle = theta < _SMALL_ANGLE_THRESHOLD
        theta_sq = theta * theta
        theta_4 = theta_sq * theta_sq

        # sin(θ)/θ ≈ 1 - θ²/6 + θ⁴/120
        sin_over_theta = jnp.where(
            small_angle,
            1.0 - theta_sq * _SINC_TAYLOR_C2 + theta_4 * _SINC_TAYLOR_C4,
            jnp.sin(theta) / jnp.maximum(theta, _TAYLOR_EPS)
        )

        # (1-cos(θ))/θ² ≈ 1/2 - θ²/24 + θ⁴/720
        one_minus_cos_over_theta_sq = jnp.where(
            small_angle,
            _ONE_MINUS_COS_C0 - theta_sq * _ONE_MINUS_COS_C2 + theta_4 * _ONE_MINUS_COS_C4,
            (1.0 - jnp.cos(theta)) / jnp.maximum(theta_sq, _TAYLOR_EPS)
        )

        return sin_over_theta, one_minus_cos_over_theta_sq

    def _matrix_exp_so3(self, omega: Array) -> Array:
        """Compute SO(3) matrix exponential using Rodrigues formula.

        Computes exp(hat(omega)) where hat(omega) is the skew-symmetric matrix
        corresponding to rotation vector omega.

        Args:
            omega: Rotation vector of shape (..., 3).

        Returns:
            Rotation matrix of shape (..., 3, 3).
        """
        theta = jnp.linalg.norm(omega, axis=-1, keepdims=True)
        sin_over_theta, one_minus_cos_over_theta_sq = self._compute_rodrigues_coefficients(theta)

        # Skew-symmetric matrix
        K = self._skew_symmetric(omega)

        # Rodrigues formula: R = I + sin(θ)/θ * K + (1-cos(θ))/θ² * K²
        I = jnp.eye(3)
        if omega.ndim > 1:
            I = jnp.broadcast_to(I, (*omega.shape[:-1], 3, 3))

        return I + sin_over_theta[..., None] * K + one_minus_cos_over_theta_sq[..., None] * (K @ K)

    def _matrix_log_so3(self, R: Array) -> Array:
        """Compute SO(3) matrix logarithm.

        Extracts rotation vector omega such that R = exp(hat(omega)).

        Args:
            R: Rotation matrix of shape (..., 3, 3).

        Returns:
            Rotation vector of shape (..., 3).
        """
        # Extract rotation angle from trace
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)  # Handle numerical errors
        theta = jnp.arccos(cos_theta)

        # Handle different cases
        # Case 1: theta ≈ 0 (near identity)
        small_angle = jnp.abs(theta) < 1e-8

        # Case 2: theta ≈ π (near singularity)
        near_pi = jnp.abs(theta - jnp.pi) < 1e-8

        # Case 3: regular case
        sin_theta = jnp.sin(theta)

        # For small angles, use first-order approximation
        # omega ≈ 1/2 * [R₃₂-R₂₃, R₁₃-R₃₁, R₂₁-R₁₂]
        antisym_part = jnp.stack(
            [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], axis=-1
        )

        omega_small = 0.5 * antisym_part

        # For regular case: omega = theta/(2*sin(theta)) * antisymmetric_part
        omega_regular = theta[..., None] / (2.0 * jnp.maximum(jnp.abs(sin_theta[..., None]), 1e-12)) * antisym_part

        # For near π case, need special handling
        # Find the eigenvector corresponding to eigenvalue 1
        diag_R = jnp.diagonal(R, axis1=-2, axis2=-1)
        max_diag_idx = jnp.argmax(diag_R, axis=-1)

        # Extract the appropriate column as the axis
        axis_candidates = jnp.transpose(R, axes=(*range(R.ndim - 2), -1, -2))  # Transpose to get columns

        # Select the column with largest diagonal element
        if R.ndim == 2:  # Single matrix case
            axis = axis_candidates[max_diag_idx]
        else:  # Batch case
            batch_indices = jnp.arange(max_diag_idx.size).reshape(max_diag_idx.shape)
            axis = axis_candidates[batch_indices, max_diag_idx]

        # Normalize and scale by pi
        axis = axis / jnp.maximum(jnp.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
        omega_pi = jnp.pi * axis

        # Select appropriate result based on conditions
        omega = jnp.where(small_angle[..., None], omega_small, jnp.where(near_pi[..., None], omega_pi, omega_regular))

        return omega

    def _compute_v_matrix(self, omega: Array) -> Array:
        """Compute V matrix for SE(3) exponential map.

        V = I + (1-cos(θ))/θ² * K + (θ-sin(θ))/θ³ * K²
        where K is the skew-symmetric matrix of omega.

        Args:
            omega: Rotation vector of shape (..., 3).

        Returns:
            V matrix of shape (..., 3, 3).
        """
        theta = jnp.linalg.norm(omega, axis=-1, keepdims=True)
        small_angle = theta < _SMALL_ANGLE_THRESHOLD

        theta_sq = theta * theta
        theta_4 = theta_sq * theta_sq

        # Reuse coefficients from Rodrigues formula
        _, one_minus_cos_over_theta_sq = self._compute_rodrigues_coefficients(theta)

        # (θ-sin(θ))/θ³ ≈ 1/6 - θ²/120 + θ⁴/5040
        theta_minus_sin_over_theta_cubed = jnp.where(
            small_angle,
            1.0/6.0 - theta_sq/120.0 + theta_4/5040.0,
            (theta - jnp.sin(theta)) / jnp.maximum(theta * theta_sq, _TAYLOR_EPS)
        )

        # Build V matrix
        I = jnp.eye(3)
        if omega.ndim > 1:
            I = jnp.broadcast_to(I, (*omega.shape[:-1], 3, 3))

        K = self._skew_symmetric(omega)
        return I + one_minus_cos_over_theta_sq[..., None] * K + theta_minus_sin_over_theta_cubed[..., None] * (K @ K)

    def _compute_v_inverse_matrix(self, omega: Array) -> Array:
        """Compute V^(-1) matrix for SE(3) logarithm map.

        Args:
            omega: Rotation vector of shape (..., 3).

        Returns:
            V^(-1) matrix of shape (..., 3, 3).
        """
        theta = jnp.linalg.norm(omega, axis=-1, keepdims=True)
        small_angle = theta < _SMALL_ANGLE_THRESHOLD

        theta_sq = theta * theta
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        # Taylor expansion for small angles: B ≈ 1/12 - θ²/720
        B_small = _V_INVERSE_C0 - theta_sq * _V_INVERSE_C2

        # Regular case: B = (2*sin(θ)-θ*(1+cos(θ)))/(2*θ²*sin(θ))
        numerator = 2.0 * sin_theta - theta * (1.0 + cos_theta)
        denominator = 2.0 * theta_sq * jnp.maximum(jnp.abs(sin_theta), _TAYLOR_EPS)
        B_regular = numerator / denominator

        B = jnp.where(small_angle, B_small, B_regular)

        # Build V^(-1) matrix
        I = jnp.eye(3)
        if omega.ndim > 1:
            I = jnp.broadcast_to(I, (*omega.shape[:-1], 3, 3))

        K = self._skew_symmetric(omega)
        return I - 0.5 * K + B[..., None] * (K @ K)

    def exp_tangent(self, xi: Array) -> ManifoldPoint:
        """SE(3) exponential map from se(3) to SE(3).

        Maps 6D tangent vector xi = [omega, rho] to SE(3) transform.

        Args:
            xi: Tangent vector of shape (..., 6) where first 3 components
                are rotation (omega) and last 3 are translation (rho).

        Returns:
            SE(3) transform of shape (..., 7) as [qw, qx, qy, qz, x, y, z].
        """
        # Split tangent vector into rotation and translation parts
        omega = xi[..., :3]  # rotation part
        rho = xi[..., 3:]  # translation part

        # Compute rotation matrix from rotation vector
        R = self._matrix_exp_so3(omega)

        # Compute V matrix and apply to translation
        V = self._compute_v_matrix(omega)
        t = jnp.einsum("...ij,...j->...i", V, rho)  # V @ rho

        # Convert rotation matrix to quaternion
        q = self._rotation_matrix_to_quaternion(R)

        # Combine quaternion and translation
        return jnp.concatenate([q, t], axis=-1)

    def log_tangent(self, x: ManifoldPoint) -> Array:
        """SE(3) logarithm map from SE(3) to se(3).

        Maps SE(3) transform to 6D tangent vector xi = [omega, rho].

        Args:
            x: SE(3) transform of shape (..., 7) as [qw, qx, qy, qz, x, y, z].

        Returns:
            Tangent vector of shape (..., 6) where first 3 components
            are rotation (omega) and last 3 are translation (rho).
        """
        # Extract quaternion and translation
        q = x[..., :4]  # quaternion part
        t = x[..., 4:]  # translation part

        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(q)

        # Get rotation vector from matrix logarithm
        omega = self._matrix_log_so3(R)

        # Compute V^(-1) and apply to translation to recover original rho
        V_inv = self._compute_v_inverse_matrix(omega)
        rho = jnp.einsum("...ij,...j->...i", V_inv, t)  # V_inv @ t

        # Combine rotation and translation parts
        return jnp.concatenate([omega, rho], axis=-1)

    def __repr__(self) -> str:
        """String representation of SE(3) manifold."""
        return f"SE3(atol={self.atol})"
