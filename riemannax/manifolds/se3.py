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
_SINC_TAYLOR_C2 = 1.0 / 6.0  # -theta²/6
_SINC_TAYLOR_C4 = 1.0 / 120.0  # theta⁴/120
_ONE_MINUS_COS_C0 = 0.5  # 1/2
_ONE_MINUS_COS_C2 = 1.0 / 24.0  # -theta²/24
_ONE_MINUS_COS_C4 = 1.0 / 720.0  # theta⁴/720
_V_INVERSE_C0 = 1.0 / 12.0  # 1/12 for small angle V^(-1)
_V_INVERSE_C2 = 1.0 / 720.0  # -theta²/720 for small angle V^(-1)


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

    def validate_tangent(self, x: Array, v: Array, atol: float = 1e-6) -> bool | Array:
        """Validate tangent vector in se(3) Lie algebra.

        Tangent vectors should be 6-dimensional: (omega, rho) where
        omega ∈ so(3) (3D rotation vector) and rho ∈ R³ (3D translation vector).

        Args:
            x: Base point on SE(3) manifold (not used for validation)
            v: Tangent vector to validate
            atol: Absolute tolerance for validation (unused for SE(3))

        Returns:
            True if vector is valid tangent vector in se(3), False otherwise
        """
        try:
            v = jnp.asarray(v)

            # Check dimensions
            if v.ndim == 1:
                result = v.shape[0] == 6
            elif v.ndim == 2:
                # Batch case
                result = v.shape[1] == 6
            else:
                result = False

            # Return JAX array directly if in traced context to avoid TracerBoolConversionError
            try:
                return bool(result)
            except TypeError:
                # In JAX traced context, return the array directly
                return result

        except (ValueError, TypeError):
            return False

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
            jnp.sin(theta) / jnp.maximum(theta, _TAYLOR_EPS),
        )

        # (1-cos(θ))/θ² ≈ 1/2 - θ²/24 + θ⁴/720
        one_minus_cos_over_theta_sq = jnp.where(
            small_angle,
            _ONE_MINUS_COS_C0 - theta_sq * _ONE_MINUS_COS_C2 + theta_4 * _ONE_MINUS_COS_C4,
            (1.0 - jnp.cos(theta)) / jnp.maximum(theta_sq, _TAYLOR_EPS),
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
            1.0 / 6.0 - theta_sq / 120.0 + theta_4 / 5040.0,
            (theta - jnp.sin(theta)) / jnp.maximum(theta * theta_sq, _TAYLOR_EPS),
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

    def compose(self, g1: Array, g2: Array) -> Array:
        """Compose two SE(3) group elements.

        Args:
            g1: First SE(3) element as (qw, qx, qy, qz, tx, ty, tz)
            g2: Second SE(3) element as (qw, qx, qy, qz, tx, ty, tz)

        Returns:
            Composed element g1 * g2 as (qw, qx, qy, qz, tx, ty, tz)
        """
        # Handle both single and batched inputs
        if g1.ndim == 1:
            # Single input case
            q1, t1 = g1[:4], g1[4:]
            q2, t2 = g2[:4], g2[4:]
        else:
            # Batched input case
            q1, t1 = g1[..., :4], g1[..., 4:]
            q2, t2 = g2[..., :4], g2[..., 4:]

        # Quaternion multiplication: q_result = q1 * q2
        qw1, qx1, qy1, qz1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        qw2, qx2, qy2, qz2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        qw_result = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx_result = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy_result = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz_result = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

        q_result = jnp.stack([qw_result, qx_result, qy_result, qz_result], axis=-1)
        q_result = self._quaternion_normalize(q_result)

        # Translation composition: t_result = R1 * t2 + t1
        R1 = self._quaternion_to_rotation_matrix(q1)
        t_result = R1 @ t2 + t1 if g1.ndim == 1 else jnp.einsum("...ij,...j->...i", R1, t2) + t1

        return jnp.concatenate([q_result, t_result], axis=-1)

    def inverse(self, g: Array) -> Array:
        """Compute inverse of SE(3) group element.

        Args:
            g: SE(3) element as (qw, qx, qy, qz, tx, ty, tz)

        Returns:
            Inverse element g^(-1) as (qw, qx, qy, qz, tx, ty, tz)
        """
        # Handle both single and batched inputs
        if g.ndim == 1:
            q, t = g[:4], g[4:]
        else:
            q, t = g[..., :4], g[..., 4:]

        # Quaternion inverse (conjugate for unit quaternion)
        if g.ndim == 1:
            q_inv = jnp.array([q[0], -q[1], -q[2], -q[3]])
        else:
            q_inv = jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)

        # Translation inverse: t_inv = -R^T * t = -R_inv * t
        R_inv = self._quaternion_to_rotation_matrix(q_inv)
        t_inv = -R_inv @ t if g.ndim == 1 else -jnp.einsum("...ij,...j->...i", R_inv, t)

        return jnp.concatenate([q_inv, t_inv], axis=-1)

    def proj(self, x: Array, v: Array) -> Array:
        """Project vector onto tangent space at point x.

        For SE(3), this projects a 7D Euclidean gradient (w.r.t. quaternion + translation)
        onto the 6D se(3) Lie algebra tangent space.

        Args:
            x: Point on SE(3) manifold as (qw, qx, qy, qz, tx, ty, tz)
            v: Vector to project - can be 7D (Euclidean) or 6D (already in tangent)

        Returns:
            Projected vector in 6D tangent space (omega, rho)
        """
        # If already 6D, assume it's already in tangent space
        if v.shape[-1] == 6:
            return v

        # If 7D, project from Euclidean gradient to tangent space
        if v.shape[-1] == 7:
            # Extract quaternion and translation parts of the gradient
            q_grad = v[..., :4]  # Gradient w.r.t. quaternion
            t_grad = v[..., 4:7]  # Gradient w.r.t. translation

            # Extract current quaternion from x
            q = x[..., :4]

            # Project quaternion gradient to so(3) tangent space
            # For quaternion q = (qw, qx, qy, qz), the tangent projection is:
            # omega = 2 * (qw * q_vec_grad - qx * qw_grad, qy * qw_grad - qw * qy_grad, qz * qw_grad - qw * qz_grad)
            # Simplified: omega = 2 * (q_w * q_vec_grad - q_vec * q_w_grad)
            qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            qw_grad, qx_grad, qy_grad, qz_grad = q_grad[..., 0], q_grad[..., 1], q_grad[..., 2], q_grad[..., 3]

            # Project to so(3) tangent space (3D rotation tangent)
            omega_x = 2 * (qw * qx_grad - qx * qw_grad)
            omega_y = 2 * (qw * qy_grad - qy * qw_grad)
            omega_z = 2 * (qw * qz_grad - qz * qw_grad)

            omega = jnp.stack([omega_x, omega_y, omega_z], axis=-1)

            # Translation part remains the same
            rho = t_grad

            # Combine into 6D tangent vector
            return jnp.concatenate([omega, rho], axis=-1)

        # For other dimensions, raise error
        raise ValueError(f"Cannot project vector of shape {v.shape} to SE(3) tangent space")

    def inner(self, x: Array, v1: Array, v2: Array) -> Array:
        """Compute inner product of tangent vectors at point x.

        Uses canonical inner product on se(3) Lie algebra.

        Args:
            x: Point on SE(3) manifold
            v1: First tangent vector
            v2: Second tangent vector

        Returns:
            Inner product scalar
        """
        # Canonical inner product on se(3): <v1, v2> = tr(v1^T * v2)
        # For 6D vectors (omega, rho), this is just dot product
        return jnp.dot(v1, v2)

    def retr(self, x: Array, v: Array) -> Array:
        """Retraction: exponential map from tangent space to manifold.

        Args:
            x: Base point on SE(3) manifold
            v: Tangent vector at x

        Returns:
            Point on manifold after retraction
        """
        # SE(3) retraction via group exponential
        exp_v = self.exp_tangent(v)
        return self.compose(x, exp_v)

    def dist(self, x: Array, y: Array) -> Array:
        """Compute Riemannian distance between two points.

        Args:
            x: First point on SE(3) manifold
            y: Second point on SE(3) manifold

        Returns:
            Geodesic distance
        """
        # Distance via logarithm map: ||log(x^(-1) * y)||
        x_inv = self.inverse(x)
        xy_inv = self.compose(x_inv, y)
        log_xy = self.log_tangent(xy_inv)

        # L2 norm in se(3) Lie algebra
        return jnp.linalg.norm(log_xy)

    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport tangent vector from x to y.

        For SE(3) as a Lie group, parallel transport is trivial
        (tangent vectors are transported without change).

        Args:
            x: Source point
            y: Target point
            v: Tangent vector at x

        Returns:
            Parallel transported vector at y
        """
        # For Lie groups, parallel transport is identity
        return v

    def exp(self, x: Array, v: Array) -> Array:
        """Exponential map: move from point x along tangent vector v.

        Args:
            x: Point on SE(3) manifold as (qw, qx, qy, qz, tx, ty, tz)
            v: Tangent vector in 6D se(3) algebra (omega, rho)

        Returns:
            New point on SE(3) reached by following geodesic from x in direction v
        """
        # Convert tangent vector to SE(3) transformation
        delta = self.exp_tangent(v)

        # Compose with current point: result = x * delta (right multiplication)
        return self.compose(x, delta)

    def log(self, x: Array, y: Array) -> Array:
        """Logarithmic map: compute tangent vector from x to y.

        Args:
            x: Base point on SE(3) manifold
            y: Target point on SE(3) manifold

        Returns:
            Tangent vector in 6D se(3) algebra from x to y
        """
        # Compute relative transformation: delta = x^(-1) * y
        x_inv = self.inverse(x)
        delta = self.compose(x_inv, y)

        # Convert to tangent vector
        return self.log_tangent(delta)

    def __repr__(self) -> str:
        """String representation of SE(3) manifold."""
        return f"SE3(atol={self.atol})"
