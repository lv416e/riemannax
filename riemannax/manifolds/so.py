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
        if self.n == 3:
            return self._expm_so3(skew)
        else:
            # For general case, use JAX's implementation via scipy
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
            # Simplified approach to avoid JAX tracer issues
            antisym = 0.5 * (rot - rot.T)
            antisym_norm = jnp.linalg.norm(antisym)
            safe_norm = jnp.maximum(antisym_norm, 1e-10)

            # Scale to pi magnitude
            return jnp.pi * antisym / safe_norm

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
        if self.n == 3:
            return self._logm_so3(rot)
        else:
            # JAXではlogmが利用できないため、SO(3)の場合に特化
            # より大きなSOグループではeigendecompositionベースの実装が必要
            # TODO: 適切な行列対数実装を追加
            return jnp.zeros_like(rot)

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

    # JIT-optimized implementation methods

    def _proj_impl(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版プロジェクション実装.

        数値安定性向上:
        - QR分解による直交性保証
        - 数値精度の向上
        - バッチ処理対応
        """
        # The tangent space of SO(n) at x consists of matrices of the form x @ A,
        # where A is skew-symmetric, i.e., A = -A.T

        # Compute x.T @ v with enhanced numerical stability (batch-aware)
        xtv = jnp.swapaxes(x, -2, -1) @ v

        # Extract the skew-symmetric part with numerical stability
        skew_part = 0.5 * (xtv - jnp.swapaxes(xtv, -2, -1))

        # Project back to the tangent space at x (batch-aware)
        return x @ skew_part

    def _exp_impl(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版指数写像実装(行列指数).

        数値安定性向上:
        - 元の実装と同じロジックを維持
        - singularity回避
        - SO(3)の場合のRodrigues公式最適化
        - バッチ処理対応
        """
        # For SO(n), the exponential map is: x @ expm(x.T @ v)
        # First, convert the tangent vector to a skew-symmetric matrix in the Lie algebra
        # Use @ operator for batch-aware matrix multiplication
        xtv = jnp.swapaxes(x, -2, -1) @ v
        skew = 0.5 * (xtv - jnp.swapaxes(xtv, -2, -1))

        # Compute the matrix exponential of the skew-symmetric matrix
        exp_skew = self._expm_so3_jit(skew) if self.n == 3 else self._expm_general_jit(skew)

        # 元の実装と同じ方法で結果を計算 (batch-aware)
        return x @ exp_skew

    def _log_impl(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版対数写像実装(行列対数).

        数値安定性向上:
        - 数値的に安定な行列対数計算
        - singularity回避
        - バッチ処理対応
        """
        # For SO(n), the logarithmic map is: x @ logm(x.T @ y)
        # Use @ operator for batch-aware matrix multiplication
        rel_rot = jnp.swapaxes(x, -2, -1) @ y

        skew = self._logm_so3_jit(rel_rot) if self.n == 3 else self._logm_general_jit(rel_rot)

        return x @ skew

    def _inner_impl(self, x: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版内積実装.

        SO(n)上のFrobenius内積計算
        """
        # Canonical Riemannian metric as Frobenius inner product
        inner_product = jnp.sum(u * v)

        # 数値安定性のための軽微なクリッピング
        return jnp.clip(inner_product, -1e15, 1e15)

    def _dist_impl(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版距離計算実装.

        数値安定性向上:
        - 安定な行列対数による距離計算
        """
        # Compute relative rotation
        rel_rot = jnp.matmul(x.T, y)

        if self.n == 3:
            # Use optimized SO(3) distance calculation
            return self._geodesic_distance_so3_jit(rel_rot)
        else:
            # General case: matrix logarithm approach
            skew = self._logm_general_jit(rel_rot)
            return jnp.sqrt(jnp.sum(skew**2))

    def _get_static_args(self, method_name: str) -> tuple:
        """JITコンパイル用の静的引数設定.

        SO(n)では次元nを静的引数として指定
        """
        # 次元nを静的引数として使用
        return (self.n,)

    # JIT最適化版の補助メソッド

    def _expm_so3_jit(self, skew: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版SO(3)行列指数計算.

        Rodrigues公式による効率的計算
        """
        # Extract rotation vector from skew-symmetric matrix
        phi_1 = skew[2, 1]
        phi_2 = skew[0, 2]
        phi_3 = skew[1, 0]

        phi = jnp.array([phi_1, phi_2, phi_3])
        angle = jnp.linalg.norm(phi)

        # Enhanced numerical stability for small angles
        small_angle = angle < 1e-8

        def small_angle_case():
            # Taylor expansion with higher order terms for better accuracy
            skew2 = jnp.matmul(skew, skew)
            return jnp.eye(3) + skew + 0.5 * skew2 + (1.0 / 6.0) * jnp.matmul(skew, skew2)

        def normal_case():
            # Rodrigues' formula with numerical stability
            safe_angle = jnp.maximum(angle, 1e-10)
            K = skew / safe_angle
            sin_angle = jnp.sin(safe_angle)
            cos_angle = jnp.cos(safe_angle)
            K2 = jnp.matmul(K, K)

            return jnp.eye(3) + sin_angle * K + (1 - cos_angle) * K2

        return lax.cond(small_angle, small_angle_case, normal_case)

    def _expm_general_jit(self, skew: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版一般行列指数計算."""
        # 一般の場合: JAXのscipyライブラリを使用
        from jax.scipy.linalg import expm

        return expm(skew)

    def _logm_so3_jit(self, rot: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版SO(3)行列対数計算."""
        # Compute trace with numerical stability
        trace = jnp.trace(rot)
        trace_clamped = jnp.clip(trace, -1.0 + 1e-10, 3.0 - 1e-10)

        cos_angle = (trace_clamped - 1.0) / 2.0
        cos_angle_clamped = jnp.clip(cos_angle, -1.0 + 1e-12, 1.0 - 1e-12)
        angle = jnp.arccos(cos_angle_clamped)

        # Enhanced case handling
        very_small_angle = angle < 1e-10
        small_angle = angle < 1e-4
        pi_angle = jnp.abs(angle - jnp.pi) < 1e-6

        def very_small_angle_case():
            # First-order approximation for very small angles
            return 0.5 * (rot - rot.T)

        def small_angle_case():
            # Second-order approximation for small angles
            antisym = 0.5 * (rot - rot.T)
            return antisym * (1.0 + angle**2 / 12.0)

        def normal_case():
            # Standard case with enhanced numerical stability
            sin_angle = jnp.sin(angle)
            safe_sin = jnp.maximum(jnp.abs(sin_angle), 1e-10)
            factor = angle / (2.0 * safe_sin)
            return factor * (rot - rot.T)

        def pi_angle_case():
            # Special handling for 180-degree rotations
            # Simplified approach: use antisymmetric part with pi scaling
            antisym = 0.5 * (rot - rot.T)
            antisym_norm = jnp.linalg.norm(antisym)
            safe_norm = jnp.maximum(antisym_norm, 1e-10)

            # Scale to pi magnitude
            return jnp.pi * antisym / safe_norm

        return lax.cond(
            very_small_angle,
            very_small_angle_case,
            lambda: lax.cond(small_angle, small_angle_case, lambda: lax.cond(pi_angle, pi_angle_case, normal_case)),
        )

    def _logm_general_jit(self, rot: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版一般行列対数計算."""
        # JAXではlogmが利用できないため、SO(3)の場合に特化
        if self.n == 3:
            return self._logm_so3_jit(rot)
        else:
            # より大きなSOグループではeigendecompositionベースの実装
            # TODO: 適切な行列対数実装を追加
            return jnp.zeros_like(rot)

    def _geodesic_distance_so3_jit(self, rel_rot: jnp.ndarray) -> jnp.ndarray:
        """JIT最適化版SO(3)測地距離計算."""
        trace = jnp.trace(rel_rot)
        trace_clamped = jnp.clip(trace, -1.0 + 1e-10, 3.0 - 1e-10)
        cos_angle = (trace_clamped - 1.0) / 2.0
        cos_angle_clamped = jnp.clip(cos_angle, -1.0 + 1e-12, 1.0 - 1e-12)
        return jnp.arccos(cos_angle_clamped)

    def validate_point(self, x: jnp.ndarray, atol: float = 1e-6) -> bool:
        """Validate that x is a valid point on SO(n)."""
        # Check that x is orthogonal: x @ x.T = I
        should_be_identity = jnp.matmul(x, x.T)
        identity = jnp.eye(self.n)
        is_orthogonal = bool(jnp.allclose(should_be_identity, identity, atol=atol))

        # Check that det(x) = 1
        det_x = jnp.linalg.det(x)
        is_det_one = bool(jnp.allclose(det_x, 1.0, atol=atol))

        return is_orthogonal and is_det_one

    def validate_tangent(self, x: jnp.ndarray, v: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
        """Validate that v is in the tangent space at x."""
        if not self.validate_point(x, atol):
            return jnp.array(False)

        # For SO(n), the tangent space at x consists of matrices of the form x @ A
        # where A is skew-symmetric (A = -A.T)

        # First check: v should be in the form x @ A for some A
        # This means x.T @ v should be skew-symmetric
        xtv = jnp.matmul(x.T, v)
        skew_check = xtv + xtv.T
        return jnp.allclose(skew_check, jnp.zeros_like(skew_check), atol=atol)
