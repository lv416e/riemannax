"""Implementation of the Symmetric Positive Definite (SPD) manifold.

This module provides operations for optimization on the manifold of symmetric
positive definite matrices, which is fundamental in covariance estimation,
signal processing, and many machine learning applications.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import expm, solve

from .base import Manifold


def _matrix_log(x):
    """Compute matrix logarithm using eigendecomposition for SPD matrices.

    For SPD matrices, we can use eigendecomposition: X = Q @ diag(λ) @ Q^T
    Then log(X) = Q @ diag(log(λ)) @ Q^T
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(x)
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, 1e-12)
    log_eigenvals = jnp.log(eigenvals)
    return eigenvecs @ jnp.diag(log_eigenvals) @ eigenvecs.T


def _matrix_sqrt(x):
    """Compute matrix square root using eigendecomposition for SPD matrices.

    For SPD matrices, we can use eigendecomposition: X = Q @ diag(λ) @ Q^T
    Then sqrt(X) = Q @ diag(sqrt(λ)) @ Q^T
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(x)
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, 1e-12)
    sqrt_eigenvals = jnp.sqrt(eigenvals)
    return eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.T


class SymmetricPositiveDefinite(Manifold):
    """Symmetric Positive Definite manifold SPD(n) with affine-invariant metric.

    The manifold of nxn symmetric positive definite matrices:
    SPD(n) = {X ∈ R^(nxn) : X = X^T, X ≻ 0}

    This implementation uses the affine-invariant Riemannian metric, which makes
    the manifold complete and provides nice theoretical properties.
    """

    def __init__(self, n):
        """Initialize the SPD manifold.

        Args:
            n: Size of the matrices (nxn).
        """
        super().__init__()  # JIT関連の初期化
        self.n = n

    def proj(self, x, v):
        """Project matrix v onto the tangent space of SPD at point x.

        The tangent space at x consists of symmetric matrices.
        For the affine-invariant metric, we use the simple symmetric part:
        proj_x(v) = sym(v) = (v + v^T) / 2

        Args:
            x: Point on SPD manifold (nxn symmetric positive definite matrix).
            v: Matrix in the ambient space R^(nxn).

        Returns:
            The projection of v onto the tangent space at x.
        """
        # For the affine-invariant metric, the tangent space is just symmetric matrices
        return 0.5 * (v + v.T)

    def exp(self, x, v):
        """Apply the exponential map to move from point x along tangent vector v.

        For the affine-invariant metric on SPD:
        exp_x(v) = x @ expm(x^(-1/2) @ v @ x^(-1/2)) @ x^(1/2)

        Args:
            x: Point on SPD manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by following the geodesic from x in direction v.
        """
        # Compute x^(-1/2) using eigendecomposition
        x_sqrt = _matrix_sqrt(x)
        x_inv_sqrt = solve(x_sqrt, jnp.eye(self.n), assume_a="pos")

        # Transform tangent vector to matrix exponential form
        v_transformed = x_inv_sqrt @ v @ x_inv_sqrt

        # Apply matrix exponential
        exp_v = expm(v_transformed)

        # Transform back to SPD manifold
        return x_sqrt @ exp_v @ x_sqrt

    def log(self, x, y):
        """Apply the logarithmic map to find the tangent vector from x to y.

        For the affine-invariant metric on SPD:
        log_x(y) = x^(1/2) @ logm(x^(-1/2) @ y @ x^(-1/2)) @ x^(1/2)

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.

        Returns:
            The tangent vector v at x such that exp_x(v) = y.
        """
        # Compute x^(-1/2) and x^(1/2)
        x_sqrt = _matrix_sqrt(x)
        x_inv_sqrt = solve(x_sqrt, jnp.eye(self.n), assume_a="pos")

        # Transform to matrix logarithm form
        y_transformed = x_inv_sqrt @ y @ x_inv_sqrt

        # Apply matrix logarithm
        log_y = _matrix_log(y_transformed)

        # Transform back to tangent space
        return x_sqrt @ log_y @ x_sqrt

    def inner(self, x, u, v):
        """Compute the Riemannian inner product between tangent vectors u and v.

        For the affine-invariant metric:
        <u, v>_x = tr(x^(-1) @ u @ x^(-1) @ v)

        Args:
            x: Point on SPD manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x.
        """
        x_inv = solve(x, jnp.eye(self.n), assume_a="pos")
        return jnp.trace(x_inv @ u @ x_inv @ v)

    def transp(self, x, y, v):
        """Parallel transport vector v from tangent space at x to tangent space at y.

        For the affine-invariant metric, we use a simplified approach:
        P_x→y(v) = (y/x)^(1/2) @ v @ (y/x)^(1/2)
        This is an approximation that preserves the tangent space structure.

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        # Simplified parallel transport using square root scaling
        x_inv = solve(x, jnp.eye(self.n), assume_a="pos")
        scaling = _matrix_sqrt(y @ x_inv)
        return scaling @ v @ scaling.T

    def dist(self, x, y):
        """Compute the Riemannian distance between points x and y.

        For the affine-invariant metric:
        d(x, y) = ||log_x(y)||_x = sqrt(tr(logm(x^(-1/2) @ y @ x^(-1/2))^2))

        Args:
            x: First point on SPD manifold.
            y: Second point on SPD manifold.

        Returns:
            The geodesic distance between x and y.
        """
        # Use the logarithmic map
        log_xy = self.log(x, y)

        # Compute the norm in the Riemannian metric
        return jnp.sqrt(self.inner(x, log_xy, log_xy))

    def random_point(self, key, *shape):
        """Generate random point(s) on the SPD manifold.

        Generates SPD matrices by creating random matrices and computing A @ A^T + ε*I
        to ensure positive definiteness.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random SPD matrix/matrices with specified shape.
        """
        if len(shape) == 0:
            # Single matrix
            A = jr.normal(key, (self.n, self.n))
            return A @ A.T + 1e-6 * jnp.eye(self.n)
        else:
            # Batch of matrices
            full_shape = (*shape, self.n, self.n)
            A = jr.normal(key, full_shape)
            # Use einsum for batched matrix multiplication
            AAt = jnp.einsum("...ij,...kj->...ik", A, A)
            return AAt + 1e-6 * jnp.eye(self.n)

    def random_tangent(self, key, x, *shape):
        """Generate random tangent vector(s) at point x.

        Generates random symmetric matrices in the tangent space.

        Args:
            key: JAX PRNG key.
            x: Point on SPD manifold.
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        if len(shape) == 0:
            # Single tangent vector
            v_raw = jr.normal(key, (self.n, self.n))
            return self.proj(x, v_raw)
        else:
            # Batch of tangent vectors
            full_shape = (*shape, self.n, self.n)
            v_raw = jr.normal(key, full_shape)

            # Apply projection to each matrix in the batch
            def proj_single(v):
                return self.proj(x, v)

            return jax.vmap(proj_single)(v_raw)

    def _is_in_manifold(self, x, tolerance=1e-6):
        """Check if a matrix is in the SPD manifold.

        Args:
            x: Matrix to check.
            tolerance: Numerical tolerance for checks.

        Returns:
            Boolean indicating if x is symmetric positive definite.
        """
        # Check symmetry
        is_symmetric = jnp.allclose(x, x.T, atol=tolerance)

        # Check positive definiteness via eigenvalues
        eigenvals = jnp.linalg.eigvals(x)
        is_positive_definite = jnp.all(eigenvals > tolerance)

        return is_symmetric and is_positive_definite

    # JIT-optimized implementation methods

    def _proj_impl(self, x, v):
        """JIT最適化版プロジェクション実装.

        数値安定性向上:
        - 対称性の効率的保証
        """
        # For SPD manifolds, tangent space consists of symmetric matrices
        return 0.5 * (v + v.T)

    def _exp_impl(self, x, v):
        """JIT最適化版指数写像実装.

        数値安定性向上:
        - 固有値分解による安定な実装
        - 条件数制御による数値安定性確保
        """
        # Use eigendecomposition for better JAX compatibility
        return self._exp_impl_eigen(x, v)

    def _exp_impl_eigen(self, x, v):
        """Eigendecomposition-based exponential map fallback with batch support."""
        # Compute matrix square root via eigendecomposition (batch-aware)
        eigenvals_x, eigenvecs_x = jnp.linalg.eigh(x)
        # Ensure positive eigenvalues for numerical stability
        eigenvals_x = jnp.maximum(eigenvals_x, 1e-12)
        sqrt_eigenvals_x = jnp.sqrt(eigenvals_x)

        # x^(1/2) and x^(-1/2) (batch-aware diagonal operations)
        sqrt_diag = (
            jnp.apply_along_axis(jnp.diag, -1, sqrt_eigenvals_x)
            if len(sqrt_eigenvals_x.shape) > 1
            else jnp.diag(sqrt_eigenvals_x)
        )
        x_sqrt = eigenvecs_x @ sqrt_diag @ jnp.swapaxes(eigenvecs_x, -2, -1)

        inv_sqrt_eigenvals_x = 1.0 / sqrt_eigenvals_x
        inv_sqrt_diag = (
            jnp.apply_along_axis(jnp.diag, -1, inv_sqrt_eigenvals_x)
            if len(inv_sqrt_eigenvals_x.shape) > 1
            else jnp.diag(inv_sqrt_eigenvals_x)
        )
        x_inv_sqrt = eigenvecs_x @ inv_sqrt_diag @ jnp.swapaxes(eigenvecs_x, -2, -1)

        # Transform tangent vector (batch-aware)
        v_transformed = x_inv_sqrt @ v @ x_inv_sqrt

        # Apply matrix exponential
        exp_v = expm(v_transformed)

        # Transform back (batch-aware)
        return x_sqrt @ exp_v @ x_sqrt

    def _log_impl(self, x, y):
        """JIT最適化版対数写像実装.

        数値安定性向上:
        - 固有値分解による安定な実装
        - 条件数制御による安定性確保
        """
        # Use eigendecomposition for better JAX compatibility
        return self._log_impl_eigen(x, y)

    def _log_impl_eigen(self, x, y):
        """Eigendecomposition-based logarithmic map fallback."""
        # Compute matrix square root via eigendecomposition
        eigenvals_x, eigenvecs_x = jnp.linalg.eigh(x)
        eigenvals_x = jnp.maximum(eigenvals_x, 1e-12)
        sqrt_eigenvals_x = jnp.sqrt(eigenvals_x)

        # x^(1/2) and x^(-1/2)
        x_sqrt = eigenvecs_x @ jnp.diag(sqrt_eigenvals_x) @ eigenvecs_x.T
        inv_sqrt_eigenvals_x = 1.0 / sqrt_eigenvals_x
        x_inv_sqrt = eigenvecs_x @ jnp.diag(inv_sqrt_eigenvals_x) @ eigenvecs_x.T

        # Transform y
        y_transformed = x_inv_sqrt @ y @ x_inv_sqrt

        # Apply matrix logarithm
        eigenvals_y, eigenvecs_y = jnp.linalg.eigh(y_transformed)
        eigenvals_y = jnp.maximum(eigenvals_y, 1e-12)
        log_eigenvals_y = jnp.log(eigenvals_y)
        log_y = eigenvecs_y @ jnp.diag(log_eigenvals_y) @ eigenvecs_y.T

        # Transform back
        return x_sqrt @ log_y @ x_sqrt

    def _inner_impl(self, x, u, v):
        """JIT最適化版内積実装.

        SPD多様体上のaffine-invariant内積
        バッチ処理対応
        """
        # Compute x^(-1) using direct solve for JAX compatibility (batch-aware)
        # Create identity matrix with correct batch shape
        batch_shape = x.shape[:-2] if x.ndim > 2 else ()
        eye_shape = (*batch_shape, self.n, self.n)
        identity = jnp.broadcast_to(jnp.eye(self.n), eye_shape)

        x_inv = solve(x, identity, assume_a="pos")

        # Compute <u, v>_x = tr(x^(-1) @ u @ x^(-1) @ v) (batch-aware)
        temp = x_inv @ u @ x_inv @ v
        # Use jnp.trace with axis specification for batch processing
        inner_product = jnp.trace(temp, axis1=-2, axis2=-1)
        return jnp.clip(inner_product, -1e15, 1e15)  # Numerical stability  # Numerical stability

    def _dist_impl(self, x, y):
        """JIT最適化版距離計算実装.

        数値安定性向上:
        - Cholesky分解による効率的計算
        """
        # Use the logarithmic map and inner product
        log_xy = self._log_impl(x, y)

        # Compute Riemannian distance: sqrt(<log_x(y), log_x(y)>_x)
        inner_product = self._inner_impl(x, log_xy, log_xy)

        # Ensure non-negative for numerical stability
        inner_product = jnp.maximum(inner_product, 0.0)

        distance = jnp.sqrt(inner_product)
        return jnp.where(distance < 1e-12, 0.0, distance)

    def _get_static_args(self, method_name: str) -> tuple:
        """JITコンパイル用の静的引数設定.

        SPD多様体では行列サイズ n を静的引数として指定
        """
        return (self.n,)
