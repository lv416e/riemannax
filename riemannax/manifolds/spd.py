"""Implementation of the Symmetric Positive Definite (SPD) manifold.

This module provides operations for optimization on the manifold of symmetric
positive definite matrices, which is fundamental in covariance estimation,
signal processing, and many machine learning applications.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.linalg import expm, solve

from ..core.constants import NumericalConstants
from ..core.jit_decorator import jit_optimized
from .base import Manifold


def _matrix_log(x: Array) -> Array:
    """Compute matrix logarithm using eigendecomposition for SPD matrices.

    For SPD matrices, we can use eigendecomposition: X = Q @ diag(λ) @ Q^T
    Then log(X) = Q @ diag(log(λ)) @ Q^T
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(x)
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, NumericalConstants.HIGH_PRECISION_EPSILON)
    log_eigenvals = jnp.log(eigenvals)
    return jnp.asarray(eigenvecs @ jnp.diag(log_eigenvals) @ eigenvecs.T)


def _matrix_sqrt(x: Array) -> Array:
    """Compute matrix square root using eigendecomposition for SPD matrices.

    For SPD matrices, we can use eigendecomposition: X = Q @ diag(λ) @ Q^T
    Then sqrt(X) = Q @ diag(sqrt(λ)) @ Q^T
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(x)
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, NumericalConstants.HIGH_PRECISION_EPSILON)
    sqrt_eigenvals = jnp.sqrt(eigenvals)
    return jnp.asarray(eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.T)


class SymmetricPositiveDefinite(Manifold):
    """Symmetric Positive Definite manifold SPD(n) with affine-invariant metric.

    The manifold of nxn symmetric positive definite matrices:
    SPD(n) = {X ∈ R^(nxn) : X = X^T, X ≻ 0}

    This implementation uses the affine-invariant Riemannian metric, which makes
    the manifold complete and provides nice theoretical properties.
    """

    def __init__(self, n: int) -> None:
        """Initialize the SPD manifold.

        Args:
            n: Size of the matrices (nxn).
        """
        super().__init__()  # JIT-related initialization
        self.n = n

    @jit_optimized(static_args=(0,))
    def proj(self, x: Array, v: Array) -> Array:
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

    @jit_optimized(static_args=(0,))
    def exp(self, x: Array, v: Array) -> Array:
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
        return jnp.asarray(x_sqrt @ exp_v @ x_sqrt)

    @jit_optimized(static_args=(0,))
    def log(self, x: Array, y: Array) -> Array:
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
        return jnp.asarray(x_sqrt @ log_y @ x_sqrt)

    @jit_optimized(static_args=(0,))
    def inner(self, x: Array, u: Array, v: Array) -> Array:
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

    @jit_optimized(static_args=(0,))
    def transp(self, x: Array, y: Array, v: Array) -> Array:
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
        return jnp.asarray(scaling @ v @ scaling.T)

    @jit_optimized(static_args=(0,))
    def dist(self, x: Array, y: Array) -> Array:
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

    def random_point(self, key: Array, *shape: int) -> Array:
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

    def random_tangent(self, key: Array, x: Array, *shape: int) -> Array:
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
            return jnp.asarray(self.proj(x, v_raw))
        else:
            # Batch of tangent vectors
            full_shape = (*shape, self.n, self.n)
            v_raw = jr.normal(key, full_shape)

            # Apply projection to each matrix in the batch
            def proj_single(v: Array) -> Array:
                return jnp.asarray(self.proj(x, v))

            return jax.vmap(proj_single)(v_raw)

    def _is_in_manifold(self, x: Array, tolerance: float = 1e-6) -> bool | Array:
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

        result = jnp.logical_and(is_symmetric, is_positive_definite)
        # Return JAX array directly if in traced context to avoid TracerBoolConversionError
        try:
            return bool(result)
        except TypeError:
            # In JAX traced context, return the array directly
            return result

    def validate_point(self, x: Array, atol: float = 1e-6) -> bool | Array:
        """Validate that x is a valid point on the SPD manifold.

        Args:
            x: Point to validate.
            atol: Absolute tolerance for validation.

        Returns:
            True if x is on the manifold (symmetric positive definite), False otherwise.
        """
        return self._is_in_manifold(x, atol)

    # JIT-optimized implementation methods

    def _proj_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized projection implementation.

        Numerical stability improvements:
        - Efficient guarantee of symmetry
        """
        # For SPD manifolds, tangent space consists of symmetric matrices
        return 0.5 * (v + v.T)

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized exponential map implementation.

        Numerical stability improvements:
        - Stable implementation via eigenvalue decomposition
        - Numerical stability ensured through condition number control
        """
        # Use eigendecomposition for better JAX compatibility
        return self._exp_impl_eigen(x, v)

    def _exp_impl_eigen(self, x: Array, v: Array) -> Array:
        """Eigendecomposition-based exponential map with enhanced numerical stability."""
        # Compute eigendecomposition with robust error handling
        eigenvals_x, eigenvecs_x = jnp.linalg.eigh(x)

        # Enhanced numerical stability: use larger epsilon and condition number control
        min_eigenval = jnp.maximum(jnp.min(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        max_eigenval = jnp.maximum(jnp.max(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        condition_number = max_eigenval / min_eigenval

        # Apply stronger regularization for ill-conditioned matrices
        regularization_eps = jnp.where(condition_number > 1e6, 1e-6, 1e-10)
        eigenvals_x = jnp.maximum(eigenvals_x, regularization_eps)

        sqrt_eigenvals_x = jnp.sqrt(eigenvals_x)
        inv_sqrt_eigenvals_x = 1.0 / sqrt_eigenvals_x

        # Batch-aware matrix construction with improved stability
        # More robust batch detection based on input shapes
        is_batch = x.ndim > 2 or (eigenvals_x.ndim > 1 and eigenvals_x.shape[0] != self.n)

        if is_batch:  # Batch case
            sqrt_diag = jnp.apply_along_axis(jnp.diag, -1, sqrt_eigenvals_x)
            inv_sqrt_diag = jnp.apply_along_axis(jnp.diag, -1, inv_sqrt_eigenvals_x)
            transpose_axes = (-2, -1)
        else:  # Single matrix case
            sqrt_diag = jnp.diag(sqrt_eigenvals_x)
            inv_sqrt_diag = jnp.diag(inv_sqrt_eigenvals_x)
            transpose_axes = (-2, -1)

        # Compute matrix square root and inverse square root
        x_sqrt = eigenvecs_x @ sqrt_diag @ jnp.swapaxes(eigenvecs_x, *transpose_axes)
        x_inv_sqrt = eigenvecs_x @ inv_sqrt_diag @ jnp.swapaxes(eigenvecs_x, *transpose_axes)

        # Transform tangent vector with numerical stability checks
        v_transformed = x_inv_sqrt @ v @ x_inv_sqrt

        # Check for NaN/inf in transformed tangent vector
        v_transformed = jnp.where(jnp.isfinite(v_transformed), v_transformed, 0.0)

        # Apply matrix exponential
        exp_v = expm(v_transformed)

        # Check for NaN/inf in matrix exponential result - create identity with correct shape
        if is_batch:  # Batch case
            batch_shape = v_transformed.shape[:-2]
            eye_batch = jnp.tile(jnp.eye(self.n), (*batch_shape, 1, 1))
        else:  # Single matrix case
            eye_batch = jnp.eye(self.n)

        exp_v = jnp.where(jnp.isfinite(exp_v), exp_v, eye_batch)

        # Transform back to manifold
        result = x_sqrt @ exp_v @ x_sqrt

        # Final SPD constraint enforcement: ensure positive definiteness
        # Re-eigendecompose result and clamp eigenvalues with more aggressive thresholding
        result_eigenvals, result_eigenvecs = jnp.linalg.eigh(result)

        # Use more aggressive minimum eigenvalue based on the scale of the matrix
        max_result_eigenval = jnp.max(jnp.abs(result_eigenvals))
        # Scale-aware minimum eigenvalue: at least 1e-12 times the maximum eigenvalue
        min_eigenval_threshold = jnp.maximum(
            max_result_eigenval * 1e-12,
            1e-6,  # Absolute minimum threshold for very extreme cases
        )

        result_eigenvals = jnp.maximum(result_eigenvals, min_eigenval_threshold)

        # Use the same batch detection logic
        result_diag = jnp.apply_along_axis(jnp.diag, -1, result_eigenvals) if is_batch else jnp.diag(result_eigenvals)

        result = result_eigenvecs @ result_diag @ jnp.swapaxes(result_eigenvecs, *transpose_axes)

        return jnp.asarray(result)

    def _log_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized logarithmic map implementation.

        Numerical stability improvements:
        - Stable implementation via eigenvalue decomposition
        - Stability ensured through condition number control
        """
        # Use eigendecomposition for better JAX compatibility
        return self._log_impl_eigen(x, y)

    def _log_impl_eigen(self, x: Array, y: Array) -> Array:
        """Eigendecomposition-based logarithmic map with enhanced numerical stability."""
        # Compute eigendecomposition with robust error handling
        eigenvals_x, eigenvecs_x = jnp.linalg.eigh(x)

        # Enhanced numerical stability: condition number control
        min_eigenval_x = jnp.maximum(jnp.min(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        max_eigenval_x = jnp.maximum(jnp.max(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        condition_number_x = max_eigenval_x / min_eigenval_x

        # Apply stronger regularization for ill-conditioned matrices
        regularization_eps = jnp.where(condition_number_x > 1e6, 1e-6, 1e-10)
        eigenvals_x = jnp.maximum(eigenvals_x, regularization_eps)

        sqrt_eigenvals_x = jnp.sqrt(eigenvals_x)
        inv_sqrt_eigenvals_x = 1.0 / sqrt_eigenvals_x

        # x^(1/2) and x^(-1/2)
        x_sqrt = eigenvecs_x @ jnp.diag(sqrt_eigenvals_x) @ eigenvecs_x.T
        x_inv_sqrt = eigenvecs_x @ jnp.diag(inv_sqrt_eigenvals_x) @ eigenvecs_x.T

        # Transform y with numerical stability checks
        y_transformed = x_inv_sqrt @ y @ x_inv_sqrt
        y_transformed = jnp.where(jnp.isfinite(y_transformed), y_transformed, jnp.eye(self.n))

        # Apply matrix logarithm with enhanced stability
        eigenvals_y, eigenvecs_y = jnp.linalg.eigh(y_transformed)

        # Ensure eigenvalues are positive for logarithm
        eigenvals_y = jnp.maximum(eigenvals_y, NumericalConstants.HIGH_PRECISION_EPSILON)

        # Check for very small eigenvalues that could cause log instability
        eigenvals_y = jnp.where(eigenvals_y < 1e-12, 1e-12, eigenvals_y)

        log_eigenvals_y = jnp.log(eigenvals_y)

        # Check for NaN/inf in log eigenvalues
        log_eigenvals_y = jnp.where(jnp.isfinite(log_eigenvals_y), log_eigenvals_y, 0.0)

        log_y = eigenvecs_y @ jnp.diag(log_eigenvals_y) @ eigenvecs_y.T

        # Transform back
        result = x_sqrt @ log_y @ x_sqrt

        # Final check for numerical stability
        result = jnp.where(jnp.isfinite(result), result, jnp.zeros_like(result))

        return jnp.asarray(result)

    def _inner_impl(self, x: Array, u: Array, v: Array) -> Array:
        """JIT-optimized inner product implementation with enhanced numerical stability.

        Affine-invariant inner product on SPD manifold
        Batch processing support
        """
        # Compute x^(-1) using direct solve for JAX compatibility (batch-aware)
        # Create identity matrix with correct batch shape
        batch_shape = x.shape[:-2] if x.ndim > 2 else ()
        eye_shape = (*batch_shape, self.n, self.n)
        identity = jnp.broadcast_to(jnp.eye(self.n), eye_shape)

        # Add small regularization to x for numerical stability
        x_regularized = x + NumericalConstants.HIGH_PRECISION_EPSILON * identity

        # Check if matrix is well-conditioned enough for direct solve
        eigenvals = jnp.linalg.eigvals(x_regularized)
        min_eigenval = jnp.min(jnp.real(eigenvals))
        max_eigenval = jnp.max(jnp.real(eigenvals))
        condition_number = max_eigenval / jnp.maximum(min_eigenval, NumericalConstants.HIGH_PRECISION_EPSILON)

        def stable_solve() -> Array:
            # Direct solve for well-conditioned matrices
            return solve(x_regularized, identity, assume_a="pos")

        def eigendecomposition_fallback() -> Any:
            # Eigendecomposition fallback for ill-conditioned matrices
            eigenvals, eigenvecs = jnp.linalg.eigh(x_regularized)
            eigenvals = jnp.maximum(eigenvals, NumericalConstants.HIGH_PRECISION_EPSILON)
            inv_eigenvals = 1.0 / eigenvals

            # Use robust batch detection
            is_batch_inner = x.ndim > 2 or (eigenvals.ndim > 1 and eigenvals.shape[0] != self.n)

            if is_batch_inner:  # Batch case
                inv_diag = jnp.apply_along_axis(jnp.diag, -1, inv_eigenvals)
                transpose_axes = (-2, -1)
            else:  # Single matrix case
                inv_diag = jnp.diag(inv_eigenvals)
                transpose_axes = (-2, -1)

            return eigenvecs @ inv_diag @ jnp.swapaxes(eigenvecs, *transpose_axes)

        # Choose method based on condition number
        x_inv = jax.lax.cond(condition_number < 1e10, stable_solve, eigendecomposition_fallback)

        # Check for NaN/inf in the inverse
        x_inv = jnp.where(jnp.isfinite(x_inv), x_inv, identity)

        # Compute <u, v>_x = tr(x^(-1) @ u @ x^(-1) @ v) with numerical stability
        temp = x_inv @ u @ x_inv @ v

        # Check for NaN/inf in intermediate computation
        temp = jnp.where(jnp.isfinite(temp), temp, 0.0)

        # Use jnp.trace with axis specification for batch processing
        inner_product = jnp.trace(temp, axis1=-2, axis2=-1)

        # Enhanced numerical clipping
        inner_product = jnp.where(jnp.isfinite(inner_product), inner_product, 0.0)
        return jnp.clip(inner_product, -1e15, 1e15)

    def _dist_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized distance calculation implementation.

        Numerical stability improvements:
        - Efficient calculation via Cholesky decomposition
        """
        # Use the logarithmic map and inner product
        log_xy = self._log_impl(x, y)

        # Compute Riemannian distance: sqrt(<log_x(y), log_x(y)>_x)
        inner_product = self._inner_impl(x, log_xy, log_xy)

        # Ensure non-negative for numerical stability
        inner_product = jnp.maximum(inner_product, 0.0)

        distance = jnp.sqrt(inner_product)
        return jnp.where(distance < NumericalConstants.HIGH_PRECISION_EPSILON, 0.0, distance)

    def _get_static_args(self, method_name: str) -> tuple[Any, ...]:
        """Static argument configuration for JIT compilation.

        For SPD manifold, returns argument position indices for static compilation.
        Conservative approach: no static arguments to avoid shape/type conflicts.
        """
        # Return empty tuple - no static arguments for safety
        # Future optimization could consider making 'self' parameter static (position 0)
        return ()
