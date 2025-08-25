"""Geodesic connection and true parallel transport for SPD manifolds.  # ruff: noqa: RUF002.

This module implements the mathematically exact parallel transport on the
manifold of Symmetric Positive Definite (SPD) matrices with the affine-invariant
Riemannian metric. This replaces approximate methods with geodesic-based transport.

Key features:
- Matrix exponential, logarithm, and square root via eigendecomposition
- True parallel transport using geodesic connection (not approximations)
- Geodesic computation with proper parametrization
- Numerical stability handling for ill-conditioned matrices
- Full JAX ecosystem compatibility (JIT, vmap, grad)

Mathematical Foundation:
For SPD manifold P(n) with affine-invariant metric ⟨U, V⟩_X = tr(X^(-1)U X^(-1)V),
the geodesic between X and Y is:
    gamma(t) = X^(1/2) exp(t log(X^(-1/2) Y X^(-1/2))) X^(1/2)

The exact parallel transport of tangent vector V from X to Y is:
    P_{X→Y}(V) = Y^(1/2) (Y^(-1/2) X Y^(-1/2))^(1/2) X^(-1/2) V X^(-1/2)
                 (Y^(-1/2) X Y^(-1/2))^(1/2) Y^(1/2)

References:
- Pennec, X. (2006). Intrinsic Statistics on Riemannian Manifolds
- Arsigny, V. et al. (2007). Geometric Means in a Novel Vector Space Structure
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from riemannax.core.numerical_stability import NumericalStabilityLayer


class GeodesicConnection:
    """Implements geodesic connection and exact parallel transport for SPD manifolds.

    This class provides mathematically exact parallel transport on the manifold of
    SPD matrices, replacing approximation methods with geodesic-based calculations.
    All operations use eigendecomposition for numerical stability and are designed
    to work with JAX transformations.

    The implementation handles:
    - Matrix functions (exp, log, sqrt) via eigendecomposition
    - Exact parallel transport using geodesic connection
    - Numerical stability for ill-conditioned matrices
    - JAX transformations (JIT, vmap, grad)

    Examples:
        >>> conn = GeodesicConnection()
        >>> X = jnp.eye(3) * 2.0
        >>> Y = jnp.array([[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]])
        >>> V = jnp.array([[0.1, 0.05, 0.], [0.05, -0.1, 0.], [0., 0., 0.08]])
        >>> transported = conn.parallel_transport(X, Y, V)
        >>> geodesic_midpoint = conn.geodesic(X, Y, 0.5)
    """

    def __init__(self, stability_layer: NumericalStabilityLayer | None = None) -> None:
        """Initialize the geodesic connection.

        Args:
            stability_layer: Optional numerical stability layer for regularization.
                           If None, a default layer is created.
        """
        self.stability_layer = stability_layer or NumericalStabilityLayer()

    def matrix_sqrt_spd(self, X: jnp.ndarray) -> jnp.ndarray:
        """Compute matrix square root for SPD matrix using eigendecomposition.

        For SPD matrix X, computes X^(1/2) such that X^(1/2) @ X^(1/2) = X.
        Uses eigendecomposition: X = Q Λ Q^T => X^(1/2) = Q Λ^(1/2) Q^T

        Args:
            X: SPD matrix of shape (n, n).

        Returns:
            Matrix square root X^(1/2) of same shape as X.

        Note:
            For numerical stability, eigenvalues are clamped to avoid
            near-zero values that could cause instability.
        """
        # Apply regularization if needed for numerical stability
        X_reg = self.stability_layer.regularize_matrix(X)

        # Symmetric eigendecomposition (more stable than eig for symmetric matrices)
        eigenvals, eigenvecs = jnp.linalg.eigh(X_reg)

        # Ensure positive eigenvalues (clamp very small values)
        eigenvals = jnp.maximum(eigenvals, 1e-16)

        # Compute sqrt of eigenvalues
        sqrt_eigenvals = jnp.sqrt(eigenvals)

        # Reconstruct matrix: Q sqrt(Λ) Q^T
        result = eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.T

        return jnp.array(result)

    def matrix_inv_sqrt_spd(self, X: jnp.ndarray) -> jnp.ndarray:
        """Compute matrix inverse square root for SPD matrix.

        For SPD matrix X, computes X^(-1/2) such that X^(-1/2) @ X @ X^(-1/2) = I.
        Uses eigendecomposition: X = Q Λ Q^T => X^(-1/2) = Q Λ^(-1/2) Q^T

        Args:
            X: SPD matrix of shape (n, n).

        Returns:
            Matrix inverse square root X^(-1/2) of same shape as X.
        """
        # Apply regularization if needed for numerical stability
        X_reg = self.stability_layer.regularize_matrix(X)

        # Symmetric eigendecomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(X_reg)

        # Ensure positive eigenvalues and avoid division by zero
        eigenvals = jnp.maximum(eigenvals, 1e-16)

        # Compute inverse sqrt of eigenvalues
        inv_sqrt_eigenvals = 1.0 / jnp.sqrt(eigenvals)

        # Reconstruct matrix: Q Λ^(-1/2) Q^T
        result = eigenvecs @ jnp.diag(inv_sqrt_eigenvals) @ eigenvecs.T

        return jnp.array(result)

    def matrix_exp_symmetric(self, A: jnp.ndarray) -> jnp.ndarray:
        """Compute matrix exponential for symmetric matrix.

        For symmetric matrix A, computes exp(A) using eigendecomposition.
        Since A is symmetric: A = Q Λ Q^T => exp(A) = Q exp(Λ) Q^T

        Args:
            A: Symmetric matrix of shape (n, n).

        Returns:
            Matrix exponential exp(A) of same shape as A.

        Note:
            The result is always SPD when A is symmetric, since exp(λ) > 0
            for all real λ.
        """
        # Symmetric eigendecomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(A)

        # Compute exp of eigenvalues element-wise
        exp_eigenvals = jnp.exp(eigenvals)

        # Reconstruct matrix: Q exp(Λ) Q^T
        result = eigenvecs @ jnp.diag(exp_eigenvals) @ eigenvecs.T

        return jnp.array(result)

    def matrix_log_spd(self, X: jnp.ndarray) -> jnp.ndarray:
        """Compute matrix logarithm for SPD matrix.

        For SPD matrix X, computes log(X) using eigendecomposition.
        Since X is SPD: X = Q Λ Q^T => log(X) = Q log(Λ) Q^T

        Args:
            X: SPD matrix of shape (n, n).

        Returns:
            Matrix logarithm log(X) of same shape as X (symmetric).

        Note:
            The logarithm is well-defined for SPD matrices since all
            eigenvalues are positive.
        """
        # Apply regularization if needed for numerical stability
        X_reg = self.stability_layer.regularize_matrix(X)

        # Symmetric eigendecomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(X_reg)

        # Ensure positive eigenvalues for logarithm
        eigenvals = jnp.maximum(eigenvals, 1e-16)

        # Compute log of eigenvalues element-wise
        log_eigenvals = jnp.log(eigenvals)

        # Reconstruct matrix: Q log(Λ) Q^T
        result = eigenvecs @ jnp.diag(log_eigenvals) @ eigenvecs.T

        return jnp.array(result)

    def geodesic(self, X: jnp.ndarray, Y: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute point on geodesic between two SPD matrices.

        Computes the geodesic path gamma(t) connecting X and Y on the SPD manifold
        with affine-invariant metric. The geodesic is given by:
        gamma(t) = X^(1/2) exp(t log(X^(-1/2) Y X^(-1/2))) X^(1/2)

        Args:
            X: Starting SPD matrix of shape (n, n).
            Y: Ending SPD matrix of shape (n, n).
            t: Parameter in [0, 1] where gamma(0) = X and gamma(1) = Y.

        Returns:
            Point gamma(t) on geodesic of same shape as X and Y.

        Note:
            This is the unique shortest path between X and Y on the
            SPD manifold with respect to the affine-invariant metric.
        """
        # Compute matrix square root and inverse square root of X
        sqrt_X = self.matrix_sqrt_spd(X)
        inv_sqrt_X = self.matrix_inv_sqrt_spd(X)

        # Compute the argument for the matrix exponential
        # log_arg = X^(-1/2) Y X^(-1/2)
        log_arg = inv_sqrt_X @ Y @ inv_sqrt_X

        # Compute matrix logarithm
        log_matrix = self.matrix_log_spd(log_arg)

        # Scale by parameter t
        scaled_log = t * log_matrix

        # Compute matrix exponential
        exp_matrix = self.matrix_exp_symmetric(scaled_log)

        # Final geodesic point: X^(1/2) exp(t log(...)) X^(1/2)
        result = sqrt_X @ exp_matrix @ sqrt_X

        return result

    def parallel_transport(self, X: jnp.ndarray, Y: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Exact parallel transport of tangent vector on SPD manifold.

        Implements the mathematically exact parallel transport of tangent
        vector V from point X to point Y using the geodesic connection.
        This replaces approximation methods with the true Riemannian connection.

        The exact formula is:
        P_{X→Y}(V) = Y^(1/2) (Y^(-1/2) X Y^(-1/2))^(1/2) X^(-1/2) V X^(-1/2)
                     (Y^(-1/2) X Y^(-1/2))^(1/2) Y^(1/2)

        Args:
            X: Source SPD matrix of shape (n, n).
            Y: Target SPD matrix of shape (n, n).
            V: Tangent vector at X (symmetric matrix) of shape (n, n).

        Returns:
            Transported tangent vector at Y of same shape as V.

        Note:
            This preserves the Riemannian metric: ⟨U, V⟩_X = ⟨P(U), P(V)⟩_Y
            where P denotes parallel transport from X to Y.
        """
        # Compute matrix square roots and inverse square roots
        self.matrix_sqrt_spd(X)
        inv_sqrt_X = self.matrix_inv_sqrt_spd(X)
        sqrt_Y = self.matrix_sqrt_spd(Y)
        inv_sqrt_Y = self.matrix_inv_sqrt_spd(Y)

        # Compute the transformation matrix: Y^(-1/2) X Y^(-1/2)
        transform_matrix = inv_sqrt_Y @ X @ inv_sqrt_Y

        # Compute its square root
        sqrt_transform = self.matrix_sqrt_spd(transform_matrix)

        # Apply the exact parallel transport formula
        # Step 1: X^(-1/2) V X^(-1/2)
        normalized_V = inv_sqrt_X @ V @ inv_sqrt_X

        # Step 2: Apply transformation matrices
        # sqrt_transform @ normalized_V @ sqrt_transform
        transformed_V = sqrt_transform @ normalized_V @ sqrt_transform

        # Step 3: Final transformation Y^(1/2) @ transformed_V @ Y^(1/2)
        result = sqrt_Y @ transformed_V @ sqrt_Y

        return result

    def parallel_transport_approximate(self, X: jnp.ndarray, Y: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """Approximate parallel transport (for comparison purposes).

        Implements the approximate parallel transport formula that was
        previously used: P_x→y(v) = (y/x)^(1/2) @ v @ (y/x)^(1/2)

        This method is provided for comparison with the exact method and
        for backward compatibility.

        Args:
            X: Source SPD matrix of shape (n, n).
            Y: Target SPD matrix of shape (n, n).
            V: Tangent vector at X (symmetric matrix) of shape (n, n).

        Returns:
            Approximately transported tangent vector at Y of same shape as V.

        Note:
            This is a first-order approximation that ignores curvature terms.
            The exact method should be preferred for mathematical correctness.
        """
        # Compute X^(-1) and Y @ X^(-1)
        X_inv = jnp.linalg.inv(X)
        Y_over_X = Y @ X_inv

        # Compute (Y/X)^(1/2)
        sqrt_Y_over_X = self.matrix_sqrt_spd(Y_over_X)

        # Apply approximate formula
        result = sqrt_Y_over_X @ V @ sqrt_Y_over_X

        return result

    def geodesic_distance(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """Compute geodesic distance between two SPD matrices.

        The geodesic distance with respect to the affine-invariant metric is:
        d(X, Y) = ||log(X^(-1/2) Y X^(-1/2))||_F

        where ||·||_F is the Frobenius norm.

        Args:
            X: First SPD matrix of shape (n, n).
            Y: Second SPD matrix of shape (n, n).

        Returns:
            Geodesic distance as scalar.

        Note:
            This distance is invariant under congruent transformations:
            d(AXA^T, AYA^T) = d(X, Y) for any invertible A.
        """
        # Compute X^(-1/2)
        inv_sqrt_X = self.matrix_inv_sqrt_spd(X)

        # Compute log(X^(-1/2) Y X^(-1/2))
        log_arg = inv_sqrt_X @ Y @ inv_sqrt_X
        log_matrix = self.matrix_log_spd(log_arg)

        # Compute Frobenius norm
        distance = jnp.linalg.norm(log_matrix, "fro")

        return jnp.array(distance)

    def is_spd_compatible(self, X: jnp.ndarray, tolerance: float = 1e-8) -> bool:
        """Check if matrix is compatible with SPD manifold operations.

        Verifies that the matrix satisfies the requirements for SPD manifold
        operations (symmetric and positive definite).

        Args:
            X: Matrix to check of shape (n, n).
            tolerance: Numerical tolerance for checks.

        Returns:
            True if matrix is suitable for SPD manifold operations.
        """
        return self.stability_layer.validate_spd_properties(X, tolerance)


# JIT-compiled convenience functions for performance
def parallel_transport_jit(X: jnp.ndarray, Y: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled convenience function for parallel transport.

    Args:
        X: Source SPD matrix.
        Y: Target SPD matrix.
        V: Tangent vector to transport.

    Returns:
        Transported tangent vector.
    """
    connection = GeodesicConnection()
    return connection.parallel_transport(X, Y, V)


def geodesic_jit(X: jnp.ndarray, Y: jnp.ndarray, t: float) -> jnp.ndarray:
    """JIT-compiled convenience function for geodesic computation.

    Args:
        X: Starting SPD matrix.
        Y: Ending SPD matrix.
        t: Geodesic parameter.

    Returns:
        Point on geodesic.
    """
    connection = GeodesicConnection()
    return connection.geodesic(X, Y, t)


# Apply JIT compilation
parallel_transport_compiled = jax.jit(parallel_transport_jit)
geodesic_compiled = jax.jit(geodesic_jit)
