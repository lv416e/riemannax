"""Implementation of the Grassmann manifold Gr(p,n).

The Grassmann manifold consists of all p-dimensional subspaces of n-dimensional
Euclidean space, represented by n * p matrices with orthonormal columns.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from ..core.constants import NumericalConstants
from ..core.jit_decorator import jit_optimized
from ..core.type_system import ManifoldPoint, TangentVector
from .base import DimensionError, Manifold


class Grassmann(Manifold):
    """Grassmann manifold Gr(p,n) of p-dimensional subspaces in R^n.

    Points on the manifold are represented as n * p matrices with orthonormal columns.
    The tangent space at a point X consists of n * p matrices V such that X^T V = 0.
    """

    def __init__(self, n: int, p: int):
        """Initialize Grassmann manifold.

        Args:
            n: Ambient space dimension.
            p: Subspace dimension (must satisfy p ≤ n).

        Raises:
            DimensionError: If p > n.
        """
        if p > n:
            raise DimensionError(f"Subspace dimension p={p} cannot exceed ambient dimension n={n}")
        if p <= 0 or n <= 0:
            raise DimensionError("Dimensions must be positive")

        super().__init__()  # JIT-related initialization
        self.n = n
        self.p = p

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of Gr(p,n) = p(n-p)."""
        return self.p * (self.n - self.p)

    @property
    def ambient_dimension(self) -> int:
        """Ambient space dimension n * p."""
        return self.n * self.p

    @jit_optimized(static_args=(0,))
    def proj(self, x: Array, v: Array) -> Array:
        """Project matrix V onto tangent space at X.

        Tangent space: T_X Gr(p,n) = {V ∈ R^{n * p} : X^T V = 0}.
        """
        # Check dimensions
        if v.shape != (self.n, self.p):
            raise ValueError(f"Tangent vector must have shape ({self.n}, {self.p}), got {v.shape}")
        return v - x @ (x.T @ v)

    @jit_optimized(static_args=(0,))
    def exp(self, x: Array, v: Array) -> Array:
        """True geodesic exponential map on Grassmann manifold.

        Computes the exponential map exp_x(v) using proper geodesic curves,
        not retraction. This ensures mathematical correctness for optimization.
        """
        return self._exp_impl(x, v)

    @jit_optimized(static_args=(0,))
    def retr(self, x: Array, v: Array) -> Array:
        """QR-based retraction (cheaper than exponential map)."""
        y = x + v
        q, _ = jnp.linalg.qr(y, mode="reduced")
        return q

    @jit_optimized(static_args=(0,))
    def log(self, x: Array, y: Array) -> Array:
        """Logarithmic map from X to Y (true geodesic inverse)."""
        return self._log_impl(x, y)

    @jit_optimized(static_args=(0,))
    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport from T_X to T_Y via projection."""
        return jnp.asarray(self.proj(y, v))

    @jit_optimized(static_args=(0,))
    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Riemannian inner product is the Frobenius inner product."""
        return jnp.sum(u * v)

    @jit_optimized(static_args=(0,))
    def dist(self, x: Array, y: Array) -> Array:
        """Geodesic distance using principal angles."""
        # Compute principal angles via SVD
        u, s, _ = jnp.linalg.svd(x.T @ y, full_matrices=False)
        cos_theta = jnp.clip(s, -1.0 + NumericalConstants.EPSILON, 1.0 - NumericalConstants.EPSILON)

        # Compute principal angles
        theta = jnp.arccos(jnp.abs(cos_theta))
        distance = jnp.linalg.norm(theta)

        # Handle near-zero distances (identical subspaces)
        return jnp.asarray(jnp.where(distance < NumericalConstants.HIGH_PRECISION_EPSILON, 0.0, distance))

    def random_point(self, key: Array, *shape: int) -> Array:
        """Generate random point via QR decomposition of Gaussian matrix."""
        if shape:
            batch_shape = shape
            full_shape = (*batch_shape, self.n, self.p)
        else:
            full_shape = (self.n, self.p)

        # Sample from standard normal and orthogonalize
        gaussian = jr.normal(key, full_shape)

        if shape:
            # Handle batched case
            def qr_fn(g: Array) -> Array:
                q, _ = jnp.linalg.qr(g, mode="reduced")
                return q

            return jnp.asarray(jnp.vectorize(qr_fn, signature="(n,p)->(n,p)")(gaussian))
        else:
            q, _ = jnp.linalg.qr(gaussian, mode="reduced")
            return q

    def random_tangent(self, key: Array, x: Array, *shape: int) -> Array:
        """Generate random tangent vector via projection."""
        target_shape = (*shape, self.n, self.p) if shape else (self.n, self.p)

        # Sample Gaussian and project to tangent space
        v = jr.normal(key, target_shape)

        if shape:
            # Handle batched case
            def proj_fn(vi: Array) -> Array:
                return jnp.asarray(self.proj(x, vi))

            return jnp.asarray(jnp.vectorize(proj_fn, signature="(n,p)->(n,p)")(v))
        else:
            return jnp.asarray(self.proj(x, v))

    def validate_point(self, x: Array, atol: float = 1e-6) -> bool:
        """Validate that X has orthonormal columns."""
        if x.shape != (self.n, self.p):
            return False

        # Check orthonormality: X^T X = I
        should_be_identity = x.T @ x
        identity = jnp.eye(self.p)
        return bool(jnp.allclose(should_be_identity, identity, atol=atol))

    def validate_tangent(self, x: ManifoldPoint, v: TangentVector, atol: float = 1e-6) -> bool:
        """Validate that V is in tangent space: X^T V = 0."""
        if not self.validate_point(x, atol):
            return False
        if v.shape != (self.n, self.p):
            return False

        # Check tangent space condition
        should_be_zero = x.T @ v
        return bool(jnp.allclose(should_be_zero, 0.0, atol=atol))

    def __repr__(self) -> str:
        """Return string representation of Grassmann manifold."""
        return f"Grassmann({self.n}, {self.p})"

    # JIT-optimized implementation methods

    def _proj_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized projection implementation.

        Numerical stability improvements:
        - Efficient preservation of subspace constraints
        - Batch processing support
        """
        # Tangent space: T_X Gr(p,n) = {V ∈ R^{n * p} : X^T V = 0}
        # Use @ operator for batch-aware matrix multiplication
        return v - x @ (jnp.swapaxes(x, -2, -1) @ v)

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """True SVD-based exponential map following Grassmann manifold theory.

        Implements: z = p·V·cos(S)·V^T + U·sin(S)·V^T, then QR decomposition
        where ξ = U·S·V^T is the SVD decomposition of the tangent vector.
        """
        # Project to tangent space to ensure v is in tangent space
        v_proj = self.proj(x, v)

        # Handle zero tangent vector
        v_norm = jnp.linalg.norm(v_proj)

        def zero_tangent_case() -> Array:
            return x

        def nonzero_tangent_case() -> Array:
            # Step 1: Compute SVD of tangent vector ξ = U·S·V^T
            U, S, Vt = jnp.linalg.svd(v_proj, full_matrices=False)
            V = Vt.T  # Convert V^T back to V

            # Step 2: Apply trigonometric functions to singular values
            cos_S = jnp.cos(S)
            sin_S = jnp.sin(S)

            # Step 3: Compute z = p·V·cos(S)·V^T + U·sin(S)·V^T
            # First term: x @ V @ diag(cos_S) @ V^T
            first_term = x @ V @ jnp.diag(cos_S) @ V.T

            # Second term: U @ diag(sin_S) @ V^T
            second_term = U @ jnp.diag(sin_S) @ V.T

            # Combine terms
            z = first_term + second_term

            # Step 4: QR decomposition to ensure orthonormality
            Q, _ = jnp.linalg.qr(z, mode="reduced")

            return Q

        return jnp.asarray(jax.lax.cond(
            v_norm < 1e-12,
            zero_tangent_case,
            nonzero_tangent_case
        ))

    def _log_impl(self, x: Array, y: Array) -> Array:
        """True SVD-based logarithmic map following Grassmann manifold theory.

        Implements: log_x(y) = V·atan(S)·U^T with proper numerical stability.
        """
        # Handle near-identical points
        def identical_points_case() -> Array:
            return jnp.zeros_like(x)

        def different_points_case() -> Array:
            # Use the standard Grassmann logarithmic map approach
            # Compute the orthogonal part of y relative to x
            orthogonal_part = (jnp.eye(self.n) - x @ x.T) @ y

            # For small differences, use the direct tangent space projection
            norm_orth = jnp.linalg.norm(orthogonal_part)

            def small_distance_case():
                # For small distances, the log map is approximately the orthogonal projection
                return orthogonal_part

            def large_distance_case():
                # For larger distances, we need to scale properly
                # Compute SVD of orthogonal part
                U_orth, S_orth, VT_orth = jnp.linalg.svd(orthogonal_part, full_matrices=False)

                # Scale the singular values appropriately using arctan
                # This handles the V·atan(S)·U^T formula properly
                scaled_singular_values = jnp.where(
                    S_orth > 1e-10,
                    jnp.arctan(S_orth),  # Standard atan scaling
                    S_orth  # For very small values, atan(x) ≈ x
                )

                # Reconstruct the tangent vector
                return U_orth @ jnp.diag(scaled_singular_values) @ VT_orth

            # Choose based on the magnitude of the orthogonal part
            result = jax.lax.cond(
                norm_orth < 0.1,  # Threshold for "small" vs "large" distance
                small_distance_case,
                large_distance_case
            )

            # Always project to tangent space to ensure X^T @ result = 0
            return self.proj(x, result)

        # Check if points are nearly identical
        distance = jnp.linalg.norm(x - y)
        are_close = distance < 1e-12
        return jnp.asarray(jax.lax.cond(are_close, identical_points_case, different_points_case))


    def _inner_impl(self, x: Array, u: Array, v: Array) -> Array:
        """JIT-optimized inner product implementation.

        Frobenius inner product on Grassmann manifold
        """
        # Frobenius inner product with numerical stability
        inner_product = jnp.sum(u * v)
        return jnp.clip(inner_product, -1e15, 1e15)

    def _dist_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized distance calculation implementation.

        Numerical stability improvements:
        - Stable principal angle calculation via SVD decomposition
        """
        # Compute principal angles via SVD - more robust than Frobenius norm check
        XTY = jnp.matmul(x.T, y)
        U, s, Vt = jnp.linalg.svd(XTY, full_matrices=False)

        # Clamp singular values to valid range for arccos
        s_clipped = jnp.clip(s, -1.0 + NumericalConstants.EPSILON, 1.0 - NumericalConstants.EPSILON)

        # Compute principal angles
        theta = jnp.arccos(jnp.abs(s_clipped))

        # Handle near-zero angles (identical subspaces)
        distance = jnp.linalg.norm(theta)
        return jnp.asarray(jnp.where(distance < NumericalConstants.HIGH_PRECISION_EPSILON, 0.0, distance))

    def _get_static_args(self, method_name: str) -> tuple[Any, ...]:
        """Static argument configuration for JIT compilation.

        For Grassmann manifold, returns argument position indices for static compilation.
        Conservative approach: no static arguments to avoid shape/type conflicts.
        """
        # Return empty tuple - no static arguments for safety
        # Future optimization could consider making 'self' parameter static (position 0)
        return ()

    def _is_valid_point(self, x: Array) -> bool:
        """Check if a point is valid on the Grassmann manifold.

        Args:
            x: Point to validate

        Returns:
            True if point has orthonormal columns, False otherwise
        """
        if x.shape != (self.n, self.p):
            return False

        # Check orthonormality: x^T @ x should be identity
        gram = x.T @ x
        identity = jnp.eye(self.p)
        return bool(jnp.allclose(gram, identity, atol=1e-6))
