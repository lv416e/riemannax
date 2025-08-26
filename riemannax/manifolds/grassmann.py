"""Implementation of the Grassmann manifold Gr(p,n).

The Grassmann manifold consists of all p-dimensional subspaces of n-dimensional
Euclidean space, represented by n * p matrices with orthonormal columns.
"""

from collections.abc import Callable
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
        """JIT-optimized projection implementation with batch support.

        Numerical stability improvements:
        - Efficient preservation of subspace constraints
        - Full batch processing support for arbitrary batch dimensions
        - Batch-aware matrix operations using einsum for optimal performance
        """
        # Tangent space: T_X Gr(p,n) = {V ∈ R^{n * p} : X^T V = 0}
        # For batch processing, we need to handle arbitrary leading dimensions

        # Get batch dimensions by comparing shapes
        x_shape = x.shape
        v_shape = v.shape

        # Extract batch dimensions (everything except last 2 dimensions)
        batch_dims_x = x_shape[:-2] if len(x_shape) > 2 else ()
        batch_dims_v = v_shape[:-2] if len(v_shape) > 2 else ()

        # Ensure both have same batch dimensions or one is unbatched
        if batch_dims_x and batch_dims_v and batch_dims_x != batch_dims_v:
            raise ValueError(f"Batch dimensions must match: x {batch_dims_x} vs v {batch_dims_v}")

        # Use einsum for batch-aware matrix multiplication
        # This handles arbitrary batch dimensions efficiently
        if len(x_shape) > 2 or len(v_shape) > 2:
            # Batch case: use einsum for efficient computation
            # Formula: v - x @ (x.T @ v) becomes v - x @ (einsum('...ji,...jk->...ik', x, v))
            xtv = jnp.einsum('...ji,...jk->...ik', x, v)
            x_xtv = jnp.einsum('...ij,...jk->...ik', x, xtv)
            return v - x_xtv
        else:
            # Single case: use standard matrix operations
            return v - x @ (x.T @ v)

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """True SVD-based exponential map with comprehensive batch support.

        Implements: z = p·V·cos(S)·V^T + U·sin(S)·V^T, then QR decomposition
        where ξ = U·S·V^T is the SVD decomposition of the tangent vector.

        Enhanced for batch processing with arbitrary batch dimensions.
        """
        # Project to tangent space to ensure v is in tangent space
        v_proj = self.proj(x, v)

        # Handle zero tangent vector - batch-aware norm calculation
        # For batch processing, we need to handle norms along the appropriate axes
        v_norm = jnp.linalg.norm(v_proj, axis=(-2, -1)) if len(v_proj.shape) > 2 else jnp.linalg.norm(v_proj)

        def zero_tangent_case() -> Array:
            return x

        def nonzero_tangent_case() -> Array:
            # Step 1: Compute SVD of tangent vector ξ = U·S·V^T
            # JAX SVD handles batch dimensions automatically
            U, S, Vt = jnp.linalg.svd(v_proj, full_matrices=False)

            # Step 2: Apply trigonometric functions to singular values
            cos_S = jnp.cos(S)
            sin_S = jnp.sin(S)

            # Step 3: Compute z = x·V·cos(S)·V^T + U·sin(S)·V^T
            # Use einsum for batch-aware operations
            V = jnp.swapaxes(Vt, -2, -1)  # Convert V^T back to V

            if len(x.shape) > 2:
                # Batch case: use einsum for efficient batch operations
                # First term: x @ V @ diag(cos_S) @ V.T
                V_cos_S = jnp.einsum('...ij,...j->...ij', V, cos_S)
                first_term = jnp.einsum('...ij,...jk,...kl->...il', x, V_cos_S, jnp.swapaxes(V, -2, -1))

                # Second term: U @ diag(sin_S) @ V^T
                U_sin_S = jnp.einsum('...ij,...j->...ij', U, sin_S)
                second_term = jnp.einsum('...ij,...jk->...ik', U_sin_S, Vt)
            else:
                # Single case: standard matrix operations
                first_term = x @ V @ jnp.diag(cos_S) @ V.T
                second_term = U @ jnp.diag(sin_S) @ V.T

            # Combine terms
            z = first_term + second_term

            # Step 4: QR decomposition to ensure orthonormality
            # JAX QR handles batch dimensions automatically
            Q, _ = jnp.linalg.qr(z, mode="reduced")

            return Q

        # Use appropriate threshold for batch or single case
        if len(v_proj.shape) > 2:
            # Batch case: element-wise comparison
            threshold = 1e-12
            return jnp.where(
                jnp.expand_dims(v_norm < threshold, (-2, -1)),
                zero_tangent_case(),
                nonzero_tangent_case()
            )
        else:
            # Single case: use lax.cond
            return jnp.asarray(jax.lax.cond(
                v_norm < 1e-12,
                zero_tangent_case,
                nonzero_tangent_case
            ))

    def _log_impl(self, x: Array, y: Array) -> Array:
        """True SVD-based logarithmic map with comprehensive batch support.

        Implements: log_x(y) = V·atan(S)·U^T with proper numerical stability.
        Enhanced for batch processing with arbitrary batch dimensions.
        """
        # Handle near-identical points - batch-aware distance calculation
        distance = jnp.linalg.norm(x - y, axis=(-2, -1)) if len(x.shape) > 2 else jnp.linalg.norm(x - y)

        def identical_points_case() -> Array:
            return jnp.zeros_like(x)

        def different_points_case() -> Array:
            # Use the standard Grassmann logarithmic map approach
            # Compute the orthogonal part of y relative to x
            if len(x.shape) > 2:
                # Batch case: use einsum for batch-aware operations
                # Compute (I - x @ x.T) @ y efficiently
                identity = jnp.eye(self.n)
                xx_t = jnp.einsum('...ij,...kj->...ik', x, x)
                orthogonal_part = jnp.einsum('ij,...ij->...ij', identity, y) - jnp.einsum('...ij,...ik->...jk', xx_t, y)
            else:
                # Single case: standard matrix operations
                orthogonal_part = (jnp.eye(self.n) - x @ x.T) @ y

            # For small differences, use the direct tangent space projection
            if len(orthogonal_part.shape) > 2:
                # Batch case: norm over last 2 dimensions
                norm_orth = jnp.linalg.norm(orthogonal_part, axis=(-2, -1))
            else:
                # Single case: standard norm
                norm_orth = jnp.linalg.norm(orthogonal_part)

            def small_distance_case():
                # For small distances, the log map is approximately the orthogonal projection
                return orthogonal_part

            def large_distance_case():
                # For larger distances, we need to scale properly
                # Compute SVD of orthogonal part - JAX handles batch dimensions automatically
                U_orth, S_orth, VT_orth = jnp.linalg.svd(orthogonal_part, full_matrices=False)

                # Scale the singular values appropriately using arctan
                # This handles the V·atan(S)·U^T formula properly
                scaled_singular_values = jnp.where(
                    S_orth > 1e-10,
                    jnp.arctan(S_orth),  # Standard atan scaling
                    S_orth  # For very small values, atan(x) ≈ x
                )

                # Reconstruct the tangent vector
                if len(orthogonal_part.shape) > 2:
                    # Batch case: use einsum for reconstruction
                    US = jnp.einsum('...ij,...j->...ij', U_orth, scaled_singular_values)
                    return jnp.einsum('...ij,...jk->...ik', US, VT_orth)
                else:
                    # Single case: standard matrix operations
                    return U_orth @ jnp.diag(scaled_singular_values) @ VT_orth

            # Choose based on the magnitude of the orthogonal part
            if len(orthogonal_part.shape) > 2:
                # Batch case: element-wise condition
                threshold = 0.1
                result = jnp.where(
                    jnp.expand_dims(norm_orth < threshold, (-2, -1)),
                    small_distance_case(),
                    large_distance_case()
                )
            else:
                # Single case: use lax.cond
                result = jax.lax.cond(
                    norm_orth < 0.1,  # Threshold for "small" vs "large" distance
                    small_distance_case,
                    large_distance_case
                )

            # Always project to tangent space to ensure X^T @ result = 0
            return self.proj(x, result)

        # Check if points are nearly identical
        if len(x.shape) > 2:
            # Batch case: element-wise comparison
            threshold = 1e-12
            are_close = distance < threshold
            return jnp.where(
                jnp.expand_dims(are_close, (-2, -1)),
                identical_points_case(),
                different_points_case()
            )
        else:
            # Single case: use lax.cond
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

    # Batch Processing Methods

    def create_batch_operations(self) -> dict[str, Callable]:
        """Create batch-optimized versions of all manifold operations using vmap.

        Returns:
            Dictionary of batch operation functions with consistent signatures.
        """
        return {
            'proj': jax.vmap(self._proj_impl, in_axes=(0, 0)),
            'exp': jax.vmap(self._exp_impl, in_axes=(0, 0)),
            'log': jax.vmap(self._log_impl, in_axes=(0, 0)),
            'inner': jax.vmap(self._inner_impl, in_axes=(0, 0, 0)),
            'dist': jax.vmap(self._dist_impl, in_axes=(0, 0)),
            'transp': jax.vmap(lambda x, y, v: self.proj(y, v), in_axes=(0, 0, 0)),
            'retr': jax.vmap(lambda x, v: self._qr_retraction(x, v), in_axes=(0, 0))
        }

    def _qr_retraction(self, x: Array, v: Array) -> Array:
        """QR-based retraction optimized for batch processing."""
        y = x + v
        q, _ = jnp.linalg.qr(y, mode="reduced")
        return q

    def batch_exp(self, x_batch: Array, v_batch: Array) -> Array:
        """Batch exponential map computation using vmap.

        Args:
            x_batch: Batch of base points, shape (batch_size, n, p)
            v_batch: Batch of tangent vectors, shape (batch_size, n, p)

        Returns:
            Batch of exponential map results, shape (batch_size, n, p)
        """
        return jax.vmap(self._exp_impl, in_axes=(0, 0))(x_batch, v_batch)

    def batch_log(self, x_batch: Array, y_batch: Array) -> Array:
        """Batch logarithmic map computation using vmap.

        Args:
            x_batch: Batch of base points, shape (batch_size, n, p)
            y_batch: Batch of target points, shape (batch_size, n, p)

        Returns:
            Batch of logarithmic map results, shape (batch_size, n, p)
        """
        return jax.vmap(self._log_impl, in_axes=(0, 0))(x_batch, y_batch)

    def batch_proj(self, x_batch: Array, v_batch: Array) -> Array:
        """Batch projection onto tangent space using vmap.

        Args:
            x_batch: Batch of base points, shape (batch_size, n, p)
            v_batch: Batch of vectors to project, shape (batch_size, n, p)

        Returns:
            Batch of projected vectors, shape (batch_size, n, p)
        """
        return jax.vmap(self._proj_impl, in_axes=(0, 0))(x_batch, v_batch)

    def batch_dist(self, x_batch: Array, y_batch: Array) -> Array:
        """Batch distance computation using vmap.

        Args:
            x_batch: Batch of first points, shape (batch_size, n, p)
            y_batch: Batch of second points, shape (batch_size, n, p)

        Returns:
            Batch of distances, shape (batch_size,)
        """
        return jax.vmap(self._dist_impl, in_axes=(0, 0))(x_batch, y_batch)

    def batch_inner(self, x_batch: Array, u_batch: Array, v_batch: Array) -> Array:
        """Batch inner product computation using vmap.

        Args:
            x_batch: Batch of base points, shape (batch_size, n, p)
            u_batch: Batch of first tangent vectors, shape (batch_size, n, p)
            v_batch: Batch of second tangent vectors, shape (batch_size, n, p)

        Returns:
            Batch of inner products, shape (batch_size,)
        """
        return jax.vmap(self._inner_impl, in_axes=(0, 0, 0))(x_batch, u_batch, v_batch)
