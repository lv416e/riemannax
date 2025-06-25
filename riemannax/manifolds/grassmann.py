"""Implementation of the Grassmann manifold Gr(p,n).

The Grassmann manifold consists of all p-dimensional subspaces of n-dimensional
Euclidean space, represented by n * p matrices with orthonormal columns.
"""

import jax.numpy as jnp
import jax.random as jr
from jax import Array

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

    def proj(self, x: Array, v: Array) -> Array:
        """Project matrix V onto tangent space at X.

        Tangent space: T_X Gr(p,n) = {V ∈ R^{n * p} : X^T V = 0}.
        """
        return v - x @ (x.T @ v)

    def exp(self, x: Array, v: Array) -> Array:
        """Exponential map using QR decomposition of [X, V]."""
        # Simple retraction-based implementation for now
        # TODO: Implement proper geodesic exponential map
        return self.retr(x, v)

    def retr(self, x: Array, v: Array) -> Array:
        """QR-based retraction (cheaper than exponential map)."""
        y = x + v
        q, _ = jnp.linalg.qr(y, mode="reduced")
        return q

    def log(self, x: Array, y: Array) -> Array:
        """Logarithmic map from X to Y (simplified implementation)."""
        # Simple implementation: project difference to tangent space
        return self.proj(x, y - x)

    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport from T_X to T_Y via projection."""
        return self.proj(y, v)

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Riemannian inner product is the Frobenius inner product."""
        return jnp.sum(u * v)

    def dist(self, x: Array, y: Array) -> Array:
        """Geodesic distance using principal angles."""
        # Handle the case when x and y are the same point
        if jnp.allclose(x, y, atol=1e-10):
            return 0.0

        # Compute principal angles via SVD
        u, s, _ = jnp.linalg.svd(x.T @ y, full_matrices=False)
        cos_theta = jnp.clip(s, 0.0, 1.0)

        # Avoid numerical issues with arccos near 1
        theta = jnp.where(cos_theta > 1.0 - 1e-10, 0.0, jnp.arccos(cos_theta))
        return jnp.linalg.norm(theta)

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
            def qr_fn(g):
                q, _ = jnp.linalg.qr(g, mode="reduced")
                return q

            return jnp.vectorize(qr_fn, signature="(n,p)->(n,p)")(gaussian)
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
            def proj_fn(vi):
                return self.proj(x, vi)

            return jnp.vectorize(proj_fn, signature="(n,p)->(n,p)")(v)
        else:
            return self.proj(x, v)

    def validate_point(self, x: Array, atol: float = 1e-6) -> Array:
        """Validate that X has orthonormal columns."""
        if x.shape != (self.n, self.p):
            return False

        # Check orthonormality: X^T X = I
        should_be_identity = x.T @ x
        identity = jnp.eye(self.p)
        return jnp.allclose(should_be_identity, identity, atol=atol)

    def validate_tangent(self, x: Array, v: Array, atol: float = 1e-6) -> Array:
        """Validate that V is in tangent space: X^T V = 0."""
        if not self.validate_point(x, atol):
            return False
        if v.shape != (self.n, self.p):
            return False

        # Check tangent space condition
        should_be_zero = x.T @ v
        return jnp.allclose(should_be_zero, 0.0, atol=atol)

    def __repr__(self) -> str:
        """Return string representation of Grassmann manifold."""
        return f"Grassmann({self.n}, {self.p})"
