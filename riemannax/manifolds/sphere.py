"""Implementation of the sphere manifold S^n with its Riemannian geometry.

This module provides operations for optimization on the unit sphere, a fundamental
manifold in Riemannian geometry with applications in directional statistics,
rotation representations, and constrained optimization.
"""

import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax

from .base import Manifold


class Sphere(Manifold):
    """Sphere manifold S^n embedded in R^(n+1) with the canonical Riemannian metric.

    The n-dimensional sphere S^n consists of all unit vectors in R^(n+1), i.e.,
    all points x ∈ R^(n+1) such that ||x|| = 1.
    """

    def proj(self, x, v):
        """Project vector v onto the tangent space of the sphere at point x.

        The tangent space at x consists of all vectors orthogonal to x.
        The projection removes the component of v parallel to x.

        Special case: if v is all zeros, this projects x onto the sphere (normalization).

        Args:
            x: Point on the sphere (unit vector).
            v: Vector in the ambient space R^(n+1).

        Returns:
            The orthogonal projection of v onto the tangent space at x,
            or the normalized x if v is zero (for manifold projection).
        """
        # Special case: if v is zero, project x onto sphere (normalization)
        v_norm = jnp.linalg.norm(v)
        is_zero_v = v_norm < 1e-10

        def normalize_x():
            # Project x onto the sphere by normalization
            x_norm = jnp.linalg.norm(x)
            return x / jnp.maximum(x_norm, 1e-10)

        def project_tangent():
            # Remove the component of v parallel to x
            return v - jnp.sum(x * v) * x

        return lax.cond(is_zero_v, normalize_x, project_tangent)

    def exp(self, x, v):
        """Compute the exponential map on the sphere.

        For the sphere, the exponential map corresponds to following a great circle
        in the direction of the tangent vector v.

        Args:
            x: Point on the sphere (unit vector).
            v: Tangent vector at x (orthogonal to x).

        Returns:
            The point on the sphere reached by following the geodesic from x in direction v.
        """
        # Compute the norm of the tangent vector
        v_norm = jnp.linalg.norm(v)
        # Handle numerical stability for small vectors
        safe_norm = jnp.maximum(v_norm, 1e-10)
        # Follow the great circle
        return jnp.cos(safe_norm) * x + jnp.sin(safe_norm) * v / safe_norm

    def log(self, x, y):
        """Compute the logarithmic map on the sphere.

        For two points x and y on the sphere, this finds the tangent vector v at x
        such that following the geodesic in that direction for distance ||v|| reaches y.

        Args:
            x: Starting point on the sphere (unit vector).
            y: Target point on the sphere (unit vector).

        Returns:
            The tangent vector at x that points toward y along the geodesic.
        """
        # Project y-x onto the tangent space at x
        v = self.proj(x, y - x)
        # Compute the norm of the projected vector
        v_norm = jnp.linalg.norm(v)
        # Handle numerical stability
        safe_norm = jnp.maximum(v_norm, 1e-10)
        # Compute the angle between x and y (geodesic distance)
        theta = jnp.arccos(jnp.clip(jnp.dot(x, y), -1.0, 1.0))
        # Scale the direction vector by the geodesic distance
        return theta * v / safe_norm

    def retr(self, x, v):
        """Compute the retraction on the sphere.

        For the sphere, a simple retraction is normalization of x + v.

        Args:
            x: Point on the sphere (unit vector).
            v: Tangent vector at x (orthogonal to x).

        Returns:
            The point on the sphere reached by the retraction.
        """
        # Simple retraction by normalization
        y = x + v
        return y / jnp.linalg.norm(y)

    def transp(self, x, y, v):
        """Parallel transport on the sphere from x to y.

        Parallel transport preserves the inner product and the norm of the vector.

        Args:
            x: Starting point on the sphere (unit vector).
            y: Target point on the sphere (unit vector).
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        # Get the tangent vector that takes x to y
        log_xy = self.log(x, y)
        log_xy_norm = jnp.linalg.norm(log_xy)

        # Handle the case when x and y are very close or antipodal
        is_small = log_xy_norm < 1e-10

        def small_case():
            # If x and y are close, approximate with projection
            return self.proj(y, v)

        def normal_case():
            # Normal case: compute parallel transport
            u = log_xy / log_xy_norm
            return v - (jnp.dot(v, u) / (1 + jnp.dot(x, y))) * (u + y)

        return lax.cond(is_small, small_case, normal_case)

    def inner(self, x, u, v):
        """Compute the Riemannian inner product on the sphere.

        On the sphere, the Riemannian metric is simply the Euclidean inner product
        in the ambient space restricted to the tangent space.

        Args:
            x: Point on the sphere (unit vector).
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x in the Riemannian metric.
        """
        return jnp.dot(u, v)

    def dist(self, x, y):
        """Compute the geodesic distance between points on the sphere.

        The geodesic distance is the length of the shortest path along the sphere,
        which is the arc length of the great circle connecting x and y.

        Args:
            x: First point on the sphere (unit vector).
            y: Second point on the sphere (unit vector).

        Returns:
            The geodesic distance between x and y.
        """
        # The geodesic distance is the arc length, which is the angle between x and y
        cos_angle = jnp.clip(jnp.dot(x, y), -1.0, 1.0)
        return jnp.arccos(cos_angle)

    def random_point(self, key, *shape):
        """Generate random point(s) on the sphere.

        Points are sampled uniformly from the sphere using the standard normal
        distribution in the ambient space followed by normalization.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) on the sphere with specified shape.
        """
        if not shape:
            # Default to generating a single point
            ambient_dim = 3  # Default to S^2 embedded in R^3
            shape = (ambient_dim,)

        # Sample from standard normal distribution
        samples = jr.normal(key, shape)

        # Normalize to get points on the sphere
        return samples / jnp.linalg.norm(samples, axis=-1, keepdims=True)

    def random_tangent(self, key, x, *shape):
        """Generate random tangent vector(s) at point x.

        Tangent vectors are sampled from a normal distribution in the ambient
        space and then projected onto the tangent space at x.

        Args:
            key: JAX PRNG key.
            x: Point on the sphere (unit vector).
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        if not shape:
            # Default to the shape of x
            shape = x.shape

        # Sample from standard normal distribution
        ambient_vectors = jr.normal(key, shape)

        # Project onto the tangent space at x
        tangent_vectors = self.proj(x, ambient_vectors)

        return tangent_vectors

    def validate_point(self, x, atol: float = 1e-6) -> bool:
        """Validate that x is a valid point on the sphere."""
        # Check that x is a unit vector
        norm = jnp.linalg.norm(x)
        return bool(jnp.allclose(norm, 1.0, atol=atol))

    def validate_tangent(self, x, v, atol: float = 1e-6) -> jnp.ndarray:
        """Validate that v is in the tangent space at x."""
        if not self.validate_point(x, atol):
            return jnp.array(False)
        # Check that v is orthogonal to x
        dot_product = jnp.dot(x, v)
        return jnp.allclose(dot_product, 0.0, atol=atol)

    @property
    def dimension(self) -> int:
        """Dimension of the sphere (n for S^n)."""
        return 2  # Default to S^2 (dimension 2)

    @property
    def ambient_dimension(self) -> int:
        """Ambient space dimension (n+1 for S^n)."""
        return 3  # Default to R^3 for S^2

    # JIT-optimized implementation methods

    def _proj_impl(self, x: Array, v: Array) -> Array:
        """JIT最適化版プロジェクション実装.

        数値安定性向上:
        - ゼロ除算防止のための安全な正規化
        - 小さなノルムに対する適切な処理
        - バッチ処理対応
        """
        # Special case: if v is zero, project x onto sphere (normalization)
        v_norm = jnp.linalg.norm(v, axis=-1)
        is_zero_v = v_norm < 1e-10

        # Normalize x onto sphere (batch-aware)
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        safe_x_norm = jnp.maximum(x_norm, 1e-10)  # ゼロ除算防止
        normalized_x = x / safe_x_norm

        # Project tangent (batch-aware)
        dot_product = jnp.sum(x * v, axis=-1, keepdims=True)
        # クリッピングによる数値安定化
        clipped_dot = jnp.clip(dot_product, -1e10, 1e10)
        tangent_projection = v - clipped_dot * x

        # Use jnp.where for batch-compatible conditional processing
        is_zero_v = jnp.expand_dims(is_zero_v, axis=-1)
        return jnp.where(is_zero_v, normalized_x, tangent_projection)

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """JIT最適化版指数写像実装.

        数値安定性向上:
        - 小さなベクトルに対する安全な処理
        - 三角関数の数値安定性確保
        """
        # Compute the norm of the tangent vector with enhanced stability
        v_norm = jnp.linalg.norm(v)

        # ゼロ除算防止のための安全なノルム
        safe_norm = jnp.maximum(v_norm, 1e-10)

        # 小さなベクトルの場合の特別処理
        is_small = v_norm < 1e-8

        def small_vector_case():
            # 小さなベクトルの場合は1次近似を使用し、球面制約を保持
            result = x + v
            # 球面制約を保持するため正規化
            result_norm = jnp.linalg.norm(result)
            safe_result_norm = jnp.maximum(result_norm, 1e-10)
            return result / safe_result_norm

        def normal_case():
            # 通常の指数写像計算
            cos_norm = jnp.cos(v_norm)
            sin_norm = jnp.sin(v_norm)
            return cos_norm * x + sin_norm * v / safe_norm

        return lax.cond(is_small, small_vector_case, normal_case)

    def _log_impl(self, x: Array, y: Array) -> Array:
        """JIT最適化版対数写像実装.

        数値安定性向上:
        - 内積のクリッピング
        - 対極点での安定な処理
        - バッチ処理対応
        """
        # Project y-x onto the tangent space at x
        diff = y - x
        v = self._proj_impl(x, diff)

        # Compute the norm of the projected vector with stability (batch-aware)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        safe_norm = jnp.maximum(v_norm, 1e-10)

        # Compute the angle between x and y with clipping for stability (batch-aware)
        dot_product = jnp.sum(x * y, axis=-1)  # Use sum for batch processing
        clipped_dot = jnp.clip(dot_product, -1.0 + 1e-10, 1.0 - 1e-10)
        theta = jnp.arccos(clipped_dot)

        # 対極点近傍での特別処理 (batch-aware)
        is_antipodal = jnp.abs(dot_product + 1.0) < 1e-8

        # Expand theta and is_antipodal for broadcasting
        theta = jnp.expand_dims(theta, axis=-1)
        is_antipodal = jnp.expand_dims(is_antipodal, axis=-1)

        # Use jnp.where for batch-compatible conditional processing
        antipodal_result = jnp.pi * v / safe_norm
        normal_result = theta * v / safe_norm

        return jnp.where(is_antipodal, antipodal_result, normal_result)

    def _inner_impl(self, x: Array, u: Array, v: Array) -> Array:
        """JIT最適化版内積実装.

        球面上では単純なユークリッド内積だが、
        数値安定性のためクリッピングを追加
        バッチ処理対応
        """
        # Use batch-aware dot product
        dot_product = jnp.sum(u * v, axis=-1)
        # 極値でのクリッピング(必要に応じて)
        return jnp.clip(dot_product, -1e10, 1e10)

    def _dist_impl(self, x: Array, y: Array) -> Array:
        """JIT最適化版距離計算実装.

        数値安定性向上:
        - 内積のクリッピング
        - arccos の定義域での安全な計算
        """
        # The geodesic distance is the arc length (angle between x and y)
        dot_product = jnp.dot(x, y)

        # クリッピングによる数値安定化 - arccos の定義域 [-1, 1] を確保
        clipped_dot = jnp.clip(dot_product, -1.0 + 1e-15, 1.0 - 1e-15)

        return jnp.arccos(clipped_dot)

    def _get_static_args(self, method_name: str) -> tuple:
        """JITコンパイル用の静的引数設定.

        Sphere多様体では次元が動的に変わる可能性があるため、
        現時点では静的引数は使用しない。

        将来的には次元数を静的引数として指定可能
        """
        # 現在の実装では静的引数なし
        # 将来的には dimension を静的引数として使用可能
        return ()
