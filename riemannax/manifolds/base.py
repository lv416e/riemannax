"""Abstract base classes for Riemannian manifold implementations.

This module defines the core interfaces for Riemannian manifolds, establishing
the contract that concrete manifold implementations must satisfy.
"""

import logging
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import Array

from ..core.performance import PerformanceMonitor

# JIT最適化関連のインポート
from ..core.safe_jit import SafeJITWrapper

logger = logging.getLogger(__name__)


class ManifoldError(Exception):
    """Base exception for manifold-related errors."""

    pass


class DimensionError(ManifoldError):
    """Exception for dimension mismatches."""

    pass


class Manifold:
    """Abstract base class for Riemannian manifolds.

    This class defines the essential operations required for optimization on
    Riemannian manifolds, including tangent space projections and exponential/logarithmic maps.

    Enhanced with JIT optimization support for high-performance computing.
    """

    def __init__(self):
        """Initialize manifold with JIT optimization support."""
        # JIT最適化関連の初期化
        self._jit_compiled_methods: dict[str, Callable] = {}
        self._safe_jit_wrapper = SafeJITWrapper(fallback_enabled=True)
        self._jit_enabled = True

        # JIT最適化の遅延初期化(最初のメソッド呼び出し時)
        self._jit_initialized = False

        # パフォーマンス監視
        self._performance_tracking = False

    def _compile_core_methods(self) -> None:
        """JITコアメソッドのコンパイル.

        各多様体の主要操作をJIT最適化してキャッシュに保存します。
        """
        try:
            # コンパイル対象メソッドのリスト
            core_methods = ["proj", "exp", "log", "transp", "inner", "dist"]

            for method_name in core_methods:
                impl_method_name = f"_{method_name}_impl"

                # 実装メソッドが存在する場合のみJITコンパイル
                if hasattr(self, impl_method_name):
                    impl_method = getattr(self, impl_method_name)
                    static_args = self._get_static_args(method_name)

                    # SafeJITWrapperを使用してJIT最適化
                    jit_method = self._safe_jit_wrapper.safe_jit(
                        impl_method, static_argnums=static_args if static_args else None
                    )

                    self._jit_compiled_methods[method_name] = jit_method
                    logger.debug(f"Compiled JIT method: {method_name}")

            self._jit_initialized = True
            logger.info(f"Compiled {len(self._jit_compiled_methods)} JIT methods for {self.__class__.__name__}")

        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
            # JITコンパイル失敗時はフォールバックを使用
            self._jit_enabled = False

    def _call_jit_method(self, method_name: str, *args, **kwargs) -> Array:
        """JIT最適化メソッドの統一呼び出し.

        Args:
            method_name: 呼び出すメソッド名
            *args: メソッド引数
            **kwargs: キーワード引数

        Returns:
            JIT最適化された結果
        """
        # JIT初期化の遅延実行
        if not self._jit_initialized:
            try:
                self._compile_core_methods()
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}")
                self._jit_enabled = False

        # JIT最適化版が利用可能な場合
        if self._jit_enabled and method_name in self._jit_compiled_methods:
            try:
                if self._performance_tracking:
                    with PerformanceMonitor.measure(f"{self.__class__.__name__}.{method_name}"):
                        return self._jit_compiled_methods[method_name](*args, **kwargs)
                else:
                    return self._jit_compiled_methods[method_name](*args, **kwargs)

            except Exception as e:
                logger.warning(f"JIT execution failed for {method_name}: {e}")
                # フォールバック実行

        # フォールバック: 実装メソッドの直接呼び出し
        impl_method_name = f"_{method_name}_impl"
        if hasattr(self, impl_method_name):
            impl_method = getattr(self, impl_method_name)
            return impl_method(*args, **kwargs)
        else:
            raise NotImplementedError(f"Neither JIT nor implementation method found for {method_name}")

    def _get_static_args(self, method_name: str) -> tuple[int, ...]:
        """静的引数インデックスの取得.

        Args:
            method_name: メソッド名

        Returns:
            静的引数のインデックスタプル
        """
        # デフォルト実装では静的引数なし
        # サブクラスでオーバーライドして次元数等を静的引数に設定
        return ()

    def clear_jit_cache(self) -> None:
        """JITキャッシュのクリア."""
        self._jit_compiled_methods.clear()
        self._jit_initialized = False
        logger.debug(f"Cleared JIT cache for {self.__class__.__name__}")

    def enable_performance_tracking(self, enabled: bool = True) -> None:
        """パフォーマンス追跡の有効化/無効化.

        Args:
            enabled: 追跡の有効化フラグ
        """
        self._performance_tracking = enabled

    def get_performance_report(self) -> dict[str, Any]:
        """パフォーマンスレポートの取得.

        Returns:
            パフォーマンス統計辞書
        """
        return PerformanceMonitor.get_speedup_report()

    def _reset_jit_cache(self) -> None:
        """JITキャッシュのリセット(テスト用)."""
        self.clear_jit_cache()

    def proj(self, x: Array, v: Array) -> Array:
        """Project a vector from ambient space to the tangent space at point x.

        Args:
            x: Point on the manifold.
            v: Vector in the ambient space to be projected.

        Returns:
            The projection of v onto the tangent space at x.
        """
        try:
            return self._call_jit_method("proj", x, v)
        except NotImplementedError:
            raise NotImplementedError("Subclasses must implement projection operation") from None

    def exp(self, x: Array, v: Array) -> Array:
        """Apply the exponential map to move from point x along tangent vector v.

        The exponential map takes a point x on the manifold and a tangent vector v at x,
        and returns the point on the manifold reached by following the geodesic in the
        direction of v for a distance of ||v||.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by following the geodesic from x in direction v.
        """
        try:
            return self._call_jit_method("exp", x, v)
        except NotImplementedError:
            raise NotImplementedError("Subclasses must implement exponential map") from None

    def log(self, x: Array, y: Array) -> Array:
        """Apply the logarithmic map to find the tangent vector that maps x to y.

        The logarithmic map is the inverse of the exponential map. It takes two points
        x and y on the manifold and returns the tangent vector v at x such that the
        exponential map of v at x gives y.

        Args:
            x: Starting point on the manifold.
            y: Target point on the manifold.

        Returns:
            The tangent vector v at x such that exp(x, v) = y.
        """
        try:
            return self._call_jit_method("log", x, y)
        except NotImplementedError:
            raise NotImplementedError("Subclasses must implement logarithmic map") from None

    def retr(self, x: Array, v: Array) -> Array:
        """Apply retraction to move from point x along tangent vector v.

        Retraction is a cheaper approximation of the exponential map that maintains
        essential properties for optimization algorithms.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by the retraction from x in direction v.
        """
        return self.exp(x, v)

    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport vector v from tangent space at x to tangent space at y.

        Parallel transport moves a tangent vector along a geodesic while preserving
        its length and angle with the geodesic.

        Args:
            x: Starting point on the manifold.
            y: Target point on the manifold.
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        try:
            return self._call_jit_method("transp", x, y, v)
        except NotImplementedError:
            raise NotImplementedError("Subclasses must implement parallel transport") from None

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Compute the Riemannian inner product between tangent vectors u and v at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x in the Riemannian metric.
        """
        try:
            return self._call_jit_method("inner", x, u, v)
        except NotImplementedError:
            raise NotImplementedError("Subclasses must implement Riemannian inner product") from None

    def dist(self, x: Array, y: Array) -> Array:
        """Compute the Riemannian distance between points x and y on the manifold.

        Args:
            x: First point on the manifold.
            y: Second point on the manifold.

        Returns:
            The geodesic distance between x and y.
        """
        v = self.log(x, y)
        return jnp.sqrt(self.inner(x, v, v))

    def norm(self, x: Array, v: Array) -> Array:
        """Compute the norm of tangent vector v at point x.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The norm ||v||_x in the Riemannian metric.
        """
        return jnp.sqrt(self.inner(x, v, v))

    def random_point(self, key: Array, *shape: int) -> Array:
        """Generate random point(s) on the manifold.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) on the manifold with specified shape.
        """
        raise NotImplementedError("Subclasses must implement random point generation")

    def random_tangent(self, key: Array, x: Array, *shape: int) -> Array:
        """Generate random tangent vector(s) at point x.

        Args:
            key: JAX PRNG key.
            x: Point on the manifold.
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        raise NotImplementedError("Subclasses must implement random tangent generation")

    def curvature_tensor(self, x: Array, u: Array, v: Array, w: Array) -> Array:
        """Compute the Riemann curvature tensor R(u,v)w at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.
            w: Third tangent vector at x.

        Returns:
            The curvature tensor R(u,v)w at x.
        """
        raise NotImplementedError("Curvature tensor computation not implemented")

    def sectional_curvature(self, x: Array, u: Array, v: Array) -> Array:
        """Compute the sectional curvature of the plane spanned by u and v at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The sectional curvature K(u,v) at x.
        """
        raise NotImplementedError("Sectional curvature computation not implemented")

    def injectivity_radius(self, x: Array) -> Array:
        """Compute the injectivity radius at point x.

        Args:
            x: Point on the manifold.

        Returns:
            The injectivity radius at x.
        """
        raise NotImplementedError("Injectivity radius computation not implemented")

    def validate_point(self, x: Array) -> bool:
        """Validate that x is a valid point on the manifold.

        Args:
            x: Point to validate.

        Returns:
            True if x is on the manifold, False otherwise.
        """
        raise NotImplementedError("Point validation not implemented")

    def validate_tangent(self, x: Array, v: Array, atol: float = 1e-6) -> Array:
        """Validate that v is a valid tangent vector at point x.

        Args:
            x: Point on the manifold.
            v: Vector to validate.
            atol: Absolute tolerance for validation.

        Returns:
            True if v is in the tangent space at x, False otherwise.
        """
        # Default implementation: check if v equals its projection
        proj_v = self.proj(x, v)
        return jnp.allclose(v, proj_v, atol=atol)

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the manifold."""
        raise NotImplementedError("Subclasses must define manifold dimension")

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the ambient space."""
        raise NotImplementedError("Subclasses must define ambient dimension")

    def __repr__(self) -> str:
        """String representation of the manifold."""
        return f"{self.__class__.__name__}()"
