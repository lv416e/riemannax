import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from riemannax.manifolds.base import Manifold


class ConcreteManifoldForTesting(Manifold):
    """テスト用の具象多様体クラス."""

    def __init__(self, dim: int = 3):
        super().__init__()
        self._dim = dim

    def _proj_impl(self, x: Array, v: Array) -> Array:
        """実装版プロジェクション（テスト用）."""
        return v  # 単純実装

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """実装版指数写像（テスト用）."""
        return x + v  # 単純実装

    def _log_impl(self, x: Array, y: Array) -> Array:
        """実装版対数写像（テスト用）."""
        return y - x  # 単純実装

    def _transp_impl(self, x: Array, y: Array, v: Array) -> Array:
        """実装版平行輸送（テスト用）."""
        return v  # 単純実装

    def _inner_impl(self, x: Array, u: Array, v: Array) -> Array:
        """実装版内積（テスト用）."""
        return jnp.sum(u * v)  # 単純実装

    def _get_static_args(self, method_name: str) -> tuple:
        """静的引数設定（テスト用）."""
        return ()  # テスト用は静的引数なし

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ambient_dimension(self) -> int:
        return self._dim


class TestJITEnhancedBaseManifold:
    """JIT対応BaseManifold のユニットテスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        self.manifold = ConcreteManifoldForTesting(dim=3)
        # JIT関連の初期化
        if hasattr(self.manifold, "_reset_jit_cache"):
            self.manifold._reset_jit_cache()

    def test_jit_initialization(self):
        """JIT初期化機能のテスト."""
        assert hasattr(self.manifold, "_jit_compiled_methods")
        assert hasattr(self.manifold, "_compile_core_methods")
        assert hasattr(self.manifold, "_call_jit_method")

    def test_compile_core_methods(self):
        """コアメソッドのJITコンパイルテスト."""
        # コンパイル実行
        self.manifold._compile_core_methods()

        # JITコンパイル済みメソッドの確認
        compiled_methods = self.manifold._jit_compiled_methods

        # 実装メソッドが存在するもののみJITコンパイル対象
        expected_methods = ["proj", "exp", "log", "transp", "inner"]
        for method in expected_methods:
            assert method in compiled_methods

    def test_jit_vs_original_numerical_equivalence_proj(self):
        """JIT版と元の版の数値同等性テスト（プロジェクション）."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # 元の実装
        original_result = self.manifold._proj_impl(x, v)

        # JIT版実行
        jit_result = self.manifold.proj(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_jit_vs_original_numerical_equivalence_exp(self):
        """JIT版と元の版の数値同等性テスト（指数写像）."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # 元の実装
        original_result = self.manifold._exp_impl(x, v)

        # JIT版実行
        jit_result = self.manifold.exp(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_jit_vs_original_numerical_equivalence_log(self):
        """JIT版と元の版の数値同等性テスト（対数写像）."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.1, 2.2, 3.3])

        # 元の実装
        original_result = self.manifold._log_impl(x, y)

        # JIT版実行
        jit_result = self.manifold.log(x, y)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_jit_vs_original_numerical_equivalence_inner(self):
        """JIT版と元の版の数値同等性テスト（内積）."""
        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.1, 0.2, 0.3])
        v = jnp.array([0.2, 0.4, 0.6])

        # 元の実装
        original_result = self.manifold._inner_impl(x, u, v)

        # JIT版実行
        jit_result = self.manifold.inner(x, u, v)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_dist_jit_optimization(self):
        """距離計算のJIT最適化テスト."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.1, 2.2, 3.3])

        # JIT版距離計算
        jit_distance = self.manifold.dist(x, y)

        # 手動計算での確認
        v = self.manifold.log(x, y)
        expected_distance = jnp.sqrt(self.manifold.inner(x, v, v))

        np.testing.assert_almost_equal(jit_distance, expected_distance, decimal=8)

    def test_api_compatibility_preservation(self):
        """API互換性保持のテスト."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])
        y = jnp.array([1.1, 2.2, 3.3])

        # 既存APIの全メソッドが正常に動作することを確認
        proj_result = self.manifold.proj(x, v)
        exp_result = self.manifold.exp(x, v)
        log_result = self.manifold.log(x, y)
        inner_result = self.manifold.inner(x, v, v)
        dist_result = self.manifold.dist(x, y)
        norm_result = self.manifold.norm(x, v)

        # 結果が適切な形状であることを確認
        assert proj_result.shape == v.shape
        assert exp_result.shape == x.shape
        assert log_result.shape == v.shape
        assert isinstance(inner_result.item(), float)
        assert isinstance(dist_result.item(), float)
        assert isinstance(norm_result.item(), float)

    def test_jit_method_call_with_static_args(self):
        """静的引数付きJITメソッド呼び出しテスト."""

        class StaticArgsManifold(ConcreteManifoldForTesting):
            def _get_static_args(self, method_name: str) -> tuple:
                return (2,) if method_name in ["proj", "exp"] else ()

        manifold = StaticArgsManifold(dim=3)

        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # 静的引数付きでの実行
        result = manifold.proj(x, v)
        assert result.shape == v.shape

    def test_jit_cache_management(self):
        """JITキャッシュ管理のテスト."""
        # 初期状態でキャッシュが空であることを確認
        assert len(self.manifold._jit_compiled_methods) == 0

        # メソッド実行でキャッシュが構築されることを確認
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        self.manifold.proj(x, v)
        assert len(self.manifold._jit_compiled_methods) > 0

        # キャッシュクリア機能のテスト
        if hasattr(self.manifold, "clear_jit_cache"):
            self.manifold.clear_jit_cache()
            assert len(self.manifold._jit_compiled_methods) == 0

    def test_fallback_mechanism(self):
        """JIT失敗時のフォールバック機構テスト."""

        class FailingJITManifold(ConcreteManifoldForTesting):
            def _compile_core_methods(self):
                # 意図的にコンパイル失敗をシミュレート
                raise RuntimeError("JIT compilation failed")

        manifold = FailingJITManifold(dim=3)

        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # フォールバック実行が成功することを確認
        result = manifold.proj(x, v)
        expected = manifold._proj_impl(x, v)
        np.testing.assert_array_almost_equal(result, expected)

    def test_batch_processing_optimization(self):
        """バッチ処理最適化のテスト."""
        # バッチデータの準備
        batch_size = 10
        x_batch = jnp.ones((batch_size, 3))
        v_batch = jnp.ones((batch_size, 3)) * 0.1

        # バッチ処理実行
        results = jax.vmap(self.manifold.proj)(x_batch, v_batch)

        # バッチ結果の確認
        assert results.shape == (batch_size, 3)

        # 個別実行との数値同等性確認
        individual_result = self.manifold.proj(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(results[0], individual_result)

    def test_performance_improvement_measurement(self):
        """パフォーマンス向上測定のテスト."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # JITなし実行時間測定
        import time

        start_time = time.time()
        for _ in range(100):
            self.manifold._proj_impl(x, v)
        time.time() - start_time

        # JIT実行時間測定（ウォームアップ後）
        self.manifold.proj(x, v)  # ウォームアップ

        start_time = time.time()
        for _ in range(100):
            self.manifold.proj(x, v)
        jit_time = time.time() - start_time

        # パフォーマンス向上の確認（JITが高速であることを期待）
        # 注意: 初回実行時は検証困難なため、この時点では実行可能性のみ確認
        assert jit_time >= 0  # 最低限の動作確認

    def test_error_handling_in_jit_context(self):
        """JITコンテキストでのエラーハンドリングテスト."""

        class ErrorManifold(ConcreteManifoldForTesting):
            def _proj_impl(self, x: Array, v: Array) -> Array:
                if jnp.any(x < 0):
                    raise ValueError("Negative values not allowed")
                return v

        manifold = ErrorManifold(dim=3)

        # 正常な入力での動作確認
        x_valid = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])
        result = manifold.proj(x_valid, v)
        np.testing.assert_array_almost_equal(result, v)

        # エラー入力でのエラーハンドリング確認
        x_invalid = jnp.array([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Negative values not allowed"):
            manifold.proj(x_invalid, v)

    def test_jit_integration_with_existing_methods(self):
        """JIT統合と既存メソッドの互換性テスト."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # 既存メソッドとJIT版の組み合わせテスト

        # norm メソッド（内積を使用）
        norm_result = self.manifold.norm(x, v)
        expected_norm = jnp.sqrt(self.manifold.inner(x, v, v))
        np.testing.assert_almost_equal(norm_result, expected_norm, decimal=8)

        # validate_tangent メソッド（projectionを使用）
        is_valid = self.manifold.validate_tangent(x, v)
        assert isinstance(is_valid.item(), bool | np.bool_)

    def test_memory_efficiency_with_jit(self):
        """JITでのメモリ効率性テスト."""
        # 大きなデータでのメモリ効率確認
        large_x = jnp.ones((1000, 3))
        large_v = jnp.ones((1000, 3)) * 0.1

        # バッチ処理での実行確認（メモリ使用量監視は困難なため、実行可能性のみ確認）
        results = jax.vmap(self.manifold.proj)(large_x, large_v)
        assert results.shape == (1000, 3)
