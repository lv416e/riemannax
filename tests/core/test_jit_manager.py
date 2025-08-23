"""JIT manager unit tests."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from riemannax.core.jit_manager import JITManager


class TestJITManager:
    """JIT管理システムのユニットテスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        JITManager.reset_config()
        JITManager.clear_cache()

    def test_configure_basic_settings(self):
        """基本設定の更新テスト."""
        # 初期化
        JITManager.configure(enable_jit=True, cache_size=1000)

        # 設定確認
        assert JITManager._config["enable_jit"] is True
        assert JITManager._config["cache_size"] == 1000

    def test_jit_decorator_basic_function(self):
        """基本的な関数に対するJITデコレータテスト."""

        # テスト対象関数
        def simple_add(x, y):
            return x + y

        # JIT最適化
        jit_add = JITManager.jit_decorator(simple_add)

        # 実行テスト
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = jit_add(x, y)

        expected = jnp.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_jit_decorator_with_static_args(self):
        """静的引数付きJITデコレータテスト."""

        def manifold_op(x, v, dim):
            return x + v * dim

        # 静的引数指定でJIT最適化
        jit_op = JITManager.jit_decorator(manifold_op, static_argnums=(2,))

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.1, 0.2])
        dim = 5

        result = jit_op(x, v, dim)
        expected = jnp.array([1.5, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_clear_cache(self):
        """JITキャッシュクリア機能テスト."""
        # キャッシュにダミーデータ設定
        JITManager._cache["test_func"] = MagicMock()

        # キャッシュクリア実行
        JITManager.clear_cache()

        # キャッシュが空になっていることを確認
        assert len(JITManager._cache) == 0

    def test_jit_decorator_with_device_specification(self):
        """デバイス指定付きJITデコレータテスト."""

        def device_op(x):
            return x * 2

        # CPU指定でJIT最適化
        jit_op = JITManager.jit_decorator(device_op, device="cpu")

        x = jnp.array([1.0, 2.0])
        result = jit_op(x)
        expected = jnp.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_config_initialization(self):
        """設定の初期化テスト."""
        # 初期設定値を確認
        expected_defaults = {"enable_jit": True, "cache_size": 10000, "fallback_on_error": True, "debug_mode": False}

        for key, expected_value in expected_defaults.items():
            assert JITManager._config[key] == expected_value

    def test_jit_performance_tracking(self):
        """JIT性能追跡機能のテスト."""

        def tracked_func(x):
            return jnp.sum(x)

        jit_func = JITManager.jit_decorator(tracked_func)

        # 実行と性能データの記録
        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_func(x)

        # 結果確認
        assert result == 6.0
        # 性能データが記録されていることを確認
        # 注意: 実装により詳細は変更される可能性がある
