"""Safe JIT wrapper unit tests."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from riemannax.core.safe_jit import SafeJITWrapper


class TestSafeJITWrapper:
    """安全JIT実行システムのユニットテスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        SafeJITWrapper.reset_failure_logs()

    def test_safe_jit_successful_execution(self):
        """正常なJIT実行のテスト."""

        def simple_add(x, y):
            return x + y

        wrapper = SafeJITWrapper()
        safe_func = wrapper.safe_jit(simple_add)

        # 正常実行
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = safe_func(x, y)

        expected = jnp.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_jit_with_fallback_on_error(self):
        """JIT失敗時のフォールバック実行テスト."""

        def problematic_function(x):
            # JITコンパイルで問題を起こす可能性がある関数
            if hasattr(x, "at"):
                raise RuntimeError("Simulated JIT compilation error")
            return x * 2

        def fallback_function(x):
            return x * 2

        wrapper = SafeJITWrapper(fallback_enabled=True)
        safe_func = wrapper.safe_jit(problematic_function, fallback_func=fallback_function)

        # フォールバック実行が期待される
        x = jnp.array([1.0, 2.0])
        result = safe_func(x)

        expected = jnp.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    @patch("jax.jit")
    def test_jit_compilation_failure_with_mock(self, mock_jit):
        """JITコンパイル失敗のモックテスト."""

        def test_func(x):
            return x * 3

        def fallback_func(x):
            return x * 3

        # JITコンパイル失敗をシミュレート
        mock_jit_func = MagicMock()
        mock_jit_func.side_effect = RuntimeError("JIT compilation failed")
        mock_jit.return_value = mock_jit_func

        wrapper = SafeJITWrapper(fallback_enabled=True)
        safe_func = wrapper.safe_jit(test_func, fallback_func=fallback_func)

        # フォールバック実行
        x = jnp.array([2.0, 4.0])
        result = safe_func(x)

        expected = jnp.array([6.0, 12.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_failure_report_generation(self):
        """失敗レポート生成のテスト."""

        def failing_function(x):
            raise ValueError("Test error for failure report")

        wrapper = SafeJITWrapper(fallback_enabled=False)
        safe_func = wrapper.safe_jit(failing_function)

        # 失敗実行
        x = jnp.array([1.0])
        with pytest.raises(Exception):
            safe_func(x)

        # 失敗レポート取得
        failure_report = wrapper.get_failure_report()

        assert "total_failures" in failure_report
        assert "recent_failures" in failure_report
        assert failure_report["total_failures"] > 0

    def test_max_retries_configuration(self):
        """最大リトライ数設定のテスト."""
        retry_count = 0

        def flaky_function(x):
            nonlocal retry_count
            retry_count += 1
            if retry_count <= 2:  # 最初の2回は失敗
                raise RuntimeError("Flaky error")
            return x * 2

        wrapper = SafeJITWrapper(max_retries=3, fallback_enabled=False)
        safe_func = wrapper.safe_jit(flaky_function)

        # リトライ後に成功することを期待
        x = jnp.array([1.0, 2.0])
        result = safe_func(x)

        expected = jnp.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)
        assert retry_count == 3  # 3回目で成功

    def test_static_args_with_safe_jit(self):
        """静的引数付きSafe JITのテスト."""

        def manifold_operation(x, v, dim):
            return x + v * dim

        wrapper = SafeJITWrapper()
        safe_func = wrapper.safe_jit(manifold_operation, static_argnums=(2,))

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.1, 0.2])
        dim = 5

        result = safe_func(x, v, dim)
        expected = jnp.array([1.5, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fallback_disabled_raises_exception(self):
        """フォールバック無効時の例外発生テスト."""

        def always_failing_func(x):
            raise RuntimeError("Always fails")

        wrapper = SafeJITWrapper(fallback_enabled=False)
        safe_func = wrapper.safe_jit(always_failing_func)

        x = jnp.array([1.0])
        with pytest.raises(RuntimeError, match="Always fails"):
            safe_func(x)

    def test_compilation_time_tracking(self):
        """コンパイル時間追跡のテスト."""

        def tracked_function(x):
            return jnp.sum(x)

        wrapper = SafeJITWrapper(track_compilation_time=True)
        safe_func = wrapper.safe_jit(tracked_function)

        # 初回実行(コンパイル発生)
        x = jnp.array([1.0, 2.0, 3.0])
        safe_func(x)

        # コンパイル時間が記録されているかチェック
        compilation_stats = wrapper.get_compilation_statistics()
        assert "total_compilations" in compilation_stats
        assert compilation_stats["total_compilations"] > 0

    def test_error_categorization(self):
        """エラー分類機能のテスト."""

        def compilation_error_func(x):
            raise RuntimeError("XLA compilation failed")

        def memory_error_func(x):
            raise MemoryError("Out of GPU memory")

        def type_error_func(x):
            raise TypeError("Invalid argument type")

        wrapper = SafeJITWrapper()

        # 様々なエラーで実行
        for func, _expected_category in [
            (compilation_error_func, "compilation"),
            (memory_error_func, "memory"),
            (type_error_func, "type"),
        ]:
            safe_func = wrapper.safe_jit(func, fallback_func=lambda x: x)
            try:
                safe_func(jnp.array([1.0]))
            except:
                pass  # エラーは期待される

        # エラー分類レポート取得
        failure_report = wrapper.get_failure_report()
        assert "error_categories" in failure_report

    def test_performance_degradation_detection(self):
        """性能劣化検出のテスト."""

        def fast_function(x):
            return x * 2

        def slow_function(x):
            # 意図的に遅い処理をシミュレート
            time.sleep(0.1)
            return x * 2

        wrapper = SafeJITWrapper(performance_monitoring=True, fallback_enabled=True)

        # 高速関数とフォールバックを設定
        safe_func = wrapper.safe_jit(fast_function, fallback_func=slow_function)

        x = jnp.array([1.0, 2.0])
        safe_func(x)

        # 性能統計取得
        perf_stats = wrapper.get_performance_statistics()
        assert "execution_times" in perf_stats

    def test_context_manager_interface(self):
        """コンテキストマネージャーインターフェースのテスト."""

        def test_function(x):
            return x + 1

        with SafeJITWrapper(fallback_enabled=True) as wrapper:
            safe_func = wrapper.safe_jit(test_function)

            x = jnp.array([1.0, 2.0])
            result = safe_func(x)

            expected = jnp.array([2.0, 3.0])
            np.testing.assert_array_almost_equal(result, expected)

        # コンテキスト終了後の状態確認
        assert wrapper.is_closed()

    def test_debug_mode_detailed_logging(self):
        """デバッグモード詳細ログのテスト."""

        def debug_function(x):
            return jnp.exp(x)

        wrapper = SafeJITWrapper(debug_mode=True)
        safe_func = wrapper.safe_jit(debug_function)

        x = jnp.array([0.0, 1.0])
        safe_func(x)

        # デバッグ情報取得
        debug_info = wrapper.get_debug_info()
        assert "jit_trace_info" in debug_info
        assert "execution_details" in debug_info
