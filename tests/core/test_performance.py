import time

from riemannax.core.performance import PerformanceMonitor


class TestPerformanceMonitor:
    """パフォーマンス監視システムのユニットテスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        PerformanceMonitor.reset_metrics()

    def test_measure_context_manager(self):
        """時間測定コンテキストマネージャーのテスト."""
        operation_name = "test_operation"

        with PerformanceMonitor.measure(operation_name):
            time.sleep(0.1)

        # 測定結果が記録されていることを確認
        metrics = PerformanceMonitor.get_metrics()
        assert operation_name in metrics
        assert len(metrics[operation_name]["execution_times"]) == 1
        assert metrics[operation_name]["execution_times"][0] >= 0.1

    def test_multiple_measurements(self):
        """複数回測定のテスト."""
        operation_name = "repeated_operation"

        # 3回測定実行
        for _i in range(3):
            with PerformanceMonitor.measure(operation_name):
                time.sleep(0.05)

        metrics = PerformanceMonitor.get_metrics()
        assert len(metrics[operation_name]["execution_times"]) == 3
        for exec_time in metrics[operation_name]["execution_times"]:
            assert exec_time >= 0.05

    def test_compilation_time_recording(self):
        """コンパイル時間記録のテスト."""
        func_name = "test_jit_function"
        compile_time = 1.234

        PerformanceMonitor.compilation_time(func_name, compile_time)

        metrics = PerformanceMonitor.get_metrics()
        assert func_name in metrics
        assert metrics[func_name]["compilation_time"] == compile_time

    def test_speedup_report_basic(self):
        """基本的な速度向上レポートのテスト."""
        # JIT最適化前の実行時間
        with PerformanceMonitor.measure("operation_no_jit"):
            time.sleep(0.1)

        # JIT最適化後の実行時間(高速化シミュレーション)
        with PerformanceMonitor.measure("operation_jit"):
            time.sleep(0.02)

        # 速度向上レポート取得
        report = PerformanceMonitor.get_speedup_report()

        # レポート構造の確認
        assert "summary" in report
        assert "details" in report
        assert "total_operations" in report["summary"]

    def test_speedup_calculation(self):
        """速度向上計算のテスト."""
        # 基準操作(遅い)
        baseline_op = "slow_operation"
        with PerformanceMonitor.measure(baseline_op):
            time.sleep(0.1)

        # 最適化操作(速い)
        optimized_op = "fast_operation"
        with PerformanceMonitor.measure(optimized_op):
            time.sleep(0.02)

        # 速度向上計算
        speedup = PerformanceMonitor.calculate_speedup(baseline_op, optimized_op)

        # 5倍程度の速度向上を期待
        assert speedup >= 4.0
        assert speedup <= 6.0

    def test_performance_threshold_check(self):
        """パフォーマンス閾値チェックのテスト."""
        operation_name = "threshold_test"
        target_speedup = 2.0  # より現実的な目標値

        # 高速操作をシミュレート
        with PerformanceMonitor.measure(operation_name):
            time.sleep(0.02)

        # 閾値達成チェック
        achieved = PerformanceMonitor.check_performance_target(
            operation_name,
            baseline_time=0.05,  # 基準時間: 50ms
            target_speedup=target_speedup,  # 目標: 2倍高速化
        )

        # 約2.5倍速度向上を達成している想定 (50ms -> 20ms)
        assert achieved is True

    def test_reset_metrics(self):
        """メトリクス初期化のテスト."""
        # データ登録
        with PerformanceMonitor.measure("test_op"):
            time.sleep(0.01)

        PerformanceMonitor.compilation_time("test_func", 1.0)

        # 初期化実行
        PerformanceMonitor.reset_metrics()

        # データがクリアされていることを確認
        metrics = PerformanceMonitor.get_metrics()
        assert len(metrics) == 0

    def test_get_average_execution_time(self):
        """平均実行時間取得のテスト."""
        operation_name = "avg_test"

        # 複数回実行(異なる時間)
        sleep_times = [0.01, 0.02, 0.03]
        for sleep_time in sleep_times:
            with PerformanceMonitor.measure(operation_name):
                time.sleep(sleep_time)

        # 平均実行時間取得
        avg_time = PerformanceMonitor.get_average_execution_time(operation_name)

        # 期待される平均時間周辺であることを確認
        expected_avg = sum(sleep_times) / len(sleep_times)
        assert abs(avg_time - expected_avg) < 0.01

    def test_performance_statistics(self):
        """パフォーマンス統計情報のテスト."""
        operation_name = "stats_test"

        # 複数回実行
        for i in range(5):
            with PerformanceMonitor.measure(operation_name):
                time.sleep(0.01 + i * 0.01)

        # 統計情報取得
        stats = PerformanceMonitor.get_performance_statistics(operation_name)

        # 統計情報の構造確認
        assert "count" in stats
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats

        # 実行回数の確認
        assert stats["count"] == 5
