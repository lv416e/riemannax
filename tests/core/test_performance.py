import time

from riemannax.core.performance import PerformanceMonitor


class TestPerformanceMonitor:
    """Unit tests for performance monitoring system."""

    def setup_method(self):
        """Setup before each test execution."""
        PerformanceMonitor.reset_metrics()

    def test_measure_context_manager(self):
        """Test time measurement context manager."""
        operation_name = "test_operation"

        with PerformanceMonitor.measure(operation_name):
            time.sleep(0.1)

        # Confirm measurement results are recorded
        metrics = PerformanceMonitor.get_metrics()
        assert operation_name in metrics
        assert len(metrics[operation_name]["execution_times"]) == 1
        assert metrics[operation_name]["execution_times"][0] >= 0.1

    def test_multiple_measurements(self):
        """Test multiple measurements."""
        operation_name = "repeated_operation"

        # Execute 3 measurements
        for _i in range(3):
            with PerformanceMonitor.measure(operation_name):
                time.sleep(0.05)

        metrics = PerformanceMonitor.get_metrics()
        assert len(metrics[operation_name]["execution_times"]) == 3
        for exec_time in metrics[operation_name]["execution_times"]:
            assert exec_time >= 0.05

    def test_compilation_time_recording(self):
        """Test compilation time recording."""
        func_name = "test_jit_function"
        compile_time = 1.234

        PerformanceMonitor.compilation_time(func_name, compile_time)

        metrics = PerformanceMonitor.get_metrics()
        assert func_name in metrics
        assert metrics[func_name]["compilation_time"] == compile_time

    def test_speedup_report_basic(self):
        """Test basic speedup report."""
        # Execution time before JIT optimization
        with PerformanceMonitor.measure("operation_no_jit"):
            time.sleep(0.1)

        # Execution time after JIT optimization (speedup simulation)
        with PerformanceMonitor.measure("operation_jit"):
            time.sleep(0.02)

        # Get speedup report
        report = PerformanceMonitor.get_speedup_report()

        # Check report structure
        assert "summary" in report
        assert "details" in report
        assert "total_operations" in report["summary"]

    def test_speedup_calculation(self):
        """Test speedup calculation."""
        # Baseline operation (slow)
        baseline_op = "slow_operation"
        with PerformanceMonitor.measure(baseline_op):
            time.sleep(0.1)

        # Optimized operation (fast)
        optimized_op = "fast_operation"
        with PerformanceMonitor.measure(optimized_op):
            time.sleep(0.02)

        # Calculate speedup
        speedup = PerformanceMonitor.calculate_speedup(baseline_op, optimized_op)

        # Expect about 5x speedup (with tolerance for system timing variations)
        assert speedup >= 3.5  # Slightly more lenient lower bound
        assert speedup <= 6.0

    def test_performance_threshold_check(self):
        """Test performance threshold check."""
        operation_name = "threshold_test"
        target_speedup = 2.0  # More realistic target value

        # Simulate fast operation (use shorter sleep for more reliable timing)
        with PerformanceMonitor.measure(operation_name):
            time.sleep(0.01)  # 10ms

        # Check threshold achievement
        achieved = PerformanceMonitor.check_performance_target(
            operation_name,
            baseline_time=0.05,  # Baseline time: 50ms
            target_speedup=target_speedup,  # Target: 2x speedup (need < 25ms)
        )

        # Should achieve 5x speedup (50ms -> 10ms), well above 2x target
        assert achieved is True

    def test_reset_metrics(self):
        """Test metrics initialization."""
        # Register data
        with PerformanceMonitor.measure("test_op"):
            time.sleep(0.01)

        PerformanceMonitor.compilation_time("test_func", 1.0)

        # Execute initialization
        PerformanceMonitor.reset_metrics()

        # Confirm data is cleared
        metrics = PerformanceMonitor.get_metrics()
        assert len(metrics) == 0

    def test_get_average_execution_time(self):
        """Test getting average execution time."""
        operation_name = "avg_test"

        # Multiple executions (different times)
        sleep_times = [0.01, 0.02, 0.03]
        for sleep_time in sleep_times:
            with PerformanceMonitor.measure(operation_name):
                time.sleep(sleep_time)

        # Get average execution time
        avg_time = PerformanceMonitor.get_average_execution_time(operation_name)

        # Confirm it's around the expected average time
        expected_avg = sum(sleep_times) / len(sleep_times)
        assert abs(avg_time - expected_avg) < 0.01

    def test_performance_statistics(self):
        """Test performance statistics information."""
        operation_name = "stats_test"

        # Multiple executions
        for i in range(5):
            with PerformanceMonitor.measure(operation_name):
                time.sleep(0.01 + i * 0.01)

        # Get statistics information
        stats = PerformanceMonitor.get_performance_statistics(operation_name)

        # Check statistics information structure
        assert "count" in stats
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats

        # Check execution count
        assert stats["count"] == 5
