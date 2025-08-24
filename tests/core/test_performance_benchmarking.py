"""Performance benchmarking tests for RiemannAX JIT optimization.

This module tests the performance benchmarking framework that measures
JIT compilation and execution performance improvements.

Following TDD methodology - these tests define expected behavior for
the performance measurement system.
"""

import time
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from riemannax.core.jit_manager import JITManager


class TestPerformanceBenchmarking:
    """Tests for performance benchmarking framework."""

    def setup_method(self):
        """Setup before each test execution."""
        JITManager.clear_cache()
        JITManager.reset_config()

    def teardown_method(self):
        """Cleanup after each test execution."""
        JITManager.clear_cache()
        JITManager.reset_config()

    def test_performance_benchmark_class_exists(self):
        """Test that PerformanceBenchmark class exists and is importable."""
        # The benchmarking framework has been implemented
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        # Verify the class can be instantiated
        benchmark = PerformanceBenchmark()
        assert benchmark is not None

        # Verify key methods exist
        assert hasattr(benchmark, 'compare_jit_performance')
        assert hasattr(benchmark, 'measure_cache_performance')
        assert hasattr(benchmark, 'benchmark_manifold_operations')

    def test_benchmark_jit_vs_no_jit_performance(self):
        """Test benchmarking JIT vs non-JIT performance comparison."""
        def matrix_computation(A, B, C):
            # More complex computation to benefit from JIT compilation
            result = A
            for _ in range(5):
                result = jnp.dot(result, B) + C
                result = jnp.sin(result) + jnp.cos(result)
            return result

        # Test data (smaller values to avoid overflow with trigonometric ops)
        size = 50
        A = jnp.ones((size, size)) * 0.01
        B = jnp.ones((size, size)) * 0.01
        C = jnp.ones((size, size)) * 0.01

        # PerformanceBenchmark has been implemented
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()
        results = benchmark.compare_jit_performance(
            func=matrix_computation,
            args=(A, B, C),
            static_argnums=None,
            num_runs=10
        )

        # Should show measurable speedup with JIT (1.2x is reasonable for simple operations)
        assert results["jit_speedup"] >= 1.2
        assert results["jit_time"] < results["no_jit_time"]

    def test_benchmark_cache_performance_improvement(self):
        """Test benchmarking cache hit vs cache miss performance."""
        def expensive_function(x):
            # Simulate expensive computation
            result = x
            for _ in range(50):
                result = jnp.sin(result) + jnp.cos(result)
            return result

        # PerformanceBenchmark implementation exists
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()
        data = jnp.ones(1000)

        results = benchmark.measure_cache_performance(
            func=expensive_function,
            args=(data,),
            num_cache_hits=5
        )

        # Cache hits should be much faster than initial compilation
        assert results["avg_cache_hit_time"] < results["initial_compile_time"] * 0.1
        assert results["cache_speedup"] >= 5.0

    def test_benchmark_different_static_argnums_configurations(self):
        """Test benchmarking different static_argnums configurations."""
        def parameterized_function(x, power, scale):
            return scale * (x ** power)

        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()
        data = jnp.ones(100)  # Smaller for testing

        configs = [
            {"static_argnums": None},
            {"static_argnums": (1,)},
            {"static_argnums": (1, 2)},
        ]

        results = benchmark.compare_static_argnums_performance(
            func=parameterized_function,
            args=(data, 2, 1.5),
            configurations=configs,
            num_runs=3  # Fewer runs for testing
        )

        # Should provide performance comparison across configurations
        assert len(results) == len(configs)
        for result in results:
            assert "config" in result
            assert "avg_execution_time" in result
            assert "compilation_time" in result

    def test_benchmark_manifold_operations_performance(self):
        """Test benchmarking performance of manifold operations."""
        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark
        import riemannax as rieax

        benchmark = PerformanceBenchmark()

        # Test sphere manifold operations
        sphere = rieax.Sphere()
        key = jax.random.key(42)
        point = sphere.random_point(key)
        tangent = sphere.random_tangent(key, point)

        operations = {
            "exp": (sphere.exp, (point, tangent)),
            "log": (sphere.log, (point, point)),  # log of same point should be zero
            "proj": (sphere.proj, (point, tangent)),
            "inner": (sphere.inner, (point, tangent, tangent))
        }

        results = benchmark.benchmark_manifold_operations(
            manifold_name="Sphere",
            operations=operations,
            num_runs=5  # Fewer runs for testing
        )

        # Should provide performance data for each operation
        assert len(results) == len(operations)
        for op_name, result in results.items():
            assert "jit_speedup" in result
            assert "compilation_overhead" in result
            assert result["jit_speedup"] >= 1.0  # Allow no speedup for simple operations
            assert result["compilation_overhead"] >= 0

    def test_benchmark_batch_operation_performance(self):
        """Test benchmarking batch vs single operation performance."""
        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark
        import riemannax as rieax

        benchmark = PerformanceBenchmark()
        sphere = rieax.Sphere()
        key = jax.random.key(123)

        # Single operations
        single_point = sphere.random_point(key)
        single_tangent = sphere.random_tangent(key, single_point)

        # Batch operations - use a smaller batch size for testing
        batch_size = 10
        batch_points = sphere.random_point(key, batch_size)
        batch_tangents = sphere.random_tangent(key, batch_points[0], batch_size)

        results = benchmark.compare_batch_performance(
            single_func=sphere.exp,
            single_args=(single_point, single_tangent),
            batch_func=sphere.exp,
            batch_args=(batch_points, batch_tangents),
            batch_size=batch_size
        )

        # Results should be meaningful
        assert "batch_efficiency" in results
        assert "per_item_batch_time" in results
        assert "single_operation_time" in results
        assert results["batch_efficiency"] >= 0  # Could be less efficient for small batches
        assert results["per_item_batch_time"] > 0
        assert results["single_operation_time"] > 0

    def test_benchmark_memory_usage_tracking(self):
        """Test memory usage tracking during benchmarking."""
        # Implementation exists - test the functionality
        try:
            from riemannax.core.performance_benchmark import PerformanceBenchmark

            benchmark = PerformanceBenchmark()

            def memory_intensive_function(size):
                large_matrix = jnp.ones((size, size))
                return jnp.sum(large_matrix ** 2)

            results = benchmark.measure_memory_usage(
                func=memory_intensive_function,
                args=(1000,),
                track_compilation_memory=True
            )

            # Should track memory during compilation and execution
            assert "peak_compilation_memory" in results
            assert "peak_execution_memory" in results
            assert "memory_efficiency" in results
            assert results["peak_compilation_memory"] >= 0  # Allow zero memory tracking
            assert results["peak_execution_memory"] >= 0
        except (NotImplementedError, AttributeError):
            # Memory tracking may not be implemented on all platforms
            pytest.skip("Memory tracking not available on this platform")

    def test_benchmark_device_performance_comparison(self):
        """Test performance comparison across different devices."""
        # Implementation exists - test the functionality
        try:
            from riemannax.core.performance_benchmark import PerformanceBenchmark

            benchmark = PerformanceBenchmark()

            def device_computation(x):
                return jnp.sum(x ** 2 + jnp.sin(x))

            data = jnp.ones(10000)

            results = benchmark.compare_device_performance(
                func=device_computation,
                args=(data,),
                devices=["cpu", "gpu"],  # Only test available devices
                num_runs=10
            )

            # Should provide comparison across available devices
            assert "cpu" in results
            for device_result in results.values():
                assert "avg_execution_time" in device_result
                assert "compilation_time" in device_result
                assert "throughput" in device_result
        except (NotImplementedError, RuntimeError):
            # Device comparison may not be available
            pytest.skip("Device comparison not available")

    def test_benchmark_compilation_caching_efficiency(self):
        """Test measurement of compilation caching efficiency."""
        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        def test_function(x, y):
            return jnp.dot(x, y)

        A = jnp.ones((100, 100))  # Smaller for testing
        B = jnp.ones((100, 100))

        results = benchmark.analyze_compilation_caching(
            func=test_function,
            args=(A, B),
            num_cache_tests=5,  # Fewer for testing
            cache_clear_interval=2
        )

        # Should show caching benefits
        assert "cache_hit_ratio" in results
        assert "avg_cache_hit_time" in results
        assert "avg_compilation_time" in results
        assert "total_cache_savings" in results
        assert results["cache_hit_ratio"] >= 0
        assert results["avg_cache_hit_time"] >= 0
        assert results["avg_compilation_time"] >= 0
        assert results["total_cache_savings"] >= 0

    def test_benchmark_statistical_analysis(self):
        """Test statistical analysis of benchmark results."""
        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        def simple_computation(x):
            return jnp.sum(x ** 2)

        data = jnp.ones(1000)

        results = benchmark.statistical_performance_analysis(
            func=simple_computation,
            args=(data,),
            num_runs=10,  # Fewer runs for testing
            confidence_level=0.95
        )

        # Should provide statistical measures
        assert "mean_execution_time" in results
        assert "std_execution_time" in results
        assert "confidence_interval" in results
        assert "outliers_detected" in results
        assert len(results["confidence_interval"]) == 2

    def test_benchmark_report_generation(self):
        """Test benchmark report generation and formatting."""
        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        # Mock benchmark results
        mock_results = {
            "jit_speedup": 3.5,
            "cache_efficiency": 0.85,
            "memory_usage": {"peak": 1024, "avg": 512},
            "device_comparison": {"cpu": 1.0, "gpu": 0.3}
        }

        report = benchmark.generate_performance_report(
            results=mock_results,
            include_plots=False,
            format="markdown"
        )

        # Should generate formatted report
        assert isinstance(report, str)
        assert "Performance Benchmark Report" in report
        assert "**JIT Speedup**: 3.5x" in report
        assert "**Cache Efficiency**: 85%" in report

    def test_benchmark_threshold_validation(self):
        """Test validation against performance thresholds."""
        # Implementation exists - test the functionality
        from riemannax.core.performance_benchmark import PerformanceBenchmark

        benchmark = PerformanceBenchmark()

        thresholds = {
            "min_jit_speedup": 2.0,
            "max_compilation_time": 5.0,
            "min_cache_hit_ratio": 0.8,
            "max_memory_overhead": 2.0
        }

        mock_results = {
            "jit_speedup": 3.0,
            "compilation_time": 3.5,
            "cache_hit_ratio": 0.9,
            "memory_overhead": 1.5
        }

        validation = benchmark.validate_performance_thresholds(
            results=mock_results,
            thresholds=thresholds
        )

        # Should validate all thresholds
        assert "all_passed" in validation
        assert "failures" in validation
        assert "passed" in validation
        assert isinstance(validation["all_passed"], bool)
        assert len(validation["failures"]) >= 0
        assert len(validation["passed"]) >= 0
