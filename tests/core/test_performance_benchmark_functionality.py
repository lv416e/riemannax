"""Working tests for PerformanceBenchmark functionality.

This module tests the implemented PerformanceBenchmark class functionality
with realistic expectations and thresholds.
"""

import jax
import jax.numpy as jnp

from riemannax.core.performance_benchmark import PerformanceBenchmark


class TestPerformanceBenchmarkFunctionality:
    """Tests for implemented PerformanceBenchmark functionality."""

    def setup_method(self):
        """Setup before each test execution."""
        from riemannax.core.jit_manager import JITManager

        JITManager.clear_cache()
        JITManager.reset_config()

    def test_class_instantiation(self):
        """Test that PerformanceBenchmark class can be instantiated."""
        benchmark = PerformanceBenchmark()
        assert benchmark is not None
        assert hasattr(benchmark, "compare_jit_performance")
        assert hasattr(benchmark, "measure_cache_performance")

    def test_jit_performance_comparison(self):
        """Test JIT vs non-JIT performance comparison works."""

        def simple_computation(x):
            return jnp.sum(x**2)

        benchmark = PerformanceBenchmark()
        data = jnp.ones(100)

        results = benchmark.compare_jit_performance(
            func=simple_computation, args=(data,), static_argnums=None, num_runs=3
        )

        # Check result structure
        assert "jit_speedup" in results
        assert "compilation_time" in results
        assert "jit_time" in results
        assert "no_jit_time" in results

        # Basic performance expectations
        assert results["jit_speedup"] > 0
        assert results["compilation_time"] >= 0
        assert results["jit_time"] > 0
        assert results["no_jit_time"] > 0

    def test_cache_performance_measurement(self):
        """Test cache performance measurement functionality."""

        def test_function(x):
            return jnp.sum(x**2)

        benchmark = PerformanceBenchmark()
        data = jnp.ones(50)

        results = benchmark.measure_cache_performance(func=test_function, args=(data,), num_cache_hits=3)

        # Check result structure
        assert "initial_compile_time" in results
        assert "avg_cache_hit_time" in results
        assert "cache_speedup" in results

        # Basic expectations
        assert results["initial_compile_time"] >= 0
        assert results["avg_cache_hit_time"] >= 0
        assert results["cache_speedup"] >= 0

    def test_static_argnums_comparison(self):
        """Test static_argnums configuration comparison."""

        def parameterized_func(x, power):
            return x**power

        benchmark = PerformanceBenchmark()
        data = jnp.ones(50)

        configs = [{"static_argnums": None}, {"static_argnums": (1,)}]

        results = benchmark.compare_static_argnums_performance(
            func=parameterized_func, args=(data, 2), configurations=configs, num_runs=2
        )

        assert len(results) == 2
        for result in results:
            assert "config" in result
            assert "compilation_time" in result
            assert "avg_execution_time" in result

    def test_manifold_operations_benchmarking(self):
        """Test manifold operations benchmarking."""
        import riemannax as rieax

        benchmark = PerformanceBenchmark()
        sphere = rieax.Sphere()
        jax.random.key(42)

        # Smaller data for faster testing
        point = jnp.array([1.0, 0.0, 0.0])
        tangent = jnp.array([0.0, 0.1, 0.0])

        operations = {"proj": (sphere.proj, (point, tangent))}

        results = benchmark.benchmark_manifold_operations(manifold_name="Sphere", operations=operations, num_runs=2)

        assert "proj" in results
        assert "jit_speedup" in results["proj"]
        assert "compilation_overhead" in results["proj"]

    def test_memory_usage_tracking(self):
        """Test memory usage tracking functionality."""

        def memory_function(x):
            return jnp.sum(x**2)

        benchmark = PerformanceBenchmark()
        data = jnp.ones(100)

        results = benchmark.measure_memory_usage(func=memory_function, args=(data,), track_compilation_memory=True)

        assert "peak_compilation_memory" in results
        assert "peak_execution_memory" in results
        assert "memory_efficiency" in results
        assert results["peak_compilation_memory"] >= 0
        assert results["peak_execution_memory"] >= 0

    def test_caching_analysis(self):
        """Test compilation caching analysis."""

        def cache_test_func(x):
            return jnp.sum(x**2)

        benchmark = PerformanceBenchmark()
        data = jnp.ones(50)

        results = benchmark.analyze_compilation_caching(
            func=cache_test_func, args=(data,), num_cache_tests=6, cache_clear_interval=2
        )

        assert "cache_hit_ratio" in results
        assert "avg_cache_hit_time" in results
        assert "avg_compilation_time" in results
        assert 0 <= results["cache_hit_ratio"] <= 1

    def test_statistical_analysis(self):
        """Test statistical performance analysis."""

        def stats_func(x):
            return jnp.sum(x**2)

        benchmark = PerformanceBenchmark()
        data = jnp.ones(50)

        results = benchmark.statistical_performance_analysis(
            func=stats_func, args=(data,), num_runs=10, confidence_level=0.95
        )

        assert "mean_execution_time" in results
        assert "std_execution_time" in results
        assert "confidence_interval" in results
        assert "outliers_detected" in results
        assert len(results["confidence_interval"]) == 2

    def test_report_generation(self):
        """Test performance report generation."""
        benchmark = PerformanceBenchmark()

        mock_results = {
            "jit_speedup": 2.5,
            "cache_efficiency": 0.8,
            "memory_usage": {"peak": 1024, "avg": 512},
            "device_comparison": {"cpu": 1.0},
        }

        report = benchmark.generate_performance_report(results=mock_results, output_format="markdown")

        assert isinstance(report, str)
        assert "Performance Benchmark Report" in report
        assert "2.5x" in report

    def test_threshold_validation(self):
        """Test performance threshold validation."""
        benchmark = PerformanceBenchmark()

        results = {"jit_speedup": 3.0, "compilation_time": 2.0, "cache_hit_ratio": 0.9}

        thresholds = {"min_jit_speedup": 2.0, "max_compilation_time": 3.0, "min_cache_hit_ratio": 0.8}

        validation = benchmark.validate_performance_thresholds(results=results, thresholds=thresholds)

        assert "all_passed" in validation
        assert "passed" in validation
        assert "failures" in validation
        assert validation["all_passed"] is True
