"""Tests for the Performance Benchmarking Suite."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.performance_benchmark import (
    BenchmarkSummary,
    PerformanceBenchmark,
    PerformanceResult,
    run_quick_benchmark,
)


class TestPerformanceBenchmark:
    """Test the performance benchmarking system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = PerformanceBenchmark(output_dir=self.temp_dir)

    def test_benchmark_initialization(self):
        """Test benchmark system initialization."""
        assert self.benchmark is not None
        assert len(self.benchmark.manifolds) > 0
        assert len(self.benchmark.operations) > 0
        assert len(self.benchmark.batch_sizes) > 0

        # Check that all expected manifolds are present
        expected_manifolds = [
            "sphere_3d",
            "sphere_5d",
            "sphere_10d",
            "grassmann_5_3",
            "grassmann_10_5",
            "stiefel_5_3",
            "stiefel_10_5",
            "so_3",
            "so_5",
            "spd_3",
            "spd_5",
        ]

        for manifold in expected_manifolds:
            assert manifold in self.benchmark.manifolds
            assert manifold in self.benchmark.operations

    def test_get_manifold_dims(self):
        """Test dimension retrieval for different manifolds."""
        # Sphere manifolds
        assert self.benchmark.get_manifold_dims("sphere_3d") == {"dim": 3}
        assert self.benchmark.get_manifold_dims("sphere_5d") == {"dim": 5}
        assert self.benchmark.get_manifold_dims("sphere_10d") == {"dim": 10}

        # Grassmann manifolds
        assert self.benchmark.get_manifold_dims("grassmann_5_3") == {"n": 5, "p": 3}
        assert self.benchmark.get_manifold_dims("grassmann_10_5") == {"n": 10, "p": 5}

        # SO manifolds
        assert self.benchmark.get_manifold_dims("so_3") == {"n": 3}
        assert self.benchmark.get_manifold_dims("so_5") == {"n": 5}

        # SPD manifolds
        assert self.benchmark.get_manifold_dims("spd_3") == {"n": 3}
        assert self.benchmark.get_manifold_dims("spd_5") == {"n": 5}

    def test_generate_test_data_sphere(self):
        """Test test data generation for sphere manifolds."""
        test_data = self.benchmark.generate_test_data("sphere_3d", batch_size=5)

        assert "x" in test_data
        assert "v" in test_data
        assert test_data["x"].shape == (5, 3)
        assert test_data["v"].shape == (5, 3)

    def test_generate_test_data_matrix_manifolds(self):
        """Test test data generation for matrix manifolds."""
        # Test Grassmann manifold
        test_data = self.benchmark.generate_test_data("grassmann_5_3", batch_size=3)
        assert "x" in test_data
        assert "v" in test_data
        assert test_data["x"].shape == (3, 5, 3)
        assert test_data["v"].shape == (3, 5, 3)

        # Test single sample
        test_data = self.benchmark.generate_test_data("grassmann_5_3", batch_size=1)
        assert test_data["x"].shape == (5, 3)
        assert test_data["v"].shape == (5, 3)

    def test_measure_execution_time(self):
        """Test execution time measurement."""

        # Simple function that takes some time
        def test_func():
            return np.sum(np.ones(1000))

        exec_time, comp_time = self.benchmark.measure_execution_time(test_func, warmup_runs=1, measurement_runs=3)

        assert exec_time > 0
        assert comp_time is not None
        assert comp_time > 0

    def test_benchmark_single_operation_sphere(self):
        """Test benchmarking a single operation on sphere manifold."""
        result = self.benchmark.benchmark_single_operation("sphere_3d", "exp", batch_size=5)

        assert isinstance(result, PerformanceResult)
        assert result.manifold_name == "sphere_3d"
        assert result.operation == "exp"
        assert result.batch_size == 5
        assert result.jit_time_ms >= 0
        assert result.nojit_time_ms >= 0
        assert result.speedup >= 0
        assert result.input_shape == (5, 3)

    def test_benchmark_single_operation_invalid(self):
        """Test benchmarking with invalid operation."""
        result = self.benchmark.benchmark_single_operation("sphere_3d", "invalid_operation", batch_size=5)

        assert result.jit_time_ms == float("inf")
        assert result.nojit_time_ms == float("inf")
        assert result.speedup == 0.0

    @pytest.mark.parametrize("manifold_name", ["sphere_3d", "grassmann_5_3", "stiefel_5_3", "so_3", "spd_3"])
    def test_benchmark_all_operations(self, manifold_name):
        """Test benchmarking all operations for various manifolds."""
        operations = self.benchmark.operations[manifold_name]

        for operation in operations:
            result = self.benchmark.benchmark_single_operation(manifold_name, operation, batch_size=2)

            assert isinstance(result, PerformanceResult)
            assert result.manifold_name == manifold_name
            assert result.operation == operation
            assert result.batch_size == 2

    def test_run_comprehensive_benchmark_small(self):
        """Test running a small comprehensive benchmark."""
        results = self.benchmark.run_comprehensive_benchmark(
            manifolds=["sphere_3d"], operations=["exp", "proj"], batch_sizes=[1, 5]
        )

        assert len(results) == 4  # 1 manifold x 2 operations x 2 batch sizes

        for result in results:
            assert isinstance(result, PerformanceResult)
            assert result.manifold_name == "sphere_3d"
            assert result.operation in ["exp", "proj"]
            assert result.batch_size in [1, 5]

    def test_generate_benchmark_summary_empty(self):
        """Test summary generation with empty results."""
        summary = self.benchmark.generate_benchmark_summary([])

        assert isinstance(summary, BenchmarkSummary)
        assert summary.total_benchmarks == 0
        assert summary.avg_speedup == 0.0
        assert summary.manifolds_tested == []
        assert summary.operations_tested == []

    def test_generate_benchmark_summary_with_data(self):
        """Test summary generation with data."""
        # Create mock results
        results = [
            PerformanceResult(
                manifold_name="sphere_3d",
                operation="exp",
                batch_size=5,
                jit_time_ms=1.0,
                nojit_time_ms=2.0,
                speedup=2.0,
                compilation_time_ms=5.0,
                memory_usage_mb=None,
                input_shape=(5, 3),
            ),
            PerformanceResult(
                manifold_name="sphere_3d",
                operation="proj",
                batch_size=5,
                jit_time_ms=0.5,
                nojit_time_ms=2.0,
                speedup=4.0,
                compilation_time_ms=3.0,
                memory_usage_mb=None,
                input_shape=(5, 3),
            ),
        ]

        summary = self.benchmark.generate_benchmark_summary(results)

        assert summary.total_benchmarks == 2
        assert summary.avg_speedup == 3.0  # (2.0 + 4.0) / 2
        assert summary.max_speedup == 4.0
        assert summary.min_speedup == 2.0
        assert summary.avg_jit_time_ms == 0.75  # (1.0 + 0.5) / 2
        assert summary.avg_compilation_time_ms == 4.0  # (5.0 + 3.0) / 2
        assert summary.total_time_saved_ms == 2.5  # (2.0-1.0) + (2.0-0.5)
        assert "sphere_3d" in summary.manifolds_tested
        assert "exp" in summary.operations_tested
        assert "proj" in summary.operations_tested

    def test_generate_detailed_report_empty(self):
        """Test detailed report generation with empty results."""
        report = self.benchmark.generate_detailed_report([])

        assert "No benchmark results available." in report

    def test_generate_detailed_report_with_data(self):
        """Test detailed report generation with data."""
        # Run a small benchmark first
        results = self.benchmark.run_comprehensive_benchmark(
            manifolds=["sphere_3d"], operations=["exp"], batch_sizes=[1]
        )

        report = self.benchmark.generate_detailed_report(results)

        assert "COMPREHENSIVE PERFORMANCE BENCHMARK REPORT" in report
        assert "SUMMARY STATISTICS" in report
        assert "SPHERE_3D" in report.upper()

    def test_performance_result_serialization(self):
        """Test PerformanceResult serialization."""
        result = PerformanceResult(
            manifold_name="sphere_3d",
            operation="exp",
            batch_size=5,
            jit_time_ms=1.0,
            nojit_time_ms=2.0,
            speedup=2.0,
            compilation_time_ms=5.0,
            memory_usage_mb=10.5,
            input_shape=(5, 3),
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["manifold_name"] == "sphere_3d"
        assert result_dict["operation"] == "exp"
        assert result_dict["speedup"] == 2.0

    def test_benchmark_summary_serialization(self):
        """Test BenchmarkSummary serialization."""
        summary = BenchmarkSummary(
            total_benchmarks=10,
            avg_speedup=3.5,
            max_speedup=5.0,
            min_speedup=2.0,
            avg_jit_time_ms=1.5,
            avg_compilation_time_ms=4.0,
            total_time_saved_ms=15.0,
            manifolds_tested=["sphere_3d"],
            operations_tested=["exp", "proj"],
        )

        summary_dict = summary.to_dict()

        assert isinstance(summary_dict, dict)
        assert summary_dict["total_benchmarks"] == 10
        assert summary_dict["avg_speedup"] == 3.5
        assert summary_dict["manifolds_tested"] == ["sphere_3d"]

    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        # Create some test results
        results = [
            PerformanceResult(
                manifold_name="sphere_3d",
                operation="exp",
                batch_size=1,
                jit_time_ms=1.0,
                nojit_time_ms=2.0,
                speedup=2.0,
                compilation_time_ms=5.0,
                memory_usage_mb=None,
                input_shape=(1, 3),
            )
        ]

        self.benchmark.results = results

        # Save results
        self.benchmark.save_results("test_results.json")

        # Verify file was created
        filepath = Path(self.temp_dir) / "test_results.json"
        assert filepath.exists()

        # Clear results and load
        self.benchmark.results = []
        self.benchmark.load_results("test_results.json")

        # Verify results were loaded
        assert len(self.benchmark.results) == 1
        loaded_result = self.benchmark.results[0]
        assert loaded_result.manifold_name == "sphere_3d"
        assert loaded_result.operation == "exp"
        assert loaded_result.speedup == 2.0

    def test_run_quick_benchmark(self):
        """Test the quick benchmark utility function."""
        report = run_quick_benchmark(manifolds=["sphere_3d"], batch_sizes=[1, 5])

        assert isinstance(report, str)
        assert "COMPREHENSIVE PERFORMANCE BENCHMARK REPORT" in report
        assert "SPHERE_3D" in report.upper()

    def test_benchmark_with_compilation_errors(self):
        """Test benchmark behavior with compilation errors."""
        # This is a basic test - in practice, we'd need to simulate compilation failures
        result = self.benchmark.benchmark_single_operation("sphere_3d", "exp", batch_size=1)

        # Even if there are issues, we should get a valid result object
        assert isinstance(result, PerformanceResult)

    def test_memory_monitor_context(self):
        """Test memory monitoring context manager."""
        with self.benchmark.memory_monitor() as memory_usage:
            # Simple operation
            _ = np.sum(np.ones(100))

        # Currently returns None, but should not raise errors
        assert memory_usage is None

    def test_batch_size_scaling(self):
        """Test that different batch sizes produce appropriate results."""
        small_result = self.benchmark.benchmark_single_operation("sphere_3d", "exp", batch_size=1)
        large_result = self.benchmark.benchmark_single_operation("sphere_3d", "exp", batch_size=10)

        # Both should be valid results
        assert isinstance(small_result, PerformanceResult)
        assert isinstance(large_result, PerformanceResult)

        # Input shapes should reflect batch sizes
        assert small_result.input_shape == (1, 3)
        assert large_result.input_shape == (10, 3)

    def test_all_manifold_types_supported(self):
        """Test that all manifold types can be benchmarked."""
        manifold_types = ["sphere_3d", "grassmann_5_3", "stiefel_5_3", "so_3", "spd_3"]

        for manifold_name in manifold_types:
            # Test at least one operation for each manifold type
            result = self.benchmark.benchmark_single_operation(manifold_name, "exp", batch_size=1)

            assert isinstance(result, PerformanceResult)
            assert result.manifold_name == manifold_name

    @pytest.mark.skip(reason="Skipping performance validation")
    def test_benchmark_consistency_across_runs(self):
        """Test that benchmark results are consistent across runs."""
        # Run the same benchmark twice
        result1 = self.benchmark.benchmark_single_operation("sphere_3d", "exp", batch_size=5)
        result2 = self.benchmark.benchmark_single_operation("sphere_3d", "exp", batch_size=5)

        # Results should be reasonably close (within 50% due to JIT variability)
        if result1.speedup > 0 and result2.speedup > 0:
            ratio = max(result1.speedup, result2.speedup) / min(result1.speedup, result2.speedup)
            assert ratio < 2.0  # Results should be within 2x of each other
