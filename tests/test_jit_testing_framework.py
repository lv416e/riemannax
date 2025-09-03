"""Tests for the JIT Testing Framework."""

import jax
import jax.numpy as jnp
import pytest

from tests.utils.jit_testing import JITTestFramework, quick_jit_test, quick_performance_comparison


class TestJITTestFramework:
    """Test the JIT testing framework functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = JITTestFramework()

    def test_framework_initialization(self):
        """Test that the framework initializes correctly."""
        assert self.framework is not None
        assert len(self.framework.test_manifolds) == 5
        assert "sphere" in self.framework.test_manifolds
        assert "grassmann" in self.framework.test_manifolds
        assert "stiefel" in self.framework.test_manifolds
        assert "so" in self.framework.test_manifolds
        assert "spd" in self.framework.test_manifolds

    def test_compare_jit_vs_nojit_sphere(self):
        """Test JIT vs non-JIT comparison for sphere manifold."""
        manifold = self.framework.test_manifolds["sphere"]
        key = jax.random.key(42)

        x = manifold.random_point(key, 5)
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}

        result = self.framework.compare_jit_vs_nojit(manifold, "exp", test_data)

        assert result is not None
        assert isinstance(result.max_absolute_error, float)
        assert isinstance(result.passed, bool)
        assert result.max_absolute_error >= 0
        assert result.tolerance_used > 0

    def test_performance_measurement_sphere(self):
        """Test performance measurement for sphere manifold."""
        manifold = self.framework.test_manifolds["sphere"]
        key = jax.random.key(42)

        x = manifold.random_point(key, 4)
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}

        result = self.framework.performance_measurement_detailed(manifold, "exp", test_data, runs=2, warmup_runs=1)

        assert result is not None
        assert result.jit_time > 0
        assert result.nojit_time > 0
        assert result.speedup > 0
        assert result.compilation_time is not None

    def test_batch_consistency_test_sphere(self):
        """Test batch consistency for sphere manifold."""
        manifold = self.framework.test_manifolds["sphere"]
        batch_sizes = [5, 10]

        results = self.framework.batch_consistency_test(manifold, "exp", batch_sizes)

        assert len(results) == len(batch_sizes)
        for batch_size in batch_sizes:
            assert batch_size in results
            result = results[batch_size]
            assert result.max_absolute_error >= 0
            assert isinstance(result.passed, bool)

    def test_comprehensive_manifold_test_sphere(self):
        """Test comprehensive manifold testing for sphere."""
        results = self.framework.comprehensive_manifold_test(
            "sphere", include_performance=False, include_batch_tests=True
        )

        assert results["manifold"] == "sphere"
        assert "consistency_tests" in results
        assert "batch_tests" in results

        # Check that some operations were tested
        consistency_tests = results["consistency_tests"]
        assert len(consistency_tests) > 0

        # Check that exp operation was tested (should be in all manifolds)
        if "exp" in consistency_tests:
            exp_result = consistency_tests["exp"]
            if exp_result is not None:
                assert hasattr(exp_result, "passed")

    def test_comprehensive_manifold_test_grassmann(self):
        """Test comprehensive manifold testing for Grassmann."""
        results = self.framework.comprehensive_manifold_test(
            "grassmann", include_performance=False, include_batch_tests=False
        )

        assert results["manifold"] == "grassmann"
        assert "consistency_tests" in results

        # Should have tested some operations
        assert len(results["consistency_tests"]) > 0

    def test_generate_performance_report_empty(self):
        """Test performance report generation with no data."""
        report = self.framework.generate_performance_report()
        assert "No performance data available" in report

    def test_generate_performance_report_with_data(self):
        """Test performance report generation with data."""
        # Run a simple performance test to populate history
        manifold = self.framework.test_manifolds["sphere"]
        key = jax.random.key(42)

        x = manifold.random_point(key, 3)
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}

        self.framework.performance_measurement_detailed(manifold, "exp", test_data, runs=1, warmup_runs=0)

        report = self.framework.generate_performance_report()
        assert "JIT Optimization Performance Report" in report
        assert "Average speedup" in report
        assert "Detailed Results" in report

    def test_get_consistency_summary_empty(self):
        """Test consistency summary with no data."""
        summary = self.framework.get_consistency_summary()
        assert "No consistency data available" in summary["message"]

    def test_get_consistency_summary_with_data(self):
        """Test consistency summary with data."""
        # Run a consistency test to populate history
        manifold = self.framework.test_manifolds["sphere"]
        key = jax.random.key(42)

        x = manifold.random_point(key, 3)
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}

        self.framework.compare_jit_vs_nojit(manifold, "exp", test_data)

        summary = self.framework.get_consistency_summary()
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "pass_rate" in summary
        assert summary["total_tests"] > 0

    def test_clear_history(self):
        """Test clearing test history."""
        # Add some data first
        manifold = self.framework.test_manifolds["sphere"]
        key = jax.random.key(42)

        x = manifold.random_point(key, 3)
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}

        self.framework.compare_jit_vs_nojit(manifold, "exp", test_data)
        self.framework.performance_measurement_detailed(manifold, "exp", test_data, runs=1)

        # Verify data exists
        assert len(self.framework._consistency_history) > 0
        assert len(self.framework._performance_history) > 0

        # Clear history
        self.framework.clear_history()

        # Verify data is cleared
        assert len(self.framework._consistency_history) == 0
        assert len(self.framework._performance_history) == 0

    def test_quick_jit_test(self):
        """Test the quick JIT test convenience function."""
        results = quick_jit_test("sphere")

        assert results["manifold"] == "sphere"
        assert "consistency_tests" in results
        assert len(results["consistency_tests"]) > 0

    def test_quick_performance_comparison(self):
        """Test the quick performance comparison convenience function."""
        result = quick_performance_comparison("sphere", "exp")

        assert result is not None
        assert result.jit_time > 0
        assert result.nojit_time > 0
        assert result.speedup > 0

    def test_invalid_manifold_name(self):
        """Test error handling for invalid manifold names."""
        with pytest.raises(ValueError, match="Unsupported manifold"):
            self.framework.comprehensive_manifold_test("invalid_manifold")

    def test_performance_measurement_context_manager(self):
        """Test the performance measurement context manager."""
        with self.framework.performance_measurement("test_operation", warmup_runs=0) as ctx:
            # Simulate some operation
            _ = jnp.sum(jnp.ones(100))

        # Check results after context manager finishes
        assert "execution_time" in ctx
        assert "operation" in ctx
        assert ctx["operation"] == "test_operation"
        assert ctx["execution_time"] >= 0

    def test_manifold_configs_completeness(self):
        """Test that all manifolds have proper configurations."""
        for manifold_name in self.framework.test_manifolds:
            config = self.framework.manifold_configs[manifold_name]

            # Check required fields
            assert "dims" in config
            assert "test_operations" in config
            assert "batch_sizes" in config
            assert "tolerance" in config

            # Check tolerance format
            tolerance = config["tolerance"]
            assert "rtol" in tolerance
            assert "atol" in tolerance
            assert tolerance["rtol"] > 0
            assert tolerance["atol"] > 0

            # Check operations are valid
            operations = config["test_operations"]
            assert len(operations) > 0
            assert "exp" in operations or "proj" in operations  # Should have at least one basic operation
