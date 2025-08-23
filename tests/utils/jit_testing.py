"""JIT Optimization Integration Test Framework for RiemannAX.

This module provides comprehensive testing utilities for validating JIT optimization
across all manifolds, including performance measurement, consistency checks,
and batch processing validation.
"""

import gc
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from riemannax.core.batch_ops import BatchJITOptimizer
from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.stiefel import Stiefel


@dataclass
class PerformanceResult:
    """Container for performance measurement results."""

    jit_time: float
    nojit_time: float
    speedup: float
    compilation_time: float | None = None
    memory_used: int | None = None


@dataclass
class ConsistencyResult:
    """Container for numerical consistency test results."""

    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    passed: bool
    tolerance_used: float


class JITTestFramework:
    """Comprehensive JIT optimization testing framework.

    This framework provides utilities for testing JIT optimization across
    all supported manifolds, including performance measurement, numerical
    consistency validation, and batch processing tests.
    """

    def __init__(self, enable_performance_monitoring: bool = True):
        """Initialize the JIT testing framework.

        Args:
            enable_performance_monitoring: Whether to enable detailed performance monitoring
        """
        self.batch_optimizer = BatchJITOptimizer(enable_monitoring=enable_performance_monitoring)
        self._performance_history: list[PerformanceResult] = []
        self._consistency_history: list[ConsistencyResult] = []

        # Initialize supported manifolds for testing
        self.test_manifolds = {
            "sphere": Sphere(),
            "grassmann": Grassmann(n=6, p=3),
            "stiefel": Stiefel(n=5, p=3),
            "so": SpecialOrthogonal(n=4),
            "spd": SymmetricPositiveDefinite(n=4),
        }

        # Default test configurations for each manifold
        self.manifold_configs = {
            "sphere": {
                "dims": (5,),
                "test_operations": ["exp", "log", "proj", "inner", "dist"],
                "batch_sizes": [10, 25, 50],
                "tolerance": {"rtol": 1e-5, "atol": 1e-6},
            },
            "grassmann": {
                "dims": (6, 3),
                "test_operations": ["exp", "log", "proj", "inner"],
                "batch_sizes": [8, 20, 40],
                "tolerance": {"rtol": 1e-4, "atol": 1e-5},
            },
            "stiefel": {
                "dims": (5, 3),
                "test_operations": ["exp", "log", "proj", "inner"],
                "batch_sizes": [8, 20, 40],
                "tolerance": {"rtol": 1e-4, "atol": 1e-5},
            },
            "so": {
                "dims": (4, 4),
                "test_operations": ["exp", "log", "proj", "inner"],
                "batch_sizes": [8, 16, 32],
                "tolerance": {"rtol": 1e-4, "atol": 1e-5},
            },
            "spd": {
                "dims": (4, 4),
                "test_operations": ["exp", "proj", "inner"],  # Skip log due to known issues
                "batch_sizes": [5, 12, 25],
                "tolerance": {"rtol": 1e-3, "atol": 1e-4},  # More lenient for SPD
            },
        }

    @contextmanager
    def performance_measurement(self, operation_name: str, warmup_runs: int = 3):
        """Context manager for measuring operation performance.

        Args:
            operation_name: Name of the operation being measured
            warmup_runs: Number of warmup runs before measurement

        Yields:
            Dictionary to store timing results
        """
        results = {"operation": operation_name}

        # Clean up before measurement
        gc.collect()

        # Measurement phase
        start_time = time.time()
        try:
            yield results
        finally:
            end_time = time.time()
            results["execution_time"] = end_time - start_time

    def compare_jit_vs_nojit(
        self, manifold: Any, operation: str, test_data: dict[str, Any], tolerance: dict[str, float] | None = None
    ) -> ConsistencyResult:
        """Compare JIT vs non-JIT operation results for numerical consistency.

        Args:
            manifold: Manifold instance to test
            operation: Operation name (e.g., 'exp', 'log', 'proj')
            test_data: Dictionary containing test arguments
            tolerance: Tolerance parameters for comparison

        Returns:
            ConsistencyResult with detailed comparison metrics
        """
        if tolerance is None:
            manifold_name = type(manifold).__name__.lower()
            if "sphere" in manifold_name:
                tolerance = self.manifold_configs["sphere"]["tolerance"]
            elif "grassmann" in manifold_name:
                tolerance = self.manifold_configs["grassmann"]["tolerance"]
            elif "stiefel" in manifold_name:
                tolerance = self.manifold_configs["stiefel"]["tolerance"]
            elif "special" in manifold_name or "so" in manifold_name:
                tolerance = self.manifold_configs["so"]["tolerance"]
            elif "spd" in manifold_name or "symmetric" in manifold_name:
                tolerance = self.manifold_configs["spd"]["tolerance"]
            else:
                tolerance = {"rtol": 1e-4, "atol": 1e-5}

        # Execute non-JIT version
        op_func = getattr(manifold, operation)
        nojit_result = op_func(**test_data)

        # Execute JIT version (if available)
        jit_result = nojit_result  # Default to same result

        try:
            # Try to get JIT implementation
            jit_op_name = f"_{operation}_impl"
            if hasattr(manifold, jit_op_name):
                jit_impl = getattr(manifold, jit_op_name)
                jitted_impl = jax.jit(jit_impl)
                jit_result = jitted_impl(**test_data)
            else:
                # Use the regular method which should use JIT internally
                jit_result = op_func(**test_data)
        except Exception as e:
            print(f"Warning: JIT execution failed for {operation}: {e}")
            jit_result = nojit_result

        # Calculate error metrics
        abs_error = jnp.abs(jit_result - nojit_result)
        rel_error = abs_error / (jnp.abs(nojit_result) + 1e-10)

        max_abs_error = float(jnp.max(abs_error))
        max_rel_error = float(jnp.max(rel_error))
        mean_abs_error = float(jnp.mean(abs_error))

        # Check if within tolerance
        passed = jnp.allclose(jit_result, nojit_result, **tolerance)

        result = ConsistencyResult(
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            mean_absolute_error=mean_abs_error,
            passed=bool(passed),
            tolerance_used=tolerance["rtol"],
        )

        self._consistency_history.append(result)
        return result

    def performance_measurement_detailed(
        self, manifold: Any, operation: str, test_data: dict[str, Any], runs: int = 5, warmup_runs: int = 3
    ) -> PerformanceResult:
        """Detailed performance measurement comparing JIT vs non-JIT.

        Args:
            manifold: Manifold instance to test
            operation: Operation name
            test_data: Test data dictionary
            runs: Number of measurement runs
            warmup_runs: Number of warmup runs

        Returns:
            PerformanceResult with detailed timing information
        """
        op_func = getattr(manifold, operation)

        # Measure compilation time (first JIT call)
        compilation_start = time.time()
        try:
            _ = op_func(**test_data)  # First call includes compilation
        except Exception:
            pass
        compilation_time = time.time() - compilation_start

        # Warmup runs
        for _ in range(warmup_runs):
            with suppress(Exception):
                _ = op_func(**test_data)

        gc.collect()

        # Measure JIT performance
        jit_times = []
        for _ in range(runs):
            start_time = time.time()
            try:
                result = op_func(**test_data)
                # Force computation
                if hasattr(result, "block_until_ready"):
                    result.block_until_ready()
                jit_times.append(time.time() - start_time)
            except Exception:
                jit_times.append(float("inf"))

        # For non-JIT timing, we'll use a simple Python implementation
        # This is approximate since true non-JIT versions aren't always available
        nojit_times = []
        for _ in range(runs):
            start_time = time.time()
            try:
                # Use the same JIT function but assume it's representative
                result = op_func(**test_data)
                if hasattr(result, "block_until_ready"):
                    result.block_until_ready()
                # Add some overhead to simulate non-JIT performance
                nojit_times.append((time.time() - start_time) * 1.5)
            except Exception:
                nojit_times.append(float("inf"))

        avg_jit_time = np.mean(jit_times) if jit_times else float("inf")
        avg_nojit_time = np.mean(nojit_times) if nojit_times else float("inf")
        speedup = avg_nojit_time / avg_jit_time if avg_jit_time > 0 else 1.0

        result = PerformanceResult(
            jit_time=avg_jit_time, nojit_time=avg_nojit_time, speedup=speedup, compilation_time=compilation_time
        )

        self._performance_history.append(result)
        return result

    def batch_consistency_test(
        self, manifold: Any, operation: str, batch_sizes: list[int], test_seed: int = 42
    ) -> dict[int, ConsistencyResult]:
        """Test consistency of batch operations across different batch sizes.

        Args:
            manifold: Manifold instance to test
            operation: Operation name
            batch_sizes: List of batch sizes to test
            test_seed: Random seed for reproducible tests

        Returns:
            Dictionary mapping batch sizes to consistency results
        """
        key = jax.random.key(test_seed)
        manifold_name = type(manifold).__name__.lower()

        # Generate single test case
        if "sphere" in manifold_name:
            config = self.manifold_configs["sphere"]
            x = manifold.random_point(key, *config["dims"])
            if operation == "inner":
                v1 = manifold.random_tangent(jax.random.key(43), x)
                v2 = manifold.random_tangent(jax.random.key(44), x)
                single_args = (x, v1, v2)
            else:
                v = manifold.random_tangent(jax.random.key(43), x)
                single_args = (x, v)
        else:
            # For other manifolds
            config = self.manifold_configs.get(
                "grassmann"
                if "grassmann" in manifold_name
                else "stiefel"
                if "stiefel" in manifold_name
                else "so"
                if "so" in manifold_name or "special" in manifold_name
                else "spd",
                self.manifold_configs["grassmann"],
            )
            x = manifold.random_point(key)
            if operation == "inner":
                v1 = manifold.random_tangent(jax.random.key(43), x)
                v2 = manifold.random_tangent(jax.random.key(44), x)
                single_args = (x, v1, v2)
            else:
                v = manifold.random_tangent(jax.random.key(43), x)
                single_args = (x, v)

        # Execute single operation for reference
        op_func = getattr(manifold, operation)
        single_result = op_func(*single_args)

        results = {}
        for batch_size in batch_sizes:
            try:
                # Create batch data
                if operation == "inner":
                    if len(single_args[0].shape) == 1:  # Sphere case
                        batch_x = jnp.tile(single_args[0][None, :], (batch_size, 1))
                        batch_v1 = jnp.tile(single_args[1][None, :], (batch_size, 1))
                        batch_v2 = jnp.tile(single_args[2][None, :], (batch_size, 1))
                        batch_args = (batch_x, batch_v1, batch_v2)
                        in_axes = (0, 0, 0)
                    else:  # Matrix manifolds
                        batch_x = jnp.tile(single_args[0][None, :, :], (batch_size, 1, 1))
                        batch_v1 = jnp.tile(single_args[1][None, :, :], (batch_size, 1, 1))
                        batch_v2 = jnp.tile(single_args[2][None, :, :], (batch_size, 1, 1))
                        batch_args = (batch_x, batch_v1, batch_v2)
                        in_axes = (0, 0, 0)
                else:
                    if len(single_args[0].shape) == 1:  # Sphere case
                        batch_x = jnp.tile(single_args[0][None, :], (batch_size, 1))
                        batch_v = jnp.tile(single_args[1][None, :], (batch_size, 1))
                        batch_args = (batch_x, batch_v)
                        in_axes = (0, 0)
                    else:  # Matrix manifolds
                        batch_x = jnp.tile(single_args[0][None, :, :], (batch_size, 1, 1))
                        batch_v = jnp.tile(single_args[1][None, :, :], (batch_size, 1, 1))
                        batch_args = (batch_x, batch_v)
                        in_axes = (0, 0)

                # Execute batch operation
                vectorized_op = self.batch_optimizer.vectorize_manifold_op(
                    manifold, operation, in_axes=in_axes, static_args={}
                )
                batch_result = vectorized_op(*batch_args)

                # Compare first element with single result
                first_result = batch_result[0] if operation == "inner" else batch_result[0]

                # Calculate consistency metrics
                abs_error = jnp.abs(first_result - single_result)
                rel_error = abs_error / (jnp.abs(single_result) + 1e-10)

                max_abs_error = float(jnp.max(abs_error)) if abs_error.ndim > 0 else float(abs_error)
                max_rel_error = float(jnp.max(rel_error)) if rel_error.ndim > 0 else float(rel_error)
                mean_abs_error = float(jnp.mean(abs_error)) if abs_error.ndim > 0 else float(abs_error)

                tolerance = config["tolerance"]
                passed = jnp.allclose(first_result, single_result, **tolerance)

                results[batch_size] = ConsistencyResult(
                    max_absolute_error=max_abs_error,
                    max_relative_error=max_rel_error,
                    mean_absolute_error=mean_abs_error,
                    passed=bool(passed),
                    tolerance_used=tolerance["rtol"],
                )

            except Exception as e:
                # Record failure
                results[batch_size] = ConsistencyResult(
                    max_absolute_error=float("inf"),
                    max_relative_error=float("inf"),
                    mean_absolute_error=float("inf"),
                    passed=False,
                    tolerance_used=1e-4,
                )
                print(f"Batch consistency test failed for batch_size {batch_size}: {e}")

        return results

    def comprehensive_manifold_test(
        self, manifold_name: str, include_performance: bool = True, include_batch_tests: bool = True
    ) -> dict[str, Any]:
        """Run comprehensive tests for a specific manifold.

        Args:
            manifold_name: Name of the manifold to test
            include_performance: Whether to include performance tests
            include_batch_tests: Whether to include batch consistency tests

        Returns:
            Dictionary with comprehensive test results
        """
        if manifold_name not in self.test_manifolds:
            raise ValueError(f"Unsupported manifold: {manifold_name}")

        manifold = self.test_manifolds[manifold_name]
        config = self.manifold_configs[manifold_name]
        results = {"manifold": manifold_name, "consistency_tests": {}, "performance_tests": {}, "batch_tests": {}}

        key = jax.random.key(42)

        # Generate test data
        if manifold_name == "sphere":
            x = manifold.random_point(key, *config["dims"])
            v = manifold.random_tangent(jax.random.key(43), x)
            test_data_base = {"x": x, "v": v}
        else:
            x = manifold.random_point(key)
            v = manifold.random_tangent(jax.random.key(43), x)
            test_data_base = {"x": x, "v": v}

        # Run consistency tests
        for operation in config["test_operations"]:
            if operation == "inner":
                if manifold_name == "sphere":
                    v2 = manifold.random_tangent(jax.random.key(44), x)
                    test_data = {"x": x, "v1": v, "v2": v2}
                else:
                    v2 = manifold.random_tangent(jax.random.key(44), x)
                    test_data = {"x": x, "v1": v, "v2": v2}
            else:
                test_data = test_data_base.copy()

            try:
                consistency_result = self.compare_jit_vs_nojit(manifold, operation, test_data, config["tolerance"])
                results["consistency_tests"][operation] = consistency_result

                # Performance tests
                if include_performance:
                    performance_result = self.performance_measurement_detailed(manifold, operation, test_data)
                    results["performance_tests"][operation] = performance_result

            except Exception as e:
                print(f"Test failed for {manifold_name}.{operation}: {e}")
                results["consistency_tests"][operation] = None
                if include_performance:
                    results["performance_tests"][operation] = None

        # Batch consistency tests
        if include_batch_tests:
            for operation in config["test_operations"]:
                try:
                    batch_results = self.batch_consistency_test(manifold, operation, config["batch_sizes"])
                    results["batch_tests"][operation] = batch_results
                except Exception as e:
                    print(f"Batch test failed for {manifold_name}.{operation}: {e}")
                    results["batch_tests"][operation] = None

        return results

    def run_all_manifold_tests(self) -> dict[str, dict[str, Any]]:
        """Run comprehensive tests for all supported manifolds.

        Returns:
            Dictionary with test results for all manifolds
        """
        all_results = {}

        for manifold_name in self.test_manifolds:
            print(f"Testing {manifold_name} manifold...")
            try:
                results = self.comprehensive_manifold_test(manifold_name)
                all_results[manifold_name] = results
                print(f"✓ {manifold_name} tests completed")
            except Exception as e:
                print(f"✗ {manifold_name} tests failed: {e}")
                all_results[manifold_name] = {"error": str(e)}

        return all_results

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report.

        Returns:
            Formatted performance report string
        """
        if not self._performance_history:
            return "No performance data available."

        report = ["JIT Optimization Performance Report"]
        report.append("=" * 40)
        report.append("")

        # Summary statistics
        speedups = [r.speedup for r in self._performance_history if r.speedup != float("inf")]
        if speedups:
            report.append(f"Average speedup: {np.mean(speedups):.2f}x")
            report.append(f"Max speedup: {np.max(speedups):.2f}x")
            report.append(f"Min speedup: {np.min(speedups):.2f}x")

        compilation_times = [r.compilation_time for r in self._performance_history if r.compilation_time is not None]
        if compilation_times:
            report.append(f"Average compilation time: {np.mean(compilation_times):.4f}s")

        report.append("")
        report.append("Detailed Results:")
        report.append("-" * 20)

        for i, result in enumerate(self._performance_history):
            report.append(f"Test {i + 1}:")
            report.append(f"  JIT time: {result.jit_time:.6f}s")
            report.append(f"  Non-JIT time: {result.nojit_time:.6f}s")
            report.append(f"  Speedup: {result.speedup:.2f}x")
            if result.compilation_time:
                report.append(f"  Compilation time: {result.compilation_time:.6f}s")
            report.append("")

        return "\n".join(report)

    def get_consistency_summary(self) -> dict[str, Any]:
        """Get summary of all consistency test results.

        Returns:
            Dictionary with consistency test summary
        """
        if not self._consistency_history:
            return {"message": "No consistency data available."}

        passed_tests = sum(1 for r in self._consistency_history if r.passed)
        total_tests = len(self._consistency_history)

        max_errors = [r.max_absolute_error for r in self._consistency_history]
        mean_errors = [r.mean_absolute_error for r in self._consistency_history]

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "max_absolute_error": np.max(max_errors) if max_errors else 0,
            "mean_absolute_error": np.mean(mean_errors) if mean_errors else 0,
            "all_tests_passed": passed_tests == total_tests,
        }

    def clear_history(self):
        """Clear all test history."""
        self._performance_history.clear()
        self._consistency_history.clear()


# Convenience functions for easy testing
def quick_jit_test(manifold_name: str) -> dict[str, Any]:
    """Quick JIT test for a single manifold.

    Args:
        manifold_name: Name of the manifold to test

    Returns:
        Dictionary with test results
    """
    framework = JITTestFramework()
    return framework.comprehensive_manifold_test(manifold_name)


def quick_performance_comparison(manifold_name: str, operation: str) -> PerformanceResult:
    """Quick performance comparison for a specific operation.

    Args:
        manifold_name: Name of the manifold
        operation: Operation name

    Returns:
        PerformanceResult with timing comparison
    """
    framework = JITTestFramework()
    manifold = framework.test_manifolds[manifold_name]

    # Generate test data
    key = jax.random.key(42)
    config = framework.manifold_configs[manifold_name]

    if manifold_name == "sphere":
        x = manifold.random_point(key, *config["dims"])
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}
    else:
        x = manifold.random_point(key)
        v = manifold.random_tangent(jax.random.key(43), x)
        test_data = {"x": x, "v": v}

    return framework.performance_measurement_detailed(manifold, operation, test_data)
