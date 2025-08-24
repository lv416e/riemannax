"""Performance benchmarking framework for RiemannAX JIT optimization.

This module provides comprehensive performance measurement and analysis
capabilities for JIT compilation, caching, and execution optimization.
"""

import statistics
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import jax

from .jit_manager import JITManager


class PerformanceBenchmark:
    """Comprehensive performance benchmarking framework for JIT optimization."""

    def __init__(self, warmup_runs: int = 3, precision: int = 6):
        """Initialize performance benchmark.

        Args:
            warmup_runs: Number of warmup runs before measurement
            precision: Decimal precision for timing measurements
        """
        self.warmup_runs = warmup_runs
        self.precision = precision
        self._results_history: list[dict[str, Any]] = []

    def _warmup_function(self, func: Callable[..., Any], args: tuple[Any, ...], num_runs: int | None = None) -> None:
        """Warm up function before benchmarking."""
        import contextlib

        runs = num_runs or self.warmup_runs
        for _ in range(runs):
            with contextlib.suppress(Exception):
                func(*args)

    def _time_function(self, func: Callable[..., Any], args: tuple[Any, ...], num_runs: int = 10) -> dict[str, Any]:
        """Time function execution with statistical analysis."""
        times = []

        # Warmup
        self._warmup_function(func, args)

        # Measure execution times
        for _ in range(num_runs):
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            "mean_time": round(statistics.mean(times), self.precision),
            "std_time": round(statistics.stdev(times) if len(times) > 1 else 0.0, self.precision),
            "min_time": round(min(times), self.precision),
            "max_time": round(max(times), self.precision),
            "raw_times": times,
        }

    def compare_jit_performance(
        self, func: Callable[..., Any], args: tuple[Any, ...], static_argnums: tuple[int, ...] | None = None, num_runs: int = 10
    ) -> dict[str, float | dict[str, Any]]:
        """Compare JIT vs non-JIT performance.

        Args:
            func: Function to benchmark
            args: Function arguments
            static_argnums: Static argument indices for JIT
            num_runs: Number of benchmark runs

        Returns:
            Performance comparison results
        """
        # Non-JIT version
        no_jit_results = self._time_function(func, args, num_runs)

        # JIT version
        jit_func = JITManager.jit_decorator(func, static_argnums=static_argnums)

        # Measure compilation time
        compilation_start = time.perf_counter()
        jit_func(*args)  # Trigger compilation
        compilation_time = time.perf_counter() - compilation_start

        # Measure JIT execution performance
        jit_results = self._time_function(jit_func, args, num_runs)

        # Calculate speedup
        jit_speedup = no_jit_results["mean_time"] / jit_results["mean_time"]

        return {
            "no_jit_time": no_jit_results["mean_time"],
            "jit_time": jit_results["mean_time"],
            "compilation_time": round(compilation_time, self.precision),
            "jit_speedup": round(jit_speedup, 2),
            "no_jit_stats": no_jit_results,
            "jit_stats": jit_results,
        }

    def measure_cache_performance(self, func: Callable[..., Any], args: tuple[Any, ...], num_cache_hits: int = 5) -> dict[str, Any]:
        """Measure cache performance improvements.

        Args:
            func: Function to benchmark caching
            args: Function arguments
            num_cache_hits: Number of cache hit measurements

        Returns:
            Cache performance results
        """
        JITManager.clear_cache()

        # Measure initial compilation time
        jit_func = JITManager.jit_decorator(func)
        compilation_start = time.perf_counter()
        jit_func(*args)  # Trigger compilation
        initial_compile_time = time.perf_counter() - compilation_start

        # Measure cache hit times
        cache_hit_times = []
        for _ in range(num_cache_hits):
            # Get cached function (should return same instance)
            cached_func = JITManager.jit_decorator(func)

            start_time = time.perf_counter()
            cached_func(*args)
            end_time = time.perf_counter()
            cache_hit_times.append(end_time - start_time)

        avg_cache_hit_time = statistics.mean(cache_hit_times)
        cache_speedup = initial_compile_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 0

        return {
            "initial_compile_time": round(initial_compile_time, self.precision),
            "avg_cache_hit_time": round(avg_cache_hit_time, self.precision),
            "cache_speedup": round(cache_speedup, 2),
            "cache_hit_times": cache_hit_times,
        }

    def compare_static_argnums_performance(
        self, func: Callable[..., Any], args: tuple[Any, ...], configurations: list[dict[str, Any]], num_runs: int = 10
    ) -> list[dict[str, str | float | dict[str, Any] | Any]]:
        """Compare performance across different static_argnums configurations.

        Args:
            func: Function to benchmark
            args: Function arguments
            configurations: List of JIT configurations to test
            num_runs: Number of benchmark runs per configuration

        Returns:
            Performance comparison results for each configuration
        """
        results = []

        for config in configurations:
            JITManager.clear_cache()
            static_argnums = config.get("static_argnums")

            # Measure compilation time
            jit_func = JITManager.jit_decorator(func, static_argnums=static_argnums)
            compilation_start = time.perf_counter()
            jit_func(*args)  # Trigger compilation
            compilation_time = time.perf_counter() - compilation_start

            # Measure execution performance
            execution_results = self._time_function(jit_func, args, num_runs)

            results.append(
                {
                    "config": str(config),
                    "static_argnums": static_argnums,
                    "compilation_time": round(compilation_time, self.precision),
                    "avg_execution_time": execution_results["mean_time"],
                    "execution_stats": execution_results,
                }
            )

        return results

    def benchmark_manifold_operations(
        self, manifold_name: str, operations: dict[str, tuple[Any, ...]], num_runs: int = 20
    ) -> dict[str, dict[str, Any]]:
        """Benchmark performance of manifold operations.

        Args:
            manifold_name: Name of the manifold being tested
            operations: Dictionary of operation_name -> (func, args) pairs
            num_runs: Number of benchmark runs per operation

        Returns:
            Performance results for each manifold operation
        """
        results = {}

        for op_name, (func, args) in operations.items():
            JITManager.clear_cache()

            # Non-JIT performance
            no_jit_results = self._time_function(func, args, num_runs)

            # JIT performance with compilation overhead
            jit_func = JITManager.jit_decorator(func)
            compilation_start = time.perf_counter()
            jit_func(*args)  # Trigger compilation
            compilation_overhead = time.perf_counter() - compilation_start

            jit_results = self._time_function(jit_func, args, num_runs)

            jit_speedup = no_jit_results["mean_time"] / jit_results["mean_time"]

            results[op_name] = {
                "no_jit_time": no_jit_results["mean_time"],
                "jit_time": jit_results["mean_time"],
                "jit_speedup": round(jit_speedup, 2),
                "compilation_overhead": round(compilation_overhead, self.precision),
                "efficiency": round(jit_speedup - (compilation_overhead / jit_results["mean_time"]), 2),
            }

        return results

    def compare_batch_performance(
        self, single_func: Callable[..., Any], single_args: tuple[Any, ...], batch_func: Callable[..., Any], batch_args: tuple[Any, ...], batch_size: int
    ) -> dict[str, Any]:
        """Compare batch vs single operation performance.

        Args:
            single_func: Function for single operations
            single_args: Arguments for single operation
            batch_func: Function for batch operations
            batch_args: Arguments for batch operation
            batch_size: Size of the batch

        Returns:
            Batch vs single performance comparison
        """
        # Single operation performance
        single_results = self._time_function(single_func, single_args, num_runs=10)
        single_time = single_results["mean_time"]

        # Batch operation performance
        batch_results = self._time_function(batch_func, batch_args, num_runs=10)
        batch_time = batch_results["mean_time"]

        # Calculate efficiency metrics
        per_item_batch_time = batch_time / batch_size
        batch_efficiency = single_time / per_item_batch_time

        return {
            "single_operation_time": single_time,
            "batch_operation_time": batch_time,
            "per_item_batch_time": round(per_item_batch_time, self.precision),
            "batch_efficiency": round(batch_efficiency, 2),
            "batch_size": batch_size,
        }

    def measure_memory_usage(
        self, func: Callable[..., Any], args: tuple[Any, ...], track_compilation_memory: bool = True
    ) -> dict[str, float | int]:
        """Measure memory usage during function execution.

        Args:
            func: Function to measure memory usage
            args: Function arguments
            track_compilation_memory: Whether to track compilation memory

        Returns:
            Memory usage statistics
        """
        if track_compilation_memory:
            # Start memory tracking
            tracemalloc.start()

            # JIT compilation
            jit_func = JITManager.jit_decorator(func)
            jit_func(*args)  # Trigger compilation

            compilation_memory = tracemalloc.get_traced_memory()[1]  # Peak memory
            tracemalloc.reset_peak()

            # Execution memory
            jit_func(*args)
            execution_memory = tracemalloc.get_traced_memory()[1]  # Peak memory

            tracemalloc.stop()

            memory_efficiency = compilation_memory / execution_memory if execution_memory > 0 else 1.0

            return {
                "peak_compilation_memory": compilation_memory,
                "peak_execution_memory": execution_memory,
                "memory_efficiency": round(memory_efficiency, 2),
                "memory_overhead": compilation_memory - execution_memory,
            }
        else:
            # Track execution memory only
            tracemalloc.start()
            func(*args)
            peak_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            return {
                "peak_execution_memory": peak_memory,
                "peak_compilation_memory": 0,
                "memory_efficiency": 1.0,
                "memory_overhead": 0,
            }

    def compare_device_performance(
        self, func: Callable[..., Any], args: tuple[Any, ...], devices: list[str], num_runs: int = 10
    ) -> dict[str, dict[str, Any]]:
        """Compare performance across different devices.

        Args:
            func: Function to benchmark
            args: Function arguments
            devices: List of device names to test ("cpu", "gpu", "tpu")
            num_runs: Number of benchmark runs per device

        Returns:
            Performance comparison across devices
        """
        results = {}
        available_devices = [str(device).lower() for device in jax.devices()]

        for device in devices:
            if any(device in available_device for available_device in available_devices):
                JITManager.clear_cache()

                # Create device-specific JIT function
                jit_func = JITManager.jit_decorator(func, device=device)

                # Measure compilation time
                compilation_start = time.perf_counter()
                jit_func(*args)  # Trigger compilation
                compilation_time = time.perf_counter() - compilation_start

                # Measure execution performance
                execution_results = self._time_function(jit_func, args, num_runs)

                # Calculate throughput (operations per second)
                throughput = 1.0 / execution_results["mean_time"] if execution_results["mean_time"] > 0 else 0

                results[device] = {
                    "avg_execution_time": execution_results["mean_time"],
                    "compilation_time": round(compilation_time, self.precision),
                    "throughput": round(throughput, 2),
                    "execution_stats": execution_results,
                }

        return results

    def analyze_compilation_caching(
        self, func: Callable[..., Any], args: tuple[Any, ...], num_cache_tests: int = 10, cache_clear_interval: int = 3
    ) -> dict[str, Any]:
        """Analyze compilation caching efficiency.

        Args:
            func: Function to analyze caching
            args: Function arguments
            num_cache_tests: Number of cache tests to perform
            cache_clear_interval: Interval to clear cache for miss testing

        Returns:
            Caching efficiency analysis
        """
        cache_hits = 0
        cache_misses = 0
        total_cache_hit_time = 0.0
        total_compilation_time = 0.0

        for i in range(num_cache_tests):
            if i % cache_clear_interval == 0:
                JITManager.clear_cache()  # Force cache miss

                # Measure compilation time
                compilation_start = time.perf_counter()
                jit_func = JITManager.jit_decorator(func)
                jit_func(*args)  # Trigger compilation
                compilation_time = time.perf_counter() - compilation_start

                total_compilation_time += compilation_time
                cache_misses += 1
            else:
                # Measure cache hit time
                cache_start = time.perf_counter()
                jit_func = JITManager.jit_decorator(func)
                jit_func(*args)
                cache_time = time.perf_counter() - cache_start

                total_cache_hit_time += cache_time
                cache_hits += 1

        cache_hit_ratio = cache_hits / num_cache_tests if num_cache_tests > 0 else 0
        avg_cache_hit_time = total_cache_hit_time / cache_hits if cache_hits > 0 else 0
        avg_compilation_time = total_compilation_time / cache_misses if cache_misses > 0 else 0

        total_cache_savings = total_compilation_time - total_cache_hit_time

        return {
            "cache_hit_ratio": round(cache_hit_ratio, 3),
            "avg_cache_hit_time": round(avg_cache_hit_time, self.precision),
            "avg_compilation_time": round(avg_compilation_time, self.precision),
            "total_cache_savings": round(total_cache_savings, self.precision),
            "cache_efficiency": round(avg_compilation_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 0, 2),
        }

    def statistical_performance_analysis(
        self, func: Callable[..., Any], args: tuple[Any, ...], num_runs: int = 50, confidence_level: float = 0.95
    ) -> dict[str, float | list[float] | bool]:
        """Perform statistical analysis of performance measurements.

        Args:
            func: Function to analyze statistically
            args: Function arguments
            num_runs: Number of measurement runs
            confidence_level: Confidence level for interval estimation

        Returns:
            Statistical performance analysis results
        """
        jit_func = JITManager.jit_decorator(func)
        execution_results = self._time_function(jit_func, args, num_runs)

        times = execution_results["raw_times"]
        mean_time = execution_results["mean_time"]
        std_time = execution_results["std_time"]

        # Calculate confidence interval
        import math

        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin_of_error = z_score * (std_time / math.sqrt(num_runs))
        confidence_interval = [
            round(mean_time - margin_of_error, self.precision),
            round(mean_time + margin_of_error, self.precision),
        ]

        # Detect outliers using IQR method
        sorted_times = sorted(times)
        n = len(sorted_times)
        q1 = sorted_times[n // 4]
        q3 = sorted_times[(3 * n) // 4]
        iqr = q3 - q1
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr

        outliers = [t for t in times if t < outlier_threshold_low or t > outlier_threshold_high]
        outliers_detected = len(outliers) > 0

        return {
            "mean_execution_time": mean_time,
            "std_execution_time": std_time,
            "confidence_interval": confidence_interval,
            "confidence_level": confidence_level,
            "outliers_detected": outliers_detected,
            "num_outliers": len(outliers),
            "outlier_values": outliers,
        }

    def generate_performance_report(
        self, results: dict[str, Any], include_plots: bool = False, output_format: str = "markdown"
    ) -> str:
        """Generate formatted performance benchmark report.

        Args:
            results: Benchmark results dictionary
            include_plots: Whether to include plots (placeholder)
            output_format: Output format ("markdown", "text")

        Returns:
            Formatted performance report
        """
        if output_format == "markdown":
            report = "# Performance Benchmark Report\n\n"

            if "jit_speedup" in results:
                report += "## JIT Performance\n"
                report += f"- **JIT Speedup**: {results['jit_speedup']}x\n"

            if "cache_efficiency" in results:
                report += f"- **Cache Efficiency**: {results['cache_efficiency'] * 100:.0f}%\n"

            if "memory_usage" in results:
                memory = results["memory_usage"]
                report += "\n## Memory Usage\n"
                report += f"- **Peak Memory**: {memory.get('peak', 0)} bytes\n"
                report += f"- **Average Memory**: {memory.get('avg', 0)} bytes\n"

            if "device_comparison" in results:
                device_comp = results["device_comparison"]
                report += "\n## Device Performance\n"
                for device, relative_time in device_comp.items():
                    report += f"- **{device.upper()}**: {relative_time:.1f}x relative time\n"

            report += "\n## Summary\n"
            report += "Performance benchmarking completed successfully.\n"

        else:  # text format
            report = "Performance Benchmark Report\n"
            report += "=" * 30 + "\n\n"

            if "jit_speedup" in results:
                report += f"JIT Speedup: {results['jit_speedup']}x\n"

            if "cache_efficiency" in results:
                report += f"Cache Efficiency: {results['cache_efficiency'] * 100:.0f}%\n"

        return report

    def validate_performance_thresholds(
        self, results: dict[str, float], thresholds: dict[str, float]
    ) -> dict[str, bool | list[str] | int]:
        """Validate benchmark results against performance thresholds.

        Args:
            results: Performance benchmark results
            thresholds: Performance thresholds to validate against

        Returns:
            Validation results with pass/fail status
        """
        passed = []
        failures = []

        validation_rules: dict[str, Callable[[dict[str, float], float], bool]] = {
            "min_jit_speedup": lambda r, t: r.get("jit_speedup", 0) >= t,
            "max_compilation_time": lambda r, t: r.get("compilation_time", float("inf")) <= t,
            "min_cache_hit_ratio": lambda r, t: r.get("cache_hit_ratio", 0) >= t,
            "max_memory_overhead": lambda r, t: r.get("memory_overhead", float("inf")) <= t,
        }

        for threshold_name, threshold_value in thresholds.items():
            if threshold_name in validation_rules:
                validation_func = validation_rules[threshold_name]
                if validation_func(results, threshold_value):
                    passed.append(threshold_name)
                else:
                    failures.append(
                        f"{threshold_name}: expected {threshold_value}, got {results.get(threshold_name.split('_', 1)[1], 'N/A')}"
                    )

        return {
            "all_passed": len(failures) == 0,
            "passed": passed,
            "failures": failures,
            "total_checked": len(thresholds),
        }
