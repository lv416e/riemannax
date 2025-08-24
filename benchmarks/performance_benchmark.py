"""Comprehensive Performance Benchmarking Suite for RiemannAX."""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from riemannax.core.jit_manager import JITManager
from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.stiefel import Stiefel


@dataclass
class PerformanceResult:
    """Single performance measurement result."""

    manifold_name: str
    operation: str
    batch_size: int
    jit_time_ms: float
    nojit_time_ms: float
    speedup: float
    compilation_time_ms: float | None
    memory_usage_mb: float | None
    input_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark results."""

    total_benchmarks: int
    avg_speedup: float
    max_speedup: float
    min_speedup: float
    avg_jit_time_ms: float
    avg_compilation_time_ms: float | None
    total_time_saved_ms: float
    manifolds_tested: list[str]
    operations_tested: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self, output_dir: str | None = None):
        """Initialize benchmark system.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.jit_manager = JITManager()
        self.results: list[PerformanceResult] = []

        if output_dir:
            self.output_dir: Path | None = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        # Initialize manifolds with various configurations
        self.manifolds = {
            "sphere_3d": Sphere(),
            "sphere_5d": Sphere(),
            "sphere_10d": Sphere(),
            "grassmann_5_3": Grassmann(n=5, p=3),
            "grassmann_10_5": Grassmann(n=10, p=5),
            "stiefel_5_3": Stiefel(n=5, p=3),
            "stiefel_10_5": Stiefel(n=10, p=5),
            "so_3": SpecialOrthogonal(n=3),
            "so_5": SpecialOrthogonal(n=5),
            "spd_3": SymmetricPositiveDefinite(n=3),
            "spd_5": SymmetricPositiveDefinite(n=5),
        }

        # Operations to benchmark for each manifold type
        self.operations = {
            "sphere_3d": ["exp", "log", "proj", "inner", "random_point"],
            "sphere_5d": ["exp", "log", "proj", "inner", "random_point"],
            "sphere_10d": ["exp", "log", "proj", "inner", "random_point"],
            "grassmann_5_3": ["exp", "log", "proj", "inner", "random_point"],
            "grassmann_10_5": ["exp", "log", "proj", "inner", "random_point"],
            "stiefel_5_3": ["exp", "log", "proj", "inner", "random_point"],
            "stiefel_10_5": ["exp", "log", "proj", "inner", "random_point"],
            "so_3": ["exp", "log", "proj", "inner", "random_point"],
            "so_5": ["exp", "log", "proj", "inner", "random_point"],
            "spd_3": ["exp", "log", "proj", "inner", "random_point"],
            "spd_5": ["exp", "log", "proj", "inner", "random_point"],
        }

        # Batch sizes to test
        self.batch_sizes = [1, 5, 10, 20, 50, 100]

    def get_manifold_dims(self, manifold_name: str) -> dict[str, int]:
        """Get dimension information for manifold."""
        if manifold_name.startswith("sphere"):
            if "3d" in manifold_name:
                return {"dim": 3}
            elif "5d" in manifold_name:
                return {"dim": 5}
            elif "10d" in manifold_name:
                return {"dim": 10}
        elif manifold_name.startswith("grassmann") or manifold_name.startswith("stiefel"):
            if "5_3" in manifold_name:
                return {"n": 5, "p": 3}
            elif "10_5" in manifold_name:
                return {"n": 10, "p": 5}
        elif manifold_name.startswith("so"):
            if "so_3" in manifold_name:
                return {"n": 3}
            elif "so_5" in manifold_name:
                return {"n": 5}
        elif manifold_name.startswith("spd"):
            if "spd_3" in manifold_name:
                return {"n": 3}
            elif "spd_5" in manifold_name:
                return {"n": 5}

        return {}

    def generate_test_data(self, manifold_name: str, batch_size: int) -> dict[str, jnp.ndarray]:
        """Generate test data for a specific manifold and batch size."""
        manifold = self.manifolds[manifold_name]
        dims = self.get_manifold_dims(manifold_name)
        key = jr.key(42)

        try:
            if manifold_name.startswith("sphere"):
                dim = dims["dim"]
                x = manifold.random_point(key, batch_size, dim)
                v = manifold.random_tangent(jr.key(43), x)
                return {"x": x, "v": v}

            elif manifold_name.startswith(("grassmann", "stiefel")):
                if batch_size == 1:
                    x = manifold.random_point(key)
                    v = manifold.random_tangent(jr.key(43), x)
                else:
                    # For batch operations, generate individual samples and stack
                    keys = jr.split(key, batch_size)
                    x_list = []
                    v_list = []
                    for i in range(batch_size):
                        xi = manifold.random_point(keys[i])
                        vi = manifold.random_tangent(jr.key(43 + i), xi)
                        x_list.append(xi)
                        v_list.append(vi)
                    x = jnp.stack(x_list, axis=0)
                    v = jnp.stack(v_list, axis=0)
                return {"x": x, "v": v}

            elif manifold_name.startswith(("so", "spd")):
                n = dims["n"]
                if batch_size == 1:
                    x = manifold.random_point(key, n)
                    v = manifold.random_tangent(jr.key(43), x)
                else:
                    # For batch operations, generate individual samples and stack
                    keys = jr.split(key, batch_size)
                    x_list = []
                    v_list = []
                    for i in range(batch_size):
                        xi = manifold.random_point(keys[i], n)
                        vi = manifold.random_tangent(jr.key(43 + i), xi)
                        x_list.append(xi)
                        v_list.append(vi)
                    x = jnp.stack(x_list, axis=0)
                    v = jnp.stack(v_list, axis=0)
                return {"x": x, "v": v}

            return {}

        except Exception as e:
            # If test data generation fails, return empty dict to skip this benchmark
            print(f"Warning: Test data generation failed for {manifold_name}: {e}")
            return {}

    @contextmanager
    def memory_monitor(self):
        """Context manager for memory usage monitoring."""
        # Simplified memory monitoring - in production, use jax profiling
        yield None  # For now, return None for memory usage

    def measure_execution_time(
        self, func: Any, *args: Any, warmup_runs: int = 3, measurement_runs: int = 5
    ) -> tuple[float, float | None]:
        """Measure execution time with warmup and compilation timing.

        Returns:
            (execution_time_ms, compilation_time_ms)
        """
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                _ = func(*args)
            except Exception:
                # If function fails, return invalid timing
                return float("inf"), None

        # Measure compilation time (first real run)
        start_compile = time.time()
        try:
            _ = func(*args)
        except Exception:
            return float("inf"), None
        compilation_time = (time.time() - start_compile) * 1000

        # Measure execution time
        times = []
        for _ in range(measurement_runs):
            start = time.time()
            try:
                _ = func(*args)
                times.append((time.time() - start) * 1000)  # Convert to ms
            except Exception:
                times.append(float("inf"))

        # Return median time and compilation time
        median_time = float(np.median(times))
        return median_time, compilation_time

    def benchmark_single_operation(self, manifold_name: str, operation: str, batch_size: int) -> PerformanceResult:
        """Benchmark a single operation for a specific manifold and batch size."""
        try:
            manifold = self.manifolds[manifold_name]
            test_data = self.generate_test_data(manifold_name, batch_size)

            if not test_data:
                # Return empty result for unsupported configurations
                return PerformanceResult(
                    manifold_name=manifold_name,
                    operation=operation,
                    batch_size=batch_size,
                    jit_time_ms=float("inf"),
                    nojit_time_ms=float("inf"),
                    speedup=0.0,
                    compilation_time_ms=None,
                    memory_usage_mb=None,
                    input_shape=(),
                )
        except Exception as e:
            print(f"Warning: Failed to initialize benchmark for {manifold_name}.{operation}: {e}")
            return PerformanceResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                jit_time_ms=float("inf"),
                nojit_time_ms=float("inf"),
                speedup=0.0,
                compilation_time_ms=None,
                memory_usage_mb=None,
                input_shape=(),
            )

        # Get input shape for reporting
        input_shape = next(iter(test_data.values())).shape

        try:
            # Prepare operation functions
            if operation == "exp":
                if len(test_data) >= 2:
                    x, v = list(test_data.values())[:2]

                    def nojit_func() -> Any:
                        return manifold.exp(x, v)

                    # Use manifold's JIT-decorated method directly (now handled by @jit_optimized decorator)
                    def jit_func_call() -> Any:
                        return manifold.exp(x, v)
                else:
                    raise ValueError(f"Insufficient test data for operation {operation}")
            elif operation == "log":
                if len(test_data) >= 2:
                    x, v = list(test_data.values())[:2]
                    # Generate a second point for log
                    y = manifold.exp(x, v * 0.1)  # Small step to stay on manifold

                    def nojit_func():
                        return manifold.log(x, y)

                    # Use manifold's JIT-decorated method directly
                    def jit_func_call():
                        return manifold.log(x, y)
                else:
                    raise ValueError(f"Insufficient test data for operation {operation}")
            elif operation == "proj":
                if len(test_data) >= 2:
                    x, v = list(test_data.values())[:2]

                    def nojit_func():
                        return manifold.proj(x, v)

                    # Use manifold's JIT-decorated method directly
                    def jit_func_call():
                        return manifold.proj(x, v)
                else:
                    raise ValueError(f"Insufficient test data for operation {operation}")
            elif operation == "inner":
                if len(test_data) >= 2:
                    x, v = list(test_data.values())[:2]

                    def nojit_func():
                        return manifold.inner(x, v, v)

                    # Use manifold's JIT-decorated method directly
                    def jit_func_call():
                        return manifold.inner(x, v, v)
                else:
                    raise ValueError(f"Insufficient test data for operation {operation}")
            elif operation == "random_point":
                key = jr.key(42)
                dims = self.get_manifold_dims(manifold_name)
                if manifold_name.startswith("sphere"):
                    dim = dims.get("dim", 3)

                    def nojit_func():
                        return manifold.random_point(key, batch_size, dim)

                    # Use manifold's JIT-decorated method directly
                    def jit_func_call():
                        return manifold.random_point(key, batch_size, dim)
                else:
                    # For matrix manifolds, use n parameter
                    n = dims.get("n", 3)

                    def nojit_func():
                        return manifold.random_point(key, n)

                    # Use manifold's JIT-decorated method directly
                    def jit_func_call():
                        return manifold.random_point(key, n)
            else:
                # Unsupported operation
                raise ValueError(f"Unsupported operation: {operation}")

        except Exception as e:
            print(f"Warning: Operation preparation failed for {manifold_name}.{operation}: {e}")
            return PerformanceResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                jit_time_ms=float("inf"),
                nojit_time_ms=float("inf"),
                speedup=0.0,
                compilation_time_ms=None,
                memory_usage_mb=None,
                input_shape=input_shape,
            )

        # Measure performance
        try:
            with self.memory_monitor() as memory_usage:
                # Measure non-JIT performance
                nojit_time, _ = self.measure_execution_time(nojit_func)

                # Measure JIT performance
                jit_time, compilation_time = self.measure_execution_time(jit_func_call)

            # Calculate speedup
            if jit_time > 0 and nojit_time > 0 and not np.isinf(jit_time) and not np.isinf(nojit_time):
                speedup = nojit_time / jit_time
            else:
                speedup = 0.0

            return PerformanceResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                jit_time_ms=jit_time,
                nojit_time_ms=nojit_time,
                speedup=speedup,
                compilation_time_ms=compilation_time,
                memory_usage_mb=memory_usage,
                input_shape=input_shape,
            )

        except Exception as e:
            print(f"Warning: Performance measurement failed for {manifold_name}.{operation}: {e}")
            return PerformanceResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                jit_time_ms=float("inf"),
                nojit_time_ms=float("inf"),
                speedup=0.0,
                compilation_time_ms=None,
                memory_usage_mb=None,
                input_shape=input_shape,
            )

    def run_comprehensive_benchmark(
        self,
        manifolds: list[str] | None = None,
        operations: list[str] | None = None,
        batch_sizes: list[int] | None = None,
    ) -> list[PerformanceResult]:
        """Run comprehensive benchmark across manifolds, operations, and batch sizes."""
        manifolds = manifolds or list(self.manifolds.keys())
        batch_sizes = batch_sizes or self.batch_sizes

        results = []
        total_benchmarks = 0

        # Count total benchmarks for progress tracking
        for manifold_name in manifolds:
            if manifold_name in self.operations:
                ops = operations or self.operations[manifold_name]
                total_benchmarks += len(ops) * len(batch_sizes)

        current_benchmark = 0

        for manifold_name in manifolds:
            if manifold_name not in self.operations:
                continue

            ops = operations or self.operations[manifold_name]

            for operation in ops:
                for batch_size in batch_sizes:
                    current_benchmark += 1
                    print(
                        f"Running benchmark {current_benchmark}/{total_benchmarks}: "
                        f"{manifold_name}.{operation} (batch_size={batch_size})"
                    )

                    result = self.benchmark_single_operation(manifold_name, operation, batch_size)
                    results.append(result)

        self.results.extend(results)
        return results

    def generate_benchmark_summary(self, results: list[PerformanceResult] | None = None) -> BenchmarkSummary:
        """Generate summary statistics from benchmark results."""
        if results is None:
            results = self.results

        if not results:
            return BenchmarkSummary(
                total_benchmarks=0,
                avg_speedup=0.0,
                max_speedup=0.0,
                min_speedup=0.0,
                avg_jit_time_ms=0.0,
                avg_compilation_time_ms=None,
                total_time_saved_ms=0.0,
                manifolds_tested=[],
                operations_tested=[],
            )

        valid_results = [r for r in results if r.speedup > 0 and not np.isinf(r.jit_time_ms)]

        if not valid_results:
            return BenchmarkSummary(
                total_benchmarks=len(results),
                avg_speedup=0.0,
                max_speedup=0.0,
                min_speedup=0.0,
                avg_jit_time_ms=0.0,
                avg_compilation_time_ms=None,
                total_time_saved_ms=0.0,
                manifolds_tested=list({r.manifold_name for r in results}),
                operations_tested=list({r.operation for r in results}),
            )

        speedups = [r.speedup for r in valid_results]
        jit_times = [r.jit_time_ms for r in valid_results]
        compilation_times = [r.compilation_time_ms for r in valid_results if r.compilation_time_ms is not None]

        # Calculate time saved
        time_saved = sum(r.nojit_time_ms - r.jit_time_ms for r in valid_results)

        return BenchmarkSummary(
            total_benchmarks=len(results),
            avg_speedup=float(np.mean(speedups)),
            max_speedup=float(np.max(speedups)),
            min_speedup=float(np.min(speedups)),
            avg_jit_time_ms=float(np.mean(jit_times)),
            avg_compilation_time_ms=float(np.mean(compilation_times)) if compilation_times else None,
            total_time_saved_ms=time_saved,
            manifolds_tested=list({r.manifold_name for r in results}),
            operations_tested=list({r.operation for r in results}),
        )

    def generate_detailed_report(self, results: list[PerformanceResult] | None = None) -> str:
        """Generate detailed performance report."""
        if results is None:
            results = self.results

        if not results:
            return "No benchmark results available."

        summary = self.generate_benchmark_summary(results)

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary section
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total benchmarks: {summary.total_benchmarks}")
        report.append(f"Average speedup: {summary.avg_speedup:.2f}x")
        report.append(f"Max speedup: {summary.max_speedup:.2f}x")
        report.append(f"Min speedup: {summary.min_speedup:.2f}x")
        report.append(f"Average JIT time: {summary.avg_jit_time_ms:.2f} ms")
        if summary.avg_compilation_time_ms:
            report.append(f"Average compilation time: {summary.avg_compilation_time_ms:.2f} ms")
        report.append(f"Total time saved: {summary.total_time_saved_ms:.2f} ms")
        report.append("")

        # Manifolds tested
        report.append("MANIFOLDS TESTED")
        report.append("-" * 40)
        for manifold in summary.manifolds_tested:
            report.append(f"- {manifold}")
        report.append("")

        # Operations tested
        report.append("OPERATIONS TESTED")
        report.append("-" * 40)
        for operation in summary.operations_tested:
            report.append(f"- {operation}")
        report.append("")

        # Detailed results by manifold
        report.append("DETAILED RESULTS BY MANIFOLD")
        report.append("-" * 40)

        manifolds = {r.manifold_name for r in results}
        for manifold in sorted(manifolds):
            manifold_results = [r for r in results if r.manifold_name == manifold]
            valid_results = [r for r in manifold_results if r.speedup > 0 and not np.isinf(r.jit_time_ms)]

            report.append(f"\n{manifold.upper()}")
            report.append("-" * len(manifold))

            if valid_results:
                avg_speedup = np.mean([r.speedup for r in valid_results])
                report.append(f"Average speedup: {avg_speedup:.2f}x")

                # Best and worst operations
                best = max(valid_results, key=lambda x: x.speedup)
                worst = min(valid_results, key=lambda x: x.speedup)
                report.append(f"Best: {best.operation} (batch={best.batch_size}) - {best.speedup:.2f}x speedup")
                report.append(f"Worst: {worst.operation} (batch={worst.batch_size}) - {worst.speedup:.2f}x speedup")
            else:
                report.append("No valid results for this manifold")

        return "\n".join(report)

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        if not self.output_dir:
            print("No output directory configured. Results not saved.")
            return

        filepath = self.output_dir / filename

        # Convert results to dict format
        results_dict = {
            "summary": self.generate_benchmark_summary().to_dict(),
            "results": [result.to_dict() for result in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to: {filepath}")

    def load_results(self, filename: str = "benchmark_results.json"):
        """Load benchmark results from JSON file."""
        if not self.output_dir:
            print("No output directory configured. Cannot load results.")
            return

        filepath = self.output_dir / filename

        if not filepath.exists():
            print(f"Results file not found: {filepath}")
            return

        with open(filepath) as f:
            data = json.load(f)

        # Convert back to PerformanceResult objects
        self.results = []
        for result_dict in data.get("results", []):
            result = PerformanceResult(**result_dict)
            self.results.append(result)

        print(f"Results loaded from: {filepath}")


def run_quick_benchmark(manifolds: list[str] | None = None, batch_sizes: list[int] | None = None) -> str:
    """Quick benchmark utility function."""
    benchmark = PerformanceBenchmark()

    # Use smaller batch sizes for quick benchmark
    if batch_sizes is None:
        batch_sizes = [1, 10, 50]

    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(manifolds=manifolds, batch_sizes=batch_sizes)

    return benchmark.generate_detailed_report(results)
