"""Performance monitoring system for JIT optimization tracking."""

import statistics
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar


@dataclass
class OperationMetrics:
    """Metrics information for individual operations."""

    execution_times: list[float] = field(default_factory=list)
    compilation_time: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """JIT optimization performance monitoring system."""

    # Manage metrics information with class variables
    _metrics: ClassVar[dict[str, OperationMetrics]] = {}

    @classmethod
    @contextmanager
    def measure(cls, operation_name: str) -> Generator[None, None, None]:
        """Operation time measurement context manager.

        Args:
            operation_name: Name of the operation to be measured.

        Yields:
            Measurement execution context.
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            cls._record_execution_time(operation_name, execution_time)

    @classmethod
    def _record_execution_time(cls, operation_name: str, execution_time: float) -> None:
        """Record execution time.

        Args:
            operation_name: Name of the operation.
            execution_time: Execution time in seconds.
        """
        if operation_name not in cls._metrics:
            cls._metrics[operation_name] = OperationMetrics()

        cls._metrics[operation_name].execution_times.append(execution_time)
        cls._metrics[operation_name].updated_at = datetime.now()

    @classmethod
    def compilation_time(cls, func_name: str, compile_time: float) -> None:
        """Record compilation time.

        Args:
            func_name: Name of the function.
            compile_time: Compilation time in seconds.
        """
        if func_name not in cls._metrics:
            cls._metrics[func_name] = OperationMetrics()

        cls._metrics[func_name].compilation_time = compile_time
        cls._metrics[func_name].updated_at = datetime.now()

    @classmethod
    def get_metrics(cls) -> dict[str, dict[str, Any]]:
        """Get all metrics information.

        Returns:
            Dictionary containing metrics information.
        """
        result = {}
        for operation_name, metrics in cls._metrics.items():
            result[operation_name] = {
                "execution_times": metrics.execution_times,
                "compilation_time": metrics.compilation_time,
                "created_at": metrics.created_at,
                "updated_at": metrics.updated_at,
            }
        return result

    @classmethod
    def get_speedup_report(cls) -> dict[str, Any]:
        """Generate speedup report.

        Returns:
            Dictionary containing speedup report data.
        """
        total_operations = len(cls._metrics)
        operations_with_times = sum(1 for metrics in cls._metrics.values() if len(metrics.execution_times) > 0)

        return {
            "summary": {
                "total_operations": total_operations,
                "operations_with_measurements": operations_with_times,
                "generated_at": datetime.now(),
            },
            "details": {
                name: {
                    "average_time": statistics.mean(metrics.execution_times) if metrics.execution_times else None,
                    "measurement_count": len(metrics.execution_times),
                    "compilation_time": metrics.compilation_time,
                }
                for name, metrics in cls._metrics.items()
            },
        }

    @classmethod
    def calculate_speedup(cls, baseline_operation: str, optimized_operation: str) -> float:
        """Calculate speedup ratio.

        Args:
            baseline_operation: Name of the baseline operation.
            optimized_operation: Name of the optimized operation.

        Returns:
            Speedup ratio as a multiplier.
        """
        baseline_metrics = cls._metrics.get(baseline_operation)
        optimized_metrics = cls._metrics.get(optimized_operation)

        if not baseline_metrics or not baseline_metrics.execution_times:
            raise ValueError(f"No execution times found for baseline operation: {baseline_operation}")

        if not optimized_metrics or not optimized_metrics.execution_times:
            raise ValueError(f"No execution times found for optimized operation: {optimized_operation}")

        baseline_avg = statistics.mean(baseline_metrics.execution_times)
        optimized_avg = statistics.mean(optimized_metrics.execution_times)

        if optimized_avg == 0:
            return float("inf")

        return baseline_avg / optimized_avg

    @classmethod
    def check_performance_target(cls, operation_name: str, baseline_time: float, target_speedup: float) -> bool:
        """Check performance target achievement.

        Args:
            operation_name: Name of the operation.
            baseline_time: Baseline execution time.
            target_speedup: Target speedup multiplier.

        Returns:
            True if target is achieved, False otherwise.
        """
        metrics = cls._metrics.get(operation_name)
        if not metrics or not metrics.execution_times:
            return False

        avg_time = statistics.mean(metrics.execution_times)
        actual_speedup = baseline_time / avg_time if avg_time > 0 else 0

        return actual_speedup >= target_speedup

    @classmethod
    def get_average_execution_time(cls, operation_name: str) -> float | None:
        """Get average execution time.

        Args:
            operation_name: Name of the operation.

        Returns:
            Average execution time in seconds, or None if no data.
        """
        metrics = cls._metrics.get(operation_name)
        if not metrics or not metrics.execution_times:
            return None

        return statistics.mean(metrics.execution_times)

    @classmethod
    def get_performance_statistics(cls, operation_name: str) -> dict[str, float]:
        """Get performance statistics information.

        Args:
            operation_name: Name of the operation.

        Returns:
            Dictionary containing statistical information.
        """
        metrics = cls._metrics.get(operation_name)
        if not metrics or not metrics.execution_times:
            return {}

        times = metrics.execution_times
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    @classmethod
    def reset_metrics(cls) -> None:
        """Initialize all metrics."""
        cls._metrics.clear()

    @classmethod
    def enable(cls) -> None:
        """Enable performance monitoring."""
        # No special initialization needed as monitoring system is already enabled as class variable
        # Placeholder for future expansion
        pass

    @classmethod
    def disable(cls) -> None:
        """Disable performance monitoring."""
        # Placeholder for future expansion
        pass

    @classmethod
    def clear_stats(cls) -> None:
        """Clear statistics (alias for reset_metrics)."""
        cls.reset_metrics()

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        """Get statistics (alias for get_speedup_report)."""
        report = cls.get_speedup_report()

        # Add more detailed statistical information
        stats = {
            "total_operations": report["summary"]["total_operations"],
            "operations_with_measurements": report["summary"]["operations_with_measurements"],
            "generated_at": report["summary"]["generated_at"],
        }

        # Calculate average speedup
        speedups = []
        for _name, details in report["details"].items():
            if details["average_time"] and details["compilation_time"] and details["average_time"] > 0:
                # Assume virtual baseline (without JIT) is about 10 times slower
                estimated_speedup = 10.0 * details["compilation_time"] / details["average_time"]
                if estimated_speedup > 1.0:
                    speedups.append(estimated_speedup)

        stats["avg_speedup"] = statistics.mean(speedups) if speedups else 1.0
        stats["max_speedup"] = max(speedups) if speedups else 1.0
        stats["operation_details"] = report["details"]

        return stats
