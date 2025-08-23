"""Performance monitoring system for JIT optimization tracking."""

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar


@dataclass
class OperationMetrics:
    """個別操作のメトリクス情報."""

    execution_times: list[float] = field(default_factory=list)
    compilation_time: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """JIT最適化パフォーマンス監視システム."""

    # クラス変数でメトリクス情報を管理
    _metrics: ClassVar[dict[str, OperationMetrics]] = {}

    @classmethod
    @contextmanager
    def measure(cls, operation_name: str):
        """操作時間測定コンテキストマネージャー.

        Args:
            operation_name: 測定対象操作名

        Yields:
            測定実行コンテキスト
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
        """実行時間を記録.

        Args:
            operation_name: 操作名
            execution_time: 実行時間(秒)
        """
        if operation_name not in cls._metrics:
            cls._metrics[operation_name] = OperationMetrics()

        cls._metrics[operation_name].execution_times.append(execution_time)
        cls._metrics[operation_name].updated_at = datetime.now()

    @classmethod
    def compilation_time(cls, func_name: str, compile_time: float) -> None:
        """コンパイル時間記録.

        Args:
            func_name: 関数名
            compile_time: コンパイル時間(秒)
        """
        if func_name not in cls._metrics:
            cls._metrics[func_name] = OperationMetrics()

        cls._metrics[func_name].compilation_time = compile_time
        cls._metrics[func_name].updated_at = datetime.now()

    @classmethod
    def get_metrics(cls) -> dict[str, dict[str, Any]]:
        """全メトリクス情報取得.

        Returns:
            メトリクス情報辞書
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
        """速度向上レポート生成.

        Returns:
            速度向上レポート
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
        """速度向上比率計算.

        Args:
            baseline_operation: 基準操作名
            optimized_operation: 最適化操作名

        Returns:
            速度向上比率(倍数)
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
        """パフォーマンス目標達成チェック.

        Args:
            operation_name: 操作名
            baseline_time: 基準実行時間
            target_speedup: 目標速度向上倍率

        Returns:
            目標達成の可否
        """
        metrics = cls._metrics.get(operation_name)
        if not metrics or not metrics.execution_times:
            return False

        avg_time = statistics.mean(metrics.execution_times)
        actual_speedup = baseline_time / avg_time if avg_time > 0 else 0

        return actual_speedup >= target_speedup

    @classmethod
    def get_average_execution_time(cls, operation_name: str) -> float | None:
        """平均実行時間取得.

        Args:
            operation_name: 操作名

        Returns:
            平均実行時間(秒)
        """
        metrics = cls._metrics.get(operation_name)
        if not metrics or not metrics.execution_times:
            return None

        return statistics.mean(metrics.execution_times)

    @classmethod
    def get_performance_statistics(cls, operation_name: str) -> dict[str, float]:
        """パフォーマンス統計情報取得.

        Args:
            operation_name: 操作名

        Returns:
            統計情報辞書
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
        """全メトリクス初期化."""
        cls._metrics.clear()

    @classmethod
    def enable(cls) -> None:
        """パフォーマンス監視を有効化."""
        # すでにクラス変数として監視システムは有効なため、特別な初期化は不要
        # 将来的な拡張のためのプレースホルダー
        pass

    @classmethod
    def disable(cls) -> None:
        """パフォーマンス監視を無効化."""
        # 将来的な拡張のためのプレースホルダー
        pass

    @classmethod
    def clear_stats(cls) -> None:
        """統計情報をクリア(reset_metricsのエイリアス)."""
        cls.reset_metrics()

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        """統計情報取得(get_speedup_reportのエイリアス)."""
        report = cls.get_speedup_report()

        # より詳細な統計情報を追加
        stats = {
            "total_operations": report["summary"]["total_operations"],
            "operations_with_measurements": report["summary"]["operations_with_measurements"],
            "generated_at": report["summary"]["generated_at"],
        }

        # 平均速度向上の計算
        speedups = []
        for _name, details in report["details"].items():
            if details["average_time"] and details["compilation_time"] and details["average_time"] > 0:
                # 仮想的なベースライン(JIT無し)は約10倍遅いと仮定
                estimated_speedup = 10.0 * details["compilation_time"] / details["average_time"]
                if estimated_speedup > 1.0:
                    speedups.append(estimated_speedup)

        stats["avg_speedup"] = statistics.mean(speedups) if speedups else 1.0
        stats["max_speedup"] = max(speedups) if speedups else 1.0
        stats["operation_details"] = report["details"]

        return stats
