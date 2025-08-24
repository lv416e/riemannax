"""Safe JIT execution system with fallback mechanisms."""

import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, ClassVar, Literal

import jax

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """JIT実行失敗の記録."""

    timestamp: datetime
    function_name: str
    error_type: str
    error_message: str
    error_category: str
    stack_trace: str
    retry_count: int = 0


@dataclass
class CompilationStats:
    """コンパイル統計情報."""

    total_compilations: int = 0
    total_compilation_time: float = 0.0
    average_compilation_time: float = 0.0
    successful_compilations: int = 0
    failed_compilations: int = 0


class SafeJITWrapper:
    """安全なJIT実行ラッパーシステム."""

    # クラス変数で失敗ログを管理
    _global_failure_logs: ClassVar[list[FailureRecord]] = []

    def __init__(
        self,
        fallback_enabled: bool = True,
        max_retries: int = 3,
        track_compilation_time: bool = False,
        performance_monitoring: bool = False,
        debug_mode: bool = False,
    ):
        """SafeJITWrapperの初期化.

        Args:
            fallback_enabled: フォールバック機能の有効化
            max_retries: 最大リトライ回数
            track_compilation_time: コンパイル時間追跡
            performance_monitoring: パフォーマンス監視
            debug_mode: デバッグモード
        """
        self.fallback_enabled = fallback_enabled
        self.max_retries = max_retries
        self.track_compilation_time = track_compilation_time
        self.performance_monitoring = performance_monitoring
        self.debug_mode = debug_mode

        self.failure_logs: list[FailureRecord] = []
        self.compilation_stats = CompilationStats()
        self.performance_data: dict[str, list[float]] = {}
        self.debug_info: dict[str, Any] = {}
        self._closed = False

    def safe_jit(
        self,
        func: Callable[..., Any],
        fallback_func: Callable[..., Any] | None = None,
        static_argnums: tuple[int, ...] | None = None,
        **jit_kwargs: Any,
    ) -> Callable[..., Any]:
        """安全なJITデコレータ.

        Args:
            func: JIT対象関数
            fallback_func: フォールバック関数
            static_argnums: 静的引数インデックス
            **jit_kwargs: 追加JIT引数

        Returns:
            安全JIT関数
        """
        if fallback_func is None:
            fallback_func = func

        func_name = getattr(func, "__name__", "anonymous")

        # JIT設定準備
        jit_options = {}
        if static_argnums is not None:
            jit_options["static_argnums"] = static_argnums
        jit_options.update(jit_kwargs)

        # コンパイル時間追跡用
        if self.track_compilation_time:
            compile_start_time = time.time()

        try:
            # JIT最適化関数の作成
            jit_func = jax.jit(func, static_argnums=static_argnums)

            if self.track_compilation_time:
                compile_time = time.time() - compile_start_time
                self._update_compilation_stats(compile_time, True)

        except Exception as e:
            if self.track_compilation_time:
                compile_time = time.time() - compile_start_time
                self._update_compilation_stats(compile_time, False)

            self._log_failure(func_name, e, "compilation", 0)

            if not self.fallback_enabled:
                raise

            logger.warning(f"JIT compilation failed for {func_name}, using fallback")
            return self._create_fallback_wrapper(fallback_func, func_name)

        @wraps(func)
        def safe_wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_count = 0
            last_error = None

            while retry_count <= self.max_retries:
                try:
                    # パフォーマンス監視
                    if self.performance_monitoring:
                        start_time = time.time()

                    # デバッグモード
                    if self.debug_mode:
                        self.debug_info[func_name] = {
                            "args_shapes": [getattr(arg, "shape", None) for arg in args if hasattr(arg, "shape")],
                            "execution_attempt": retry_count + 1,
                        }

                    # JIT実行
                    result = jit_func(*args, **kwargs)

                    # パフォーマンス記録
                    if self.performance_monitoring:
                        execution_time = time.time() - start_time
                        if func_name not in self.performance_data:
                            self.performance_data[func_name] = []
                        self.performance_data[func_name].append(execution_time)

                    return result

                except Exception as e:
                    last_error = e
                    retry_count += 1

                    error_category = self._categorize_error(e)
                    self._log_failure(func_name, e, error_category, retry_count)

                    if retry_count > self.max_retries:
                        break

                    # リトライ前の短い待機
                    if retry_count < self.max_retries:
                        time.sleep(0.01 * retry_count)

            # 最終的にフォールバック実行
            if self.fallback_enabled:
                logger.warning(f"JIT execution failed for {func_name} after {self.max_retries} retries, using fallback")
                return fallback_func(*args, **kwargs)
            else:
                if last_error is not None:
                    raise last_error
                else:
                    raise RuntimeError(f"JIT execution failed for {func_name} with unknown error")

        return safe_wrapper

    def _create_fallback_wrapper(self, fallback_func: Callable[..., Any], func_name: str) -> Callable[..., Any]:
        """フォールバック関数のラッパー作成."""

        @wraps(fallback_func)
        def fallback_wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.performance_monitoring:
                start_time = time.time()
                result = fallback_func(*args, **kwargs)
                execution_time = time.time() - start_time

                if func_name not in self.performance_data:
                    self.performance_data[func_name] = []
                self.performance_data[func_name].append(execution_time)

                return result
            else:
                return fallback_func(*args, **kwargs)

        return fallback_wrapper

    def _categorize_error(self, error: Exception) -> str:
        """エラーの分類."""
        type(error).__name__.lower()
        error_message = str(error).lower()

        if "compilation" in error_message or "xla" in error_message:
            return "compilation"
        elif "memory" in error_message or isinstance(error, MemoryError):
            return "memory"
        elif isinstance(error, TypeError):
            return "type"
        elif "device" in error_message or "gpu" in error_message:
            return "device"
        else:
            return "unknown"

    def _log_failure(self, func_name: str, error: Exception, category: str, retry_count: int) -> None:
        """失敗の記録."""
        failure_record = FailureRecord(
            timestamp=datetime.now(),
            function_name=func_name,
            error_type=type(error).__name__,
            error_message=str(error),
            error_category=category,
            stack_trace=traceback.format_exc(),
            retry_count=retry_count,
        )

        self.failure_logs.append(failure_record)
        SafeJITWrapper._global_failure_logs.append(failure_record)

    def _update_compilation_stats(self, compile_time: float, success: bool) -> None:
        """コンパイル統計の更新."""
        self.compilation_stats.total_compilations += 1
        self.compilation_stats.total_compilation_time += compile_time

        if success:
            self.compilation_stats.successful_compilations += 1
        else:
            self.compilation_stats.failed_compilations += 1

        if self.compilation_stats.total_compilations > 0:
            self.compilation_stats.average_compilation_time = (
                self.compilation_stats.total_compilation_time / self.compilation_stats.total_compilations
            )

    def get_failure_report(self) -> dict[str, Any]:
        """失敗レポート生成.

        Returns:
            失敗レポート辞書
        """
        total_failures = len(self.failure_logs)
        recent_failures = [
            {
                "timestamp": record.timestamp,
                "function": record.function_name,
                "error_type": record.error_type,
                "error_message": record.error_message,
                "category": record.error_category,
                "retry_count": record.retry_count,
            }
            for record in self.failure_logs[-10:]  # 最新10件
        ]

        # エラーカテゴリ別の統計
        category_counts: dict[str, int] = {}
        for record in self.failure_logs:
            category = record.error_category
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total_failures": total_failures,
            "recent_failures": recent_failures,
            "error_categories": category_counts,
            "report_generated_at": datetime.now(),
        }

    def get_compilation_statistics(self) -> dict[str, Any]:
        """コンパイル統計取得.

        Returns:
            コンパイル統計辞書
        """
        return {
            "total_compilations": self.compilation_stats.total_compilations,
            "successful_compilations": self.compilation_stats.successful_compilations,
            "failed_compilations": self.compilation_stats.failed_compilations,
            "total_compilation_time": self.compilation_stats.total_compilation_time,
            "average_compilation_time": self.compilation_stats.average_compilation_time,
        }

    def get_performance_statistics(self) -> dict[str, Any]:
        """パフォーマンス統計取得.

        Returns:
            パフォーマンス統計辞書
        """
        return {
            "execution_times": self.performance_data.copy(),
            "monitored_functions": list(self.performance_data.keys()),
            "total_executions": sum(len(times) for times in self.performance_data.values()),
        }

    def get_debug_info(self) -> dict[str, Any]:
        """デバッグ情報取得.

        Returns:
            デバッグ情報辞書
        """
        return {
            "jit_trace_info": self.debug_info.copy(),
            "execution_details": {
                "fallback_enabled": self.fallback_enabled,
                "max_retries": self.max_retries,
                "debug_mode": self.debug_mode,
            },
        }

    def is_closed(self) -> bool:
        """クローズ状態確認.

        Returns:
            クローズ状態
        """
        return self._closed

    def __enter__(self) -> "SafeJITWrapper":
        """コンテキストマネージャー開始."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """コンテキストマネージャー終了."""
        self._closed = True
        return False

    @classmethod
    def reset_failure_logs(cls) -> None:
        """グローバル失敗ログの初期化."""
        cls._global_failure_logs.clear()
