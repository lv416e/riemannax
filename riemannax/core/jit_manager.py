"""JIT Management System for RiemannAX optimization."""

from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar

import jax


class JITManager:
    """JIT最適化の中央管理システム."""

    # クラス変数で設定とキャッシュを管理
    _config: ClassVar[dict[str, Any]] = {
        "enable_jit": True,
        "cache_size": 10000,
        "fallback_on_error": True,
        "debug_mode": False,
    }

    _cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def configure(cls, **kwargs) -> None:
        """JIT設定の更新.

        Args:
            **kwargs: 設定パラメータ
                - enable_jit: JIT最適化の有効/無効
                - cache_size: キャッシュサイズ
                - fallback_on_error: エラー時のフォールバック
                - debug_mode: デバッグモード
        """
        cls._config.update(kwargs)

    @classmethod
    def jit_decorator(
        cls, func: Callable, static_argnums: tuple[int, ...] | None = None, device: str | None = None
    ) -> Callable:
        """統一JITデコレータ.

        Args:
            func: JIT最適化対象の関数
            static_argnums: 静的引数のインデックス
            device: 実行デバイス指定 (cpu, gpu, tpu)

        Returns:
            JIT最適化された関数
        """
        if not cls._config["enable_jit"]:
            return func

        # JIT設定準備
        jit_kwargs = {}
        if static_argnums is not None:
            jit_kwargs["static_argnums"] = static_argnums

        # デバイス指定がある場合の処理
        if device is not None:
            # 現在は基本実装のみ(デバイス指定は将来の拡張)
            pass

        # JIT最適化関数の作成
        jit_func = jax.jit(func, static_argnums=static_argnums)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return jit_func(*args, **kwargs)
            except Exception as e:
                if cls._config["fallback_on_error"]:
                    # フォールバック実行
                    return func(*args, **kwargs)
                else:
                    raise e

        return wrapper

    @classmethod
    def clear_cache(cls) -> None:
        """JITキャッシュクリア."""
        cls._cache.clear()

    @classmethod
    def get_config(cls) -> dict[str, Any]:
        """現在の設定を取得.

        Returns:
            現在の設定辞書
        """
        return cls._config.copy()

    @classmethod
    def reset_config(cls) -> None:
        """設定をデフォルトに戻す."""
        cls._config = {"enable_jit": True, "cache_size": 10000, "fallback_on_error": True, "debug_mode": False}
