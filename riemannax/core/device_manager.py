"""Device management system for hardware optimization."""

import logging
from datetime import datetime
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class DeviceManager:
    """ハードウェア最適化デバイス管理システム."""

    # クラス変数でデバイス情報と設定を管理
    _device_info: ClassVar[dict[str, Any]] = {}
    _current_config: ClassVar[dict[str, Any]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls) -> None:
        """デバイス情報の初期化と自動検出."""
        try:
            # JAXデバイスの検出
            devices = jax.devices()

            cls._device_info = {
                "available_devices": devices,
                "current_device": devices[0] if devices else None,
                "device_count": len(devices),
                "platforms": list({device.platform for device in devices}),
            }

            cls._initialized = True
            logger.info(f"Initialized with {len(devices)} devices: {[d.platform for d in devices]}")

        except Exception as e:
            logger.warning(f"Device initialization failed: {e}")
            # フォールバック: 最小限の構成
            cls._device_info = {
                "available_devices": [],
                "current_device": None,
                "device_count": 0,
                "platforms": ["cpu"],
            }
            cls._initialized = True

    @classmethod
    def get_device_info(cls) -> dict[str, Any]:
        """デバイス情報取得.

        Returns:
            デバイス情報辞書
        """
        if not cls._initialized:
            cls.initialize()
        return cls._device_info.copy()

    @classmethod
    def set_default_device(cls, device: str) -> None:
        """デフォルトデバイスの設定.

        Args:
            device: デバイス種別 ('cpu', 'gpu', 'auto')
        """
        if not cls._initialized:
            cls.initialize()

        available_devices = cls._device_info.get("available_devices", [])

        if device.lower() == "auto":
            # 自動選択: 最適なデバイスを選択
            preferred = cls.get_preferred_device()
            if preferred:
                cls._device_info["current_device"] = preferred
                logger.info(f"Auto-selected device: {preferred.platform}")

        elif device.lower() == "cpu":
            # CPU強制選択
            cpu_device = None
            for dev in available_devices:
                if dev.platform.lower() == "cpu":
                    cpu_device = dev
                    break

            if cpu_device:
                cls._device_info["current_device"] = cpu_device
                logger.info("Forced CPU device selection")
            else:
                logger.warning("CPU device not found, using first available device")

        elif device.lower() in ["gpu", "cuda"]:
            # GPU強制選択
            gpu_device = None
            for dev in available_devices:
                if dev.platform.lower() in ["gpu", "cuda"]:
                    gpu_device = dev
                    break

            if gpu_device:
                cls._device_info["current_device"] = gpu_device
                logger.info(f"Selected GPU device: {gpu_device}")
            else:
                logger.warning("GPU device not found, falling back to CPU")
                cls.fallback_to_cpu()

        else:
            logger.warning(f"Unknown device type '{device}', using auto selection")
            cls.set_default_device("auto")

        # 設定適用
        cls.configure_for_device()

    @classmethod
    def has_gpu(cls) -> bool:
        """GPU利用可能性チェック.

        Returns:
            GPU利用可能の可否
        """
        if not cls._initialized:
            cls.initialize()

        return any(
            device.platform.lower() in ["gpu", "cuda"] for device in cls._device_info.get("available_devices", [])
        )

    @classmethod
    def has_tpu(cls) -> bool:
        """TPU利用可能性チェック.

        Returns:
            TPU利用可能の可否
        """
        if not cls._initialized:
            cls.initialize()

        return any(device.platform.lower() == "tpu" for device in cls._device_info.get("available_devices", []))

    @classmethod
    def get_optimal_compilation_config(cls) -> dict[str, Any]:
        """最適コンパイル設定取得.

        Returns:
            最適化設定辞書
        """
        if not cls._initialized:
            cls.initialize()

        current_device = cls._device_info.get("current_device")
        device_platform = getattr(current_device, "platform", "cpu").lower()

        # デバイス別最適化設定
        base_config: dict[str, Any] = {"device": device_platform, "backend": device_platform, "xla_options": {}}

        if device_platform in ["gpu", "cuda"]:
            base_config["xla_options"].update(
                {"memory_optimization": True, "gpu_memory_fraction": 0.9, "allow_growth": True}
            )

            # 複数GPU検出時の設定
            gpu_count = sum(
                1 for d in cls._device_info.get("available_devices", []) if d.platform.lower() in ["gpu", "cuda"]
            )
            if gpu_count > 1:
                base_config["xla_options"]["multi_gpu"] = True

        elif device_platform == "tpu":
            base_config["xla_options"].update({"tpu_optimization": True, "sharding_strategy": "auto"})
        else:
            # CPU設定
            base_config["xla_options"].update({"cpu_parallel": True, "vectorization": True})

        return base_config

    @classmethod
    def configure_for_device(cls) -> None:
        """選択デバイス用の設定適用."""
        if not cls._initialized:
            cls.initialize()

        optimal_config = cls.get_optimal_compilation_config()
        current_device = cls._device_info.get("current_device")

        cls._current_config = {
            "device_type": optimal_config["backend"],
            "jit_settings": {"backend": optimal_config["backend"], "device_assignment": current_device},
            "xla_options": optimal_config["xla_options"],
            "applied_at": datetime.now(),
        }

    @classmethod
    def get_current_config(cls) -> dict[str, Any] | None:
        """現在の設定取得.

        Returns:
            現在の設定辞書
        """
        return cls._current_config.copy() if cls._current_config else None

    @classmethod
    def get_device_capabilities(cls) -> dict[str, Any]:
        """デバイス機能取得.

        Returns:
            デバイス機能辞書
        """
        if not cls._initialized:
            cls.initialize()

        current_device = cls._device_info.get("current_device")

        capabilities = {
            "supports_jit": True,  # JAXは常にJIT対応
            "supports_vmap": True,  # JAXは常にvmap対応
            "max_memory": cls._estimate_device_memory(),
            "compute_units": getattr(current_device, "core_count", 1) if current_device else 1,
            "precision_support": ["float32", "float64", "bfloat16"],
        }

        return capabilities

    @classmethod
    def estimate_memory_usage(cls, array_shape: tuple[int, ...], dtype=jnp.float32) -> int:
        """メモリ使用量推定.

        Args:
            array_shape: 配列形状
            dtype: データ型

        Returns:
            推定メモリ使用量(バイト)
        """
        # 要素数計算
        num_elements = 1
        for dim in array_shape:
            num_elements *= dim

        # データ型別バイト数
        dtype_sizes = {jnp.float32: 4, jnp.float64: 8, jnp.int32: 4, jnp.int64: 8, jnp.complex64: 8, jnp.complex128: 16}

        element_size = dtype_sizes.get(dtype, 4)  # デフォルト: float32
        return num_elements * element_size

    @classmethod
    def get_preferred_device(cls):
        """優先デバイス取得.

        Returns:
            優先デバイス
        """
        if not cls._initialized:
            cls.initialize()

        devices = cls._device_info.get("available_devices", [])
        if not devices:
            return None

        # 優先順位: TPU > GPU > CPU
        for device in devices:
            if device.platform.lower() == "tpu":
                return device

        for device in devices:
            if device.platform.lower() in ["gpu", "cuda"]:
                return device

        # CPUフォールバック
        return devices[0]

    @classmethod
    def fallback_to_cpu(cls) -> bool:
        """CPUフォールバック実行.

        Returns:
            フォールバック成功の可否
        """
        try:
            if not cls._initialized:
                cls.initialize()

            devices = cls._device_info.get("available_devices", [])
            cpu_device = None

            for device in devices:
                if device.platform.lower() == "cpu":
                    cpu_device = device
                    break

            if cpu_device:
                cls._device_info["current_device"] = cpu_device
                cls.configure_for_device()
                return True
            else:
                logger.warning("No CPU device found for fallback")
                return False

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}")
            return False

    @classmethod
    def get_performance_metrics(cls) -> dict[str, Any]:
        """デバイス性能メトリクス取得.

        Returns:
            性能メトリクス辞書
        """
        if not cls._initialized:
            cls.initialize()

        current_device = cls._device_info.get("current_device")
        platform = getattr(current_device, "platform", "unknown").lower()

        metrics = {
            "device_platform": platform,
            "compute_capability": cls._estimate_compute_capability(platform),
        }

        if platform in ["gpu", "cuda"]:
            metrics.update({"memory_bandwidth": "estimated_high", "tensor_cores": True, "estimated_flops": "high"})
        elif platform == "tpu":
            metrics.update(
                {"memory_bandwidth": "estimated_very_high", "matrix_units": True, "estimated_flops": "very_high"}
            )
        else:
            metrics.update({"memory_bandwidth": "estimated_medium", "estimated_flops": "medium"})

        return metrics

    @classmethod
    def reset(cls) -> None:
        """デバイス管理状態の初期化."""
        cls._device_info = {}
        cls._current_config = {}
        cls._initialized = False

    @classmethod
    def _estimate_device_memory(cls) -> str:
        """デバイスメモリ推定."""
        if cls.has_tpu():
            return "32GB+"
        elif cls.has_gpu():
            return "4GB-80GB"
        else:
            return "system_ram"

    @classmethod
    def _estimate_compute_capability(cls, platform: str) -> str:
        """計算能力推定."""
        capability_map = {"tpu": "very_high", "gpu": "high", "cuda": "high", "cpu": "medium"}
        return capability_map.get(platform, "unknown")
