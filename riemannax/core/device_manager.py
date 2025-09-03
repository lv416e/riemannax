"""Device management system for hardware optimization."""

import logging
from datetime import datetime
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class DeviceManager:
    """Hardware optimization device management system."""

    # Manage device information and configuration through class variables
    _device_info: ClassVar[dict[str, Any]] = {}
    _current_config: ClassVar[dict[str, Any]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize device information and automatic detection."""
        try:
            # JAX device detection
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
            # Fallback: minimal configuration
            cls._device_info = {
                "available_devices": [],
                "current_device": None,
                "device_count": 0,
                "platforms": ["cpu"],
            }
            cls._initialized = True

    @classmethod
    def get_device_info(cls) -> dict[str, Any]:
        """Get device information.

        Returns:
            Device information dictionary
        """
        if not cls._initialized:
            cls.initialize()
        return cls._device_info.copy()

    @classmethod
    def set_default_device(cls, device: str) -> None:
        """Set default device.

        Args:
            device: Device type ('cpu', 'gpu', 'auto')
        """
        if not cls._initialized:
            cls.initialize()

        available_devices = cls._device_info.get("available_devices", [])

        if device.lower() == "auto":
            # Automatic selection: select optimal device
            preferred = cls.get_preferred_device()
            if preferred:
                cls._device_info["current_device"] = preferred
                logger.info(f"Auto-selected device: {preferred.platform}")

        elif device.lower() == "cpu":
            # Force CPU selection
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
            # Force GPU selection
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

        # Apply configuration
        cls.configure_for_device()

    @classmethod
    def has_gpu(cls) -> bool:
        """Check GPU availability.

        Returns:
            Whether GPU is available
        """
        if not cls._initialized:
            cls.initialize()

        return any(
            device.platform.lower() in ["gpu", "cuda"] for device in cls._device_info.get("available_devices", [])
        )

    @classmethod
    def has_tpu(cls) -> bool:
        """Check TPU availability.

        Returns:
            Whether TPU is available
        """
        if not cls._initialized:
            cls.initialize()

        return any(device.platform.lower() == "tpu" for device in cls._device_info.get("available_devices", []))

    @classmethod
    def get_optimal_compilation_config(cls) -> dict[str, Any]:
        """Get optimal compilation configuration.

        Returns:
            Optimization configuration dictionary
        """
        if not cls._initialized:
            cls.initialize()

        current_device = cls._device_info.get("current_device")
        device_platform = getattr(current_device, "platform", "cpu").lower()

        # Device-specific optimization settings
        base_config: dict[str, Any] = {"device": device_platform, "backend": device_platform, "xla_options": {}}

        if device_platform in ["gpu", "cuda"]:
            base_config["xla_options"].update(
                {"memory_optimization": True, "gpu_memory_fraction": 0.9, "allow_growth": True}
            )

            # Configuration for multiple GPU detection
            gpu_count = sum(
                1 for d in cls._device_info.get("available_devices", []) if d.platform.lower() in ["gpu", "cuda"]
            )
            if gpu_count > 1:
                base_config["xla_options"]["multi_gpu"] = True

        elif device_platform == "tpu":
            base_config["xla_options"].update({"tpu_optimization": True, "sharding_strategy": "auto"})
        else:
            # CPU configuration
            base_config["xla_options"].update({"cpu_parallel": True, "vectorization": True})

        return base_config

    @classmethod
    def configure_for_device(cls) -> None:
        """Apply configuration for selected device."""
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
        """Get current configuration.

        Returns:
            Current configuration dictionary
        """
        return cls._current_config.copy() if cls._current_config else None

    @classmethod
    def get_device_capabilities(cls) -> dict[str, Any]:
        """Get device capabilities.

        Returns:
            Device capabilities dictionary
        """
        if not cls._initialized:
            cls.initialize()

        current_device = cls._device_info.get("current_device")

        capabilities = {
            "supports_jit": True,  # JAX always supports JIT
            "supports_vmap": True,  # JAX always supports vmap
            "max_memory": cls._estimate_device_memory(),
            "compute_units": getattr(current_device, "core_count", 1) if current_device else 1,
            "precision_support": ["float32", "float64", "bfloat16"],
        }

        return capabilities

    @classmethod
    def estimate_memory_usage(cls, array_shape: tuple[int, ...], dtype: Any = jnp.float32) -> int:
        """Estimate memory usage.

        Args:
            array_shape: Array shape
            dtype: Data type

        Returns:
            Estimated memory usage in bytes
        """
        # Calculate number of elements
        num_elements = 1
        for dim in array_shape:
            num_elements *= dim

        # Bytes per data type
        dtype_sizes = {jnp.float32: 4, jnp.float64: 8, jnp.int32: 4, jnp.int64: 8, jnp.complex64: 8, jnp.complex128: 16}

        element_size = dtype_sizes.get(dtype, 4)  # Default: float32
        return num_elements * element_size

    @classmethod
    def get_preferred_device(cls) -> Any:
        """Get preferred device.

        Returns:
            Preferred device
        """
        if not cls._initialized:
            cls.initialize()

        devices = cls._device_info.get("available_devices", [])
        if not devices:
            return None

        # Priority order: TPU > GPU > CPU
        for device in devices:
            if device.platform.lower() == "tpu":
                return device

        for device in devices:
            if device.platform.lower() in ["gpu", "cuda"]:
                return device

        # CPU fallback
        return devices[0]

    @classmethod
    def fallback_to_cpu(cls) -> bool:
        """Execute CPU fallback.

        Returns:
            Whether fallback was successful
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
        """Get device performance metrics.

        Returns:
            Performance metrics dictionary
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
        """Reset device management state."""
        cls._device_info = {}
        cls._current_config = {}
        cls._initialized = False

    @classmethod
    def _estimate_device_memory(cls) -> str:
        """Estimate device memory."""
        if cls.has_tpu():
            return "32GB+"
        elif cls.has_gpu():
            return "4GB-80GB"
        else:
            return "system_ram"

    @classmethod
    def _estimate_compute_capability(cls, platform: str) -> str:
        """Estimate compute capability."""
        capability_map = {"tpu": "very_high", "gpu": "high", "cuda": "high", "cpu": "medium"}
        return capability_map.get(platform, "unknown")
