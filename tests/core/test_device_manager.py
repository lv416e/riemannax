"""Device manager unit tests."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from riemannax.core.device_manager import DeviceManager


class TestDeviceManager:
    """Unit tests for device management system."""

    def setup_method(self):
        """Setup before each test execution."""
        DeviceManager.reset()

    def test_initialize_device_detection(self):
        """Test device auto-detection initialization."""
        DeviceManager.initialize()

        # Confirm device information is set after initialization
        device_info = DeviceManager.get_device_info()

        assert "available_devices" in device_info
        assert "current_device" in device_info
        assert "device_count" in device_info

        # Confirm at least CPU is available
        assert len(device_info["available_devices"]) >= 1
        assert any("cpu" in str(device).lower() for device in device_info["available_devices"])

    @patch("jax.devices")
    def test_gpu_detection_with_mock(self, mock_devices):
        """Mock test for GPU detection."""
        # Mock GPU environment
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_gpu.device_kind = "GPU"
        mock_devices.return_value = [mock_gpu]

        DeviceManager.initialize()
        device_info = DeviceManager.get_device_info()

        # Confirm GPU detection
        assert device_info["device_count"] == 1
        assert DeviceManager.has_gpu()

    @patch("jax.devices")
    def test_tpu_detection_with_mock(self, mock_devices):
        """Mock test for TPU detection."""
        # Mock TPU environment
        mock_tpu = MagicMock()
        mock_tpu.platform = "tpu"
        mock_tpu.device_kind = "TPU"
        mock_devices.return_value = [mock_tpu]

        DeviceManager.initialize()

        # Confirm TPU detection
        assert DeviceManager.has_tpu()
        device_info = DeviceManager.get_device_info()
        assert device_info["device_count"] == 1

    def test_get_optimal_compilation_config_cpu(self):
        """Test getting optimal compilation config for CPU."""
        DeviceManager.initialize()

        config = DeviceManager.get_optimal_compilation_config()

        # Check basic settings
        assert isinstance(config, dict)
        assert "device" in config
        assert "backend" in config
        assert "xla_options" in config

    @patch("jax.devices")
    def test_get_optimal_compilation_config_gpu(self, mock_devices):
        """Test getting optimal compilation config for GPU."""
        # Mock GPU environment
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_gpu.device_kind = "GPU"
        mock_devices.return_value = [mock_gpu]

        DeviceManager.initialize()
        config = DeviceManager.get_optimal_compilation_config()

        # Check GPU-specific settings
        assert config["backend"] in ["gpu", "cuda"]
        assert "memory_optimization" in config["xla_options"]

    def test_configure_for_device_cpu(self):
        """Test applying device configuration for CPU."""
        DeviceManager.initialize()
        DeviceManager.configure_for_device()

        # Check state after configuration applied
        current_config = DeviceManager.get_current_config()
        assert current_config is not None
        assert "jit_settings" in current_config

    @patch("jax.devices")
    def test_configure_for_device_gpu(self, mock_devices):
        """Test applying device configuration for GPU."""
        # Mock GPU environment
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_devices.return_value = [mock_gpu]

        DeviceManager.initialize()
        DeviceManager.configure_for_device()

        current_config = DeviceManager.get_current_config()
        assert current_config["device_type"] == "gpu"

    def test_device_capability_check(self):
        """Test device capability check."""
        DeviceManager.initialize()

        # Basic capability check
        capabilities = DeviceManager.get_device_capabilities()

        assert isinstance(capabilities, dict)
        assert "supports_jit" in capabilities
        assert "supports_vmap" in capabilities
        assert "max_memory" in capabilities

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        DeviceManager.initialize()

        # Memory estimation for sample array
        array_shape = (1000, 1000)
        dtype = jnp.float32

        estimated_memory = DeviceManager.estimate_memory_usage(array_shape, dtype)

        # Expect about 4MB of memory usage
        expected_bytes = 1000 * 1000 * 4  # float32 is 4 bytes
        assert abs(estimated_memory - expected_bytes) < 1000  # Error tolerance

    def test_device_selection_preference(self):
        """Test device selection priority."""
        DeviceManager.initialize()

        # Check current device selection logic
        preferred_device = DeviceManager.get_preferred_device()

        assert preferred_device is not None
        assert hasattr(preferred_device, "platform")

    @patch("jax.devices")
    def test_multi_gpu_handling(self, mock_devices):
        """Test multi-GPU environment handling."""
        # Mock multiple GPUs
        mock_gpu1 = MagicMock()
        mock_gpu1.platform = "gpu"
        mock_gpu1.id = 0
        mock_gpu2 = MagicMock()
        mock_gpu2.platform = "gpu"
        mock_gpu2.id = 1
        mock_devices.return_value = [mock_gpu1, mock_gpu2]

        DeviceManager.initialize()

        device_info = DeviceManager.get_device_info()
        assert device_info["device_count"] == 2

        # Check multi-GPU support settings
        config = DeviceManager.get_optimal_compilation_config()
        assert "multi_gpu" in config["xla_options"]

    def test_fallback_to_cpu(self):
        """Test CPU fallback mechanism."""
        # Device initialization
        DeviceManager.initialize()

        # Force CPU fallback execution
        success = DeviceManager.fallback_to_cpu()

        assert success is True
        device_info = DeviceManager.get_device_info()
        current_device = device_info["current_device"]
        assert "cpu" in str(current_device).lower()

    def test_device_performance_metrics(self):
        """Test device performance metrics."""
        DeviceManager.initialize()

        # Get performance metrics
        metrics = DeviceManager.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "compute_capability" in metrics
        assert "memory_bandwidth" in metrics or "estimated_flops" in metrics
