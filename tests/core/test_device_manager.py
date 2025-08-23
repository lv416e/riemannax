"""Device manager unit tests."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from riemannax.core.device_manager import DeviceManager


class TestDeviceManager:
    """デバイス管理システムのユニットテスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        DeviceManager.reset()

    def test_initialize_device_detection(self):
        """デバイス自動検出の初期化テスト."""
        DeviceManager.initialize()

        # 初期化後にデバイス情報が設定されることを確認
        device_info = DeviceManager.get_device_info()

        assert "available_devices" in device_info
        assert "current_device" in device_info
        assert "device_count" in device_info

        # 少なくともCPUは利用可能であることを確認
        assert len(device_info["available_devices"]) >= 1
        assert any("cpu" in str(device).lower() for device in device_info["available_devices"])

    @patch("jax.devices")
    def test_gpu_detection_with_mock(self, mock_devices):
        """GPU検出のモックテスト."""
        # GPU環境をモック
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_gpu.device_kind = "GPU"
        mock_devices.return_value = [mock_gpu]

        DeviceManager.initialize()
        device_info = DeviceManager.get_device_info()

        # GPU検出の確認
        assert device_info["device_count"] == 1
        assert DeviceManager.has_gpu()

    @patch("jax.devices")
    def test_tpu_detection_with_mock(self, mock_devices):
        """TPU検出のモックテスト."""
        # TPU環境をモック
        mock_tpu = MagicMock()
        mock_tpu.platform = "tpu"
        mock_tpu.device_kind = "TPU"
        mock_devices.return_value = [mock_tpu]

        DeviceManager.initialize()

        # TPU検出の確認
        assert DeviceManager.has_tpu()
        device_info = DeviceManager.get_device_info()
        assert device_info["device_count"] == 1

    def test_get_optimal_compilation_config_cpu(self):
        """CPU用最適コンパイル設定取得テスト."""
        DeviceManager.initialize()

        config = DeviceManager.get_optimal_compilation_config()

        # 基本設定の確認
        assert isinstance(config, dict)
        assert "device" in config
        assert "backend" in config
        assert "xla_options" in config

    @patch("jax.devices")
    def test_get_optimal_compilation_config_gpu(self, mock_devices):
        """GPU用最適コンパイル設定取得テスト."""
        # GPU環境をモック
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_gpu.device_kind = "GPU"
        mock_devices.return_value = [mock_gpu]

        DeviceManager.initialize()
        config = DeviceManager.get_optimal_compilation_config()

        # GPU固有設定の確認
        assert config["backend"] in ["gpu", "cuda"]
        assert "memory_optimization" in config["xla_options"]

    def test_configure_for_device_cpu(self):
        """CPU用デバイス設定適用テスト."""
        DeviceManager.initialize()
        DeviceManager.configure_for_device()

        # 設定適用後の状態確認
        current_config = DeviceManager.get_current_config()
        assert current_config is not None
        assert "jit_settings" in current_config

    @patch("jax.devices")
    def test_configure_for_device_gpu(self, mock_devices):
        """GPU用デバイス設定適用テスト."""
        # GPU環境をモック
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_devices.return_value = [mock_gpu]

        DeviceManager.initialize()
        DeviceManager.configure_for_device()

        current_config = DeviceManager.get_current_config()
        assert current_config["device_type"] == "gpu"

    def test_device_capability_check(self):
        """デバイス機能チェックテスト."""
        DeviceManager.initialize()

        # 基本機能チェック
        capabilities = DeviceManager.get_device_capabilities()

        assert isinstance(capabilities, dict)
        assert "supports_jit" in capabilities
        assert "supports_vmap" in capabilities
        assert "max_memory" in capabilities

    def test_memory_estimation(self):
        """メモリ使用量推定テスト."""
        DeviceManager.initialize()

        # サンプル配列でのメモリ推定
        array_shape = (1000, 1000)
        dtype = jnp.float32

        estimated_memory = DeviceManager.estimate_memory_usage(array_shape, dtype)

        # 4MB程度のメモリ使用を期待
        expected_bytes = 1000 * 1000 * 4  # float32は4バイト
        assert abs(estimated_memory - expected_bytes) < 1000  # 誤差許容

    def test_device_selection_preference(self):
        """デバイス選択優先度テスト."""
        DeviceManager.initialize()

        # 現在のデバイス選択ロジック確認
        preferred_device = DeviceManager.get_preferred_device()

        assert preferred_device is not None
        assert hasattr(preferred_device, "platform")

    @patch("jax.devices")
    def test_multi_gpu_handling(self, mock_devices):
        """複数GPU環境の処理テスト."""
        # 複数GPUをモック
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

        # 複数GPU対応設定の確認
        config = DeviceManager.get_optimal_compilation_config()
        assert "multi_gpu" in config["xla_options"]

    def test_fallback_to_cpu(self):
        """CPU フォールバック機構テスト."""
        # デバイス初期化
        DeviceManager.initialize()

        # 強制的にCPUフォールバックを実行
        success = DeviceManager.fallback_to_cpu()

        assert success is True
        device_info = DeviceManager.get_device_info()
        current_device = device_info["current_device"]
        assert "cpu" in str(current_device).lower()

    def test_device_performance_metrics(self):
        """デバイス性能メトリクステスト."""
        DeviceManager.initialize()

        # 性能メトリクス取得
        metrics = DeviceManager.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "compute_capability" in metrics
        assert "memory_bandwidth" in metrics or "estimated_flops" in metrics
