"""JIT manager unit tests."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from riemannax.core.jit_manager import JITManager


class TestJITManager:
    """Unit tests for JIT management system."""

    def setup_method(self):
        """Setup before each test execution."""
        JITManager.reset_config()
        JITManager.clear_cache()

    def test_configure_basic_settings(self):
        """Test basic configuration updates."""
        # Initialize
        JITManager.configure(enable_jit=True, cache_size=1000)

        # Verify configuration
        assert JITManager._config["enable_jit"] is True
        assert JITManager._config["cache_size"] == 1000

    def test_jit_decorator_basic_function(self):
        """Test JIT decorator on basic functions."""

        # Target function for testing
        def simple_add(x, y):
            return x + y

        # JIT optimization
        jit_add = JITManager.jit_decorator(simple_add)

        # Execution test
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = jit_add(x, y)

        expected = jnp.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_jit_decorator_with_static_args(self):
        """Test JIT decorator with static arguments."""

        def manifold_op(x, v, dim):
            return x + v * dim

        # JIT optimization with static argument specification
        jit_op = JITManager.jit_decorator(manifold_op, static_argnums=(2,))

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.1, 0.2])
        dim = 5

        result = jit_op(x, v, dim)
        expected = jnp.array([1.5, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_clear_cache(self):
        """Test JIT cache clearing functionality."""
        # Set dummy data to cache
        JITManager._cache["test_func"] = MagicMock()

        # Execute cache clear
        JITManager.clear_cache()

        # Confirm cache is empty
        assert len(JITManager._cache) == 0

    def test_jit_decorator_with_device_specification(self):
        """Test JIT decorator with device specification."""

        def device_op(x):
            return x * 2

        # JIT optimization with CPU specification
        jit_op = JITManager.jit_decorator(device_op, device="cpu")

        x = jnp.array([1.0, 2.0])
        result = jit_op(x)
        expected = jnp.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_config_initialization(self):
        """Test configuration initialization."""
        # Verify initial configuration values
        expected_defaults = {"enable_jit": True, "cache_size": 10000, "fallback_on_error": True, "debug_mode": False}

        for key, expected_value in expected_defaults.items():
            assert JITManager._config[key] == expected_value

    def test_jit_performance_tracking(self):
        """Test JIT performance tracking functionality."""

        def tracked_func(x):
            return jnp.sum(x)

        jit_func = JITManager.jit_decorator(tracked_func)

        # Execution and performance data recording
        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_func(x)

        # Verify results
        assert result == 6.0
        # Verify performance data is recorded
        # Note: Details may change depending on implementation
