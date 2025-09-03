"""Safe JIT wrapper unit tests."""

import time
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from riemannax.core.safe_jit import SafeJITWrapper


class TestSafeJITWrapper:
    """Unit tests for SafeJIT execution system."""

    def setup_method(self):
        """Setup before each test execution."""
        SafeJITWrapper.reset_failure_logs()

    def test_safe_jit_successful_execution(self):
        """Test successful JIT execution."""

        def simple_add(x, y):
            return x + y

        wrapper = SafeJITWrapper()
        safe_func = wrapper.safe_jit(simple_add)

        # Normal execution
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = safe_func(x, y)

        expected = jnp.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_jit_with_fallback_on_error(self):
        """Test fallback execution when JIT fails."""

        def problematic_function(x):
            # Function that may cause problems during JIT compilation
            if hasattr(x, "at"):
                raise RuntimeError("Simulated JIT compilation error")
            return x * 2

        def fallback_function(x):
            return x * 2

        wrapper = SafeJITWrapper(fallback_enabled=True)
        safe_func = wrapper.safe_jit(problematic_function, fallback_func=fallback_function)

        # Fallback execution is expected
        x = jnp.array([1.0, 2.0])
        result = safe_func(x)

        expected = jnp.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    @patch("jax.jit")
    def test_jit_compilation_failure_with_mock(self, mock_jit):
        """Mock test for JIT compilation failure."""

        def test_func(x):
            return x * 3

        def fallback_func(x):
            return x * 3

        # Simulate JIT compilation failure
        mock_jit_func = MagicMock()
        mock_jit_func.side_effect = RuntimeError("JIT compilation failed")
        mock_jit.return_value = mock_jit_func

        wrapper = SafeJITWrapper(fallback_enabled=True)
        safe_func = wrapper.safe_jit(test_func, fallback_func=fallback_func)

        # Fallback execution
        x = jnp.array([2.0, 4.0])
        result = safe_func(x)

        expected = jnp.array([6.0, 12.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_failure_report_generation(self):
        """Test failure report generation."""

        def failing_function(x):
            raise ValueError("Test error for failure report")

        wrapper = SafeJITWrapper(fallback_enabled=False)
        safe_func = wrapper.safe_jit(failing_function)

        # Failed execution
        x = jnp.array([1.0])
        with pytest.raises(Exception):
            safe_func(x)

        # Get failure report
        failure_report = wrapper.get_failure_report()

        assert "total_failures" in failure_report
        assert "recent_failures" in failure_report
        assert failure_report["total_failures"] > 0

    def test_max_retries_configuration(self):
        """Test maximum retry count configuration."""
        retry_count = 0

        def flaky_function(x):
            nonlocal retry_count
            retry_count += 1
            if retry_count <= 2:  # First 2 times fail
                raise RuntimeError("Flaky error")
            return x * 2

        wrapper = SafeJITWrapper(max_retries=3, fallback_enabled=False)
        safe_func = wrapper.safe_jit(flaky_function)

        # Expect success after retry
        x = jnp.array([1.0, 2.0])
        result = safe_func(x)

        expected = jnp.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)
        assert retry_count == 3  # Success on 3rd attempt

    def test_static_args_with_safe_jit(self):
        """Test Safe JIT with static arguments."""

        def manifold_operation(x, v, dim):
            return x + v * dim

        wrapper = SafeJITWrapper()
        safe_func = wrapper.safe_jit(manifold_operation, static_argnums=(2,))

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.1, 0.2])
        dim = 5

        result = safe_func(x, v, dim)
        expected = jnp.array([1.5, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fallback_disabled_raises_exception(self):
        """Test exception raising when fallback is disabled."""

        def always_failing_func(x):
            raise RuntimeError("Always fails")

        wrapper = SafeJITWrapper(fallback_enabled=False)
        safe_func = wrapper.safe_jit(always_failing_func)

        x = jnp.array([1.0])
        with pytest.raises(RuntimeError, match="Always fails"):
            safe_func(x)

    def test_compilation_time_tracking(self):
        """Test compilation time tracking."""

        def tracked_function(x):
            return jnp.sum(x)

        wrapper = SafeJITWrapper(track_compilation_time=True)
        safe_func = wrapper.safe_jit(tracked_function)

        # First execution (compilation occurs)
        x = jnp.array([1.0, 2.0, 3.0])
        safe_func(x)

        # Check if compilation time is recorded
        compilation_stats = wrapper.get_compilation_statistics()
        assert "total_compilations" in compilation_stats
        assert compilation_stats["total_compilations"] > 0

    def test_error_categorization(self):
        """Test error categorization functionality."""

        def compilation_error_func(x):
            raise RuntimeError("XLA compilation failed")

        def memory_error_func(x):
            raise MemoryError("Out of GPU memory")

        def type_error_func(x):
            raise TypeError("Invalid argument type")

        wrapper = SafeJITWrapper()

        # Execute with various errors
        for func, _expected_category in [
            (compilation_error_func, "compilation"),
            (memory_error_func, "memory"),
            (type_error_func, "type"),
        ]:
            safe_func = wrapper.safe_jit(func, fallback_func=lambda x: x)
            try:
                safe_func(jnp.array([1.0]))
            except:
                pass  # Errors are expected

        # Get error categorization report
        failure_report = wrapper.get_failure_report()
        assert "error_categories" in failure_report

    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""

        def fast_function(x):
            return x * 2

        def slow_function(x):
            # Simulate intentionally slow processing
            time.sleep(0.1)
            return x * 2

        wrapper = SafeJITWrapper(performance_monitoring=True, fallback_enabled=True)

        # Setup fast function and fallback
        safe_func = wrapper.safe_jit(fast_function, fallback_func=slow_function)

        x = jnp.array([1.0, 2.0])
        safe_func(x)

        # Get performance statistics
        perf_stats = wrapper.get_performance_statistics()
        assert "execution_times" in perf_stats

    def test_context_manager_interface(self):
        """Test context manager interface."""

        def test_function(x):
            return x + 1

        with SafeJITWrapper(fallback_enabled=True) as wrapper:
            safe_func = wrapper.safe_jit(test_function)

            x = jnp.array([1.0, 2.0])
            result = safe_func(x)

            expected = jnp.array([2.0, 3.0])
            np.testing.assert_array_almost_equal(result, expected)

        # Check state after context exit
        assert wrapper.is_closed()

    def test_debug_mode_detailed_logging(self):
        """Test detailed logging in debug mode."""

        def debug_function(x):
            return jnp.exp(x)

        wrapper = SafeJITWrapper(debug_mode=True)
        safe_func = wrapper.safe_jit(debug_function)

        x = jnp.array([0.0, 1.0])
        safe_func(x)

        # Get debug information
        debug_info = wrapper.get_debug_info()
        assert "jit_trace_info" in debug_info
        assert "execution_details" in debug_info
