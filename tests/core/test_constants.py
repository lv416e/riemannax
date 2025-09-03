"""Test module for configuration constants."""

from riemannax.core.constants import NumericalConstants, PerformanceThresholds


class TestNumericalConstants:
    """Test numerical constants configuration."""

    def test_epsilon_value(self):
        """Test that EPSILON has the expected value."""
        assert NumericalConstants.EPSILON == 1e-10

    def test_rtol_value(self):
        """Test that RTOL has the expected value."""
        assert NumericalConstants.RTOL == 1e-8

    def test_atol_value(self):
        """Test that ATOL has the expected value."""
        assert NumericalConstants.ATOL == 1e-10

    def test_high_precision_epsilon_value(self):
        """Test that HIGH_PRECISION_EPSILON has the expected value."""
        assert NumericalConstants.HIGH_PRECISION_EPSILON == 1e-12

    def test_constants_are_float_type(self):
        """Test that all constants are float type."""
        assert isinstance(NumericalConstants.EPSILON, float)
        assert isinstance(NumericalConstants.RTOL, float)
        assert isinstance(NumericalConstants.ATOL, float)
        assert isinstance(NumericalConstants.HIGH_PRECISION_EPSILON, float)

    def test_constants_are_positive(self):
        """Test that all tolerance constants are positive."""
        assert NumericalConstants.EPSILON > 0
        assert NumericalConstants.RTOL > 0
        assert NumericalConstants.ATOL > 0
        assert NumericalConstants.HIGH_PRECISION_EPSILON > 0

    def test_epsilon_less_than_rtol(self):
        """Test the relationship between EPSILON and RTOL."""
        assert NumericalConstants.EPSILON <= NumericalConstants.RTOL

    def test_high_precision_epsilon_less_than_epsilon(self):
        """Test that HIGH_PRECISION_EPSILON is smaller than EPSILON."""
        assert NumericalConstants.HIGH_PRECISION_EPSILON < NumericalConstants.EPSILON


class TestPerformanceThresholds:
    """Test performance threshold configuration."""

    def test_min_cpu_speedup_value(self):
        """Test that MIN_CPU_SPEEDUP has the expected value."""
        assert PerformanceThresholds.MIN_CPU_SPEEDUP == 1.1

    def test_min_gpu_speedup_value(self):
        """Test that MIN_GPU_SPEEDUP has the expected value."""
        assert PerformanceThresholds.MIN_GPU_SPEEDUP == 2.0

    def test_thresholds_are_float_type(self):
        """Test that all thresholds are float type."""
        assert isinstance(PerformanceThresholds.MIN_CPU_SPEEDUP, float)
        assert isinstance(PerformanceThresholds.MIN_GPU_SPEEDUP, float)

    def test_thresholds_are_greater_than_one(self):
        """Test that speedup thresholds are greater than 1.0 (actual speedup)."""
        assert PerformanceThresholds.MIN_CPU_SPEEDUP > 1.0
        assert PerformanceThresholds.MIN_GPU_SPEEDUP > 1.0

    def test_gpu_speedup_greater_than_cpu(self):
        """Test that GPU speedup threshold is higher than CPU."""
        assert PerformanceThresholds.MIN_GPU_SPEEDUP > PerformanceThresholds.MIN_CPU_SPEEDUP

    def test_constants_accessible_as_class_attributes(self):
        """Test that constants can be accessed as class attributes."""
        # Should not raise AttributeError
        _ = PerformanceThresholds.MIN_CPU_SPEEDUP
        _ = PerformanceThresholds.MIN_GPU_SPEEDUP
        _ = NumericalConstants.EPSILON
        _ = NumericalConstants.RTOL
        _ = NumericalConstants.ATOL


class TestConstantsEdgeCases:
    """Test edge cases and comprehensive coverage for constants module."""

    def test_constants_modification_behavior(self):
        """Test behavior when attempting to modify constants."""
        # Store original values
        original_epsilon = NumericalConstants.EPSILON
        original_cpu_speedup = PerformanceThresholds.MIN_CPU_SPEEDUP

        # Attempt to modify (this may succeed in Python but shouldn't be done)
        NumericalConstants.EPSILON = 1e-5
        PerformanceThresholds.MIN_CPU_SPEEDUP = 3.0

        # Verify modification occurred (showing they are not truly immutable)
        assert NumericalConstants.EPSILON == 1e-5
        assert PerformanceThresholds.MIN_CPU_SPEEDUP == 3.0

        # Restore original values for other tests
        NumericalConstants.EPSILON = original_epsilon
        PerformanceThresholds.MIN_CPU_SPEEDUP = original_cpu_speedup

    def test_constants_string_representation(self):
        """Test string representation of constants classes."""
        # Check that classes have proper string representations
        constants_repr = str(NumericalConstants)
        performance_repr = str(PerformanceThresholds)

        assert "NumericalConstants" in constants_repr
        assert "PerformanceThresholds" in performance_repr

    def test_all_numerical_constants_defined(self):
        """Test that all expected numerical constants are defined."""
        expected_constants = ["EPSILON", "RTOL", "ATOL", "HIGH_PRECISION_EPSILON"]

        for constant_name in expected_constants:
            assert hasattr(NumericalConstants, constant_name)
            assert isinstance(getattr(NumericalConstants, constant_name), float)

    def test_all_performance_thresholds_defined(self):
        """Test that all expected performance thresholds are defined."""
        expected_thresholds = ["MIN_CPU_SPEEDUP", "MIN_GPU_SPEEDUP"]

        for threshold_name in expected_thresholds:
            assert hasattr(PerformanceThresholds, threshold_name)
            assert isinstance(getattr(PerformanceThresholds, threshold_name), float)

    def test_constants_mathematical_relationships(self):
        """Test mathematical relationships between constants."""
        # Test ordering of precision constants
        assert NumericalConstants.HIGH_PRECISION_EPSILON < NumericalConstants.EPSILON < NumericalConstants.RTOL

        # Test absolute and relative tolerances relationship
        assert NumericalConstants.ATOL <= NumericalConstants.RTOL

    def test_performance_thresholds_realistic_values(self):
        """Test that performance thresholds have realistic values."""
        # CPU speedup should be modest but meaningful
        assert 1.0 <= PerformanceThresholds.MIN_CPU_SPEEDUP <= 10.0

        # GPU speedup should be significant
        assert 1.5 <= PerformanceThresholds.MIN_GPU_SPEEDUP <= 100.0

        # GPU should have higher threshold than CPU
        ratio = PerformanceThresholds.MIN_GPU_SPEEDUP / PerformanceThresholds.MIN_CPU_SPEEDUP
        assert 1.5 <= ratio <= 10.0

    def test_constants_precision_consistency(self):
        """Test that precision constants are internally consistent."""
        # High precision should be at least 100x more precise than standard
        precision_ratio = NumericalConstants.EPSILON / NumericalConstants.HIGH_PRECISION_EPSILON
        assert precision_ratio >= 100.0

        # Relative tolerance should be reasonable compared to absolute
        if NumericalConstants.ATOL > 0:
            rtol_atol_ratio = NumericalConstants.RTOL / NumericalConstants.ATOL
            assert 0.01 <= rtol_atol_ratio <= 1000.0

    def test_constants_module_attributes(self):
        """Test module-level attributes and structure."""
        import riemannax.core.constants as constants_module

        # Check that main classes are available at module level
        assert hasattr(constants_module, "NumericalConstants")
        assert hasattr(constants_module, "PerformanceThresholds")

        # Check that classes are actually classes
        assert isinstance(constants_module.NumericalConstants, type)
        assert isinstance(constants_module.PerformanceThresholds, type)
