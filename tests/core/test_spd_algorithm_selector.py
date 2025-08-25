"""Tests for SPD Algorithm Selector.

This module tests the adaptive algorithm selection logic for SPD manifold operations.
Tests verify that the selector chooses appropriate algorithms based on matrix size
and condition number.
"""

import pytest

from riemannax.core.spd_algorithm_selector import SPDAlgorithmSelector


class TestSPDAlgorithmSelector:
    """Test suite for SPD Algorithm Selector."""

    def test_select_method_small_matrix_good_condition(self) -> None:
        """Test algorithm selection for small matrices with good condition number."""
        selector = SPDAlgorithmSelector()

        # Small matrix (n=500) with good condition number (1e5)
        method = selector.select_method(n=500, condition_number=1e5)

        assert method == "eigendecomposition"

    def test_select_method_large_matrix_good_condition(self) -> None:
        """Test algorithm selection for large matrices with good condition number."""
        selector = SPDAlgorithmSelector()

        # Large matrix (n=1500) with good condition number (1e8)
        method = selector.select_method(n=1500, condition_number=1e8)

        assert method == "cholesky"

    def test_select_method_large_matrix_bad_condition(self) -> None:
        """Test algorithm selection for large matrices with poor condition number."""
        selector = SPDAlgorithmSelector()

        # Large matrix (n=2000) with poor condition number (1e13)
        method = selector.select_method(n=2000, condition_number=1e13)

        assert method == "eigendecomposition"

    def test_select_method_edge_case_condition_threshold(self) -> None:
        """Test algorithm selection at condition number threshold."""
        selector = SPDAlgorithmSelector()

        # Exactly at condition threshold (1e12)
        method = selector.select_method(n=1500, condition_number=1e12)

        assert method == "cholesky"

        # Just above condition threshold
        method = selector.select_method(n=1500, condition_number=1e12 + 1)

        assert method == "eigendecomposition"

    def test_select_method_edge_case_size_threshold(self) -> None:
        """Test algorithm selection at size threshold."""
        selector = SPDAlgorithmSelector()

        # Exactly at size threshold (n=1000) with good condition
        method = selector.select_method(n=1000, condition_number=1e8)

        assert method == "eigendecomposition"

        # Just above size threshold
        method = selector.select_method(n=1001, condition_number=1e8)

        assert method == "cholesky"

    def test_select_method_return_type(self) -> None:
        """Test that select_method returns a string."""
        selector = SPDAlgorithmSelector()

        result = selector.select_method(n=500, condition_number=1e5)

        assert isinstance(result, str)
        assert result in ["eigendecomposition", "cholesky"]

    def test_select_method_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        selector = SPDAlgorithmSelector()

        with pytest.raises((ValueError, TypeError)):
            selector.select_method(n=-1, condition_number=1e5)

        with pytest.raises((ValueError, TypeError)):
            selector.select_method(n=500, condition_number=-1.0)

        with pytest.raises((ValueError, TypeError)):
            selector.select_method(n=0, condition_number=1e5)
