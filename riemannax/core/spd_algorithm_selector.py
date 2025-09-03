"""Adaptive Algorithm Selector for SPD Manifold Operations.

This module provides intelligent algorithm selection for symmetric positive definite
(SPD) matrix operations based on matrix size and condition number.

The selector chooses between eigendecomposition and Cholesky decomposition methods
to optimize for performance and numerical stability.
"""

from typing import Literal


class SPDAlgorithmSelector:
    """Adaptive algorithm selector for SPD manifold operations.

    This class implements dynamic algorithm selection based on matrix properties:
    - Matrix size (n): Larger matrices benefit from Cholesky decomposition
    - Condition number: Well-conditioned matrices can use Cholesky safely

    Selection Logic:
    - If n <= 1000 and condition_number < 1e10: "eigendecomposition"
    - If condition_number < 1e12: "cholesky"
    - Else: "eigendecomposition" (for numerical stability)
    """

    def __init__(self) -> None:
        """Initialize the SPD Algorithm Selector."""
        pass

    def select_method(self, n: int, condition_number: float) -> Literal["eigendecomposition", "cholesky"]:
        """Select optimal algorithm based on matrix size and condition number.

        Args:
            n: Matrix size (number of rows/columns).
            condition_number: Estimated condition number of the matrix.

        Returns:
            String indicating the selected algorithm.

        Raises:
            ValueError: If n <= 0 or condition_number <= 0.

        Examples:
            >>> selector = SPDAlgorithmSelector()
            >>> selector.select_method(500, 1e5)
            'eigendecomposition'
            >>> selector.select_method(1500, 1e8)
            'cholesky'
        """
        # Input validation
        if n <= 0:
            raise ValueError(f"Matrix size must be positive, got n={n}")
        if condition_number <= 0:
            raise ValueError(f"Condition number must be positive, got condition_number={condition_number}")

        # Selection logic based on requirements
        if n <= 1000 and condition_number < 1e10:
            return "eigendecomposition"
        elif condition_number <= 1e12:
            return "cholesky"
        else:
            return "eigendecomposition"
