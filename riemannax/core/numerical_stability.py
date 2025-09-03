"""Numerical stability layer for SPD matrix computations.

This module provides the NumericalStabilityLayer class which implements:
- Condition number estimation for matrices
- Adaptive regularization for ill-conditioned matrices
- SPD (Symmetric Positive Definite) property validation
- Numerical stability handling for extreme cases

The implementation is JAX-native with JIT compilation support and follows
research-based best practices for numerical linear algebra stability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class NumericalStabilityLayer:
    """Provides numerical stability operations for SPD matrix computations.

    This class implements condition number estimation, adaptive regularization,
    and SPD property validation with focus on numerical stability and
    performance. All methods are JAX-native and JIT-compilable.

    Key features:
    - Condition number estimation using SVD
    - Adaptive regularization: x + ε * I where ε is condition-dependent
    - SPD validation: symmetry check + positive eigenvalue verification
    - Robust handling of edge cases (NaN, Inf, singular matrices)
    - Full JAX ecosystem compatibility (jit, vmap, grad)

    Examples:
        >>> stability = NumericalStabilityLayer()
        >>> matrix = jnp.eye(3) * 2.0
        >>> cond_num = stability.estimate_condition_number(matrix)
        >>> is_spd = stability.validate_spd_properties(matrix, tolerance=1e-8)
        >>> regularized = stability.regularize_matrix(matrix, condition_threshold=1e10)
    """

    def __init__(self) -> None:
        """Initialize the numerical stability layer."""
        pass

    def estimate_condition_number(self, x: jnp.ndarray) -> jnp.ndarray:
        """Estimate the condition number of a matrix using SVD.

        Uses singular value decomposition to compute the condition number as
        the ratio of largest to smallest singular values. Handles edge cases
        like singular matrices and provides numerically stable computation.

        Args:
            x: Square matrix to analyze. Shape (n, n).

        Returns:
            Condition number as scalar. Returns large finite value for
            near-singular matrices.

        Note:
            This method uses SVD which is more stable than eigendecomposition
            for condition number estimation. For singular matrices, returns
            a large but finite value instead of infinity for numerical stability.
        """
        # Input validation
        assert x.ndim == 2, f"Expected 2D matrix, got {x.ndim}D array"
        assert x.shape[0] == x.shape[1], f"Expected square matrix, got shape {x.shape}"

        # Handle NaN/Inf inputs gracefully
        has_nan_inf = jnp.any(jnp.isnan(x)) | jnp.any(jnp.isinf(x))

        def compute_condition() -> jnp.ndarray:
            try:
                s = jnp.linalg.svd(x, compute_uv=False)

                # Filter out effectively zero singular values
                s_filtered = jnp.where(s > 1e-16, s, 1e-16)

                # Condition number is ratio of largest to smallest singular value
                condition_number = s_filtered[0] / s_filtered[-1]

                # Clamp to reasonable range to avoid numerical issues
                condition_number = jnp.clip(condition_number, 1.0, 1e16)

                return condition_number

            except (ValueError, RuntimeError):
                # Fallback for numerical issues
                return jnp.array(1e16)

        return jnp.where(has_nan_inf, jnp.array(1e16), compute_condition())

    def validate_spd_properties(self, x: jnp.ndarray, tolerance: float = 1e-8) -> bool:
        """Validate if a matrix satisfies SPD (Symmetric Positive Definite) properties.

        Checks both symmetry and positive definiteness:
        1. Symmetry: ||A - A^T||_F < tolerance * ||A||_F
        2. Positive definiteness: all eigenvalues > tolerance

        Args:
            x: Matrix to validate. Shape (n, n).
            tolerance: Numerical tolerance for checks. Must be positive.

        Returns:
            True if matrix is SPD within tolerance, False otherwise.

        Note:
            Uses eigendecomposition for positive definiteness check as it's
            more reliable than Cholesky for this validation purpose.
        """
        assert tolerance > 0, f"Tolerance must be positive, got {tolerance}"
        assert x.ndim == 2, f"Expected 2D matrix, got {x.ndim}D array"
        assert x.shape[0] == x.shape[1], f"Expected square matrix, got shape {x.shape}"

        # Handle NaN/Inf inputs
        has_nan_inf = jnp.any(jnp.isnan(x)) | jnp.any(jnp.isinf(x))

        # Check symmetry: ||A - A^T||_F < tolerance * ||A||_F
        x_transpose = x.T
        asymmetry = jnp.linalg.norm(x - x_transpose, "fro")
        matrix_norm = jnp.linalg.norm(x, "fro")

        # Avoid division by zero
        relative_asymmetry = jnp.where(matrix_norm > 1e-16, asymmetry / matrix_norm, asymmetry)

        is_symmetric = relative_asymmetry < tolerance

        # Check positive definiteness via eigenvalues
        def check_eigenvals() -> jnp.ndarray:
            try:
                # Use symmetric eigendecomposition for efficiency
                eigenvals = jnp.linalg.eigvals(x)
                eigenvals_real = jnp.real(eigenvals)  # Should be real for symmetric matrices

                # All eigenvalues must be positive
                return jnp.all(eigenvals_real > tolerance)
            except (ValueError, RuntimeError):
                return jnp.array(False)

        is_positive_definite = check_eigenvals()

        # Combine all conditions
        is_spd = (~has_nan_inf) & is_symmetric & is_positive_definite

        return bool(is_spd)

    def regularize_matrix(self, x: jnp.ndarray, condition_threshold: float = 1e12) -> jnp.ndarray:
        """Apply adaptive regularization to improve matrix conditioning.

        Implements regularization: x_reg = x + ε * I where ε is chosen
        adaptively based on the condition number. The regularization strength
        increases with worse conditioning to ensure numerical stability.

        Algorithm:
        1. Estimate condition number of input matrix
        2. If condition > threshold, compute adaptive regularization strength
        3. Apply regularization: x_reg = x + ε * I
        4. ε = alpha * ||x||_F where alpha depends on condition number

        Args:
            x: Matrix to regularize. Shape (n, n).
            condition_threshold: Condition number threshold above which
                                regularization is applied. Must be positive.

        Returns:
            Regularized matrix with improved conditioning. Same shape as input.

        Note:
            The regularization strength is chosen to balance numerical stability
            with preservation of the original matrix structure. Uses Frobenius
            norm scaling for invariance to matrix scaling.
        """
        # Note: Input validation removed for JIT compatibility
        # Validation should be done at call site when not using JIT

        # Estimate current condition number
        current_condition = self.estimate_condition_number(x)

        # Only regularize if needed
        needs_regularization = current_condition > condition_threshold

        # Compute adaptive regularization strength
        matrix_norm = jnp.linalg.norm(x, "fro")

        # Regularization strength increases with condition number
        # Use log scale to handle extreme condition numbers
        log_condition_excess = jnp.log10(jnp.maximum(current_condition / condition_threshold, 1.0))

        # Base regularization strength as fraction of matrix norm
        # Scaling: 1e-12 for threshold condition, up to 1e-6 for very ill-conditioned
        base_strength = 1e-12
        condition_multiplier = jnp.power(10.0, log_condition_excess * 0.5)
        regularization_strength = base_strength * condition_multiplier

        # Ensure minimum regularization for singular matrices
        regularization_strength = jnp.maximum(regularization_strength, base_strength)

        # Scale by matrix norm for invariance
        epsilon = regularization_strength * matrix_norm

        # Ensure minimum absolute regularization
        epsilon = jnp.maximum(epsilon, 1e-16)

        # Apply regularization conditionally
        identity = jnp.eye(x.shape[0])
        regularization_term = epsilon * identity

        # Only add regularization if needed
        regularized_matrix: jnp.ndarray = jnp.where(needs_regularization, x + regularization_term, x)

        return regularized_matrix


# Convenience functions for direct use
def estimate_condition_number_func(x: jnp.ndarray) -> jnp.ndarray:
    """Convenience function to estimate matrix condition number.

    Args:
        x: Square matrix to analyze.

    Returns:
        Condition number estimate.
    """
    stability_layer = NumericalStabilityLayer()
    result: jnp.ndarray = stability_layer.estimate_condition_number(x)
    return result


def validate_spd_properties_func(x: jnp.ndarray, tolerance: float = 1e-8) -> bool:
    """Convenience function to validate SPD properties.

    Args:
        x: Matrix to validate.
        tolerance: Numerical tolerance.

    Returns:
        True if matrix is SPD.
    """
    stability_layer = NumericalStabilityLayer()
    return stability_layer.validate_spd_properties(x, tolerance)


def regularize_matrix_func(x: jnp.ndarray, condition_threshold: float = 1e12) -> jnp.ndarray:
    """Convenience function to regularize matrix.

    Args:
        x: Matrix to regularize.
        condition_threshold: Conditioning threshold.

    Returns:
        Regularized matrix.
    """
    # Input validation for non-JIT usage
    assert condition_threshold > 0, f"Condition threshold must be positive, got {condition_threshold}"
    assert x.ndim == 2, f"Expected 2D matrix, got {x.ndim}D array"
    assert x.shape[0] == x.shape[1], f"Expected square matrix, got shape {x.shape}"

    stability_layer = NumericalStabilityLayer()
    result: jnp.ndarray = stability_layer.regularize_matrix(x, condition_threshold)
    return result


# JIT-safe internal function without validation
def _regularize_matrix_jit_safe(x: jnp.ndarray, condition_threshold: float = 1e12) -> jnp.ndarray:
    """JIT-safe regularization function without input validation."""
    stability_layer = NumericalStabilityLayer()
    return stability_layer.regularize_matrix(x, condition_threshold)


# JIT compiled versions
estimate_condition_number = jax.jit(estimate_condition_number_func)
regularize_matrix = jax.jit(_regularize_matrix_jit_safe)
validate_spd_properties = validate_spd_properties_func  # Cannot be JIT compiled due to bool return type
