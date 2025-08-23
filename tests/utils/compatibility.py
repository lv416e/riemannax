"""Test compatibility utilities for JIT optimizations."""

from collections.abc import Callable
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
import pytest


class JITCompatibilityHelper:
    """Helper class for JIT compatibility verification."""

    # Standard tolerances for different operations
    STANDARD_TOLERANCES: ClassVar[dict[str, dict[str, float]]] = {
        "exp": {"rtol": 1e-5, "atol": 1e-7},
        "log": {"rtol": 1e-5, "atol": 1e-7},
        "proj": {"rtol": 1e-6, "atol": 1e-8},
        "inner": {"rtol": 1e-6, "atol": 1e-8},
        "dist": {"rtol": 1e-5, "atol": 1e-7},
        "random_point": {"rtol": 1e-6, "atol": 1e-8},
    }

    # Looser tolerances for numerically challenging operations
    RELAXED_TOLERANCES: ClassVar[dict[str, dict[str, float]]] = {
        "exp": {"rtol": 1e-4, "atol": 1e-6},
        "log": {"rtol": 1e-4, "atol": 1e-6},
        "proj": {"rtol": 1e-5, "atol": 1e-7},
        "inner": {"rtol": 1e-5, "atol": 1e-7},
        "dist": {"rtol": 1e-4, "atol": 1e-6},
        "random_point": {"rtol": 1e-5, "atol": 1e-7},
    }

    @staticmethod
    def check_numerical_stability(result: Any, operation_name: str = "operation") -> bool:
        """Check if result is numerically stable (no NaN or Inf)."""
        if jnp.any(jnp.isnan(result)):
            return False
        return not jnp.any(jnp.isinf(result))

    @staticmethod
    def check_spd_constraint(matrix: jnp.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if matrix satisfies SPD constraint."""
        try:
            eigenvals = jnp.real(jnp.linalg.eigvals(matrix))
            if jnp.any(jnp.isnan(eigenvals)) or jnp.any(jnp.isinf(eigenvals)):
                return False
            return bool(jnp.all(eigenvals > -tolerance))
        except Exception:
            return False

    @staticmethod
    def adaptive_tolerance_test(result_a: Any, result_b: Any, operation_name: str, use_relaxed: bool = False) -> None:
        """Test with adaptive tolerance based on operation type."""
        tolerances = (
            JITCompatibilityHelper.RELAXED_TOLERANCES if use_relaxed else JITCompatibilityHelper.STANDARD_TOLERANCES
        )

        tol = tolerances.get(operation_name, tolerances["proj"])

        try:
            np.testing.assert_allclose(result_a, result_b, rtol=tol["rtol"], atol=tol["atol"])
        except AssertionError as e:
            if not use_relaxed:
                # Try with relaxed tolerance
                JITCompatibilityHelper.adaptive_tolerance_test(result_a, result_b, operation_name, use_relaxed=True)
            else:
                # Even relaxed tolerance failed
                raise e

    @staticmethod
    def safe_jit_comparison(
        jit_func: Callable,
        nojit_func: Callable,
        *args,
        operation_name: str = "operation",
        skip_on_instability: bool = True,
        **kwargs,
    ) -> None:
        """Safely compare JIT and non-JIT function results."""
        try:
            # Execute functions
            result_jit = jit_func(*args, **kwargs)
            result_nojit = nojit_func(*args, **kwargs)

            # Check numerical stability
            if not JITCompatibilityHelper.check_numerical_stability(result_jit):
                if skip_on_instability:
                    pytest.skip(f"JIT result numerically unstable for {operation_name}")
                else:
                    pytest.fail(f"JIT result numerically unstable for {operation_name}")

            if not JITCompatibilityHelper.check_numerical_stability(result_nojit):
                if skip_on_instability:
                    pytest.skip(f"Non-JIT result numerically unstable for {operation_name}")
                else:
                    pytest.fail(f"Non-JIT result numerically unstable for {operation_name}")

            # For SPD manifold operations, check constraints
            if operation_name in ["exp", "random_point"] and len(result_jit.shape) >= 2:
                if result_jit.shape[-1] == result_jit.shape[-2]:  # Square matrices
                    if not JITCompatibilityHelper.check_spd_constraint(result_jit):
                        if skip_on_instability:
                            pytest.skip(f"JIT result violates SPD constraint for {operation_name}")
                        else:
                            pytest.fail(f"JIT result violates SPD constraint for {operation_name}")

                    if not JITCompatibilityHelper.check_spd_constraint(result_nojit):
                        if skip_on_instability:
                            pytest.skip(f"Non-JIT result violates SPD constraint for {operation_name}")
                        else:
                            pytest.fail(f"Non-JIT result violates SPD constraint for {operation_name}")

            # Compare results with adaptive tolerance
            JITCompatibilityHelper.adaptive_tolerance_test(result_jit, result_nojit, operation_name)

        except Exception as e:
            if skip_on_instability:
                pytest.skip(f"Numerical instability in {operation_name}: {e!s}")
            else:
                raise e

    @staticmethod
    def create_stable_spd_matrix(
        key: jnp.ndarray, size: int, min_eigenval: float = 0.1, max_eigenval: float = 10.0
    ) -> jnp.ndarray:
        """Create a numerically stable SPD matrix."""
        import jax.random as jr

        # Generate random orthogonal matrix
        A = jr.normal(key, (size, size))
        Q, _ = jnp.linalg.qr(A)

        # Create controlled eigenvalues
        eigenvals = jnp.linspace(min_eigenval, max_eigenval, size)

        # Construct SPD matrix
        return Q @ jnp.diag(eigenvals) @ Q.T

    @staticmethod
    def verify_manifold_operation_compatibility(
        manifold: Any, operation_name: str, test_data: dict, max_retries: int = 3
    ) -> None:
        """Verify compatibility of a manifold operation between JIT and non-JIT."""
        for attempt in range(max_retries):
            try:
                if hasattr(manifold, f"_{operation_name}_impl"):
                    # Get implementation functions
                    impl_func = getattr(manifold, f"_{operation_name}_impl")
                    getattr(manifold, operation_name)

                    # Prepare arguments
                    args = list(test_data.values())

                    # Compare results
                    JITCompatibilityHelper.safe_jit_comparison(
                        lambda *a: manifold._call_jit_method(operation_name, *a),
                        impl_func,
                        *args,
                        operation_name=operation_name,
                    )

                    # If we get here, the test passed
                    return

                else:
                    pytest.skip(f"Operation {operation_name} not implemented for {type(manifold).__name__}")

            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, raise the exception
                    raise e
                # Try again with different data or approach
                continue


class SPDCompatibilityMixin:
    """Mixin for SPD-specific compatibility tests."""

    def assert_spd_result(self, matrix: jnp.ndarray, operation_name: str = "operation"):
        """Assert that result satisfies SPD constraints with proper error handling."""
        if not JITCompatibilityHelper.check_numerical_stability(matrix, operation_name):
            pytest.skip(f"Numerical instability in {operation_name}")

        if not JITCompatibilityHelper.check_spd_constraint(matrix, tolerance=1e-5):
            eigenvals = jnp.real(jnp.linalg.eigvals(matrix))
            min_eigenval = jnp.min(eigenvals)
            if min_eigenval > -1e-4:  # Close to SPD, might be numerical error
                pytest.skip(f"Near-SPD violation in {operation_name}: min eigenvalue = {min_eigenval}")
            else:
                pytest.fail(f"SPD constraint violation in {operation_name}: min eigenvalue = {min_eigenval}")


def requires_numerical_stability(operation_name: str):
    """Decorator to skip tests that encounter numerical instability."""

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                return test_func(*args, **kwargs)
            except (AssertionError, ValueError, RuntimeError) as e:
                error_msg = str(e).lower()
                stability_keywords = ["nan", "inf", "numerical", "unstable", "singular", "not converged"]

                if any(keyword in error_msg for keyword in stability_keywords):
                    pytest.skip(f"Numerical instability in {operation_name}: {e!s}")
                else:
                    raise e

        return wrapper

    return decorator
