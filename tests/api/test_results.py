"""Tests for optimization result standardization."""

import pytest
import jax.numpy as jnp
from enum import Enum

from riemannax.api.results import OptimizationResult, ConvergenceStatus


class TestOptimizationResult:
    """Test OptimizationResult class functionality."""

    def test_optimization_result_creation(self):
        """Test basic OptimizationResult creation with required fields."""
        result = OptimizationResult(
            optimized_params=jnp.array([1.0, 0.0]),
            objective_value=0.5,
            convergence_status=ConvergenceStatus.CONVERGED,
            iteration_count=10
        )

        assert result.optimized_params.shape == (2,)
        assert result.objective_value == 0.5
        assert result.convergence_status == ConvergenceStatus.CONVERGED
        assert result.iteration_count == 10

    def test_optimization_result_with_metadata(self):
        """Test OptimizationResult with optional metadata."""
        metadata = {
            "final_gradient_norm": 1e-6,
            "algorithm": "riemannian_sgd",
            "manifold_type": "sphere"
        }

        result = OptimizationResult(
            optimized_params=jnp.array([0.6, 0.8]),
            objective_value=0.25,
            convergence_status=ConvergenceStatus.CONVERGED,
            iteration_count=5,
            metadata=metadata
        )

        assert result.metadata["final_gradient_norm"] == 1e-6
        assert result.metadata["algorithm"] == "riemannian_sgd"
        assert result.metadata["manifold_type"] == "sphere"

    def test_convergence_status_enum(self):
        """Test ConvergenceStatus enum values."""
        assert ConvergenceStatus.CONVERGED.value == "converged"
        assert ConvergenceStatus.MAX_ITERATIONS.value == "max_iterations_reached"
        assert ConvergenceStatus.FAILED.value == "failed"
        assert ConvergenceStatus.STOPPED.value == "stopped_by_user"

    def test_optimization_result_backward_compatibility(self):
        """Test compatibility with existing OptimizeResult interface."""
        result = OptimizationResult(
            optimized_params=jnp.array([1.0, 0.0]),
            objective_value=0.5,
            convergence_status=ConvergenceStatus.CONVERGED,
            iteration_count=10
        )

        # Should have the fields that OptimizeResult has
        assert hasattr(result, 'x')  # Should alias optimized_params
        assert hasattr(result, 'fun')  # Should alias objective_value
        assert hasattr(result, 'success')  # Should derive from convergence_status
        assert hasattr(result, 'niter')  # Should alias iteration_count
        assert hasattr(result, 'message')  # Should provide descriptive message

    def test_success_property_for_different_statuses(self):
        """Test success property derivation from convergence status."""
        converged_result = OptimizationResult(
            optimized_params=jnp.array([1.0]),
            objective_value=0.1,
            convergence_status=ConvergenceStatus.CONVERGED,
            iteration_count=5
        )
        assert converged_result.success == True

        failed_result = OptimizationResult(
            optimized_params=jnp.array([1.0]),
            objective_value=0.1,
            convergence_status=ConvergenceStatus.FAILED,
            iteration_count=10
        )
        assert failed_result.success == False

        max_iter_result = OptimizationResult(
            optimized_params=jnp.array([1.0]),
            objective_value=0.1,
            convergence_status=ConvergenceStatus.MAX_ITERATIONS,
            iteration_count=100
        )
        assert max_iter_result.success == False
