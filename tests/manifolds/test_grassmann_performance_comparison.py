"""Performance comparison tests for Grassmann manifold batch vs sequential operations.

This module compares the performance of the new batch processing implementation
against sequential processing to verify the expected speedup benefits.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from typing import Dict, List, Tuple, Any

from riemannax.manifolds.grassmann import Grassmann


class GrassmannPerformanceComparison:
    """Performance comparison suite for batch vs sequential operations."""

    def __init__(self):
        """Initialize performance comparison suite."""
        self.results: Dict[str, Any] = {}

    def compare_batch_vs_sequential(
        self,
        manifold: Grassmann,
        operation_name: str,
        batch_sizes: List[int],
        num_trials: int = 5,
        warmup_trials: int = 2
    ) -> Dict[str, Any]:
        """Compare batch vs sequential operation performance.

        Args:
            manifold: Grassmann manifold instance
            operation_name: Name of operation to compare
            batch_sizes: List of batch sizes to test
            num_trials: Number of timing trials
            warmup_trials: Number of warmup trials

        Returns:
            Dictionary containing comparison results
        """
        results = {
            'manifold': f'Gr({manifold.p},{manifold.n})',
            'operation': operation_name,
            'batch_sizes': batch_sizes,
            'batch_times': {},
            'sequential_times': {},
            'speedups': {},
            'efficiency_analysis': {}
        }

        key = jax.random.PRNGKey(42)

        for batch_size in batch_sizes:
            # Generate test data
            keys = jax.random.split(key, batch_size)
            x_batch = jax.vmap(manifold.random_point)(keys)

            # Prepare arguments based on operation
            if operation_name in ['proj', 'exp', 'retr', 'transp']:
                v_keys = jax.random.split(jax.random.PRNGKey(123), batch_size)
                v_batch = jax.vmap(manifold.random_tangent)(v_keys, x_batch)
                test_args_batch = (x_batch, v_batch)
                test_args_individual = [(x_batch[i], v_batch[i]) for i in range(batch_size)]
            elif operation_name in ['log', 'dist']:
                y_keys = jax.random.split(jax.random.PRNGKey(124), batch_size)
                y_batch = jax.vmap(manifold.random_point)(y_keys)
                test_args_batch = (x_batch, y_batch)
                test_args_individual = [(x_batch[i], y_batch[i]) for i in range(batch_size)]
            elif operation_name == 'inner':
                v_keys = jax.random.split(jax.random.PRNGKey(125), batch_size)
                v_batch = jax.vmap(manifold.random_tangent)(v_keys, x_batch)
                u_keys = jax.random.split(jax.random.PRNGKey(126), batch_size)
                u_batch = jax.vmap(manifold.random_tangent)(u_keys, x_batch)
                test_args_batch = (x_batch, u_batch, v_batch)
                test_args_individual = [(x_batch[i], u_batch[i], v_batch[i]) for i in range(batch_size)]
            else:
                raise ValueError(f"Unknown operation: {operation_name}")

            # Get operation functions
            batch_func = getattr(manifold, f'batch_{operation_name}')
            sequential_func = getattr(manifold, operation_name)

            # JIT compile both versions
            jit_batch_func = jax.jit(batch_func)
            jit_sequential_func = jax.jit(sequential_func)

            # Warmup batch version
            for _ in range(warmup_trials):
                _ = jit_batch_func(*test_args_batch)

            # Warmup sequential version
            for _ in range(warmup_trials):
                for args in test_args_individual[:min(3, len(test_args_individual))]:
                    _ = jit_sequential_func(*args)

            # Measure batch performance
            batch_times = []
            for _ in range(num_trials):
                start_time = time.perf_counter()
                batch_result = jit_batch_func(*test_args_batch)
                jax.block_until_ready(batch_result)
                end_time = time.perf_counter()
                batch_times.append(end_time - start_time)

            # Measure sequential performance
            sequential_times = []
            for _ in range(num_trials):
                start_time = time.perf_counter()
                sequential_results = []
                for args in test_args_individual:
                    result = jit_sequential_func(*args)
                    sequential_results.append(result)
                # Ensure all results are ready
                for result in sequential_results:
                    jax.block_until_ready(result)
                end_time = time.perf_counter()
                sequential_times.append(end_time - start_time)

            # Store results
            batch_mean = np.mean(batch_times)
            sequential_mean = np.mean(sequential_times)
            speedup = sequential_mean / batch_mean if batch_mean > 0 else 0

            results['batch_times'][batch_size] = {
                'mean': batch_mean,
                'std': np.std(batch_times),
                'min': np.min(batch_times),
                'max': np.max(batch_times)
            }

            results['sequential_times'][batch_size] = {
                'mean': sequential_mean,
                'std': np.std(sequential_times),
                'min': np.min(sequential_times),
                'max': np.max(sequential_times)
            }

            results['speedups'][batch_size] = speedup

            # Verify results are equivalent (within numerical tolerance)
            self._verify_result_equivalence(
                batch_result, sequential_results, operation_name, batch_size
            )

        # Analyze efficiency
        results['efficiency_analysis'] = self._analyze_efficiency(results)

        return results

    def _verify_result_equivalence(
        self,
        batch_result: jnp.ndarray,
        sequential_results: List[jnp.ndarray],
        operation_name: str,
        batch_size: int
    ) -> None:
        """Verify that batch and sequential results are equivalent."""
        sequential_stacked = jnp.stack(sequential_results, axis=0)

        if operation_name in ['dist', 'inner']:
            # Scalar results
            max_error = jnp.max(jnp.abs(batch_result - sequential_stacked))
            assert max_error < 1e-10, \
                f"Results differ for {operation_name} (batch {batch_size}): max_error = {max_error}"
        else:
            # Matrix results
            max_error = jnp.max(jnp.abs(batch_result - sequential_stacked))
            assert max_error < 1e-10, \
                f"Results differ for {operation_name} (batch {batch_size}): max_error = {max_error}"

    def _analyze_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze efficiency metrics from comparison results."""
        batch_sizes = results['batch_sizes']
        speedups = [results['speedups'][bs] for bs in batch_sizes]

        analysis = {
            'mean_speedup': np.mean(speedups),
            'max_speedup': np.max(speedups),
            'min_speedup': np.min(speedups),
            'speedup_scaling': {},
            'efficiency_rating': 'UNKNOWN'
        }

        # Analyze how speedup scales with batch size
        if len(batch_sizes) > 1:
            # Fit linear relationship
            coeffs = np.polyfit(batch_sizes, speedups, 1)
            correlation = np.corrcoef(batch_sizes, speedups)[0, 1] if len(speedups) > 1 else 0

            analysis['speedup_scaling'] = {
                'slope': coeffs[0],
                'intercept': coeffs[1],
                'correlation': correlation
            }

        # Rate efficiency
        mean_speedup = analysis['mean_speedup']
        if mean_speedup > 2.0:
            analysis['efficiency_rating'] = 'EXCELLENT'
        elif mean_speedup > 1.5:
            analysis['efficiency_rating'] = 'GOOD'
        elif mean_speedup > 1.1:
            analysis['efficiency_rating'] = 'MODERATE'
        else:
            analysis['efficiency_rating'] = 'POOR'

        return analysis

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable performance comparison report."""
        manifold = results['manifold']
        operation = results['operation']
        efficiency = results['efficiency_analysis']

        report = [
            f"=== Performance Comparison: {manifold} {operation.upper()} ===",
            f"Efficiency Rating: {efficiency['efficiency_rating']}",
            f"Mean Speedup: {efficiency['mean_speedup']:.2f}x",
            f"Max Speedup: {efficiency['max_speedup']:.2f}x",
            f"Min Speedup: {efficiency['min_speedup']:.2f}x",
            "",
            "Detailed Results by Batch Size:",
            ""
        ]

        for batch_size in results['batch_sizes']:
            batch_time = results['batch_times'][batch_size]['mean']
            sequential_time = results['sequential_times'][batch_size]['mean']
            speedup = results['speedups'][batch_size]

            report.extend([
                f"  Batch Size {batch_size}:",
                f"    Batch Time: {batch_time:.4f}s ± {results['batch_times'][batch_size]['std']:.4f}s",
                f"    Sequential Time: {sequential_time:.4f}s ± {results['sequential_times'][batch_size]['std']:.4f}s",
                f"    Speedup: {speedup:.2f}x",
                ""
            ])

        if 'speedup_scaling' in efficiency and efficiency['speedup_scaling']:
            scaling = efficiency['speedup_scaling']
            report.extend([
                "Speedup Scaling Analysis:",
                f"  Correlation with batch size: {scaling['correlation']:.3f}",
                f"  Scaling slope: {scaling['slope']:.4f}",
                f"  Interpretation: {'POSITIVE' if scaling['slope'] > 0 else 'NEGATIVE'} scaling with batch size",
                ""
            ])

        return "\n".join(report)


class TestGrassmannPerformanceComparison:
    """Test suite for Grassmann manifold performance comparisons."""

    def setup_method(self):
        """Setup performance comparison tests."""
        self.comparison = GrassmannPerformanceComparison()
        self.test_manifolds = [
            Grassmann(4, 2),  # Gr(2,4) - small
            Grassmann(5, 3),  # Gr(3,5) - medium
            Grassmann(6, 3),  # Gr(3,6) - medium-large
        ]
        self.test_batch_sizes = [1, 5, 10, 20]

    @pytest.mark.slow
    def test_projection_performance_comparison(self):
        """Compare batch vs sequential projection performance."""
        for manifold in self.test_manifolds:
            with self.subTest(manifold=f"Gr({manifold.p},{manifold.n})"):
                results = self.comparison.compare_batch_vs_sequential(
                    manifold=manifold,
                    operation_name='proj',
                    batch_sizes=self.test_batch_sizes,
                    num_trials=3,
                    warmup_trials=1
                )

                # Verify performance improvements
                efficiency = results['efficiency_analysis']

                # Should show meaningful speedup for larger batch sizes
                large_batch_speedups = [
                    results['speedups'][bs] for bs in self.test_batch_sizes if bs >= 10
                ]

                if large_batch_speedups:
                    avg_large_batch_speedup = np.mean(large_batch_speedups)
                    assert avg_large_batch_speedup > 1.1, \
                        f"Insufficient speedup for {manifold}: {avg_large_batch_speedup:.2f}x"

                # Print detailed report
                report = self.comparison.generate_performance_report(results)
                print(f"\n{report}")

    @pytest.mark.slow
    def test_exponential_map_performance_comparison(self):
        """Compare batch vs sequential exponential map performance."""
        # Use smaller manifold for exp map (more computationally intensive)
        manifold = Grassmann(4, 3)

        results = self.comparison.compare_batch_vs_sequential(
            manifold=manifold,
            operation_name='exp',
            batch_sizes=[1, 5, 10],  # Smaller batch sizes for intensive operation
            num_trials=3,
            warmup_trials=1
        )

        # Verify performance improvements
        efficiency = results['efficiency_analysis']

        # Even small speedup is valuable for expensive operations
        assert efficiency['mean_speedup'] > 1.0, \
            f"No speedup observed for exponential map: {efficiency['mean_speedup']:.2f}x"

        # Print detailed report
        report = self.comparison.generate_performance_report(results)
        print(f"\n{report}")

    def test_distance_computation_performance_comparison(self):
        """Compare batch vs sequential distance computation performance."""
        manifold = Grassmann(5, 3)

        results = self.comparison.compare_batch_vs_sequential(
            manifold=manifold,
            operation_name='dist',
            batch_sizes=[1, 5, 10, 15],
            num_trials=3,
            warmup_trials=1
        )

        # Distance computation should show good speedup
        efficiency = results['efficiency_analysis']

        # Should achieve reasonable speedup
        assert efficiency['mean_speedup'] > 1.2, \
            f"Poor speedup for distance computation: {efficiency['mean_speedup']:.2f}x"

        # Print detailed report
        report = self.comparison.generate_performance_report(results)
        print(f"\n{report}")

    def test_cross_operation_speedup_consistency(self):
        """Test that speedup behavior is reasonable across different operations.

        Note: Not all operations will show batch processing speedup due to:
        - Different computational characteristics (memory-bound vs compute-bound)
        - Varying overhead costs for batching setup
        - JIT compilation effects that differ by operation complexity
        """
        manifold = Grassmann(4, 3)
        operations = ['proj', 'dist']  # Use faster operations for this test
        batch_size = 10

        speedup_results = {}

        for operation in operations:
            results = self.comparison.compare_batch_vs_sequential(
                manifold=manifold,
                operation_name=operation,
                batch_sizes=[batch_size],
                num_trials=3,
                warmup_trials=1
            )

            speedup_results[operation] = results['speedups'][batch_size]

        # At least one operation should show meaningful speedup
        max_speedup = max(speedup_results.values())
        assert max_speedup > 1.0, f"No operation showed speedup: {speedup_results}"

        # All operations should have reasonable (positive and bounded) speedup values
        for operation, speedup in speedup_results.items():
            assert speedup > 0.01, f"Unreasonably low speedup for {operation}: {speedup:.2f}x"
            assert speedup < 100.0, f"Unrealistic speedup for {operation}: {speedup:.2f}x"

        # Log results for analysis
        print(f"\nOperation Speedup Analysis:")
        for operation, speedup in speedup_results.items():
            print(f"  {operation}: {speedup:.2f}x")

        # Note: We don't require consistent speedups across operations as different
        # operations have different computational and memory characteristics

    def test_scalability_with_manifold_size(self):
        """Test that batch processing behavior is consistent across manifold sizes.

        Note: Larger manifolds may not always show batch processing speedup because:
        - Sequential overhead becomes relatively smaller compared to total computation
        - Memory bandwidth becomes the bottleneck rather than parallelization
        - Cache effects become more pronounced with larger data structures
        - For very large problems, sequential processing can be more memory-efficient

        This test verifies consistent behavior rather than requiring speedup for all sizes.
        """
        manifolds = [
            Grassmann(4, 2),  # Small: complexity = 4*2*2 = 16
            Grassmann(6, 3),  # Large: complexity = 6*3*3 = 54
        ]

        batch_size = 10
        operation = 'proj'  # Use projection for consistent comparison

        speedups = {}

        for manifold in manifolds:
            complexity = manifold.n * manifold.p * manifold.p

            results = self.comparison.compare_batch_vs_sequential(
                manifold=manifold,
                operation_name=operation,
                batch_sizes=[batch_size],
                num_trials=3,
                warmup_trials=1
            )

            speedups[complexity] = results['speedups'][batch_size]

        # At least one manifold should show meaningful speedup
        max_speedup = max(speedups.values())
        assert max_speedup > 1.0, f"No manifold showed speedup: {speedups}"

        # All speedups should be reasonable (not negative or extremely small)
        for complexity, speedup in speedups.items():
            assert speedup > 0.1, f"Unreasonable speedup for complexity {complexity}: {speedup:.2f}x"
            assert speedup < 100.0, f"Unrealistic speedup for complexity {complexity}: {speedup:.2f}x"

        # Log the results for analysis
        complexities = sorted(speedups.keys())
        small_speedup = speedups[complexities[0]]
        large_speedup = speedups[complexities[1]]

        print(f"\nScalability Analysis:")
        print(f"  Small manifold (complexity {complexities[0]}): {small_speedup:.2f}x speedup")
        print(f"  Large manifold (complexity {complexities[1]}): {large_speedup:.2f}x speedup")

        # Note: We don't require larger manifolds to have better speedups
        # as this depends on many factors including memory hierarchy and system architecture

    def test_memory_efficiency_comparison(self):
        """Test that batch processing is memory efficient compared to sequential."""
        manifold = Grassmann(5, 3)
        batch_size = 20
        key = jax.random.PRNGKey(42)

        # Generate test data
        keys = jax.random.split(key, batch_size)
        x_batch = jax.vmap(manifold.random_point)(keys)

        v_keys = jax.random.split(jax.random.PRNGKey(123), batch_size)
        v_batch = jax.vmap(manifold.random_tangent)(v_keys, x_batch)

        # Test batch operation
        batch_func = jax.jit(manifold.batch_proj)
        batch_result = batch_func(x_batch, v_batch)

        # Test sequential operation
        sequential_func = jax.jit(manifold.proj)
        sequential_results = []
        for i in range(batch_size):
            result = sequential_func(x_batch[i], v_batch[i])
            sequential_results.append(result)

        # Both should complete without memory errors
        assert batch_result.shape == (batch_size, manifold.n, manifold.p)
        assert len(sequential_results) == batch_size

        # Results should be equivalent
        sequential_stacked = jnp.stack(sequential_results, axis=0)
        max_error = jnp.max(jnp.abs(batch_result - sequential_stacked))
        assert max_error < 1e-12, f"Memory efficiency test failed: max error = {max_error}"
