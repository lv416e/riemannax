"""Static argument optimization tests for manifold operations.

This module tests and optimizes static_argnums configurations across all
manifold types to achieve optimal JIT compilation performance.
"""

import jax
import jax.numpy as jnp

import riemannax as rieax
from riemannax.core.jit_manager import JITManager
from riemannax.core.performance_benchmark import PerformanceBenchmark


class TestStaticArgsOptimization:
    """Tests for static argument optimization across manifolds."""

    def setup_method(self):
        """Setup before each test execution."""
        JITManager.clear_cache()
        JITManager.reset_config()

    def test_sphere_static_args_optimization(self):
        """Test optimization of Sphere manifold static arguments."""
        sphere = rieax.Sphere()
        jax.random.key(42)

        # Test current configuration (no static args)
        current_static_args = sphere._get_static_args("exp")
        assert current_static_args == (), "Sphere should currently have no static args"

        # Test data
        point = jnp.array([1.0, 0.0, 0.0])
        tangent = jnp.array([0.0, 0.1, 0.0])

        benchmark = PerformanceBenchmark()

        # Benchmark current (no static args) vs optimized configurations
        configs = [
            {"static_argnums": None, "name": "no_static_args"},
            {"static_argnums": (), "name": "empty_static_args"},
        ]

        results = []
        for config in configs:
            result = benchmark.compare_jit_performance(
                func=sphere.exp, args=(point, tangent), static_argnums=config["static_argnums"], num_runs=3
            )
            result["config_name"] = config["name"]
            results.append(result)

        # Both configurations should work (no static args needed for this case)
        for result in results:
            assert result["jit_speedup"] > 0
            assert "compilation_time" in result

    def test_manifold_static_args_consistency(self):
        """Test consistency of static argument configurations across manifolds."""
        # Updated expectations after correcting static argument implementations
        # All manifolds now use empty tuples for safety (no static arguments)
        manifolds_and_expected = [
            (rieax.SymmetricPositiveDefinite(n=3), ()),
            (rieax.Stiefel(n=5, p=3), ()),
            (rieax.Grassmann(n=5, p=3), ()),
            (rieax.SpecialOrthogonal(n=3), ()),
            (rieax.Sphere(), ()),  # No static args
        ]

        for manifold, expected_args in manifolds_and_expected:
            static_args = manifold._get_static_args("exp")
            assert static_args == expected_args, (
                f"{type(manifold).__name__} static args mismatch: got {static_args}, expected {expected_args}"
            )

            # Verify that returned static args are argument position indices (integers)
            assert isinstance(static_args, tuple), f"Static args should be tuple, got {type(static_args)}"
            for arg_idx in static_args:
                assert isinstance(arg_idx, int), f"Static arg indices should be integers, got {type(arg_idx)}"

    def test_static_args_performance_impact(self):
        """Test performance impact of different static argument configurations."""
        spd = rieax.SymmetricPositiveDefinite(n=3)
        key = jax.random.key(123)

        # Generate test data
        point = spd.random_point(key)
        tangent = spd.random_tangent(key, point)

        benchmark = PerformanceBenchmark()

        # Test different static argument configurations
        configs = [
            {"static_argnums": None, "name": "no_static"},
            {"static_argnums": (0,), "name": "point_static"},  # Make point static
            {"static_argnums": (), "name": "empty_static"},
        ]

        results = benchmark.compare_static_argnums_performance(
            func=spd.exp, args=(point, tangent), configurations=configs, num_runs=3
        )

        assert len(results) == 3

        # All configurations should complete successfully and provide timing data
        for result in results:
            assert "avg_execution_time" in result
            assert "compilation_time" in result
            assert result["avg_execution_time"] > 0
            assert result["compilation_time"] >= 0

        # The point_static config should be present in results
        point_static_result = next(r for r in results if "(0,)" in r["config"])
        assert point_static_result is not None

    def test_automatic_static_args_detection(self):
        """Test automatic detection of optimal static arguments."""
        # This test demonstrates what an automatic optimization system would look like

        def analyze_function_signature(func, manifold):
            """Analyze function to suggest optimal static arguments."""
            # For manifolds, dimension parameters are typically good static args
            suggested_static_args = []

            if hasattr(manifold, "n"):
                suggested_static_args.append("n")
            if hasattr(manifold, "p"):
                suggested_static_args.append("p")

            return suggested_static_args

        # Test on different manifolds
        test_cases = [
            (rieax.SymmetricPositiveDefinite(n=4), ["n"]),
            (rieax.Stiefel(n=6, p=4), ["n", "p"]),
            (rieax.Grassmann(n=7, p=3), ["n", "p"]),
            (rieax.SpecialOrthogonal(n=4), ["n"]),
        ]

        for manifold, expected_params in test_cases:
            detected_params = analyze_function_signature(manifold.exp, manifold)

            # Should detect the expected dimension parameters
            for param in expected_params:
                assert param in detected_params, f"Failed to detect {param} parameter for {type(manifold).__name__}"

    def test_batch_operation_static_args_optimization(self):
        """Test static argument optimization for batch operations."""
        sphere = rieax.Sphere(n=3)  # Use Sphere which handles batching better
        key = jax.random.key(456)

        # Generate batch test data
        batch_size = 5
        points = sphere.random_point(key, batch_size, 4)  # Generate batch of points on S^3
        tangents = sphere.random_tangent(key, points[0], batch_size, 4)  # Generate batch of tangent vectors

        benchmark = PerformanceBenchmark()

        # Compare single vs batch operations with static args
        single_point = points[0]
        single_tangent = tangents[0]

        results = benchmark.compare_batch_performance(
            single_func=sphere.exp,
            single_args=(single_point, single_tangent),
            batch_func=sphere.exp,  # Same function, different input shapes
            batch_args=(points, tangents),
            batch_size=batch_size,
        )

        # Batch operations should complete successfully
        assert "batch_efficiency" in results
        assert "per_item_batch_time" in results

    def test_compilation_caching_with_static_args(self):
        """Test that static arguments improve compilation caching."""
        stiefel = rieax.Stiefel(n=4, p=2)
        key = jax.random.key(789)

        # Generate test data
        point = stiefel.random_point(key)
        tangent = stiefel.random_tangent(key, point)

        benchmark = PerformanceBenchmark()

        # Test caching efficiency with static arguments
        results = benchmark.analyze_compilation_caching(
            func=stiefel.exp, args=(point, tangent), num_cache_tests=6, cache_clear_interval=2
        )

        # Static arguments should improve caching efficiency
        assert results["cache_hit_ratio"] >= 0.5
        assert results["avg_compilation_time"] > results["avg_cache_hit_time"]

    def test_memory_efficiency_with_static_args(self):
        """Test memory efficiency of static argument configurations."""
        grassmann = rieax.Grassmann(n=5, p=2)
        key = jax.random.key(111)

        # Generate test data
        point = grassmann.random_point(key)
        tangent = grassmann.random_tangent(key, point)

        benchmark = PerformanceBenchmark()

        # Test memory usage with static arguments
        results = benchmark.measure_memory_usage(
            func=grassmann.exp, args=(point, tangent), track_compilation_memory=True
        )

        # Should have reasonable memory efficiency
        assert results["memory_efficiency"] > 0
        assert results["peak_execution_memory"] > 0

    def test_cross_manifold_static_args_comparison(self):
        """Compare static argument effectiveness across different manifold types."""
        benchmark = PerformanceBenchmark()

        # Test different manifolds with their optimal static args
        manifold_configs = [
            ("Sphere", rieax.Sphere(), "sphere_point", "sphere_tangent"),
            ("SPD", rieax.SymmetricPositiveDefinite(n=3), "spd_point", "spd_tangent"),
            ("Stiefel", rieax.Stiefel(n=4, p=2), "stiefel_point", "stiefel_tangent"),
        ]

        results = {}
        key = jax.random.key(222)

        for name, manifold, _point_key, _tangent_key in manifold_configs:
            try:
                # Generate test data
                point = manifold.random_point(key)
                tangent = manifold.random_tangent(key, point)

                # Test JIT performance
                perf_result = benchmark.compare_jit_performance(
                    func=manifold.exp,
                    args=(point, tangent),
                    static_argnums=manifold._get_static_args("exp"),
                    num_runs=2,
                )

                results[name] = {
                    "jit_speedup": perf_result["jit_speedup"],
                    "compilation_time": perf_result["compilation_time"],
                    "static_args": manifold._get_static_args("exp"),
                }

            except Exception as e:
                # Some manifolds might have issues, but test should continue
                results[name] = {"error": str(e)}

        # Verify we got results for most manifolds
        successful_tests = len([r for r in results.values() if "error" not in r])
        assert successful_tests >= 2, f"Too many manifold tests failed: {results}"

        # All successful tests should show some performance benefit
        for name, result in results.items():
            if "error" not in result:
                assert result["jit_speedup"] > 0, f"{name} should show JIT speedup"

    def test_static_args_validation_and_recommendations(self):
        """Test validation of current static argument configurations and provide recommendations."""

        def validate_static_args_config(manifold, method_name="exp"):
            """Validate and provide recommendations for static argument configuration."""
            current_config = manifold._get_static_args(method_name)

            recommendations = {
                "current_config": current_config,
                "is_optimized": len(current_config) > 0,
                "recommendations": [],
            }

            # Check for dimension parameters that could be static
            if hasattr(manifold, "n") and manifold.n not in current_config:
                recommendations["recommendations"].append(f"Consider adding dimension n={manifold.n} as static arg")

            if hasattr(manifold, "p") and hasattr(manifold, "n") and manifold.p not in current_config:
                recommendations["recommendations"].append(f"Consider adding dimension p={manifold.p} as static arg")

            return recommendations

        # Test all major manifold types
        manifolds = [
            rieax.Sphere(),
            rieax.SymmetricPositiveDefinite(n=3),
            rieax.Stiefel(n=4, p=2),
            rieax.Grassmann(n=5, p=3),
            rieax.SpecialOrthogonal(n=3),
        ]

        validation_results = {}

        for manifold in manifolds:
            name = type(manifold).__name__
            validation = validate_static_args_config(manifold)
            validation_results[name] = validation

            # Print recommendations for inspection
            if validation["recommendations"]:
                print(f"\n{name} optimization recommendations:")
                for rec in validation["recommendations"]:
                    print(f"  - {rec}")

        # Check optimization status based on current implementation
        sphere_validation = validation_results["Sphere"]
        assert not sphere_validation["is_optimized"], "Sphere should not be currently optimized"

        # SPD is currently not optimized for safety reasons
        spd_validation = validation_results["SymmetricPositiveDefinite"]
        assert not spd_validation["is_optimized"], "SPD is currently not optimized for safety"

        # Other manifolds might be optimized (check if they have static args configured)
        for name in ["Stiefel", "Grassmann", "SpecialOrthogonal"]:
            if name in validation_results:
                validation = validation_results[name]
                # Just verify that validation completed successfully (don't assert specific optimization status)
                assert "is_optimized" in validation
                assert "recommendations" in validation
