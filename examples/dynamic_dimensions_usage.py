#!/usr/bin/env python3
"""Dynamic Dimension Usage Examples for RiemannAX.

This example demonstrates how to use RiemannAX manifolds with dynamic dimensions,
showcasing the factory pattern and performance characteristics across different
manifold sizes.

Examples include:
- Creating manifolds with various dimensions using factory functions
- Performance comparison across different dimensions
- Type-safe operations with proper validation
- JIT compilation benefits for larger manifolds
"""


from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from riemannax.core.constants import NumericalConstants
from riemannax.core.performance_benchmark import PerformanceBenchmark

# Import factory functions for dynamic manifold creation
from riemannax.manifolds import create_grassmann, create_so, create_spd, create_sphere, create_stiefel


def demonstrate_sphere_dimensions() -> None:
    """Demonstrate sphere manifolds with various dimensions."""
    print("=== Dynamic Sphere Dimensions ===\n")

    # Test different sphere dimensions
    dimensions = [1, 2, 3, 5, 10, 50, 100]

    key = jr.PRNGKey(42)

    for n in dimensions:
        print(f"Sphere S^{n} (embedded in R^{n + 1}):")

        # Create sphere using factory function
        sphere = create_sphere(n)

        # Generate random point and tangent vector
        point = sphere.random_point(key)
        tangent = sphere.random_tangent(key, point)

        # Perform basic operations
        exp_point = sphere.exp(point, tangent * 0.1)  # Small step
        log_vector = sphere.log(point, exp_point)
        distance = sphere.dist(point, exp_point)

        print(f"  - Dimension: {sphere.dimension}")
        print(f"  - Ambient dimension: {sphere.ambient_dimension}")
        print(f"  - Point shape: {point.shape}")
        print(f"  - Distance (exp/log consistency): {distance:.6f}")
        print(f"  - Log vector norm: {jnp.linalg.norm(log_vector):.6f}")

        # Validate manifold constraints
        assert sphere.validate_point(point), f"Invalid point for S^{n}"
        assert sphere.validate_tangent(point, tangent), f"Invalid tangent for S^{n}"

        print("  ✅ All operations successful and constraints satisfied\n")

        # Update key for next iteration
        key = jr.split(key)[0]


def demonstrate_grassmann_dimensions() -> None:
    """Demonstrate Grassmann manifolds with various dimensions."""
    print("=== Dynamic Grassmann Dimensions ===\n")

    # Test different Grassmann manifold configurations
    configs = [
        (2, 5),  # Gr(2,5) - 2D subspaces in R^5
        (3, 8),  # Gr(3,8) - 3D subspaces in R^8
        (5, 12),  # Gr(5,12) - 5D subspaces in R^12
        (10, 20),  # Gr(10,20) - 10D subspaces in R^20
    ]

    key = jr.PRNGKey(42)

    for p, n in configs:
        print(f"Grassmann Gr({p},{n}) - {p}D subspaces in R^{n}:")

        # Create Grassmann manifold
        grassmann = create_grassmann(p, n)

        # Generate random point (p x n matrix with orthonormal columns)
        point = grassmann.random_point(key)
        tangent = grassmann.random_tangent(key, point)

        # Perform operations
        exp_point = grassmann.exp(point, tangent * 0.01)  # Very small step for numerical stability
        distance = grassmann.dist(point, exp_point)

        print(f"  - Parameters: p={grassmann.p}, n={grassmann.n}")
        print(f"  - Manifold dimension: {grassmann.dimension}")
        print(f"  - Point shape: {point.shape}")
        print(f"  - Orthonormality check: {jnp.allclose(point.T @ point, jnp.eye(p), atol=1e-6)}")
        print(f"  - Distance: {distance:.6f}")

        # Validate constraints
        assert grassmann.validate_point(point), f"Invalid point for Gr({p},{n})"

        print("  ✅ Orthonormality and manifold constraints satisfied\n")

        key = jr.split(key)[0]


def demonstrate_stiefel_dimensions() -> None:
    """Demonstrate Stiefel manifolds with various dimensions."""
    print("=== Dynamic Stiefel Dimensions ===\n")

    configs = [
        (2, 3),  # St(2,3) - 2 orthonormal vectors in R^3
        (3, 5),  # St(3,5) - 3 orthonormal vectors in R^5
        (4, 8),  # St(4,8) - 4 orthonormal vectors in R^8
        (5, 5),  # St(5,5) - special case (full rank)
    ]

    key = jr.PRNGKey(42)

    for p, n in configs:
        print(f"Stiefel St({p},{n}) - {p} orthonormal vectors in R^{n}:")

        stiefel = create_stiefel(p, n)

        point = stiefel.random_point(key)
        tangent = stiefel.random_tangent(key, point)

        # Check orthonormality
        gram_matrix = point.T @ point
        is_orthonormal = jnp.allclose(gram_matrix, jnp.eye(p), atol=1e-6)

        print(f"  - Parameters: p={stiefel.p}, n={stiefel.n}")
        print(f"  - Manifold dimension: {stiefel.dimension}")
        print(f"  - Point shape: {point.shape}")
        print(f"  - Orthonormality: {is_orthonormal}")

        # Tangent space constraint: X^T Y + Y^T X = 0 where Y is tangent at X
        tangent_constraint = point.T @ tangent + tangent.T @ point
        is_tangent_valid = jnp.allclose(tangent_constraint, jnp.zeros((p, p)), atol=1e-6)
        print(f"  - Tangent space constraint: {is_tangent_valid}")

        assert stiefel.validate_point(point), f"Invalid point for St({p},{n})"

        print("  ✅ All constraints satisfied\n")

        key = jr.split(key)[0]


def demonstrate_so_dimensions() -> None:
    """Demonstrate Special Orthogonal manifolds SO(n)."""
    print("=== Dynamic SO(n) Dimensions ===\n")

    dimensions = [2, 3, 4, 5, 8]
    key = jr.PRNGKey(42)

    for n in dimensions:
        print(f"Special Orthogonal SO({n}) - {n}x{n} rotation matrices:")

        so = create_so(n)

        point = so.random_point(key)

        # Check orthogonality and determinant
        is_orthogonal = jnp.allclose(point @ point.T, jnp.eye(n), atol=1e-6)
        det_value = jnp.linalg.det(point)
        is_special = jnp.allclose(det_value, 1.0, atol=1e-6)

        print(f"  - Matrix size: {n}x{n}")
        print(f"  - Manifold dimension: {so.dimension}")
        print(f"  - Orthogonality: {is_orthogonal}")
        print(f"  - Determinant = 1: {is_special} (det = {det_value:.6f})")

        assert so.validate_point(point), f"Invalid rotation matrix for SO({n})"

        print("  ✅ Rotation matrix properties satisfied\n")

        key = jr.split(key)[0]


def demonstrate_spd_dimensions() -> None:
    """Demonstrate Symmetric Positive Definite manifolds SPD(n)."""
    print("=== Dynamic SPD(n) Dimensions ===\n")

    dimensions = [2, 3, 4, 5, 8]
    key = jr.PRNGKey(42)

    for n in dimensions:
        print(f"SPD({n}) - {n}x{n} symmetric positive definite matrices:")

        spd = create_spd(n)

        point = spd.random_point(key)

        # Check symmetry
        is_symmetric = jnp.allclose(point, point.T, atol=1e-6)

        # Check positive definiteness via eigenvalues
        eigenvals = jnp.linalg.eigvals(point)
        is_positive_definite = jnp.all(eigenvals > 1e-8)
        min_eigenval = jnp.min(eigenvals)

        print(f"  - Matrix size: {n}x{n}")
        print(f"  - Manifold dimension: {spd.dimension}")
        print(f"  - Symmetry: {is_symmetric}")
        print(f"  - Positive definite: {is_positive_definite} (min eigenval: {min_eigenval:.6f})")

        assert spd.validate_point(point), f"Invalid SPD matrix for SPD({n})"

        print("  ✅ SPD properties satisfied\n")

        key = jr.split(key)[0]


def benchmark_dimension_scaling() -> None:
    """Benchmark performance scaling with manifold dimensions."""
    print("=== Performance Scaling Analysis ===\n")

    benchmark = PerformanceBenchmark()

    # Test sphere performance across dimensions
    print("Sphere Performance Scaling:")
    dimensions = [10, 50, 100, 200]

    key = jr.PRNGKey(42)

    results: dict[int, dict[str, float]] = {}

    for n in dimensions:
        sphere = create_sphere(n)
        point = sphere.random_point(key)
        tangent = sphere.random_tangent(key, point)

        # Benchmark exponential map
        try:
            perf_results: dict[str, Any] = benchmark.compare_jit_performance(sphere.exp, args=(point, tangent), num_runs=10)

            jit_speedup = float(perf_results.get("jit_speedup", 0))
            compilation_time = float(perf_results.get("compilation_time", 0))
            jit_time = float(perf_results.get("jit_time", 0))

            results[n] = {"jit_speedup": jit_speedup, "compilation_time": compilation_time, "jit_time": jit_time}

            print(
                f"  S^{n}: JIT speedup = {jit_speedup:.2f}x, "
                f"compilation = {compilation_time:.4f}s, "
                f"execution = {jit_time:.6f}s"
            )

        except Exception as e:
            print(f"  S^{n}: Benchmark failed - {e}")

        key = jr.split(key)[0]

    # Analyze scaling trends
    print("\nPerformance Analysis:")
    if len(results) >= 2:
        dims = sorted(results.keys())
        speedups: list[float] = [results[d]["jit_speedup"] for d in dims]

        print(f"  - JIT speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")

        # Check if any dimension meets our target
        target_met = [
            d for d in dims if results[d]["jit_speedup"] >= NumericalConstants.RTOL * 10000
        ]  # Using a practical target
        if target_met:
            print(f"  - Dimensions meeting performance targets: {target_met}")
        else:
            print("  - Note: JIT benefits may be limited on CPU for mathematical operations")
            print("  - GPU acceleration typically shows more significant speedups")

    print()


def demonstrate_type_safety() -> None:
    """Demonstrate type safety and validation features."""
    print("=== Type Safety and Validation ===\n")

    # Test dimension validation
    print("1. Dimension Validation:")

    try:
        # These should work
        create_sphere(3)
        create_grassmann(2, 5)
        create_stiefel(3, 4)
        print("  ✅ Valid dimensions accepted")

        # These should raise errors
        try:
            create_sphere(-1)
            print("  ❌ Should have rejected negative dimension")
        except ValueError as e:
            print(f"  ✅ Correctly rejected negative dimension: {e}")

        try:
            create_grassmann(5, 3)  # p >= n
            print("  ❌ Should have rejected invalid Grassmann dimensions")
        except ValueError as e:
            print(f"  ✅ Correctly rejected invalid Grassmann dimensions: {e}")

    except Exception as e:
        print(f"  ❌ Unexpected error in validation: {e}")

    print("\n2. Point and Tangent Validation:")

    sphere = create_sphere(2)
    key = jr.PRNGKey(42)

    # Valid point and tangent
    valid_point = sphere.random_point(key)
    valid_tangent = sphere.random_tangent(key, valid_point)

    print(f"  ✅ Valid point check: {sphere.validate_point(valid_point)}")
    print(f"  ✅ Valid tangent check: {sphere.validate_tangent(valid_point, valid_tangent)}")

    # Invalid point (not unit norm)
    invalid_point = valid_point * 2.0  # Scale to break unit norm
    print(f"  ✅ Invalid point correctly rejected: {not sphere.validate_point(invalid_point)}")

    # Invalid tangent (not orthogonal)
    invalid_tangent = valid_point  # Point itself is not orthogonal to itself
    print(f"  ✅ Invalid tangent correctly rejected: {not sphere.validate_tangent(valid_point, invalid_tangent)}")

    print()


def main():
    """Run all dynamic dimension usage examples."""
    print("RiemannAX Dynamic Dimensions Usage Examples")
    print("=" * 50)
    print()

    # Enable JIT compilation for performance demonstrations
    jax.config.update("jax_enable_x64", True)  # Use double precision for numerical accuracy

    try:
        demonstrate_sphere_dimensions()
        demonstrate_grassmann_dimensions()
        demonstrate_stiefel_dimensions()
        demonstrate_so_dimensions()
        demonstrate_spd_dimensions()
        benchmark_dimension_scaling()
        demonstrate_type_safety()

        print("🎉 All dynamic dimension examples completed successfully!")
        print()
        print("Key Takeaways:")
        print("- Factory functions enable easy creation of manifolds with any valid dimension")
        print("- Type validation ensures mathematical constraints are satisfied")
        print("- JIT compilation provides performance benefits, especially for larger dimensions")
        print("- All manifold operations scale appropriately with dimension")
        print("- Numerical constants ensure stability across different scales")

    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
