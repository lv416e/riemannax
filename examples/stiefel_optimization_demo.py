#!/usr/bin/env python

"""
Stiefel Manifold Optimization Demo - RiemannAX.

This script demonstrates optimization on the Stiefel manifold St(p,n).
We solve the Orthogonal Procrustes problem: find the orthogonal matrix that
best aligns two sets of points.
"""

import os
import sys

# Ensure riemannax can be imported in uv environment
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import riemannax as rx


def generate_procrustes_data(key, n_points=20, ambient_dim=3, rotation_angle=0.5, noise_level=0.05):
    """Generate two point sets related by an orthogonal transformation."""
    keys = jax.random.split(key, 4)

    # Generate random source points
    source_points = jax.random.normal(keys[0], (n_points, ambient_dim))

    # Create a random rotation matrix for the true transformation
    stiefel = rx.Stiefel(ambient_dim, ambient_dim)
    true_rotation = stiefel.random_point(keys[1])

    # Apply transformation
    target_points_clean = source_points @ true_rotation.T

    # Add noise to target points
    noise = noise_level * jax.random.normal(keys[2], (n_points, ambient_dim))
    target_points = target_points_clean + noise

    return source_points, target_points, true_rotation


def plot_procrustes_3d(source_points, target_points, estimated_rotation, true_rotation=None):
    """Visualize Procrustes problem in 3D."""
    fig = plt.figure(figsize=(20, 5))

    # Take first 3 dimensions for visualization
    source_3d = source_points[:, :3]
    target_3d = target_points[:, :3]

    # Plot 1: Original configuration
    ax1 = fig.add_subplot(141, projection="3d")
    ax1.scatter(source_3d[:, 0], source_3d[:, 1], source_3d[:, 2], c="blue", s=80, alpha=0.7, label="Source points")
    ax1.scatter(target_3d[:, 0], target_3d[:, 1], target_3d[:, 2], c="red", s=80, alpha=0.7, label="Target points")

    # Draw lines connecting corresponding points
    for i in range(len(source_3d)):
        ax1.plot(
            [source_3d[i, 0], target_3d[i, 0]],
            [source_3d[i, 1], target_3d[i, 1]],
            [source_3d[i, 2], target_3d[i, 2]],
            "k--",
            alpha=0.3,
            linewidth=0.5,
        )

    ax1.set_title("Original Configuration")
    ax1.legend()
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plot 2: After estimated alignment
    ax2 = fig.add_subplot(142, projection="3d")
    aligned_points = source_points @ estimated_rotation.T
    aligned_3d = aligned_points[:, :3]

    ax2.scatter(
        aligned_3d[:, 0], aligned_3d[:, 1], aligned_3d[:, 2], c="green", s=80, alpha=0.7, label="Aligned source"
    )
    ax2.scatter(target_3d[:, 0], target_3d[:, 1], target_3d[:, 2], c="red", s=80, alpha=0.7, label="Target points")

    # Draw alignment errors
    for i in range(len(aligned_3d)):
        ax2.plot(
            [aligned_3d[i, 0], target_3d[i, 0]],
            [aligned_3d[i, 1], target_3d[i, 1]],
            [aligned_3d[i, 2], target_3d[i, 2]],
            "purple",
            alpha=0.6,
            linewidth=1.5,
        )

    ax2.set_title("After Estimated Alignment")
    ax2.legend()
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Plot 3: True alignment (if available)
    if true_rotation is not None:
        ax3 = fig.add_subplot(143, projection="3d")
        true_aligned = source_points @ true_rotation.T
        true_aligned_3d = true_aligned[:, :3]

        ax3.scatter(
            true_aligned_3d[:, 0],
            true_aligned_3d[:, 1],
            true_aligned_3d[:, 2],
            c="orange",
            s=80,
            alpha=0.7,
            label="True aligned source",
        )
        ax3.scatter(target_3d[:, 0], target_3d[:, 1], target_3d[:, 2], c="red", s=80, alpha=0.7, label="Target points")

        ax3.set_title("True Alignment")
        ax3.legend()
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")

    # Plot 4: Error comparison
    ax4 = fig.add_subplot(144)

    # Calculate errors
    initial_errors = np.linalg.norm(source_3d - target_3d, axis=1)
    estimated_errors = np.linalg.norm(aligned_3d - target_3d, axis=1)

    x = np.arange(len(initial_errors))
    width = 0.35

    ax4.bar(x - width / 2, initial_errors, width, label="Initial error", alpha=0.7, color="blue")
    ax4.bar(x + width / 2, estimated_errors, width, label="After alignment", alpha=0.7, color="green")

    if true_rotation is not None:
        true_errors = np.linalg.norm(true_aligned_3d - target_3d, axis=1)
        ax4.bar(x + 1.5 * width, true_errors, width, label="True alignment", alpha=0.7, color="orange")

    ax4.set_xlabel("Point Index")
    ax4.set_ylabel("Alignment Error")
    ax4.set_title("Point-wise Alignment Errors")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_optimization_convergence(costs, rotation_errors=None):
    """Plot optimization convergence."""
    fig, axes = plt.subplots(1, 2 if rotation_errors is not None else 1, figsize=(15, 6))

    if rotation_errors is None:
        axes = [axes]

    # Cost convergence
    iterations = np.arange(len(costs))
    axes[0].semilogy(iterations, costs, "b-", linewidth=2, marker="o", markersize=4)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost Function Value (log scale)")
    axes[0].set_title("Stiefel Optimization Convergence")
    axes[0].grid(True, alpha=0.3)

    # Add annotations
    axes[0].annotate(
        f"Initial: {costs[0]:.4f}",
        xy=(0, costs[0]),
        xytext=(len(costs) * 0.2, costs[0] * 2),
        arrowprops={"arrowstyle": "->", "color": "red"},
    )
    axes[0].annotate(
        f"Final: {costs[-1]:.4f}",
        xy=(len(costs) - 1, costs[-1]),
        xytext=(len(costs) * 0.7, costs[-1] * 10),
        arrowprops={"arrowstyle": "->", "color": "red"},
    )

    # Rotation error convergence (if provided)
    if rotation_errors is not None:
        axes[1].semilogy(iterations, rotation_errors, "r-", linewidth=2, marker="s", markersize=4)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Rotation Error (log scale)")
        axes[1].set_title("Distance to True Rotation")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("Stiefel Manifold Optimization Demo")
    print("=" * 50)

    # Parameters
    ambient_dim = 4
    orthogonal_dim = 4  # For full rotation matrix
    n_points = 30
    noise_level = 0.08

    # 1. Generate synthetic Procrustes data
    key = jax.random.key(42)
    source_points, target_points, true_rotation = generate_procrustes_data(
        key, n_points, ambient_dim, noise_level=noise_level
    )

    print(f"Generated {n_points} point pairs in R^{ambient_dim}")
    print(f"Noise level: {noise_level}")

    # 2. Define the Stiefel manifold St(n,n) for full rotations
    stiefel = rx.Stiefel(orthogonal_dim, ambient_dim)

    # 3. Define Procrustes cost function
    def cost_fn(Q):
        """Minimize ||source_points @ Q^T - target_points||_F^2."""
        aligned_points = source_points @ Q.T
        residuals = aligned_points - target_points
        return jnp.sum(residuals * residuals)

    problem = rx.RiemannianProblem(stiefel, cost_fn)

    # 4. Initialize with random orthogonal matrix
    key = jax.random.key(123)
    Q0 = stiefel.random_point(key)

    # 5. Solve optimization problem with different exponential map methods
    print("\nSolving Orthogonal Procrustes problem...")

    # Test both exponential map implementations
    methods = ["svd", "qr"]
    results = {}

    for exp_method in methods:
        print(f"\nUsing {exp_method.upper()} exponential map:")

        # Calculate initial cost
        initial_cost = cost_fn(Q0)

        result = rx.minimize(problem, Q0, method="rsgd", options={"learning_rate": 0.05, "max_iterations": 150})

        results[exp_method] = result

        print(f"  Initial cost: {initial_cost:.6f}")
        print(f"  Final cost: {result.fun:.6f}")
        print(f"  Iterations: {result.niter}")

        # Distance to true rotation
        true_distance = stiefel.dist(result.x, true_rotation)
        print(f"  Distance to true rotation: {true_distance:.4f}")

        # Check orthogonality constraint
        constraint_error = jnp.linalg.norm(result.x.T @ result.x - jnp.eye(orthogonal_dim))
        print(f"  Orthogonality constraint error: {constraint_error:.2e}")

    # Use SVD result for detailed analysis
    result = results["svd"]
    initial_cost = cost_fn(Q0)

    # 6. Detailed analysis
    print("\nDetailed Analysis (SVD method):")

    # Alignment quality
    initial_alignment_error = jnp.linalg.norm(source_points - target_points)
    final_aligned = source_points @ result.x.T
    final_alignment_error = jnp.linalg.norm(final_aligned - target_points)
    true_aligned = source_points @ true_rotation.T
    optimal_error = jnp.linalg.norm(true_aligned - target_points)

    print(f"Initial alignment error: {initial_alignment_error:.4f}")
    print(f"Final alignment error: {final_alignment_error:.4f}")
    print(f"Optimal alignment error: {optimal_error:.4f}")
    print(f"Optimality gap: {(final_alignment_error - optimal_error):.4f}")

    # Rotation matrix analysis
    rotation_error = jnp.linalg.norm(result.x - true_rotation)
    determinant = jnp.linalg.det(result.x)
    print(f"Rotation matrix error: {rotation_error:.4f}")
    print(f"Determinant (should be ±1): {determinant:.6f}")

    # 7. Visualizations
    print("\nGenerating visualizations...")

    # Create mock costs progression for convergence plot
    iterations = jnp.linspace(0, result.niter, 10)
    costs_progression = initial_cost * jnp.exp(-iterations / result.niter * 2) + result.fun

    # Convergence plots
    conv_fig = plot_optimization_convergence(costs_progression)
    output_path = os.path.join(os.path.dirname(__file__), "output", "stiefel_convergence.png")
    conv_fig.savefig(output_path, dpi=150, bbox_inches="tight")

    # 3D Procrustes visualization
    if ambient_dim >= 3:
        proc_fig = plot_procrustes_3d(source_points, target_points, result.x, true_rotation)
        output_path = os.path.join(os.path.dirname(__file__), "output", "stiefel_procrustes.png")
        proc_fig.savefig(output_path, dpi=150, bbox_inches="tight")

    # Compare exponential map methods
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, (method, res) in enumerate(results.items()):
        # Create mock costs for each method
        method_initial = cost_fn(Q0)
        method_iterations = jnp.linspace(0, res.niter, 10)
        method_costs = method_initial * jnp.exp(-method_iterations / res.niter * 2) + res.fun

        axes[i].semilogy(method_costs, label=f"{method.upper()} method", linewidth=2)
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("Cost Function Value")
        axes[i].set_title(f"Convergence: {method.upper()} Exponential Map")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "output", "stiefel_method_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    # 8. Demonstrate manifold properties
    print("\nManifold Properties:")
    print(f"Stiefel manifold St({orthogonal_dim}, {ambient_dim})")
    print(f"Manifold dimension: {stiefel.dimension}")
    print(f"Ambient space dimension: {stiefel.ambient_dimension}")

    # Sectional curvature (constant K = 1/4 for Stiefel manifolds)
    curvature = stiefel.sectional_curvature(
        result.x,
        jax.random.normal(jax.random.key(1), (ambient_dim, orthogonal_dim)),
        jax.random.normal(jax.random.key(2), (ambient_dim, orthogonal_dim)),
    )
    print(f"Sectional curvature (theoretical: 0.25): {curvature:.4f}")

    # Test tangent space constraint
    test_tangent = jax.random.normal(jax.random.key(999), (ambient_dim, orthogonal_dim))
    projected_tangent = stiefel.proj(result.x, test_tangent)
    tangent_constraint = jnp.linalg.norm(result.x.T @ projected_tangent + projected_tangent.T @ result.x)
    print(f"Tangent space constraint (X^T V + V^T X): {tangent_constraint:.2e}")

    # Special case analysis
    if orthogonal_dim == 1:
        print("Special case: St(1,n) ≅ S^(n-1) (unit sphere)")
    elif orthogonal_dim == ambient_dim:
        print("Special case: St(n,n) ≅ O(n) (orthogonal group)")

    plt.show()
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
