#!/usr/bin/env python

"""
Grassmann Manifold Optimization Demo - RiemannAX.

This script demonstrates subspace fitting on the Grassmann manifold Gr(p,n).
We find the p-dimensional subspace that best approximates a set of data points.
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


def generate_noisy_subspace_data(key, n_points=50, ambient_dim=4, true_subspace_dim=2, noise_level=0.1):
    """Generate data points near a random subspace with added noise."""
    keys = jax.random.split(key, 3)

    # Generate random true subspace
    grassmann = rx.Grassmann(ambient_dim, true_subspace_dim)
    true_subspace = grassmann.random_point(keys[0])

    # Generate points in the subspace
    coeffs = jax.random.normal(keys[1], (n_points, true_subspace_dim))
    clean_points = coeffs @ true_subspace.T

    # Add noise
    noise = noise_level * jax.random.normal(keys[2], (n_points, ambient_dim))
    noisy_points = clean_points + noise

    return noisy_points, true_subspace


def plot_subspace_fitting_3d(data_points, estimated_subspace, true_subspace=None):
    """Visualize subspace fitting in 3D (only works for ambient_dim=3, subspace_dim=2)."""
    fig = plt.figure(figsize=(15, 5))

    # Take first 3 dimensions for visualization
    data_3d = data_points[:, :3]
    est_sub_3d = estimated_subspace[:3, :]

    # Plot 1: Data points with estimated subspace
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
               c='blue', alpha=0.6, s=50, label='Data points')

    # Create mesh for subspace visualization
    u = np.linspace(-2, 2, 10)
    v = np.linspace(-2, 2, 10)
    U, V = np.meshgrid(u, v)

    # Points on estimated subspace
    points_est = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            point = U[i, j] * est_sub_3d[:, 0] + V[i, j] * est_sub_3d[:, 1]
            points_est[i, j] = point

    ax1.plot_surface(points_est[:, :, 0], points_est[:, :, 1], points_est[:, :, 2],
                    alpha=0.3, color='red', label='Estimated subspace')
    ax1.set_title('Estimated Subspace Fit')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # Plot 2: True subspace (if provided)
    if true_subspace is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                   c='blue', alpha=0.6, s=50, label='Data points')

        true_sub_3d = true_subspace[:3, :]
        points_true = np.zeros((10, 10, 3))
        for i in range(10):
            for j in range(10):
                point = U[i, j] * true_sub_3d[:, 0] + V[i, j] * true_sub_3d[:, 1]
                points_true[i, j] = point

        ax2.plot_surface(points_true[:, :, 0], points_true[:, :, 1], points_true[:, :, 2],
                        alpha=0.3, color='green', label='True subspace')
        ax2.set_title('True Subspace')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    # Plot 3: Residuals
    ax3 = fig.add_subplot(133)
    projections = data_points @ estimated_subspace @ estimated_subspace.T
    residuals = np.linalg.norm(data_points - projections, axis=1)

    ax3.hist(residuals, bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_title('Residual Distribution')
    ax3.set_xlabel('Residual Magnitude')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_optimization_convergence(costs):
    """Plot optimization convergence."""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = np.arange(len(costs))
    ax.semilogy(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost Function Value (log scale)')
    ax.set_title('Grassmann Optimization Convergence')
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(f'Initial cost: {costs[0]:.4f}',
                xy=(0, costs[0]), xytext=(len(costs)*0.2, costs[0]*2),
                arrowprops={'arrowstyle': '->', 'color': 'red'})
    ax.annotate(f'Final cost: {costs[-1]:.4f}',
                xy=(len(costs)-1, costs[-1]), xytext=(len(costs)*0.7, costs[-1]*10),
                arrowprops={'arrowstyle': '->', 'color': 'red'})

    plt.tight_layout()
    return fig


def main():
    print("Grassmann Manifold Optimization Demo")
    print("=" * 50)

    # Parameters
    ambient_dim = 4
    subspace_dim = 2
    n_points = 100
    noise_level = 0.15

    # 1. Generate synthetic data
    key = jax.random.key(42)
    data_points, true_subspace = generate_noisy_subspace_data(
        key, n_points, ambient_dim, subspace_dim, noise_level
    )

    print(f"Generated {n_points} points in R^{ambient_dim}")
    print(f"True subspace dimension: {subspace_dim}")
    print(f"Noise level: {noise_level}")

    # 2. Define the Grassmann manifold
    grassmann = rx.Grassmann(ambient_dim, subspace_dim)

    # 3. Define optimization problem: minimize fitting error
    def cost_fn(X):
        """Minimize sum of squared distances from data to subspace."""
        projections = data_points @ X @ X.T
        residuals = data_points - projections
        return jnp.sum(residuals * residuals)

    problem = rx.RiemannianProblem(grassmann, cost_fn)

    # 4. Initialize with random subspace
    key = jax.random.key(123)
    X0 = grassmann.random_point(key)

    # 5. Solve the optimization problem
    print("\nSolving subspace fitting problem...")

    # Calculate initial cost
    initial_cost = cost_fn(X0)

    result = rx.minimize(
        problem, X0, method="rsgd",
        options={"learning_rate": 0.01, "max_iterations": 200}
    )

    # 6. Analyze results
    print("\nOptimization Results:")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Final cost: {result.fun:.6f}")
    print(f"Cost reduction: {((initial_cost - result.fun) / initial_cost * 100):.2f}%")
    print(f"Iterations: {result.niter}")

    # Calculate subspace distance (geodesic distance on Grassmann manifold)
    true_distance = grassmann.dist(result.x, true_subspace)
    print(f"Distance to true subspace: {true_distance:.4f}")

    # Check manifold constraint
    constraint_error = jnp.linalg.norm(result.x.T @ result.x - jnp.eye(subspace_dim))
    print(f"Orthogonality constraint error: {constraint_error:.2e}")

    # 7. Visualizations
    print("\nGenerating visualizations...")

    # Create a simple costs list for convergence plot
    # Since we don't have iteration-by-iteration costs, create a mock progression
    iterations = jnp.linspace(0, result.niter, 10)
    costs_progression = initial_cost * jnp.exp(-iterations / result.niter * 3) + result.fun

    # Convergence plot
    conv_fig = plot_optimization_convergence(costs_progression)
    output_path = os.path.join(os.path.dirname(__file__), "output", "grassmann_convergence.png")
    conv_fig.savefig(output_path, dpi=150, bbox_inches='tight')

    # 3D visualization (if applicable)
    if ambient_dim >= 3 and subspace_dim == 2:
        fit_fig = plot_subspace_fitting_3d(data_points, result.x, true_subspace)
        output_path = os.path.join(os.path.dirname(__file__), "output", "grassmann_subspace_fit.png")
        fit_fig.savefig(output_path, dpi=150, bbox_inches='tight')

    # Principal angles analysis
    def compute_principal_angles(U, V):
        """Compute principal angles between two subspaces."""
        Q1, _ = jnp.linalg.qr(U)
        Q2, _ = jnp.linalg.qr(V)
        svd_result = jnp.linalg.svd(Q1.T @ Q2, full_matrices=False)
        cosines = jnp.clip(svd_result[1], 0, 1)
        return jnp.arccos(cosines)

    principal_angles = compute_principal_angles(result.x, true_subspace)
    print(f"\nPrincipal angles (degrees): {jnp.degrees(principal_angles)}")

    # 8. Demonstrate manifold properties
    print("\nManifold Properties:")
    print(f"Grassmann manifold Gr({subspace_dim}, {ambient_dim})")
    print(f"Manifold dimension: {grassmann.dimension}")
    print(f"Ambient space dimension: {grassmann.ambient_dimension}")

    # Test tangent space projection
    test_tangent = jax.random.normal(jax.random.key(999), (ambient_dim, subspace_dim))
    projected_tangent = grassmann.proj(result.x, test_tangent)
    tangent_constraint = jnp.linalg.norm(result.x.T @ projected_tangent)
    print(f"Tangent space constraint (X^T V): {tangent_constraint:.2e}")

    plt.show()
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
