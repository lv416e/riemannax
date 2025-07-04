#!/usr/bin/env python3
"""
SPD Manifold: Robust Covariance Matrix Estimation
=================================================

This example demonstrates robust covariance matrix estimation on the Symmetric
Positive Definite (SPD) manifold. We compare standard maximum likelihood estimation
with manifold-based robust estimation that is resilient to outliers.

Applications:
- Computer vision: Robust covariance descriptors for image classification
- Finance: Portfolio optimization with heavy-tailed return distributions
- Signal processing: Covariance matrix estimation in the presence of noise
- Machine learning: Robust Gaussian mixture model parameter estimation

Mathematical Background:
The SPD manifold P(n) = {X ∈ R^{n×n} : X = X^T, X ≻ 0} equipped with the
affine-invariant Riemannian metric provides a natural framework for covariance
matrix estimation that respects the geometric structure of positive definite matrices.

Author: RiemannAX Development Team
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import riemannax as rx


def generate_multivariate_data_with_outliers(key, n_samples=200, n_features=4, outlier_ratio=0.1):
    """Generate multivariate data with outliers for robust estimation testing."""
    keys = jax.random.split(key, 4)

    # True covariance structure with correlation
    true_cov = jnp.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.5, 1.0, 0.3, 0.0],
        [0.2, 0.3, 1.0, 0.4],
        [0.1, 0.0, 0.4, 1.0]
    ])

    # Generate normal samples
    n_clean = int(n_samples * (1 - outlier_ratio))
    clean_data = jax.random.multivariate_normal(keys[0], jnp.zeros(n_features), true_cov, (n_clean,))

    # Generate outlier samples (heavy-tailed distribution)
    n_outliers = n_samples - n_clean
    outlier_scale = 3.0  # Scale factor for outliers
    outlier_data = outlier_scale * jax.random.multivariate_normal(
        keys[1], jnp.zeros(n_features), jnp.eye(n_features), (n_outliers,)
    )

    # Combine data
    data = jnp.vstack([clean_data, outlier_data])

    # Shuffle the data
    perm = jax.random.permutation(keys[2], n_samples)
    data = data[perm]

    return data, true_cov


def mle_covariance(data):
    """Standard maximum likelihood estimation of covariance matrix."""
    n_samples = data.shape[0]
    centered_data = data - jnp.mean(data, axis=0)
    return (centered_data.T @ centered_data) / (n_samples - 1)


def robust_manifold_covariance_cost(cov_matrix, data, huber_delta=1.5):
    """
    Robust covariance estimation cost function using Huber loss.

    This cost function uses the Mahalanobis distance with Huber loss
    to reduce the influence of outliers on covariance estimation.
    """
    n_samples = data.shape[0]
    centered_data = data - jnp.mean(data, axis=0)

    # Compute Mahalanobis distances
    cov_inv = jnp.linalg.inv(cov_matrix)
    mahalanobis_sq = jnp.sum((centered_data @ cov_inv) * centered_data, axis=1)

    # Apply Huber loss to reduce outlier influence
    def huber_loss(x, delta):
        condition = jnp.abs(x) <= delta
        quadratic = 0.5 * x**2
        linear = delta * (jnp.abs(x) - 0.5 * delta)
        return jnp.where(condition, quadratic, linear)

    # Negative log-likelihood with Huber loss
    log_det_term = jnp.log(jnp.linalg.det(cov_matrix))
    huber_distances = jax.vmap(lambda x: huber_loss(jnp.sqrt(x), huber_delta))(mahalanobis_sq)

    return log_det_term + jnp.mean(huber_distances)


def optimize_covariance_manifold(data, method='radam', max_iterations=100):
    """Optimize covariance matrix on SPD manifold using robust estimation."""
    n_features = data.shape[1]
    spd = rx.SymmetricPositiveDefinite(n=n_features)

    # Initialize with identity matrix (well-conditioned starting point)
    key = jax.random.key(42)
    x0 = spd.random_point(key)

    # Define robust covariance estimation problem
    def cost_fn(cov):
        return robust_manifold_covariance_cost(cov, data)

    problem = rx.RiemannianProblem(spd, cost_fn)

    # Optimize using specified method
    learning_rates = {'rsgd': 0.01, 'radam': 0.001, 'rmom': 0.005}
    options = {
        'learning_rate': learning_rates.get(method, 0.01),
        'max_iterations': max_iterations,
        'tolerance': 1e-8
    }

    if method == 'radam':
        options.update({'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8})
    elif method == 'rmom':
        options.update({'momentum': 0.9})

    result = rx.minimize(problem, x0, method=method, options=options)
    return result


def frobenius_error(A, B):
    """Compute Frobenius norm error between two matrices."""
    return jnp.sqrt(jnp.sum((A - B)**2))


def log_euclidean_distance(A, B):
    """Compute Log-Euclidean distance between SPD matrices."""
    log_A = jnp.linalg.slogdet(A)[1]  # Log determinant
    log_B = jnp.linalg.slogdet(B)[1]
    return jnp.abs(log_A - log_B)


def demonstrate_spd_covariance_estimation():
    """Main demonstration of SPD manifold covariance estimation."""
    print("=" * 70)
    print("SPD Manifold: Robust Covariance Matrix Estimation")
    print("=" * 70)

    # Generate synthetic data with outliers
    key = jax.random.key(123)
    data, true_cov = generate_multivariate_data_with_outliers(key, n_samples=300, outlier_ratio=0.15)

    print(f"Dataset: {data.shape[0]} samples, {data.shape[1]} features, 15% outliers")
    print(f"True covariance matrix:\n{true_cov}")
    print()

    # Standard MLE estimation
    mle_cov = mle_covariance(data)
    mle_error = frobenius_error(mle_cov, true_cov)

    print("Standard Maximum Likelihood Estimation:")
    print(f"Frobenius error: {mle_error:.6f}")
    print(f"MLE covariance:\n{mle_cov}")
    print()

    # Robust manifold-based estimation with different optimizers
    methods = ['rsgd', 'radam', 'rmom']
    results = {}

    for method in methods:
        print(f"Robust SPD Manifold Estimation ({method.upper()}):")
        result = optimize_covariance_manifold(data, method=method, max_iterations=150)

        robust_cov = result.x
        robust_error = frobenius_error(robust_cov, true_cov)
        log_eucl_dist = log_euclidean_distance(robust_cov, true_cov)

        results[method] = {
            'covariance': robust_cov,
            'frobenius_error': robust_error,
            'log_euclidean_distance': log_eucl_dist,
            'final_cost': result.fun,
            'iterations': len(result.costs) if hasattr(result, 'costs') else result.nit
        }

        print(f"Frobenius error: {robust_error:.6f}")
        print(f"Log-Euclidean distance: {log_eucl_dist:.6f}")
        print(f"Final cost: {result.fun:.6f}")
        print(f"Converged in {results[method]['iterations']} iterations")
        print(f"Robust covariance ({method}):\n{robust_cov}")
        print()

    # Create comprehensive visualization
    create_covariance_estimation_plots(true_cov, mle_cov, results, data)

    # Performance summary
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Frobenius Error':<16} {'Log-Euclidean':<14} {'Improvement':<12}")
    print("-" * 70)

    baseline_error = mle_error
    print(f"{'MLE':<12} {mle_error:<16.6f} {'N/A':<14} {'Baseline':<12}")

    for method in methods:
        error = results[method]['frobenius_error']
        log_dist = results[method]['log_euclidean_distance']
        improvement = ((baseline_error - error) / baseline_error) * 100
        print(f"{method.upper():<12} {error:<16.6f} {log_dist:<14.6f} {improvement:+8.2f}%")

    # Find best method
    best_method = min(methods, key=lambda m: results[m]['frobenius_error'])
    best_error = results[best_method]['frobenius_error']
    improvement_pct = ((mle_error - best_error) / mle_error) * 100

    print("-" * 70)
    print(f"Best performing method: {best_method.upper()}")
    print(f"Improvement over MLE: {improvement_pct:.2f}%")
    print("=" * 70)


def create_covariance_estimation_plots(true_cov, mle_cov, results, data):
    """Create comprehensive visualization of covariance estimation results."""
    fig = plt.figure(figsize=(16, 12))

    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    # 1. Data scatter plot with ellipses
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)

    # Plot confidence ellipses
    def plot_confidence_ellipse(cov, color, label, alpha=0.3):
        eigenvals, eigenvecs = jnp.linalg.eigh(cov[:2, :2])
        angle = jnp.degrees(jnp.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * jnp.sqrt(eigenvals)

        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=2,
                         label=label)
        ax1.add_patch(ellipse)

    plot_confidence_ellipse(true_cov, 'green', 'True')
    plot_confidence_ellipse(mle_cov, 'red', 'MLE')
    plot_confidence_ellipse(results['radam']['covariance'], 'blue', 'Robust (Adam)')

    plt.title('Data Distribution with Confidence Ellipses')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # 2. Covariance matrices heatmap
    methods_to_plot = ['True', 'MLE', 'RSGD', 'RAdaM', 'RMom']
    covs_to_plot = [true_cov, mle_cov] + [results[m]['covariance'] for m in ['rsgd', 'radam', 'rmom']]

    for i, (method, cov) in enumerate(zip(methods_to_plot, covs_to_plot)):
        ax = plt.subplot(2, len(methods_to_plot), len(methods_to_plot) + 1 + i)
        im = plt.imshow(cov, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.title(f'{method} Covariance')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        # Add text annotations
        for i_coord in range(cov.shape[0]):
            for j_coord in range(cov.shape[1]):
                plt.text(j_coord, i_coord, f'{cov[i_coord, j_coord]:.2f}',
                        ha='center', va='center', fontsize=8)

    # 3. Error comparison
    ax6 = plt.subplot(2, 3, 6)
    methods = ['MLE', 'RSGD', 'RAdaM', 'RMom']
    frobenius_errors = [frobenius_error(mle_cov, true_cov)] + \
                      [results[m]['frobenius_error'] for m in ['rsgd', 'radam', 'rmom']]

    colors = ['red', 'orange', 'blue', 'purple']
    bars = plt.bar(methods, frobenius_errors, color=colors, alpha=0.7)
    plt.title('Covariance Estimation Error')
    plt.ylabel('Frobenius Error')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, frobenius_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{error:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / "spd_covariance_estimation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    demonstrate_spd_covariance_estimation()
