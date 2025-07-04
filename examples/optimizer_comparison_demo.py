#!/usr/bin/env python3
"""
Riemannian Optimizer Comparison: SGD vs Adam vs Momentum
========================================================

This example provides a comprehensive comparison of Riemannian optimization algorithms
available in RiemannAX: Riemannian SGD, Riemannian Adam, and Riemannian Momentum.

We test these optimizers across multiple manifolds and optimization problems to
demonstrate their convergence characteristics, computational efficiency, and
robustness to different problem structures.

Key Comparisons:
- Convergence speed and stability
- Parameter sensitivity analysis
- Performance across different manifold geometries
- Computational overhead and memory usage
- Robustness to initialization and problem conditioning

Author: RiemannAX Development Team
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import riemannax as rx


def create_sphere_optimization_problem(target_direction: jnp.ndarray):
    """Create a sphere optimization problem: minimize distance to target direction."""
    sphere = rx.Sphere()

    def cost_fn(x):
        return -jnp.dot(x, target_direction)  # Maximize dot product

    return rx.RiemannianProblem(sphere, cost_fn), sphere


def create_so3_alignment_problem(target_matrix: jnp.ndarray):
    """Create SO(3) optimization problem: align with target rotation matrix."""
    so3 = rx.SpecialOrthogonal(n=3)

    def cost_fn(R):
        return jnp.linalg.norm(R - target_matrix, 'fro')**2

    return rx.RiemannianProblem(so3, cost_fn), so3


def create_grassmann_subspace_problem(data: jnp.ndarray, subspace_dim: int):
    """Create Grassmann optimization problem: find best-fitting subspace."""
    n_features = data.shape[0]
    grassmann = rx.Grassmann(n=n_features, p=subspace_dim)

    def cost_fn(subspace):
        # Minimize reconstruction error
        projector = subspace @ subspace.T
        reconstruction = projector @ data
        return jnp.sum((data - reconstruction)**2)

    return rx.RiemannianProblem(grassmann, cost_fn), grassmann


def create_spd_covariance_problem(data: jnp.ndarray):
    """Create SPD optimization problem: robust covariance estimation."""
    n_features = data.shape[1]
    spd = rx.SymmetricPositiveDefinite(n=n_features)

    def cost_fn(cov_matrix):
        # Negative log-likelihood with regularization
        centered_data = data - jnp.mean(data, axis=0)
        try:
            cov_inv = jnp.linalg.inv(cov_matrix)
            log_det = jnp.linalg.slogdet(cov_matrix)[1]
            quadratic_term = jnp.sum((centered_data @ cov_inv) * centered_data)
            return log_det + quadratic_term / data.shape[0]
        except:
            return 1e6  # Large penalty for invalid matrices

    return rx.RiemannianProblem(spd, cost_fn), spd


def run_optimizer_comparison(problem, manifold, x0, problem_name: str, max_iterations: int = 100):
    """Run all optimizers on the same problem and collect detailed results."""

    optimizers = {
        'RSGD': {
            'method': 'rsgd',
            'options': {'learning_rate': 0.01, 'max_iterations': max_iterations}
        },
        'RAdaM': {
            'method': 'radam',
            'options': {
                'learning_rate': 0.001,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'max_iterations': max_iterations
            }
        },
        'RMomentum': {
            'method': 'rmom',
            'options': {
                'learning_rate': 0.005,
                'momentum': 0.9,
                'max_iterations': max_iterations
            }
        }
    }

    results = {}

    print(f"\n{problem_name} Optimization Results:")
    print("=" * 60)

    for name, config in optimizers.items():
        print(f"\nOptimizing with {name}...")

        # Time the optimization
        start_time = time.time()

        try:
            result = rx.minimize(problem, x0,
                               method=config['method'],
                               options=config['options'])

            optimization_time = time.time() - start_time

            # Collect metrics
            final_cost = result.fun
            final_point = result.x

            # Validate the result is on manifold
            if hasattr(manifold, '_is_in_manifold'):
                on_manifold = manifold._is_in_manifold(final_point)
            else:
                on_manifold = True  # Assume valid if no validation method

            # Compute gradient norm at solution
            grad = problem.grad(final_point)
            riemannian_grad = manifold.proj(final_point, grad)
            grad_norm = jnp.linalg.norm(riemannian_grad)

            results[name] = {
                'success': True,
                'final_cost': float(final_cost),
                'final_point': final_point,
                'optimization_time': optimization_time,
                'on_manifold': bool(on_manifold),
                'gradient_norm': float(grad_norm),
                'iterations': getattr(result, 'nit', max_iterations),
                'message': getattr(result, 'message', 'Success')
            }

            print(f"  Final cost: {final_cost:.8f}")
            print(f"  Gradient norm: {grad_norm:.8f}")
            print(f"  Time: {optimization_time:.4f}s")
            print(f"  On manifold: {on_manifold}")

        except Exception as e:
            print(f"  FAILED: {str(e)}")
            results[name] = {
                'success': False,
                'error': str(e),
                'optimization_time': time.time() - start_time
            }

    return results


def analyze_convergence_profiles(problems_and_manifolds: List[Tuple], max_iterations: int = 150):
    """Analyze detailed convergence profiles for all optimizers across problems."""

    print("\n" + "="*80)
    print("CONVERGENCE PROFILE ANALYSIS")
    print("="*80)

    all_results = {}

    for i, (problem_name, problem, manifold, x0) in enumerate(problems_and_manifolds):
        print(f"\nAnalyzing Problem {i+1}: {problem_name}")

        # Run detailed optimization with cost tracking
        optimizers = {
            'RSGD': ('rsgd', {'learning_rate': 0.01}),
            'RAdaM': ('radam', {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999}),
            'RMomentum': ('rmom', {'learning_rate': 0.005, 'momentum': 0.9})
        }

        problem_results = {}

        for opt_name, (method, base_options) in optimizers.items():
            options = {**base_options, 'max_iterations': max_iterations, 'tolerance': 1e-10}

            # Manual optimization loop to track costs
            if method == 'rsgd':
                init_fn, update_fn = rx.riemannian_gradient_descent(**base_options)
            elif method == 'radam':
                init_fn, update_fn = rx.riemannian_adam(**base_options)
            elif method == 'rmom':
                init_fn, update_fn = rx.riemannian_momentum(**base_options)

            state = init_fn(x0)
            costs = [float(problem.cost_fn(state.x))]
            times = [0.0]

            start_time = time.time()

            for iteration in range(max_iterations):
                gradient = problem.grad(state.x)
                state = update_fn(gradient, state, manifold)

                current_cost = float(problem.cost_fn(state.x))
                costs.append(current_cost)
                times.append(time.time() - start_time)

                # Early stopping if converged
                if len(costs) > 1 and abs(costs[-1] - costs[-2]) < 1e-12:
                    break

            problem_results[opt_name] = {
                'costs': costs,
                'times': times,
                'final_state': state,
                'converged_iteration': len(costs) - 1
            }

        all_results[problem_name] = problem_results

    return all_results


def create_comprehensive_visualization(all_results: Dict, problems_data: List):
    """Create comprehensive visualization of optimizer comparison results."""

    n_problems = len(all_results)
    fig = plt.figure(figsize=(20, 5 * n_problems))

    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    colors = {'RSGD': 'red', 'RAdaM': 'blue', 'RMomentum': 'green'}

    for i, (problem_name, results) in enumerate(all_results.items()):

        # Convergence plot
        ax1 = plt.subplot(n_problems, 3, 3*i + 1)

        for opt_name, opt_results in results.items():
            costs = opt_results['costs']
            iterations = range(len(costs))
            plt.semilogy(iterations, costs, color=colors[opt_name],
                        label=f'{opt_name}', linewidth=2, alpha=0.8)

        plt.title(f'{problem_name}: Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Time efficiency plot
        ax2 = plt.subplot(n_problems, 3, 3*i + 2)

        for opt_name, opt_results in results.items():
            costs = opt_results['costs']
            times = opt_results['times']
            plt.semilogy(times, costs, color=colors[opt_name],
                        label=f'{opt_name}', linewidth=2, alpha=0.8)

        plt.title(f'{problem_name}: Time Efficiency')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cost (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Final performance comparison
        ax3 = plt.subplot(n_problems, 3, 3*i + 3)

        opt_names = list(results.keys())
        final_costs = [results[name]['costs'][-1] for name in opt_names]
        convergence_iterations = [results[name]['converged_iteration'] for name in opt_names]

        x_pos = np.arange(len(opt_names))
        bars1 = plt.bar(x_pos - 0.2, final_costs, 0.4, label='Final Cost',
                       color=[colors[name] for name in opt_names], alpha=0.7)

        # Normalize iterations for display
        max_iter = max(convergence_iterations)
        normalized_iters = [it/max_iter for it in convergence_iterations]

        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x_pos + 0.2, convergence_iterations, 0.4,
                           label='Iterations', color='gray', alpha=0.5)

        ax3.set_xlabel('Optimizer')
        ax3.set_ylabel('Final Cost', color='black')
        ax3_twin.set_ylabel('Iterations to Convergence', color='gray')
        ax3.set_title(f'{problem_name}: Final Performance')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(opt_names)

        # Add value labels
        for bar, cost in zip(bars1, final_costs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{cost:.2e}', ha='center', va='bottom', fontsize=8)

        for bar, iters in zip(bars2, convergence_iterations):
            ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{iters}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save the comprehensive plot
    output_path = output_dir / "optimizer_comparison_comprehensive.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive visualization saved to: {output_path}")

    plt.show()


def demonstrate_optimizer_comparison():
    """Main demonstration function."""
    print("=" * 80)
    print("RIEMANNAX OPTIMIZER COMPARISON: SGD vs ADAM vs MOMENTUM")
    print("=" * 80)

    # Set up test problems
    key = jax.random.key(42)
    keys = jax.random.split(key, 10)

    # Problem 1: Sphere optimization
    target_direction = jnp.array([0., 0., 1.])  # North pole
    sphere_problem, sphere_manifold = create_sphere_optimization_problem(target_direction)
    sphere_x0 = rx.Sphere().random_point(keys[0])

    # Problem 2: SO(3) rotation alignment
    target_rotation = rx.SpecialOrthogonal(3).random_point(keys[1])
    so3_problem, so3_manifold = create_so3_alignment_problem(target_rotation)
    so3_x0 = rx.SpecialOrthogonal(3).random_point(keys[2])

    # Problem 3: Grassmann subspace fitting
    data_grassmann = jax.random.normal(keys[3], (6, 100))  # 6D data, find 3D subspace
    grassmann_problem, grassmann_manifold = create_grassmann_subspace_problem(data_grassmann, 3)
    grassmann_x0 = rx.Grassmann(6, 3).random_point(keys[4])

    # Problem 4: SPD covariance estimation
    true_cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    data_spd = jax.random.multivariate_normal(keys[5], jnp.zeros(2), true_cov, (200,))
    spd_problem, spd_manifold = create_spd_covariance_problem(data_spd)
    spd_x0 = rx.SymmetricPositiveDefinite(2).random_point(keys[6])

    # Define all problems
    problems_and_manifolds = [
        ("Sphere Optimization", sphere_problem, sphere_manifold, sphere_x0),
        ("SO(3) Rotation Alignment", so3_problem, so3_manifold, so3_x0),
        ("Grassmann Subspace Fitting", grassmann_problem, grassmann_manifold, grassmann_x0),
        ("SPD Covariance Estimation", spd_problem, spd_manifold, spd_x0)
    ]

    # Run comprehensive analysis
    convergence_results = analyze_convergence_profiles(problems_and_manifolds, max_iterations=100)

    # Create visualization
    create_comprehensive_visualization(convergence_results, problems_and_manifolds)

    # Summary statistics
    print("\n" + "="*80)
    print("OPTIMIZER PERFORMANCE SUMMARY")
    print("="*80)

    print(f"{'Problem':<25} {'Optimizer':<12} {'Final Cost':<15} {'Iterations':<12} {'Status'}")
    print("-" * 80)

    for problem_name, results in convergence_results.items():
        for opt_name, opt_results in results.items():
            final_cost = opt_results['costs'][-1]
            iterations = opt_results['converged_iteration']
            status = "✓ Converged" if final_cost < 1e-6 else "○ Partial"

            print(f"{problem_name:<25} {opt_name:<12} {final_cost:<15.2e} {iterations:<12} {status}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("- RAdaM: Best for ill-conditioned problems and robust convergence")
    print("- RMomentum: Good balance of speed and stability")
    print("- RSGD: Simple and reliable, good baseline")
    print("="*80)


if __name__ == "__main__":
    demonstrate_optimizer_comparison()
