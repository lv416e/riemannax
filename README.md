# RiemannAX

Hardware-accelerated Riemannian Manifold Optimization with JAX

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/riemannax.svg?cache=no)](https://pypi.org/project/riemannax/)
[![Tests](https://github.com/lv416e/riemannax/actions/workflows/tests.yml/badge.svg)](https://github.com/lv416e/riemannax/actions/workflows/tests.yml)
[![Lint](https://github.com/lv416e/riemannax/actions/workflows/lint.yml/badge.svg)](https://github.com/lv416e/riemannax/actions/workflows/lint.yml)
[![Docs](https://github.com/lv416e/riemannax/actions/workflows/docs.yml/badge.svg)](https://github.com/lv416e/riemannax/actions/workflows/docs.yml)
[![Release](https://github.com/lv416e/riemannax/actions/workflows/release.yml/badge.svg)](https://github.com/lv416e/riemannax/actions/workflows/release.yml)

## Overview

RiemannAX is a high-performance library for optimization on Riemannian manifolds, built upon JAX's ecosystem. It provides mathematically rigorous implementations of manifold structures and optimization algorithms, leveraging automatic differentiation, just-in-time compilation, and hardware acceleration to deliver exceptional computational efficiency for geometric optimization problems.

The library bridges the gap between theoretical differential geometry and practical machine learning applications, enabling researchers and practitioners to solve complex optimization problems that arise in computer vision, machine learning, and scientific computing.

## Key Features

### 🔬 **Comprehensive Manifold Library**
- **Sphere** (`S^n`): Unit hypersphere with geodesic operations
- **Special Orthogonal Group** (`SO(n)`): Rotation matrices with Lie group structure
- **Grassmann Manifold** (`Gr(p,n)`): Subspace optimization for dimensionality reduction and principal component analysis
- **Stiefel Manifold** (`St(p,n)`): Orthonormal frames with applications in orthogonal Procrustes problems
- Rigorous implementations with validation, batch operations, and numerical stability

### ⚡ **High-Performance Optimization**
- **Riemannian Gradient Descent**: First-order optimization with exponential maps and retractions
- **Automatic Differentiation**: Seamless computation of Riemannian gradients from Euclidean cost functions
- **Hardware Acceleration**: GPU/TPU support through JAX's XLA compilation
- **Batch Processing**: Vectorized operations for multiple optimization instances

### 🛠 **Robust Framework**
- **Flexible Problem Definition**: Support for custom cost functions and gradients
- **Comprehensive Validation**: Manifold constraint verification and numerical stability checks
- **Extensive Testing**: 77+ unit and integration tests ensuring mathematical correctness
- **Type Safety**: Full type annotations for Python 3.10+ compatibility

## Installation

### Standard Installation
```bash
pip install riemannax
```

### Development Installation
```bash
git clone https://github.com/lv416e/riemannax.git
cd riemannax
pip install -e ".[dev]"
```

### With UV Package Manager
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Quick Start Examples

### Sphere Optimization: Finding Optimal Directions
```python
import jax
import jax.numpy as jnp
import riemannax as rx

# Define the unit sphere manifold
sphere = rx.Sphere()

# Optimization problem: find point closest to target direction
target = jnp.array([0., 0., 1.])  # North pole
def cost_fn(x):
    return -jnp.dot(x, target)

problem = rx.RiemannianProblem(sphere, cost_fn)

# Initialize and solve
key = jax.random.key(42)
x0 = sphere.random_point(key)
result = rx.minimize(problem, x0, method='rsgd',
                    options={'learning_rate': 0.1, 'max_iterations': 100})

print(f"Optimal point: {result.x}")
print(f"Final cost: {result.fun:.6f}")
```

### Grassmann Manifold: Subspace Fitting
```python
import jax
import jax.numpy as jnp
import riemannax as rx

# Generate synthetic data in a 3D subspace of 8D space
key = jax.random.key(123)
n, p, m = 8, 3, 100
true_subspace = rx.Grassmann(n, p).random_point(key)

# Create noisy data
keys = jax.random.split(key, 3)
coeffs = jax.random.normal(keys[0], (p, m))
noise = 0.1 * jax.random.normal(keys[1], (n, m))
data = true_subspace @ coeffs + noise

# Define subspace fitting problem
def subspace_cost(x):
    projector = x @ x.T
    reconstruction = projector @ data
    return jnp.sum((data - reconstruction) ** 2)

# Optimize on Grassmann manifold
manifold = rx.Grassmann(n, p)
problem = rx.RiemannianProblem(manifold, subspace_cost)
x0 = manifold.random_point(keys[2])

result = rx.minimize(problem, x0, method='rsgd',
                    options={'learning_rate': 0.01, 'max_iterations': 200})

print(f"Reconstruction error: {result.fun:.6f}")
```

### Stiefel Manifold: Orthogonal Procrustes Problem
```python
import jax
import jax.numpy as jnp
import riemannax as rx

# Setup Procrustes problem: find optimal orthogonal transformation
key = jax.random.key(789)
n, p = 6, 4
keys = jax.random.split(key, 3)

A = jax.random.normal(keys[0], (n, p))
B = jax.random.normal(keys[1], (n, p))

# Minimize ||A - BQ||_F^2 over orthogonal matrices Q
def procrustes_cost(Q):
    return jnp.sum((A - B @ Q) ** 2)

# Optimize on Stiefel manifold (orthogonal group)
manifold = rx.Stiefel(p, p)
problem = rx.RiemannianProblem(manifold, procrustes_cost)
x0 = manifold.random_point(keys[2])

result = rx.minimize(problem, x0, method='rsgd',
                    options={'learning_rate': 0.1, 'max_iterations': 100})

print(f"Procrustes cost: {result.fun:.6f}")
print(f"Orthogonality check: {jnp.allclose(result.x.T @ result.x, jnp.eye(p))}")
```

## Advanced Usage

### Custom Gradient Functions
```python
# Define Euclidean gradient for automatic projection
def euclidean_grad(x):
    return jax.grad(cost_fn)(x)

problem = rx.RiemannianProblem(manifold, cost_fn, euclidean_grad_fn=euclidean_grad)
```

### Batch Optimization
```python
# Optimize multiple instances simultaneously
batch_size = 10
x0_batch = manifold.random_point(key, batch_size)

# Vectorized cost function
def batch_cost(x_batch):
    return jax.vmap(cost_fn)(x_batch)

batch_problem = rx.RiemannianProblem(manifold, batch_cost)
```

### Exponential Map vs. Retraction
```python
# Use exponential map for geodesically exact optimization
result_exp = rx.minimize(problem, x0, method='rsgd', use_retraction=False)

# Use retraction for computational efficiency
result_retr = rx.minimize(problem, x0, method='rsgd', use_retraction=True)
```

## Comprehensive Examples

Explore detailed implementations in the `examples/` directory:

- **`sphere_optimization_demo.py`**: Sphere optimization with visualization
- **`grassmann_optimization_demo.py`**: Subspace fitting and principal angles analysis
- **`stiefel_optimization_demo.py`**: Orthogonal Procrustes with multiple exponential map methods
- **`manifolds_comparison_demo.py`**: Comparative analysis across all manifolds
- **`notebooks/`**: Interactive Jupyter notebooks with step-by-step tutorials

## Testing and Development

### Running Tests
```bash
# Quick test suite
make test

# With coverage analysis
make coverage

# Specific test categories
pytest tests/manifolds/     # Manifold implementations
pytest tests/optimizers/    # Optimization algorithms
pytest tests/integration/   # End-to-end workflows
```

### Development Workflow
```bash
# Install development dependencies
pip install -e ".[dev]"

# Code formatting and linting
make format
make lint

# Type checking
make typecheck

# Documentation building
make docs
```

## Performance Characteristics

RiemannAX leverages JAX's XLA compilation for exceptional performance:

- **GPU Acceleration**: Automatic device placement and parallel execution
- **JIT Compilation**: First-call compilation overhead with subsequent near-C performance
- **Memory Efficiency**: In-place operations and optimized memory layouts
- **Batch Processing**: Vectorized operations across multiple problem instances

Typical performance improvements over CPU-based alternatives:
- 10-100x speedup on GPU for large-scale problems
- 2-5x speedup on CPU through XLA optimization
- Linear scaling with batch size for parallel optimization

## Mathematical Foundation

RiemannAX implements manifolds with rigorous differential geometric operations:

### Manifold Interface
Each manifold provides:
- **Exponential Map** (`exp`): Geodesic curves from tangent vectors
- **Logarithmic Map** (`log`): Inverse of exponential map
- **Retraction** (`retr`): Computationally efficient approximation to exponential map
- **Parallel Transport** (`transp`): Moving tangent vectors along manifold
- **Riemannian Metric** (`inner`): Tangent space inner products
- **Projection** (`proj`): Orthogonal projection onto tangent space

### Numerical Stability
- Robust QR-based orthogonalization for Stiefel and Grassmann manifolds
- Numerically stable distance computations using principal angles
- Careful handling of edge cases and degenerate configurations
- Comprehensive validation with appropriate floating-point tolerances

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Development setup and workflow
- Code style and testing requirements
- Documentation standards
- Pull request process

### Development Priorities
- Additional manifold implementations (Hyperbolic, Product manifolds)
- Advanced optimization algorithms (Conjugate Gradient, L-BFGS)
- Enhanced visualization and debugging tools
- Performance optimizations and benchmarking

## Citation

If you use RiemannAX in your research, please cite:

```bibtex
@software{riemannax2024,
  title={RiemannAX: Hardware-accelerated Riemannian Manifold Optimization with JAX},
  author={mary},
  year={2024},
  url={https://github.com/lv416e/riemannax}
}
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

RiemannAX draws inspiration from:
- **JAX**: Functional programming and automatic differentiation paradigms
- **Optax**: Optimization algorithm design patterns
- **Pymanopt**: Comprehensive Riemannian optimization reference
- **Geoopt**: PyTorch-based Riemannian optimization library

Special thanks to the JAX development team for creating an exceptional foundation for scientific computing.
