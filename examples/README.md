# RiemannAX Examples

This directory contains demonstration scripts showcasing the capabilities of RiemannAX, a JAX-based library for Riemannian optimization.

## Setup

### Using uv (Recommended)

```bash
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install the package and dependencies
uv pip install -e .
uv add matplotlib

# Run the demos
python examples/grassmann_optimization_demo.py
python examples/stiefel_optimization_demo.py
python examples/manifolds_comparison_demo.py
```

### Using pip

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e .
pip install matplotlib

# Run the demos
python examples/grassmann_optimization_demo.py
```

## Core Manifold Demos (Milestone 1.1)

### Grassmann Manifold Demo
**File:** `grassmann_optimization_demo.py`

Demonstrates subspace fitting on the Grassmann manifold Gr(p,n). The demo:
- Generates synthetic data points near a p-dimensional subspace in R^n
- Uses Riemannian optimization to find the best-fitting subspace
- Visualizes the optimization convergence and subspace alignment
- Computes principal angles between estimated and true subspaces
- Shows 3D visualization of subspace fitting (for applicable dimensions)

**Key Features:**
- Subspace fitting optimization problem
- Principal component analysis via manifold optimization
- Grassmann manifold geometric operations demonstration
- Convergence analysis and error metrics

### Stiefel Manifold Demo
**File:** `stiefel_optimization_demo.py`

Solves the Orthogonal Procrustes problem on the Stiefel manifold St(p,n). The demo:
- Generates two point sets related by an orthogonal transformation
- Finds the optimal orthogonal matrix aligning the point sets
- Compares SVD and QR exponential map implementations
- Visualizes point alignment before and after optimization
- Analyzes rotation matrix properties and constraint satisfaction

**Key Features:**
- Orthogonal Procrustes problem solution
- Dual exponential map implementations (SVD vs QR)
- 3D visualization of point set alignment
- Sectional curvature computation
- Special case analysis (spheres, orthogonal groups)

### Manifolds Comparison Demo
**File:** `manifolds_comparison_demo.py`

Comprehensive comparison of all Milestone 1.1 manifolds. The demo:
- Runs representative optimization problems on Sphere, Grassmann, and Stiefel manifolds
- Compares convergence behavior and constraint satisfaction
- Demonstrates manifold-specific geometric properties
- Generates side-by-side performance analysis
- Validates manifold operations and error handling

**Key Features:**
- Multi-manifold optimization comparison
- Geometric properties demonstration
- Performance metrics analysis
- Comprehensive validation testing
- Summary statistics and visualizations

## Classic Manifold Demos

### Sphere Optimization Demo
**File:** `sphere_optimization_demo.py`

Demonstrates optimization on the unit sphere S². The problem finds the point on the sphere closest to a target direction (North Pole).

**Features:**
- 3D sphere visualization from multiple viewpoints
- Gradient-based optimization on curved manifolds
- Visualization of optimization trajectory

### SO(3) Optimization Demo
**File:** `so3_optimization_demo.py`

Shows optimization on the Special Orthogonal Group SO(3) of 3D rotation matrices. Solves a rotation matrix fitting problem.

**Features:**
- 3D rotation matrix visualization as basis vector transformations
- Frobenius norm minimization on matrix manifolds
- Comparison of initial, optimal, and target rotation matrices

## Running the Examples

Each demo can be run independently:

```bash
# Core manifold demos (Milestone 1.1)
python grassmann_optimization_demo.py
python stiefel_optimization_demo.py
python manifolds_comparison_demo.py

# Classic demos
python sphere_optimization_demo.py
python so3_optimization_demo.py
```

## Output

All demos generate:
- Console output with optimization results and analysis
- Visualization plots displayed on screen
- High-resolution PNG files saved to the `output/` directory

The output files include:
- `grassmann_convergence.png` - Grassmann optimization convergence
- `grassmann_subspace_fit.png` - 3D subspace fitting visualization
- `stiefel_convergence.png` - Stiefel optimization convergence
- `stiefel_procrustes.png` - Orthogonal Procrustes visualization
- `stiefel_method_comparison.png` - SVD vs QR exponential map comparison
- `manifolds_comparison.png` - Comprehensive manifold comparison
- `sphere_optimization.png` - Sphere optimization visualization
- `so3_optimization.png` - SO(3) rotation matrix visualization

## Requirements

- JAX and JAX[CPU] or JAX[GPU]
- matplotlib for visualization
- numpy for numerical computations
- RiemannAX library (properly installed)

## Jupyter Notebooks

Interactive Jupyter notebook versions are available in the `notebooks/` subdirectory for experimentation and educational purposes.
