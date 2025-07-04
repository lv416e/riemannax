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

## New Applications and Advanced Examples

### SPD Manifold Applications
**File:** `spd_covariance_estimation.py`

Demonstrates robust covariance matrix estimation on the SPD manifold with applications in:
- Computer vision: Robust covariance descriptors
- Finance: Portfolio optimization with heavy-tailed distributions
- Signal processing: Noise-robust covariance estimation

**Key Features:**
- Comparison with standard maximum likelihood estimation
- Outlier-robust estimation using Huber loss
- Performance evaluation across different optimizers
- Comprehensive visualization of results

### Optimizer Comparison and Analysis
**File:** `optimizer_comparison_demo.py`

Comprehensive comparison of all RiemannAX optimizers (SGD, Adam, Momentum) across multiple manifolds:
- Convergence speed and stability analysis
- Parameter sensitivity studies
- Performance across different problem structures
- Detailed timing and efficiency metrics

**Key Features:**
- Cross-manifold optimization comparison
- Convergence profile analysis
- Step size and gradient norm evolution
- Performance recommendations for different scenarios

### Machine Learning Applications Showcase
**File:** `ml_applications_showcase.py`

Practical machine learning applications demonstrating RiemannAX in real-world scenarios:

1. **Geometric PCA**: Principal component analysis on Grassmann manifolds
2. **Robust Anomaly Detection**: Outlier detection using SPD manifold optimization
3. **Rotation-Invariant Features**: Feature learning on SO(3) for 3D data

**Key Features:**
- Comparison with standard Euclidean methods
- Performance improvements quantification
- Comprehensive visualization and analysis
- Production-ready implementation examples

## Advanced Tutorials

### Interactive Jupyter Notebooks
**Directory:** `notebooks/`

Enhanced notebook collection including:
- `advanced_riemannian_optimization.ipynb`: Comprehensive tutorial covering mathematical foundations, numerical considerations, and best practices
- `so3_optimization_demo.ipynb`: Interactive SO(3) rotation optimization
- `sphere_optimization_demo.ipynb`: Sphere manifold exploration

**Advanced Tutorial Features:**
- Mathematical theory with practical implementation
- Numerical stability analysis
- Performance optimization techniques
- Multi-manifold optimization examples
- Best practices and troubleshooting guide

## Running the New Examples

All new demos can be run independently:

```bash
# SPD manifold applications
python spd_covariance_estimation.py

# Optimizer comparison and analysis
python optimizer_comparison_demo.py

# Machine learning applications
python ml_applications_showcase.py

# Advanced Jupyter tutorial
jupyter notebook notebooks/advanced_riemannian_optimization.ipynb
```

## Extended Output Files

The new examples generate additional visualization files:
- `spd_covariance_estimation.png` - Robust covariance estimation analysis
- `optimizer_comparison_comprehensive.png` - Complete optimizer comparison
- `ml_applications_showcase.png` - Machine learning applications results
- `advanced_optimizer_analysis.png` - Detailed optimizer behavior analysis

## Requirements

- JAX and JAX[CPU] or JAX[GPU]
- matplotlib for visualization
- numpy for numerical computations
- RiemannAX library (properly installed)
- sklearn (for ML comparison baselines)
- jupyter (for interactive notebooks)

## Additional Dependencies for New Examples

```bash
# For ML applications
pip install scikit-learn

# For advanced visualizations
pip install seaborn

# For interactive notebooks
pip install jupyter ipykernel
```

## Jupyter Notebooks

Interactive Jupyter notebook versions are available in the `notebooks/` subdirectory for experimentation and educational purposes. The advanced tutorial notebook provides comprehensive coverage of Riemannian optimization theory and practice.
