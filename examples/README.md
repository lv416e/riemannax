# RiemannAX Demo Examples

This directory contains demonstration scripts for Riemannian manifold optimization using the RiemannAX library.

## Overview of Demos

1. **Optimization on a Sphere** (`sphere_optimization_demo.py`)
   - Solves an optimization problem on the unit sphere S² to find the point closest to the North Pole ([0,0,1]).
   - Uses Riemannian stochastic gradient descent for optimization.
   - Visualizes the sphere and optimization process using matplotlib.

2. **Optimization on the Special Orthogonal Group** (`so_optimization_demo.py`)
   - Solves an optimization problem on the SO(3) special orthogonal group (3D rotation matrices) to find the rotation matrix closest to a target rotation matrix.
   - Optimizes rotation matrices on a Riemannian manifold by minimizing the Frobenius norm.
   - Visualizes rotation matrices as basis vectors using 3D visualization.

## How to Run

To run each demo:

```bash
# Sphere optimization demo
python sphere_optimization_demo.py

# Special orthogonal group optimization demo
python so_optimization_demo.py
```

## Required Libraries

These demos require the following libraries:

- RiemannAX
- JAX
- NumPy
- Matplotlib

Installation:

```bash
pip install -e ..  # Install riemannax
pip install matplotlib  # Required for visualization
```

## Results

When running each demo, the optimization results will be displayed and visualization figures will be generated:

- `sphere_optimization.png` - Optimization results on the sphere
- `so3_optimization.png` - Optimization results on the special orthogonal group

## Advanced Usage

These demo scripts can be used as templates for implementing your own optimization problems.
You can modify the cost function and manifold to experiment with various Riemannian optimization problems.