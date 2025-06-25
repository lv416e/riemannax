# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED] - YYYY/MM/DD

### Added

- **Milestone 1.1: Core Manifolds Implementation**
  - Enhanced `Manifold` base class with comprehensive validation and error handling
  - `Grassmann` manifold Gr(p,n) for p-dimensional subspaces in R^n
  - `Stiefel` manifold St(p,n) for orthonormal p-frames in R^n
  - Dual exponential map implementations (SVD and QR methods) for Stiefel manifolds
  - Sectional curvature computation for geometric analysis
  - Comprehensive manifold constraint validation (`validate_point`, `validate_tangent`)
  - Proper dimension properties for all manifolds

- **Comprehensive Example Demonstrations**
  - `grassmann_optimization_demo.py`: Subspace fitting optimization with 3D visualization
  - `stiefel_optimization_demo.py`: Orthogonal Procrustes problem with method comparison
  - `manifolds_comparison_demo.py`: Side-by-side performance analysis of all manifolds
  - High-quality visualization outputs with convergence plots and geometric analysis

- **Enhanced Development Infrastructure**
  - Full `uv` package manager compatibility and setup instructions
  - Improved `examples/README.md` with setup guides for both `uv` and `pip`
  - Automatic output directory creation for visualization files

### Enhanced

- Updated `Sphere` manifold with validation methods for consistency
- Improved error handling throughout the manifold hierarchy
- Enhanced type annotations using `jax.Array` for Python 3.10+ compatibility

### Fixed

- Import path resolution for development environments
- Parameter ordering consistency across manifold initializations
- File path handling for cross-platform compatibility
- Constraint satisfaction thresholds for practical optimization scenarios

## [0.0.1] - 2025/04/11

### Added

- Initial implementation of core components:
    - Manifolds: `SpecialOrthogonal` (SO(n)) and `Sphere`.
    - Optimizers: `riemannian_gradient_descent`.
    - Problem definition: Base class `RiemannianProblem`.
    - Solvers: `minimize` function and `OptimizeResult` class.

### Changed

### Fixed
