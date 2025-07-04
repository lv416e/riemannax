# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Symmetric Positive Definite (SPD) Manifold Implementation**
  - `SymmetricPositiveDefinite` manifold for optimization over covariance matrices and kernel learning
  - Complete Riemannian structure with bi-invariant Log-Euclidean metric
  - Matrix logarithmic and exponential maps using eigenvalue decomposition
  - Geometric mean computation via manifold interpolation
  - Robust numerical validation for positive definiteness and symmetry constraints
  - Comprehensive test suite with edge case handling for small eigenvalues

- **Advanced Riemannian Optimization Algorithms**
  - `riemannian_adam`: Adaptive moment estimation with Riemannian transport
    - First and second moment estimates with parallel transport
    - Bias correction mechanism adapted for manifold constraints
    - Numerically stable epsilon handling and step size clipping
  - `riemannian_momentum`: Momentum-accelerated gradient descent
    - Momentum accumulation with proper tangent space transport
    - Configurable momentum coefficient and learning rate scheduling
    - Enhanced convergence properties for non-convex manifold optimization

- **Optimization Framework Enhancements**
  - Extended `minimize` solver with new optimization methods (`radam`, `rmom`)
  - Improved numerical stability across all optimizers with conservative step size limits
  - Enhanced manifold projection utilities for sphere-like manifolds
  - Comprehensive parameter validation and error handling

### Enhanced

- **Special Orthogonal Group (SO) Optimizations**
  - Specialized matrix exponential and logarithmic implementations for SO(3)
  - Improved numerical stability for small rotation angles and near-identity matrices
  - Enhanced parallel transport with proper geodesic distance computations

- **Sphere Manifold Improvements**
  - Conditional parallel transport implementation using JAX control flow
  - Enhanced projection operator supporting both tangent space and manifold normalization
  - Improved numerical handling of antipodal points and near-zero tangent vectors

- **Development and Testing Infrastructure**
  - Updated linting configuration with expanded rule set (E741 ambiguous variables)
  - Enhanced type annotations with proper `Dict` and `Any` imports

- **Strategic Planning and Documentation**
  - Added comprehensive strategic roadmap (`design/strategic_roadmap.md`)
  - Market analysis and competitive landscape assessment
  - Detailed development priorities and KPI tracking
  - Long-term vision and innovation opportunities
  - Risk management and mitigation strategies

### Fixed

- Numerical instability issues in Adam optimizer with conservative epsilon placement
- Momentum update logic corrected for proper gradient accumulation
- Bare exception handling replaced with specific `Exception` types for linting compliance
- Test suite stability with appropriately scaled learning rates and iteration counts
- Type annotation consistency across optimizer state management classes

## [0.0.2] - 2025/06/26

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
