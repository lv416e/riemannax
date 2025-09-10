# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Hyperbolic Manifolds - Computer Vision & Robotics Applications**
  - `PoincareBall`: Poincaré ball model for hyperbolic geometry
    - Complete Riemannian structure with Möbius operations and conformal mapping
    - Advanced parallel transport using conformal factor scaling
    - Sectional curvature computation and injectivity radius analysis
    - Optimized exponential/logarithmic maps with numerical stability guarantees
  - `Lorentz`: Hyperboloid model for alternative hyperbolic representation
    - Minkowski inner product geometry with proper constraint validation
    - Batch-compatible operations for large-scale hyperbolic embeddings
    - Upper hyperboloid sheet enforcement for consistent geometric properties

- **SE(3) Lie Group - Robotics & 3D Vision**
  - `SE3`: Special Euclidean group combining rotations and translations
    - Quaternion-based rotation representation with efficient composition
    - Advanced matrix exponential/logarithm using Rodrigues formula with Taylor series
    - Robust handling of small-angle singularities and near-identity transformations
    - JAX-native batch operations for simultaneous multi-pose optimization
    - Complete Lie group structure with proper tangent space projections

- **Enhanced Data Models & Validation Infrastructure**
  - `HyperbolicPoint`: Structured point representation with constraint validation
    - Automatic Poincaré ball and Lorentz model constraint checking
    - JAX-native validation methods compatible with JIT compilation
    - Enhanced error diagnostics with precise mathematical constraint information
  - `SE3Transform`: Robust SE(3) transformation validation
    - Orthogonality and determinant constraint verification for rotation matrices
    - Batch-compatible validation with enhanced numerical precision
    - Comprehensive error reporting for debugging transformation pipelines

- **Numerical Stability Manager**
  - Advanced numerical stability framework for hyperbolic and Lie group operations
  - Taylor series optimization avoiding factorial overflow in matrix exponentials
  - Model-specific stability limits and vector norm validation
  - JAX-native error handling with detailed diagnostic information

- **Factory Functions & Convenience API**
  - `create_poincare_ball()`, `create_lorentz()`, `create_se3()`: Standard constructors
  - `create_poincare_ball_for_embeddings()`: Optimized for embedding applications
  - `create_se3_for_robotics()`: Pre-configured for robotics pose estimation
  - Consistent parameter patterns and validation across all factory functions

- **Comprehensive Testing Infrastructure**
  - Extensive new test suite with over 120 functions covering mathematical properties and edge cases
  - Advanced geometric property validation (parallel transport, sectional curvature)
  - Performance benchmarking and batch operation scaling verification
  - Integration testing for computer vision and robotics applications

- **New Manifold Framework - Advanced Geometric Structures**
  - `ProductManifold`: Composite manifold implementation for M₁ × M₂ × ... × Mₖ structures
    - Component-wise operations with automatic dimension handling
    - Support for heterogeneous manifold combinations
    - Vectorized batch processing via `jax.vmap`
    - Comprehensive validation for mixed manifold types
  - `QuotientManifold`: Abstract framework for quotient manifold M/G implementations
    - Lie group action support with equivalence class management
    - Horizontal space projections orthogonal to group orbits
    - Quotient-aware geometric operations (exp, log, distance)
    - Enhanced Grassmann manifold with Gr(n,p) = St(n,p)/O(p) quotient structure

- **Mathematical Completeness - Advanced Differential Geometry**
  - **Grassmann Manifold Enhancements**:
    - `curvature_tensor()`: Riemannian curvature tensor R(u,v)w with Bianchi identity validation
    - `sectional_curvature()`: Sectional curvature K(u,v) for 2-dimensional subspaces
    - `christoffel_symbols()`: Christoffel symbols Γ(u,v) for Levi-Civita connection
    - `frechet_mean()`: Fréchet mean computation (Riemannian center of mass)
    - Mathematical property validation with comprehensive test coverage
  - **SPD Manifold Log-Euclidean Operations**:
    - `log_euclidean_exp()`: Matrix exponential in Log-Euclidean metric
    - `log_euclidean_log()`: Matrix logarithm for Log-Euclidean geometry
    - `log_euclidean_distance()`: Efficient distance computation alternative to affine-invariant
    - `log_euclidean_interpolation()`: Geodesic interpolation with parameter t ∈ [0,1]
    - `log_euclidean_mean()`: Riemannian mean via matrix logarithms
    - Computationally efficient alternative to affine-invariant metric operations

- **Optimistix Integration - Professional Optimization Framework**
  - `minimize_on_manifold()`: Main constrained optimization function
    - Integration with Optimistix solvers (BFGS, Gradient Descent, Nonlinear CG)
    - Automatic manifold constraint enforcement via retractions
    - JAX JIT compilation compatibility for high-performance optimization
    - Support for custom convergence criteria and step size controls
  - `least_squares_on_manifold()`: Least squares optimization on manifolds
    - Residual function optimization with manifold constraints
    - Automatic Jacobian computation with Riemannian gradient conversion
    - Support for overdetermined systems with geometric constraints
  - **Utility Functions**:
    - `euclidean_to_riemannian_gradient()`: Gradient space conversion utility
    - `ManifoldMinimizer`: Optimistix-compatible solver adapter class
    - `RiemannianProblemAdapter`: Problem format conversion for external solvers
  - **Comprehensive Integration Testing**: 19/19 tests passing with full compatibility validation

- **Enhanced Testing Infrastructure and Quality Assurance**
  - **Mathematical Property Validation**:
    - Curvature tensor properties (antisymmetry, Bianchi identity)
    - Sectional curvature non-negativity for Grassmann manifolds
    - Exponential/logarithmic map inverse property testing
    - Parallel transport isometry validation
  - **Performance and Numerical Stability**:
    - JIT compilation optimization testing across all manifolds
    - Batch scaling efficiency validation with correlation analysis
    - Edge case handling for near-singular configurations
    - Numerical precision testing for ill-conditioned problems
  - **External Integration Testing**:
    - Optimistix solver compatibility across different manifold types
    - Constraint enforcement validation during optimization steps
    - Convergence testing for realistic optimization problems

### Enhanced

- **Manifold Robustness and Performance**
  - Improved batch operations with simplified detection logic
  - Enhanced numerical stability in geometric operations
  - Better handling of edge cases (antipodal points, near-zero tangents)
  - Optimized memory usage in large-scale computations

- **Type Safety and Code Quality**
  - Complete type annotation coverage with `jaxtyping.Array`
  - Python 3.10+ compatibility with modern type syntax
  - Enhanced docstrings with mathematical background and references
  - Comprehensive error handling with descriptive messages

- **Development Infrastructure**
  - **Python Quality Constitution Compliance**:
    - MyPy static type checking with zero errors
    - Ruff linting with comprehensive rule set
    - Pre-commit hooks for automated quality gates
    - Pytest testing with extensive coverage
  - JAX JIT optimization framework with intelligent caching
  - Performance monitoring and benchmarking utilities

### Technical Improvements

- **JAX Integration Enhancements**
  - Advanced JIT compilation with static argument optimization
  - Device management for GPU/TPU acceleration
  - Batch processing optimization with linear scaling characteristics
  - Memory-efficient implementations for large-scale problems

- **Mathematical Rigor**
  - Comprehensive validation of manifold constraints
  - Robust numerical algorithms for geometric operations
  - Edge case handling for boundary conditions and singularities
  - Property-based testing for mathematical correctness

- **API Consistency**
  - Unified interface patterns across all manifold types
  - Consistent parameter naming and method signatures
  - Comprehensive documentation with usage examples
  - Backward compatibility preservation where possible

### Performance

- **Optimization Benchmarks**
  - JIT compilation provides 10-100x speedup for large problems
  - Linear scaling with batch size for vectorized operations
  - Memory-efficient algorithms for matrices with n > 1000
  - Hardware acceleration support (CPU/GPU/TPU) through JAX

This represents a major milestone, transforming RiemannAX from a research prototype into a production-ready library for advanced Riemannian optimization with comprehensive mathematical foundations, professional-grade integrations, and rigorous quality standards.

## [0.0.3] - 2025-07-05

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
