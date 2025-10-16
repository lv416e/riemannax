# Project Structure

## Root Directory Organization

```
riemannax/
├── riemannax/                  # Main package source code
├── tests/                      # Test suite with comprehensive coverage
├── examples/                   # Demonstration scripts and usage examples
├── benchmarks/                 # Performance benchmarking tools
├── docs/                       # Documentation and build system
├── design/                     # Design documents and roadmaps
├── .github/                    # GitHub workflows and issue templates
├── .kiro/                      # Kiro steering and spec-driven development
├── .claude/                    # Claude Code commands and configuration
├── pyproject.toml              # Project configuration and dependencies
├── Makefile                    # Development workflow automation
└── README.md                   # Project overview and quick start
```

### Key Configuration Files
- **pyproject.toml**: Centralized project configuration (dependencies, tools, build system)
- **Makefile**: Development workflow commands (test, lint, format, clean)
- **uv.lock**: Package manager lockfile for reproducible environments
- **.python-version**: Python version specification for tooling
- **.pre-commit-config.yaml**: Pre-commit hooks for code quality
- **.readthedocs.yml**: Read the Docs documentation configuration

## Core Package Structure (`riemannax/`)

### Modular Architecture
```
riemannax/
├── __init__.py                 # Public API exports and version info
├── core/                       # Core functionality and utilities
├── manifolds/                  # Manifold implementations
├── optimizers/                 # Riemannian optimization algorithms
├── solvers/                    # High-level solver interfaces
└── problems/                   # Problem definition framework
```

### Core Module (`riemannax/core/`)
**Purpose**: Foundational utilities, performance optimizations, and system management

- **JIT Management**: `jit_decorator.py`, `jit_manager.py`, `safe_jit.py`
- **Performance**: `performance.py`, `performance_benchmark.py`, `batch_ops.py`
- **Numerical Stability**: `numerical_stability.py`, `cholesky_engine.py`
- **System Utilities**: `device_manager.py`, `type_system.py`, `constants.py`
- **Algorithm Selection**: `spd_algorithm_selector.py`
- **Mathematical Utilities**: `geodesic_connection.py`

### Manifolds Module (`riemannax/manifolds/`)
**Purpose**: Implementations of specific Riemannian manifolds

**Base Infrastructure**:
- `base.py`: Abstract base class defining manifold interface
- `factory.py`: Factory pattern for manifold creation
- `data_models.py`: Structured data types for manifold operations
- `errors.py`: Manifold-specific exception classes
- `numerical_stability.py`: Cross-manifold stability utilities

**Manifold Implementations**:
- `sphere.py`: Unit hypersphere (`S^n`)
- `grassmann.py`: Grassmann manifold (`Gr(p,n)`)
- `stiefel.py`: Stiefel manifold (`St(p,n)`)
- `so.py`: Special orthogonal group (`SO(n)`)
- `spd.py`: Symmetric positive definite matrices
- `lorentz.py`: Lorentz/hyperbolic manifold
- `poincare_ball.py`: Poincaré ball model
- `se3.py`: Special Euclidean group (`SE(3)`)
- `product.py`: Product manifold constructions
- `quotient.py`: Quotient manifold constructions

### Optimizers Module (`riemannax/optimizers/`)
**Purpose**: Riemannian optimization algorithms

- `sgd.py`: Stochastic gradient descent for manifolds
- `adam.py`: Adam optimizer with Riemannian adaptations
- `momentum.py`: Momentum-based optimization
- `state.py`: Optimizer state management and utilities

### Solvers Module (`riemannax/solvers/`)
**Purpose**: High-level problem solving interfaces

- `minimize.py`: Primary optimization interface
- `optimistix_adapter.py`: Integration with Optimistix solvers

### Problems Module (`riemannax/problems/`)
**Purpose**: Problem definition and encapsulation

- `base.py`: Base problem definition classes and interfaces

## Test Structure (`tests/`)

### Hierarchical Test Organization
```
tests/
├── conftest.py                 # Shared test configuration and fixtures
├── core/                       # Core functionality tests
├── manifolds/                  # Manifold-specific tests
├── optimizers/                 # Optimizer algorithm tests
├── solvers/                    # Solver interface tests
├── problems/                   # Problem definition tests
├── utils/                      # Test utilities and helpers
├── test_integration.py         # Cross-module integration tests
├── test_performance_*.py       # Performance validation tests
└── test_*_jit.py              # JIT compilation tests
```

### Test Categories (via pytest markers)
- **manifolds**: Manifold implementation tests
- **optimizers**: Optimization algorithm tests
- **solvers**: High-level solver tests
- **slow**: Performance-intensive tests
- **integration**: Cross-module integration tests

## Documentation Structure (`docs/`)

```
docs/
├── conf.py                     # Sphinx configuration
├── build_docs.py              # Documentation build script
└── _build/                     # Generated documentation output
```

## Examples Structure (`examples/`)

**Demonstration Scripts**:
- `sphere_optimization_demo.py`: Basic sphere optimization
- `grassmann_optimization_demo.py`: Subspace fitting examples
- `stiefel_optimization_demo.py`: Orthogonal Procrustes problems
- `manifolds_comparison_demo.py`: Comparative manifold analysis
- `optimizer_comparison_demo.py`: Algorithm performance comparison
- `ml_applications_showcase.py`: Machine learning applications

**Specialized Examples**:
- `spd_covariance_estimation.py`: Covariance matrix optimization
- `so3_optimization_demo.py`: 3D rotation optimization
- `dynamic_dimensions_usage.py`: Variable dimension handling

## Code Organization Patterns

### Functional Programming Style
- **Immutable Data Structures**: JAX PyTree-based data handling
- **Pure Functions**: Side-effect-free mathematical operations
- **Function Composition**: Modular, composable algorithm components
- **Vectorization**: JAX vmap for batch processing

### Object-Oriented Design
- **Abstract Base Classes**: Consistent manifold interface via `BaseManifold`
- **Factory Pattern**: Centralized manifold creation in `factory.py`
- **Strategy Pattern**: Algorithm selection in `spd_algorithm_selector.py`
- **Data Classes**: Structured types in `data_models.py`

### Performance Architecture
- **JIT Compilation**: Selective JIT application via decorators
- **Batch Operations**: Vectorized operations in `batch_ops.py`
- **Memory Optimization**: In-place operations where mathematically valid
- **Hardware Abstraction**: Device-agnostic code with `device_manager.py`

## File Naming Conventions

### Module Files
- **Snake Case**: `manifold_name.py` (e.g., `poincare_ball.py`)
- **Descriptive Names**: Clear indication of functionality
- **Consistent Patterns**: `test_module_name.py` for tests

### Class Names
- **PascalCase**: `ClassName` (e.g., `BaseManifold`, `SphereManifold`)
- **Descriptive Suffixes**: `Manifold`, `Optimizer`, `Problem` suffixes
- **Domain-Specific**: Mathematical names (e.g., `GrassmannManifold`)

### Function Names
- **Snake Case**: `function_name` (e.g., `exponential_map`)
- **Mathematical Terms**: Standard differential geometry terminology
- **Action-Oriented**: Verb-based naming (e.g., `project_to_tangent`)

## Import Organization

### Import Hierarchy
```python
# Standard library
import functools
from typing import Optional, Tuple

# Third-party (JAX ecosystem)
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Local imports (relative)
from .base import BaseManifold
from ..core.numerical_stability import safe_divide
```

### Known Third-Party Libraries
- **JAX Ecosystem**: `jax`, `jax.numpy`, `jaxtyping`, `optax`, `equinox`
- **Scientific**: `numpy`, `scipy` (limited use)
- **Visualization**: `matplotlib`, `seaborn` (examples only)

## Key Architectural Principles

### Mathematical Rigor
- **Theoretically Grounded**: Proper differential geometry implementations
- **Numerical Stability**: Robust handling of edge cases and floating-point issues
- **Validation**: Comprehensive property testing and mathematical verification

### Performance Focus
- **JAX-Native**: Built specifically for JAX ecosystem optimization
- **Hardware Agnostic**: CPU/GPU/TPU compatibility through JAX
- **Scalable**: Linear scaling with problem size and batch dimensions

### Developer Experience
- **Type Safety**: Full type annotations with runtime checking
- **Clear Interfaces**: Consistent API across manifold implementations
- **Comprehensive Testing**: Property-based and example-driven test coverage
- **Documentation**: Extensive docstrings and usage examples

### Modularity and Extensibility
- **Plugin Architecture**: Easy addition of new manifolds and optimizers
- **Separation of Concerns**: Clear boundaries between mathematical and computational aspects
- **Composability**: Building complex manifolds from simple components
