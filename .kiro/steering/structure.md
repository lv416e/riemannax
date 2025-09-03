# Project Structure

## Root Directory Organization

```
riemannax/
├── riemannax/              # Main package source code
├── tests/                  # Comprehensive test suite
├── examples/               # Demo scripts and usage examples
├── docs/                   # Sphinx documentation
├── design/                 # Design documents and roadmaps
├── .github/                # GitHub workflows and templates
├── .kiro/                  # Spec-driven development steering
├── .claude/                # Claude Code configuration
└── [config files]         # Project configuration files
```

### **Configuration Files**
- **pyproject.toml**: Primary project configuration (PEP 621 standard)
- **pytest.ini**: Test framework configuration and markers
- **Makefile**: Development command shortcuts
- **.pre-commit-config.yaml**: Git hooks for code quality
- **.gitignore**: Version control exclusions
- **CLAUDE.md**: Claude Code development workflow instructions

## Core Package Structure (`riemannax/`)

```
riemannax/
├── __init__.py             # Package exports and version
├── core/                   # JIT optimization and performance management
│   ├── __init__.py         # Core module exports
│   ├── jit_manager.py      # JIT compilation management and caching
│   ├── jit_decorator.py    # JIT decorator with LRU caching optimization
│   ├── batch_ops.py        # Batch operation optimization
│   ├── safe_jit.py         # Safe JIT wrapper with error handling
│   ├── device_manager.py   # Hardware device management
│   ├── performance.py      # Performance monitoring utilities
│   ├── performance_benchmark.py # Performance benchmarking tools
│   ├── static_args_optimizer.py # Static argument optimization
│   ├── type_system.py      # Type system definitions and validation
│   └── constants.py        # Numerical constants and precision settings
├── manifolds/              # Riemannian manifold implementations
│   ├── __init__.py         # Manifold exports
│   ├── base.py             # Abstract base classes and interfaces
│   ├── sphere.py           # Unit sphere (S^n) manifold
│   ├── grassmann.py        # Grassmann manifold (Gr(p,n))
│   ├── stiefel.py          # Stiefel manifold (St(p,n))
│   ├── so.py               # Special orthogonal group SO(n)
│   └── spd.py              # Symmetric positive definite matrices
├── optimizers/             # Optimization algorithm implementations
│   ├── __init__.py         # Optimizer exports
│   ├── state.py            # Optimizer state management
│   ├── sgd.py              # Stochastic gradient descent
│   ├── adam.py             # Adam optimizer
│   └── momentum.py         # Momentum-based methods
├── problems/               # Problem definition utilities
│   ├── __init__.py         # Problem exports
│   └── base.py             # RiemannianProblem class
└── solvers/                # High-level solver interfaces
    ├── __init__.py         # Solver exports
    └── minimize.py         # Main minimization interface
```

## Test Structure (`tests/`)

```
tests/
├── conftest.py                            # Shared pytest fixtures
├── test_integration.py                    # End-to-end workflow tests
├── test_integration_manifolds.py          # Cross-manifold integration tests
├── test_jit_compatibility.py              # JIT system compatibility tests
├── test_end_to_end_jit.py                 # Full JIT integration tests
├── test_batch_optimization.py             # Batch operations tests
├── test_performance_benchmark.py          # Performance benchmarking tests
├── test_jit_testing_framework.py          # JIT testing framework tests
├── test_comprehensive_numerical_stability.py # Numerical stability tests
├── test_property_based_manifolds.py        # Property-based testing for manifolds
├── test_performance_validation.py          # Performance validation tests
├── test_manifold_integration.py            # Manifold integration tests
├── core/                                  # Core module tests
│   ├── test_jit_manager.py                # JIT manager tests
│   ├── test_jit_decorator.py              # JIT decorator with LRU cache tests
│   ├── test_device_manager.py             # Device manager tests
│   ├── test_safe_jit.py                   # Safe JIT wrapper tests
│   ├── test_batch_ops.py                  # Batch operations tests
│   ├── test_performance.py                # Performance monitoring tests
│   ├── test_performance_benchmark_functionality.py # Benchmark functionality tests
│   ├── test_performance_benchmarking.py    # Benchmarking tests
│   ├── test_jit_cache_functionality.py     # JIT cache tests
│   ├── test_static_args_optimization.py    # Static args optimizer tests
│   ├── test_type_system.py                # Type system validation tests
│   ├── test_constants.py                  # Numerical constants tests
│   ├── test_jit_decorator_application.py   # JIT decorator application tests
│   ├── test_base_manifold_jit_removal.py   # Base manifold JIT integration tests
│   ├── test_internationalization.py       # Internationalization support tests
│   └── test_manifold_documentation.py     # Manifold documentation tests
├── manifolds/                             # Manifold-specific tests
│   ├── test_sphere.py                     # Sphere manifold tests
│   ├── test_sphere_jit.py                 # Sphere JIT optimization tests
│   ├── test_grassmann.py                  # Grassmann manifold tests
│   ├── test_grassmann_jit.py              # Grassmann JIT optimization tests
│   ├── test_stiefel.py                    # Stiefel manifold tests
│   ├── test_stiefel_jit.py                # Stiefel JIT optimization tests
│   ├── test_so.py                         # SO(n) manifold tests
│   ├── test_so_jit.py                     # SO(n) JIT optimization tests
│   ├── test_spd.py                        # SPD manifold tests
│   ├── test_spd_jit.py                    # SPD JIT optimization tests
│   ├── test_base_jit.py                   # Base manifold JIT tests
│   └── test_manifold_factory.py           # Manifold factory function tests
├── optimizers/                            # Optimizer tests
│   ├── test_sgd.py                        # SGD tests
│   ├── test_adam.py                       # Adam tests
│   └── test_momentum.py                   # Momentum tests
├── problems/                              # Problem definition tests
│   └── test_base.py                       # RiemannianProblem tests
├── solvers/                               # Solver interface tests
│   └── test_minimize.py                   # Minimization interface tests
└── utils/                                 # Testing utilities
    ├── __init__.py                        # Utils exports
    ├── compatibility.py                   # Compatibility testing utilities
    └── jit_testing.py                     # JIT testing framework
```

### **Test Organization Patterns**
- **Mirror Structure**: Test directory structure mirrors source code organization
- **Comprehensive Coverage**: Each module has corresponding test file (49+ total test files)
- **JIT-Specific Testing**: Dedicated `*_jit.py` test files for JIT compilation validation per manifold
- **Integration Testing**: Cross-module interaction validation including JIT compatibility and end-to-end workflows
- **Performance Testing**: Dedicated benchmarking, performance validation, and monitoring tests
- **Property-Based Testing**: Mathematical property validation across manifold implementations
- **Numerical Stability**: Comprehensive numerical stability and convergence verification
- **Core Infrastructure**: Extensive testing of JIT management, device handling, and performance monitoring
- **Utilities Testing**: Shared testing infrastructure in `tests/utils/`
- **Fixture Sharing**: Common test utilities in `conftest.py`

## Examples Structure (`examples/`)

```
examples/
├── README.md                        # Examples overview and usage
├── notebooks/                       # Interactive Jupyter notebooks
├── output/                          # Generated visualizations and results
├── sphere_optimization_demo.py      # Sphere manifold optimization
├── grassmann_optimization_demo.py   # Subspace fitting examples
├── stiefel_optimization_demo.py     # Orthogonal Procrustes problems
├── so3_optimization_demo.py         # 3D rotation optimization
├── spd_covariance_estimation.py     # Covariance matrix estimation
├── manifolds_comparison_demo.py     # Comparative analysis
├── optimizer_comparison_demo.py     # Algorithm performance comparison
├── ml_applications_showcase.py      # Machine learning applications
└── dynamic_dimensions_usage.py      # Dynamic dimension handling demonstrations
```

### **Example Organization Patterns**
- **Manifold-Specific**: One demo per manifold type showing core capabilities
- **Comparative Analysis**: Cross-manifold and cross-optimizer comparisons
- **Application Focused**: Real-world use case demonstrations
- **Progressive Complexity**: From basic usage to advanced applications

## Documentation Structure (`docs/`)

```
docs/
├── conf.py                 # Sphinx configuration
├── build_docs.py          # Documentation build script
├── index.rst              # Main documentation index
├── installation.rst       # Installation instructions
├── usage.rst              # Usage guide and tutorials
├── api.rst                # API reference documentation
├── examples.rst           # Examples documentation
├── contributing.rst       # Contributor guidelines
├── strategic_roadmap.rst  # Development roadmap
├── _build/                # Generated documentation output
├── _static/               # Static assets (CSS, JS, images)
└── _templates/            # Custom Sphinx templates
```

## Code Organization Patterns

### **Manifold Implementation Pattern**
Each manifold follows a consistent structure:
```python
class ManifoldName(BaseManifold):
    def __init__(self, ...): pass           # Dimension and parameter setup
    def random_point(self, ...): pass       # Random point generation
    def inner(self, ...): pass              # Riemannian metric
    def proj(self, ...): pass               # Tangent space projection
    def exp(self, ...): pass                # Exponential map
    def log(self, ...): pass                # Logarithmic map
    def retr(self, ...): pass               # Retraction operation
    def transp(self, ...): pass             # Parallel transport
    def dist(self, ...): pass               # Geodesic distance
```

### **Optimizer Interface Pattern**
```python
class OptimizerName:
    def __init__(self, learning_rate, ...): pass
    def init_state(self, params): pass       # Initialize optimizer state
    def update(self, grads, state, params): pass  # Parameter update step
```

### **Problem Definition Pattern**
```python
problem = RiemannianProblem(
    manifold=manifold_instance,
    cost_fn=cost_function,
    euclidean_grad_fn=gradient_function  # Optional
)
```

### **Core Module Patterns**

#### **JIT Manager Usage**
```python
from riemannax.core import JITManager
# JITManager uses class methods for global configuration and caching
JITManager.configure(enable_jit=True, cache_size=10000, fallback_on_error=True)
optimized_fn = JITManager.jit_decorator(function, static_argnums=(...), device=None)
JITManager.clear_cache()  # Clear JIT cache when needed
```

#### **Device Management**
```python
from riemannax.core import DeviceManager
device_manager = DeviceManager()
device_manager.set_device('gpu')  # or 'cpu', 'tpu'
```

#### **Performance Monitoring**
```python
from riemannax.core import PerformanceMonitor
# PerformanceMonitor uses class methods for global monitoring
with PerformanceMonitor.measure('operation_name'):
    result = expensive_operation()

# Get performance metrics and reports
metrics = PerformanceMonitor.get_metrics()
report = PerformanceMonitor.get_speedup_report()
PerformanceMonitor.reset_metrics()  # Clear metrics when needed
```

## File Naming Conventions

### **Python Files**
- **Module Names**: Lowercase with underscores (e.g., `special_orthogonal.py` → `so.py` for brevity)
- **Class Names**: PascalCase matching mathematical concepts (e.g., `Grassmann`, `SpecialOrthogonal`)
- **Function Names**: Snake_case describing geometric operations (e.g., `exp_map`, `parallel_transport`)

### **Test Files**
- **Pattern**: `test_[module_name].py` (e.g., `test_sphere.py`)
- **Integration Tests**: `test_integration_[category].py`
- **Test Functions**: `test_[functionality]` (e.g., `test_exponential_map`)

### **Example Files**
- **Pattern**: `[manifold]_optimization_demo.py`
- **Comparison Files**: `[category]_comparison_demo.py`
- **Application Files**: `[application]_showcase.py`

## Import Organization

### **Internal Imports**
```python
# Standard library imports first
import math
from typing import Tuple, Optional

# Third-party imports second
import jax
import jax.numpy as jnp
import numpy as np

# Local imports last
from riemannax.manifolds.base import BaseManifold
from riemannax.problems.base import RiemannianProblem
```

### **Package Exports**
- **Flat API**: Main classes exported at package level for easy importing
- **Hierarchical Access**: Full module paths available for advanced usage
- **Version Exposure**: `__version__` available at package root

## Key Architectural Principles

### **Separation of Concerns**
- **Core**: JIT compilation, device management, performance monitoring, and optimization infrastructure
- **Manifolds**: Pure geometric operations, no optimization logic
- **Optimizers**: Generic optimization algorithms, manifold-agnostic
- **Problems**: Bridge between manifolds and cost functions
- **Solvers**: High-level interfaces combining all components

### **Functional Design**
- **Immutable Data**: JAX-style functional programming with immutable arrays
- **Pure Functions**: No side effects in core mathematical operations
- **Composability**: Easy combination of different components

### **Performance Considerations**
- **JIT Compilation**: All performance-critical functions decorated with `@jax.jit`
- **Vectorization**: Batch operations using `jax.vmap` where applicable
- **Memory Efficiency**: In-place operations where mathematically valid

### **Mathematical Rigor** 🚧 *Under Active Development*
- **Validation**: Input validation and constraint checking with manifold-specific verification
- **Numerical Stability**: Robust implementations targeting edge cases and numerical precision
- **Testing**: Comprehensive mathematical property verification across 49+ test files
- **Documentation**: Mathematical foundations and references with ongoing refinement
- **JIT Integration**: Focus on mathematically correct JIT optimization with proper static argument handling

### **Current Development Focus**
- **JAX JIT Optimization**: Comprehensive JIT compilation system with intelligent caching
- **Performance Monitoring**: Systematic performance measurement and benchmarking infrastructure
- **Type System**: Full type safety with modern Python 3.10+ type annotations
- **Numerical Constants**: Precision-controlled constants for stable geometric computations
