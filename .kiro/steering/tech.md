# Technology Stack

## Architecture

### **Core Architecture Pattern**
- **Functional Programming**: JAX-based functional approach with immutable data structures
- **Modular Design**: Separation of manifolds, optimizers, problems, and solvers
- **Mathematical Abstraction**: Clean separation between geometric operations and optimization algorithms
- **Hardware Abstraction**: Transparent GPU/TPU acceleration through JAX's device abstraction

### **Design Principles**
- **Composability**: Mix and match manifolds, cost functions, and optimization methods
- **Performance First**: JIT compilation and vectorization as primary considerations
- **Type Safety**: Comprehensive type annotations for compile-time error prevention
- **Mathematical Correctness**: Rigorous implementation of differential geometric operations

## Core Dependencies

### **Runtime Dependencies**
```toml
[dependencies]
jax = ">=0.4.0"           # Core JAX framework for autodiff and JIT
numpy = ">=1.20.0"        # Numerical computing foundation
optax = ">=0.1.0"         # Optimization algorithms and utilities
matplotlib = ">=3.10.3"   # Visualization and plotting support
```

### **Development Dependencies**
```toml
[dev-dependencies]
ruff = ">=0.9.0"          # Modern Python linter and formatter
pytest = ">=7.0.0"        # Testing framework with fixtures
black = "==23.7.0"        # Code formatter (pinned version)
mypy = ">=1.8.0"          # Static type checker
types-requests = "*"      # Type stubs for requests
types-setuptools = "*"    # Type stubs for setuptools
```

### **Documentation Dependencies**
```toml
[docs-dependencies]
sphinx = ">=7.2.0"                    # Documentation generator
sphinx-rtd-theme = ">=2.0.0"          # ReadTheDocs theme
sphinx-autodoc-typehints = ">=1.25.0" # Type hint documentation
myst-parser = ">=2.0.0"               # Markdown parser for Sphinx
nbsphinx = ">=0.9.0"                  # Jupyter notebook integration
```

## Language and Runtime

### **Python Configuration**
- **Version Requirement**: Python 3.10+ (uses modern type syntax)
- **Platform Support**: Cross-platform (macOS, Linux, Windows)
- **Virtual Environment**: Standard venv or uv for dependency isolation

### **Type System**
- **Type Checking**: MyPy with strict configuration
- **Type Coverage**: 100% type annotation coverage for public APIs
- **Type Safety**: Strict optional, warn on return any, check untyped defs

## Development Environment

### **Code Quality Tools**

#### **Ruff Configuration**
```toml
[tool.ruff]
target-version = "py310"
line-length = 120
select = ["E", "F", "W", "I", "N", "D", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["N803", "N806", "D203", "D212", "E501", "E741"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

#### **Pre-commit Configuration**
- **Basic Hooks**: Trailing whitespace, YAML/TOML validation, large file detection
- **Ruff Integration**: Automatic linting and fixing with non-zero exit on changes
- **MyPy Integration**: Type checking with comprehensive additional dependencies
- **Notebook Cleanup**: Strip output from Jupyter notebooks before commit

### **Testing Framework**

#### **pytest Configuration**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    manifolds: marks tests related to manifolds
    optimizers: marks tests related to optimizers
    solvers: marks tests related to solvers

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

#### **Test Categories**
- **Unit Tests**: Individual component testing with mathematical validation (49+ test files)
- **Integration Tests**: End-to-end workflow testing across manifolds
- **JIT Tests**: JIT compilation and optimization validation (per-manifold `*_jit.py` tests)
- **Performance Tests**: Benchmarking and speedup validation with comprehensive performance monitoring
- **Core Module Tests**: JIT manager, JIT decorator, device manager, batch operations, performance monitoring, type system, and constants
- **Numerical Stability Tests**: Comprehensive numerical stability and convergence verification
- **Property-Based Tests**: Mathematical property validation across manifold implementations
- **Compatibility Tests**: Cross-platform, dependency compatibility, and JIT compatibility validation

## Common Commands

### **Development Workflow**
```bash
# Environment Setup
pip install -e ".[dev]"              # Install with development dependencies
uv venv && source .venv/bin/activate # Alternative: UV package manager
pre-commit install                   # Enable git hooks

# Code Quality
make lint                           # Run linting (flake8, isort, black)
ruff check riemannax               # Modern linting with Ruff
ruff check --fix riemannax         # Auto-fix linting issues
make format                        # Format code (isort + black)

# Type Checking
mypy riemannax                     # Static type analysis
```

### **Testing Commands**
```bash
# Basic Testing
make test                          # Quick test suite
python -m pytest                  # Direct pytest execution

# Advanced Testing
make coverage                      # Coverage analysis with HTML report
pytest tests/manifolds/           # Test specific modules
pytest -m "not slow"              # Skip slow tests
pytest -m integration             # Integration tests only
```

### **Documentation**
```bash
# Documentation Generation
pip install -e ".[docs]"          # Install docs dependencies
make docs                          # Build Sphinx documentation
```

### **Cleanup**
```bash
make clean                         # Remove build artifacts and cache
```

## Hardware and Performance

### **JAX Hardware Support**
- **CPU**: XLA-optimized execution with 2-5x speedup
- **GPU**: CUDA support with 10-100x speedup for large problems
- **TPU**: Cloud TPU support for research-scale computations
- **Memory**: In-place operations and optimized memory layouts

### **Performance Characteristics**
- **JIT Compilation**: First-call compilation overhead, subsequent near-C performance
- **Batch Processing**: Linear scaling with batch size for parallel optimization
- **Memory Efficiency**: Functional programming with optimized memory usage
- **Numerical Stability**: IEEE-754 compliant with robust edge case handling

## Configuration Management

### **Environment Variables**
```bash
# JAX Configuration
JAX_PLATFORM_NAME=cpu|gpu|tpu      # Force platform selection
JAX_ENABLE_X64=True                # Enable 64-bit precision
JAX_DISABLE_JIT=True               # Disable JIT for debugging

# Development Configuration
PYTEST_ADDOPTS=-v                  # Verbose test output
MYPY_CACHE_DIR=.mypy_cache         # MyPy cache location
```

### **Project Configuration Files**
- **pyproject.toml**: Primary project configuration (dependencies, tools, metadata)
- **pytest.ini**: Test configuration and markers
- **Makefile**: Development command shortcuts
- **.pre-commit-config.yaml**: Git hooks configuration
- **.gitignore**: Version control exclusions
