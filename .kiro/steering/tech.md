# Technology Stack

## Architecture

RiemannAX follows a modular, functional programming architecture built on JAX's ecosystem:

- **Core Framework**: JAX for automatic differentiation, JIT compilation, and hardware acceleration
- **Mathematical Foundation**: Differential geometry and Riemannian optimization
- **Design Pattern**: Functional programming with immutable data structures
- **Performance Strategy**: JIT compilation, vectorization, and hardware acceleration (GPU/TPU)

## Core Dependencies

### Runtime Dependencies
- **JAX** (`>=0.4.0`): Core framework for automatic differentiation and hardware acceleration
- **JAXtyping** (`>=0.2.0`): Type safety and runtime type checking for JAX arrays
- **NumPy** (`>=1.20.0`): Numerical computing foundation
- **Optax** (`>=0.1.0`): Gradient processing and optimization algorithms
- **Optimistix** (`>=0.0.5`): Advanced optimization solvers and root finding
- **Equinox** (`>=0.11.0`): Neural network and PyTree utilities for JAX
- **Matplotlib** (`>=3.10.3`): Plotting and visualization

### Development Dependencies
- **Ruff** (`>=0.9.0`): Fast Python linter and formatter
- **Pytest** (`>=7.0.0`): Testing framework with parallel execution support
- **Pytest-xdist** (`>=3.0.0`): Parallel test execution
- **Pytest-timeout** (`>=2.1.0`): Test timeout management
- **Hypothesis** (`>=6.0.0`): Property-based testing
- **Black** (`==23.7.0`): Code formatting
- **MyPy** (`>=1.8.0`): Static type checking

### Documentation Dependencies
- **Sphinx** (`>=7.2.0`): Documentation generation
- **Sphinx-RTD-Theme** (`>=2.0.0`): Read the Docs theme
- **MyST-Parser** (`>=2.0.0`): Markdown support for Sphinx
- **NBSphinx** (`>=0.9.0`): Jupyter notebook integration

### Example Dependencies
- **Pandas** (`>=1.3.0`): Data manipulation and analysis
- **Seaborn** (`>=0.13.2`): Statistical visualization
- **Jupyter** (`>=1.0.0`): Interactive notebook environment

## Development Environment

### Python Version Requirements
- **Minimum**: Python 3.10
- **Supported**: Python 3.10, 3.11, 3.12
- **Recommended**: Python 3.11+ for best performance

### Package Management
- **Build System**: Setuptools (`>=61.0`) with wheel support
- **Package Discovery**: Automatic package finding from root directory
- **Installation Methods**:
  - Standard: `pip install riemannax`
  - Development: `pip install -e ".[dev]"`
  - With UV: `uv pip install -e .`

## Common Commands

### Development Workflow
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run full development workflow
make all                    # lint + test

# Testing
make test                   # Run all tests
python -m pytest          # Direct pytest execution
make coverage              # Run tests with coverage reporting

# Code Quality
make lint                  # Run linting (flake8, isort, black)
make format                # Format code (isort + black)
ruff check . --fix         # Modern linting with auto-fix
mypy riemannax            # Type checking

# Cleanup
make clean                # Remove build artifacts and cache
```

### Advanced Testing
```bash
# Parallel testing
python -m pytest -n auto

# Specific test categories
python -m pytest -m manifolds     # Manifold-specific tests
python -m pytest -m optimizers    # Optimizer tests
python -m pytest -m slow         # Include slow tests
python -m pytest -m "not slow"   # Exclude slow tests

# Property-based testing
python -m pytest --hypothesis-show-statistics

# Timeout management
python -m pytest --timeout=300   # 5-minute timeout per test
```

### Documentation
```bash
# Build documentation
cd docs && python build_docs.py
sphinx-build -b html docs docs/_build/html

# Documentation dependencies
pip install -e ".[docs]"
```

### Example Execution
```bash
# Install example dependencies
pip install -e ".[examples]"

# Run examples
python examples/sphere_optimization_demo.py
python examples/grassmann_optimization_demo.py
python examples/manifolds_comparison_demo.py
```

## Configuration Files

### Code Quality Configuration
- **pyproject.toml**:
  - Ruff configuration (line length: 120, target Python 3.10+)
  - MyPy settings (strict type checking, ignore test files)
  - Pytest configuration (markers, timeout, warnings)
  - Coverage reporting settings
- **Makefile**: Development workflow automation

### Ruff Configuration
- **Target Version**: Python 3.10+
- **Line Length**: 120 characters
- **Style Convention**: Google docstring style
- **Import Sorting**: isort integration with JAX ecosystem awareness
- **Exclusions**: Tests, examples, docs, build artifacts

### Testing Configuration
- **Test Discovery**: `test_*.py` files in `tests/` directory
- **Timeout**: 600 seconds default, signal-based timeout
- **Markers**: manifolds, optimizers, solvers, slow, integration
- **Coverage**: Source code only, excluding tests and dead code

## Hardware Acceleration

### JAX Backend Support
- **CPU**: Default backend with XLA optimization
- **GPU**: CUDA support for NVIDIA hardware
- **TPU**: Google TPU support for cloud deployment
- **Metal**: Apple Silicon GPU support (experimental)

### Performance Characteristics
- **JIT Compilation**: First-call overhead, subsequent near-C performance
- **Memory Management**: In-place operations where possible
- **Batch Processing**: Vectorized operations across problem instances
- **Scaling**: Linear performance scaling with batch size

## Environment Variables

### JAX Configuration
```bash
# Platform selection
export JAX_PLATFORM_NAME=gpu    # or cpu, tpu
export CUDA_VISIBLE_DEVICES=0   # GPU selection

# Memory management
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Debug settings
export JAX_DEBUG_NANS=true
export JAX_DISABLE_JIT=false
```

### Development Settings
```bash
# Testing
export PYTEST_TIMEOUT=600
export HYPOTHESIS_MAX_EXAMPLES=50

# Coverage reporting
export COVERAGE_CORE=sysmon
```

## Port Configuration

RiemannAX is primarily a computational library and does not require specific port configurations. However, for development:

- **Jupyter Server**: Default port 8888 for notebook examples
- **Sphinx Documentation**: Default port 8000 for local doc server
- **No Network Services**: Core library operates locally without network dependencies

## Build and Distribution

### Build Process
- **Build Backend**: Setuptools with automated package discovery
- **Version Management**: Centralized in pyproject.toml (current: 0.0.3)
- **License**: Apache-2.0
- **Distribution**: PyPI package with source distribution and wheels

### Continuous Integration
- **Testing**: Automated test suite across Python versions
- **Documentation**: Automated documentation building and deployment
- **Release**: Automated release workflow with version tagging

### Development Status
- **Maturity**: Pre-Alpha (Development Status :: 2)
- **Target Audience**: Science/Research community
- **License**: OSI Approved Apache Software License
