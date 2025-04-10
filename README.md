# RiemannAX

Hardware-accelerated Riemannian Manifold Optimization with JAX

## Overview

RiemannAX is a library for optimization on Riemannian manifolds using JAX.<br>
It provides implementations of various Riemannian manifolds and optimization algorithms<br>
that leverage JAX's automatic differentiation and GPU acceleration capabilities.

## Features

- **Manifold implementations** with differential geometric operations:
  - Sphere (S^n)
  - Special Orthogonal Group (SO(n))
  - More manifolds coming soon!

- **Optimization algorithms**:
  - Riemannian Gradient Descent (RGD)
  - More algorithms coming soon!

- **Problem solving framework**:
  - Automatic differentiation for computing Riemannian gradients
  - Support for custom gradient functions
  - Flexible solver configuration

## Installation

```bash
pip install riemannax
```

For development installation:

```bash
git clone https://github.com/riemannax/riemannax.git
cd riemannax
pip install -e ".[dev]"
```

## Usage Example

```python
import jax
import jax.numpy as jnp
import riemannax as rx

# 1. Define a manifold
sphere = rx.Sphere()

# 2. Define an optimization problem
def cost_fn(x):
    target = jnp.array([0., 0., 1.])  # North pole
    return -jnp.dot(x, target)  # Find the point closest to the north pole

problem = rx.RiemannianProblem(sphere, cost_fn)

# 3. Set the initial point
key = jax.random.PRNGKey(0)
x0 = sphere.random_point(key)

# 4. Solve the problem
result = rx.minimize(
    problem,
    x0,
    method='rsgd',
    options={'learning_rate': 0.1, 'max_iterations': 100}
)

print("Optimal point:", result.x)
print("Cost function value:", result.fun)
```

## Testing

To run the test suite:

```bash
make test
```

Or with coverage:

```bash
make coverage
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgements

RiemannAX is inspired by the design principles of JAX and Optax, as well as other Riemannian
optimization libraries like Pymanopt and Geoopt.
