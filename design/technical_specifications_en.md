# RiemannAX Technical Specifications

## Overview

This document defines the detailed technical implementation specifications for future RiemannAX expansions. It outlines design principles, API specifications, and implementation guidelines for each module.

## Manifold Implementation Specifications

### Base Class Extensions

Extend the current `Manifold` base class with the following functionality:

```python
class Manifold:
    # Existing methods
    def proj(self, x, v): ...
    def exp(self, x, v): ...
    def log(self, x, y): ...

    # New methods to be added
    def vector_transp(self, x, y, v):
        """Vector transport (default implementation)"""

    def curvature_tensor(self, x, u, v):
        """Curvature tensor computation"""

    def christoffel_symbols(self, x):
        """Christoffel symbols computation"""

    def sectional_curvature(self, x, u, v):
        """Sectional curvature computation"""

    def injectivity_radius(self, x):
        """Injectivity radius computation"""
```

### New Manifold Implementations

#### 1. Grassmann Manifold (Gr(p,n))

```python
class Grassmann(Manifold):
    """Grassmann manifold Gr(p,n) - Collection of p-dimensional subspaces in n-dimensional space"""

    def __init__(self, n: int, p: int):
        """
        Args:
            n: Ambient space dimension
            p: Subspace dimension (p < n)
        """
        self.n = n
        self.p = p
        self.shape = (n, p)

    def proj(self, x, v):
        """Projection to tangent space: T_X Gr(p,n) = {V : X^T V + V^T X = 0}"""
        return v - x @ (x.T @ v)

    def exp(self, x, v):
        """Exponential map (QR decomposition based)"""
        q, r = jnp.linalg.qr(jnp.concatenate([x, v], axis=1))
        return q[:, :self.p]

    def retr(self, x, v):
        """Retraction (QR decomposition normalization)"""
        y = x + v
        q, _ = jnp.linalg.qr(y)
        return q
```

#### 2. Stiefel Manifold (St(p,n))

```python
class Stiefel(Manifold):
    """Stiefel manifold St(p,n) - Orthonormal p-frames in n-dimensional space"""

    def __init__(self, n: int, p: int):
        self.n = n
        self.p = p
        self.shape = (n, p)

    def proj(self, x, v):
        """Projection to tangent space: T_X St(p,n) = {V : X^T V + V^T X = 0}"""
        return v - x @ (x.T @ v + v.T @ x) / 2

    def exp(self, x, v):
        """Exponential map (polar decomposition based)"""
        u, s, vt = jnp.linalg.svd(v, full_matrices=False)
        return x @ jnp.diag(jnp.cos(s)) @ vt + u @ jnp.diag(jnp.sin(s)) @ vt
```

#### 3. Hyperbolic Space

```python
class Hyperboloid(Manifold):
    """Hyperbolic space (hyperboloid model)"""

    def __init__(self, dim: int):
        self.dim = dim
        self.shape = (dim + 1,)

    def minkowski_inner(self, u, v):
        """Minkowski inner product"""
        return -u[0] * v[0] + jnp.sum(u[1:] * v[1:])

    def proj(self, x, v):
        """Projection to tangent space"""
        return v + self.minkowski_inner(x, v) * x

    def exp(self, x, v):
        """Exponential map"""
        norm_v = jnp.sqrt(self.minkowski_inner(v, v))
        return jnp.cosh(norm_v) * x + jnp.sinh(norm_v) * v / norm_v
```

## Optimization Algorithm Specifications

### Riemannian Conjugate Gradient (RCG)

```python
def riemannian_conjugate_gradient(
    beta_type: str = "polak_ribiere",
    restart_threshold: float = 0.1,
    max_inner_iterations: int = None
):
    """
    Riemannian Conjugate Gradient Method

    Args:
        beta_type: Beta computation method ('fletcher_reeves', 'polak_ribiere', 'hestenes_stiefel')
        restart_threshold: Restart threshold
        max_inner_iterations: Maximum inner iterations
    """

    def init_fn(x0):
        return RCGState(
            x=x0,
            grad=None,
            direction=None,
            iteration=0
        )

    def update_fn(gradient, state, manifold):
        # Beta coefficient computation
        if state.grad is not None:
            old_grad_transported = manifold.transp(state.x, new_x, state.grad)

            if beta_type == "fletcher_reeves":
                beta = manifold.inner(new_x, gradient, gradient) / \
                       manifold.inner(state.x, state.grad, state.grad)
            elif beta_type == "polak_ribiere":
                grad_diff = gradient - old_grad_transported
                beta = manifold.inner(new_x, gradient, grad_diff) / \
                       manifold.inner(state.x, state.grad, state.grad)
            # ... other beta computations

        # Conjugate direction update
        if state.direction is None or beta < restart_threshold:
            direction = -gradient  # Restart with gradient direction
        else:
            old_direction_transported = manifold.transp(state.x, new_x, state.direction)
            direction = -gradient + beta * old_direction_transported
            direction = manifold.proj(new_x, direction)  # Project to tangent space

        return RCGState(
            x=new_x,
            grad=gradient,
            direction=direction,
            iteration=state.iteration + 1
        )

    return init_fn, update_fn
```

### Riemannian L-BFGS

```python
def riemannian_lbfgs(
    memory_size: int = 10,
    damping: bool = True,
    scaling: bool = True
):
    """
    Riemannian L-BFGS Method

    Args:
        memory_size: History storage size
        damping: Enable damping
        scaling: Enable scaling
    """

    def init_fn(x0):
        return LBFGSState(
            x=x0,
            grad=None,
            s_history=[],  # Step history
            y_history=[],  # Gradient difference history
            rho_history=[], # 1/(s^T y) history
            iteration=0
        )

    def update_fn(gradient, state, manifold):
        # Compute Hessian inverse approximation using L-BFGS two-loop recursion
        direction = lbfgs_two_loop_recursion(
            gradient,
            state.s_history,
            state.y_history,
            state.rho_history,
            manifold
        )

        # Update history (using transport)
        if len(state.s_history) >= memory_size:
            # Remove oldest history
            state.s_history.pop(0)
            state.y_history.pop(0)
            state.rho_history.pop(0)

        return new_state

    return init_fn, update_fn
```

## Solver Extension Specifications

### Convergence Criteria

```python
class ConvergenceCriterion:
    """Base class for convergence criteria"""

    def check(self, state: OptState, manifold: Manifold) -> bool:
        raise NotImplementedError

class GradientNormCriterion(ConvergenceCriterion):
    """Convergence based on gradient norm"""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def check(self, state: OptState, manifold: Manifold) -> bool:
        grad_norm = jnp.sqrt(manifold.inner(state.x, state.grad, state.grad))
        return grad_norm < self.tolerance

class RelativeFunctionChangeCriterion(ConvergenceCriterion):
    """Convergence based on relative function value change"""

    def __init__(self, tolerance: float = 1e-8, window: int = 3):
        self.tolerance = tolerance
        self.window = window
        self.history = []

    def check(self, state: OptState, manifold: Manifold) -> bool:
        self.history.append(state.fun)
        if len(self.history) < self.window:
            return False

        relative_change = abs(self.history[-1] - self.history[-2]) / \
                         (abs(self.history[-2]) + 1e-12)
        return relative_change < self.tolerance
```

### Line Search

```python
class LineSearch:
    """Base class for line search"""

    def search(self,
               manifold: Manifold,
               problem: RiemannianProblem,
               x: jnp.ndarray,
               direction: jnp.ndarray,
               initial_step: float = 1.0) -> tuple[float, jnp.ndarray, float]:
        raise NotImplementedError

class ArmijoLineSearch(LineSearch):
    """Armijo condition line search"""

    def __init__(self,
                 c1: float = 1e-4,
                 max_iterations: int = 20,
                 backtrack_factor: float = 0.5):
        self.c1 = c1
        self.max_iterations = max_iterations
        self.backtrack_factor = backtrack_factor

    def search(self, manifold, problem, x, direction, initial_step=1.0):
        step = initial_step
        fx = problem.cost(x)
        grad = problem.grad(x)
        directional_derivative = manifold.inner(x, grad, direction)

        for _ in range(self.max_iterations):
            # Compute candidate point
            x_new = manifold.retr(x, step * direction)
            fx_new = problem.cost(x_new)

            # Check Armijo condition
            if fx_new <= fx + self.c1 * step * directional_derivative:
                return step, x_new, fx_new

            step *= self.backtrack_factor

        return step, x_new, fx_new
```

## Visualization Tool Specifications

### Optimization Trajectory Visualization

```python
class OptimizationVisualizer:
    """Optimization trajectory visualization class"""

    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        self.trajectory = []
        self.costs = []

    def update(self, x: jnp.ndarray, cost: float):
        """Record optimization step"""
        self.trajectory.append(x)
        self.costs.append(cost)

    def plot_trajectory_3d(self, fig=None, ax=None):
        """Plot 3D trajectory (for sphere/3D manifolds)"""
        if isinstance(self.manifold, Sphere) and self.manifold.shape[0] == 3:
            return self._plot_sphere_trajectory(fig, ax)
        else:
            raise NotImplementedError("3D visualization not supported for this manifold")

    def plot_convergence(self):
        """Plot convergence graphs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Cost function value evolution
        ax1.plot(self.costs)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost Function Value')
        ax1.set_title('Cost Function Convergence')
        ax1.grid(True)

        # Gradient norm evolution
        grad_norms = [self._compute_grad_norm(x) for x in self.trajectory]
        ax2.plot(grad_norms)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm Convergence')
        ax2.set_yscale('log')
        ax2.grid(True)

        return fig
```

## Performance Optimization

### JIT Optimization

```python
# JIT-optimized core functions
@jit
def manifold_optimization_step(manifold_ops, problem_ops, state, step_size):
    """JIT-optimized single optimization step"""
    x = state.x
    grad = problem_ops['grad'](x)
    direction = -grad
    new_x = manifold_ops['retr'](x, step_size * direction)
    new_cost = problem_ops['cost'](new_x)

    return state._replace(x=new_x, cost=new_cost, grad=grad)

# Batched operations
@jit
def batch_manifold_operations(manifold_ops, points, vectors):
    """Batch processing of manifold operations on multiple points"""
    return vmap(manifold_ops['exp'])(points, vectors)
```

### Memory Efficiency

```python
class ChunkedOptimizer:
    """Chunked optimizer for large-scale problems"""

    def __init__(self,
                 chunk_size: int = 1000,
                 gradient_accumulation: bool = True):
        self.chunk_size = chunk_size
        self.gradient_accumulation = gradient_accumulation

    def optimize_chunked(self,
                        problem: RiemannianProblem,
                        data: jnp.ndarray,
                        x0: jnp.ndarray,
                        optimizer,
                        num_epochs: int):
        """Chunked optimization"""
        state = optimizer.init(x0)

        for epoch in range(num_epochs):
            # Split data into chunks
            chunks = self._create_chunks(data)
            accumulated_grad = jnp.zeros_like(x0)

            for chunk in chunks:
                # Gradient computation per chunk
                chunk_grad = problem.grad_chunk(state.x, chunk)
                accumulated_grad += chunk_grad

            # Average gradients and update
            avg_grad = accumulated_grad / len(chunks)
            state = optimizer.update(avg_grad, state, problem.manifold)

        return state
```

## API Design Standards

### Consistent Naming Conventions

- **Manifolds**: PascalCase class names (e.g., `Grassmann`, `Stiefel`)
- **Methods**: snake_case for all functions and methods
- **Parameters**: snake_case with descriptive names
- **Constants**: UPPER_CASE for mathematical constants

### Error Handling

```python
class RiemannAXError(Exception):
    """Base exception for RiemannAX"""
    pass

class ManifoldError(RiemannAXError):
    """Exception for manifold-related errors"""
    pass

class ConvergenceError(RiemannAXError):
    """Exception for convergence-related issues"""
    pass

class DimensionError(RiemannAXError):
    """Exception for dimension mismatches"""
    pass
```

### Type Annotations

All new code must include comprehensive type annotations:

```python
from typing import Callable, Optional, Tuple, Union
import jax.numpy as jnp

def optimize_function(
    manifold: Manifold,
    cost_fn: Callable[[jnp.ndarray], float],
    x0: jnp.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-6
) -> OptimizeResult:
    """Type-annotated optimization function"""
    pass
```

This technical specification ensures consistent, high-quality implementation across all future RiemannAX development.
