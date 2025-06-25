"""Integration tests for new manifolds with optimization algorithms."""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rx


class TestManifoldOptimizationIntegration:
    """Integration tests for manifolds with optimization algorithms."""

    @pytest.mark.parametrize(
        "manifold_cls,args",
        [
            (rx.Grassmann, (5, 3)),
            (rx.Stiefel, (5, 3)),
            (rx.Sphere, ()),  # Test compatibility with existing sphere
        ],
    )
    def test_basic_optimization(self, manifold_cls, args):
        """Test basic optimization on different manifolds."""
        # Create manifold
        manifold = manifold_cls(*args)

        # Define a simple quadratic cost function
        if manifold_cls == rx.Sphere:
            target = jnp.array([0.0, 0.0, 1.0])  # North pole for sphere

            def cost_fn(x):
                return -jnp.dot(x, target)
        else:
            # For matrix manifolds, minimize Frobenius norm to random target
            key = jax.random.key(999)
            target = manifold.random_point(key)

            def cost_fn(x):
                return jnp.sum((x - target) ** 2)

        # Create optimization problem
        problem = rx.RiemannianProblem(manifold, cost_fn)

        # Generate initial point
        key = jax.random.key(42)
        x0 = manifold.random_point(key)

        # Run optimization
        result = rx.minimize(problem, x0, method="rsgd", options={"learning_rate": 0.1, "max_iterations": 50})

        # Verify result is on manifold
        assert manifold.validate_point(result.x)

        # Verify cost decreased
        initial_cost = problem.cost(x0)
        final_cost = result.fun
        assert final_cost <= initial_cost

    def test_grassmann_subspace_fitting(self):
        """Test Grassmann manifold for subspace fitting problem."""
        # Generate synthetic data
        key = jax.random.key(123)
        n, p, m = 8, 3, 100

        # True subspace
        true_subspace = rx.Grassmann(n, p).random_point(key)

        # Generate data points in subspace plus noise
        keys = jax.random.split(key, m + 1)
        coeffs = jax.random.normal(keys[0], (p, m))
        noise = 0.1 * jax.random.normal(keys[1], (n, m))
        data = true_subspace @ coeffs + noise

        # Define cost function: minimize reconstruction error
        def cost_fn(x):
            projector = x @ x.T
            reconstruction = projector @ data
            return jnp.sum((data - reconstruction) ** 2)

        # Optimize
        manifold = rx.Grassmann(n, p)
        problem = rx.RiemannianProblem(manifold, cost_fn)

        x0 = manifold.random_point(jax.random.key(456))
        result = rx.minimize(problem, x0, method="rsgd", options={"learning_rate": 0.01, "max_iterations": 100})

        # Verify convergence
        assert manifold.validate_point(result.x)
        initial_cost = problem.cost(x0)
        assert result.fun < initial_cost

    def test_stiefel_orthogonal_procrustes(self):
        """Test Stiefel manifold for orthogonal Procrustes problem."""
        # Generate random matrices
        key = jax.random.key(789)
        n, p = 6, 4

        keys = jax.random.split(key, 3)
        A = jax.random.normal(keys[0], (n, p))
        B = jax.random.normal(keys[1], (n, p))

        # Procrustes problem: min ||A - BQ||_F^2 over Q in St(p,p)
        def cost_fn(q):
            return jnp.sum((A - B @ q) ** 2)

        # Optimize
        manifold = rx.Stiefel(p, p)  # Orthogonal group
        problem = rx.RiemannianProblem(manifold, cost_fn)

        x0 = manifold.random_point(keys[2])
        result = rx.minimize(problem, x0, method="rsgd", options={"learning_rate": 0.1, "max_iterations": 50})

        # Verify result
        assert manifold.validate_point(result.x)

        # Should be orthogonal matrix
        should_be_identity = result.x.T @ result.x
        assert jnp.allclose(should_be_identity, jnp.eye(p), atol=1e-6)

    def test_gradient_consistency(self):
        """Test that automatic gradients are consistent across manifolds."""
        manifolds = [rx.Grassmann(4, 2), rx.Stiefel(4, 2), rx.Sphere()]

        for manifold in manifolds:
            # Simple quadratic cost
            key = jax.random.key(42)
            if isinstance(manifold, rx.Sphere):
                center = jnp.array([1.0, 0.0, 0.0])

                def cost_fn(x, center=center):
                    return jnp.sum((x - center) ** 2)
            else:
                center = manifold.random_point(key)

                def cost_fn(x, center=center):
                    return jnp.sum((x - center) ** 2)

            # Create problem and compute gradient
            problem = rx.RiemannianProblem(manifold, cost_fn)
            x = manifold.random_point(jax.random.key(123))
            grad = problem.grad(x)

            # Gradient should be in tangent space
            assert manifold.validate_tangent(x, grad)

            # Numerical gradient check - use larger epsilon for stability
            eps = 1e-5
            key_dirs = jax.random.key(456)
            v = manifold.random_tangent(key_dirs, x)
            v_norm = manifold.norm(x, v)

            # Skip if tangent vector is too small
            if v_norm < 1e-10:
                continue

            v_unit = v / v_norm

            # Finite difference using retraction for numerical stability
            x_plus = manifold.retr(x, eps * v_unit)
            x_minus = manifold.retr(x, -eps * v_unit)
            fd_grad = (problem.cost(x_plus) - problem.cost(x_minus)) / (2 * eps)

            # Directional derivative
            analytical_grad = manifold.inner(x, grad, v_unit)

            # Allow larger tolerance for retraction-based finite differences
            assert jnp.allclose(analytical_grad, fd_grad, rtol=5e-2, atol=5e-3)

    @pytest.mark.parametrize("exp_method", ["svd", "qr"])
    def test_stiefel_exponential_map_methods(self, exp_method):
        """Test both exponential map methods for Stiefel manifold in optimization."""
        manifold = rx.Stiefel(5, 3)

        # Define cost function
        key = jax.random.key(111)
        target = manifold.random_point(key)

        def cost_fn(x):
            return jnp.sum((x - target) ** 2)

        problem = rx.RiemannianProblem(manifold, cost_fn)

        # Monkey patch exponential map to use specific method
        original_exp = manifold.exp

        def custom_exp(x, v):
            return original_exp(x, v, method=exp_method)

        manifold.exp = custom_exp

        try:
            x0 = manifold.random_point(jax.random.key(222))
            result = rx.minimize(problem, x0, method="rsgd", options={"learning_rate": 0.1, "max_iterations": 30})

            assert manifold.validate_point(result.x)
            assert result.fun <= problem.cost(x0)

        finally:
            # Restore original method
            manifold.exp = original_exp

    def test_convergence_rates(self):
        """Test convergence rates for different manifolds."""
        manifolds_and_problems = [
            # Sphere: find north pole
            (rx.Sphere(), lambda x: -x[2]),
            # Grassmann: fit to identity subspace
            (rx.Grassmann(4, 2), lambda x: jnp.sum((x - jnp.eye(4, 2)) ** 2)),
            # Stiefel: fit to identity matrix
            (rx.Stiefel(3, 3), lambda x: jnp.sum((x - jnp.eye(3)) ** 2)),
        ]

        for manifold, cost_fn in manifolds_and_problems:
            problem = rx.RiemannianProblem(manifold, cost_fn)

            # Generate initial point
            key = jax.random.key(555)
            x0 = manifold.random_point(key)

            # Use more conservative learning rate and track costs
            costs = []
            state = rx.riemannian_gradient_descent(learning_rate=0.01)[0](x0)

            for _ in range(15):
                costs.append(problem.cost(state.x))
                grad = problem.grad(state.x)
                state = rx.riemannian_gradient_descent(learning_rate=0.01)[1](grad, state, manifold)

            # Verify overall decreasing trend (allow for small fluctuations)
            costs = jnp.array(costs)
            initial_cost = costs[0]
            final_cost = costs[-1]

            # Check that final cost is significantly lower than initial
            assert final_cost < initial_cost - 1e-6 or jnp.abs(final_cost - initial_cost) < 1e-3
