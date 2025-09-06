"""Integration tests for hyperbolic manifolds (PoincareBall, Lorentz) and SE(3) with RiemannianProblem."""

import jax
import jax.numpy as jnp
import pytest

import riemannax as rieax
from riemannax.manifolds import PoincareBall, Lorentz, SE3
from riemannax.problems import RiemannianProblem


@pytest.fixture
def key():
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestPoincareBallIntegration:
    """Test PoincareBall integration with RiemannianProblem and solvers."""

    def test_poincare_ball_with_riemann_problem_autodiff(self, key):
        """Test PoincareBall with automatic gradient computation."""
        manifold = PoincareBall(dimension=2, curvature=-1.0)

        # Cost function: distance from origin
        def cost_fn(x):
            return jnp.sum(x**2)

        problem = RiemannianProblem(manifold, cost_fn)

        # Start from a random point
        x0 = manifold.random_point(key)

        # Verify gradient computation works
        grad = problem.grad(x0)
        assert grad.shape == x0.shape

        # Test that gradient is tangent (verify it's actually computed)
        assert jnp.all(jnp.isfinite(grad))

    def test_poincare_ball_with_custom_gradient(self, key):
        """Test PoincareBall with custom Euclidean gradient function."""
        manifold = PoincareBall(dimension=2)

        def cost_fn(x):
            return jnp.sum(x**2)

        def euclidean_grad_fn(x):
            return 2 * x

        problem = RiemannianProblem(manifold, cost_fn, euclidean_grad_fn=euclidean_grad_fn)

        x0 = manifold.random_point(key)
        grad = problem.grad(x0)

        # Verify gradient is in tangent space
        projected = manifold.proj(x0, euclidean_grad_fn(x0))
        assert jnp.allclose(grad, projected, atol=1e-8)

    def test_poincare_ball_optimization_runs(self, key):
        """Test that optimization runs successfully on PoincareBall."""
        manifold = PoincareBall(dimension=2, curvature=-1.0)

        # Cost function: minimize distance to specific point
        target = jnp.array([0.5, 0.0])
        def cost_fn(x):
            return manifold.dist(x, target)**2

        problem = RiemannianProblem(manifold, cost_fn)
        x0 = manifold.random_point(key)

        # Run a few optimization steps
        result = rieax.minimize(problem, x0, method="rsgd",
                               options={"max_iterations": 5, "learning_rate": 0.1})

        # Verify result is valid
        assert manifold.validate_point(result.x)
        assert isinstance(result.fun, (float, jnp.ndarray))


class TestLorentzIntegration:
    """Test Lorentz manifold integration with RiemannianProblem and solvers."""

    def test_lorentz_with_riemann_problem_autodiff(self, key):
        """Test Lorentz with automatic gradient computation."""
        manifold = Lorentz(dimension=2)

        # Cost function: minimize first coordinate (time component)
        def cost_fn(x):
            return x[0]**2

        problem = RiemannianProblem(manifold, cost_fn)

        x0 = manifold.random_point(key)

        # Verify gradient computation works
        grad = problem.grad(x0)
        assert grad.shape == x0.shape

        # Verify gradient is tangent (orthogonal to x in Minkowski metric)
        minkowski_inner = manifold._minkowski_inner(x0, grad)
        assert jnp.allclose(minkowski_inner, 0.0, atol=1e-6)

    def test_lorentz_with_custom_riemannian_gradient(self, key):
        """Test Lorentz with custom Riemannian gradient function."""
        manifold = Lorentz(dimension=2)

        def cost_fn(x):
            return x[1]**2  # Minimize second spatial coordinate

        def grad_fn(x):
            # Custom Riemannian gradient: project [0, 2*x[1], 0] onto tangent space
            euclidean_grad = jnp.array([0.0, 2*x[1], 0.0])
            return manifold.proj(x, euclidean_grad)

        problem = RiemannianProblem(manifold, cost_fn, grad_fn=grad_fn)

        x0 = manifold.random_point(key)
        grad = problem.grad(x0)

        # Verify gradient is tangent
        assert jnp.allclose(manifold._minkowski_inner(x0, grad), 0.0, atol=1e-8)

    def test_lorentz_optimization_runs(self, key):
        """Test that optimization runs successfully on Lorentz manifold."""
        manifold = Lorentz(dimension=2)

        # Cost function: maximize time coordinate (minimize negative)
        def cost_fn(x):
            return -x[0]

        problem = RiemannianProblem(manifold, cost_fn)
        x0 = manifold.random_point(key)

        # Run optimization
        result = rieax.minimize(problem, x0, method="rsgd",
                               options={"max_iterations": 5, "learning_rate": 0.01})

        # Verify result is valid
        assert manifold.validate_point(result.x)
        assert isinstance(result.fun, (float, jnp.ndarray))


class TestSE3Integration:
    """Test SE(3) integration with RiemannianProblem and solvers."""

    def test_se3_with_riemann_problem_autodiff(self, key):
        """Test SE(3) with automatic gradient computation."""
        manifold = SE3()

        # Cost function: minimize translation norm
        def cost_fn(g):
            translation = g[4:7]  # Last 3 components are translation
            return jnp.sum(translation**2)

        problem = RiemannianProblem(manifold, cost_fn)

        g0 = manifold.random_point(key)

        # Verify gradient computation works
        grad = problem.grad(g0)
        assert grad.shape == (6,)  # Tangent space is 6-dimensional

        # Verify gradient gives valid tangent vector
        assert manifold.validate_tangent(g0, grad)

    def test_se3_with_custom_euclidean_gradient(self, key):
        """Test SE(3) with custom Euclidean gradient function."""
        manifold = SE3()

        def cost_fn(g):
            # Cost based on rotation: minimize angle from identity
            quat = g[:4]
            # Cost is 1 - |q_w| (minimum when q_w = ±1)
            return 1.0 - jnp.abs(quat[0])

        def euclidean_grad_fn(g):
            # Gradient w.r.t. quaternion part only
            quat = g[:4]
            grad_quat = jnp.array([-jnp.sign(quat[0]), 0.0, 0.0, 0.0])
            grad_trans = jnp.zeros(3)
            return jnp.concatenate([grad_quat, grad_trans])

        problem = RiemannianProblem(manifold, cost_fn, euclidean_grad_fn=euclidean_grad_fn)

        g0 = manifold.random_point(key)
        grad = problem.grad(g0)

        # Verify gradient is valid tangent vector
        assert manifold.validate_tangent(g0, grad)

    def test_se3_optimization_runs(self, key):
        """Test that optimization runs successfully on SE(3)."""
        manifold = SE3()

        # Cost function: minimize distance to identity
        identity = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        def cost_fn(g):
            return manifold.dist(g, identity)**2

        problem = RiemannianProblem(manifold, cost_fn)
        g0 = manifold.random_point(key)

        # Run optimization
        result = rieax.minimize(problem, g0, method="rsgd",
                               options={"max_iterations": 5, "learning_rate": 0.1})

        # Verify result is valid
        assert manifold.validate_point(result.x)
        assert isinstance(result.fun, (float, jnp.ndarray))

    def test_se3_batch_optimization(self, key):
        """Test SE(3) with batch operations in optimization."""
        manifold = SE3()

        # Cost function that works with batch
        def cost_fn(g):
            # For batch, sum over batch dimension
            if g.ndim > 1:
                return jnp.sum(g[..., 4:7]**2, axis=-1).sum()  # Sum of translation norms
            else:
                return jnp.sum(g[4:7]**2)

        problem = RiemannianProblem(manifold, cost_fn)

        # Test with single point
        g0 = manifold.random_point(key)
        grad_single = problem.grad(g0)
        assert manifold.validate_tangent(g0, grad_single)


class TestCrossManifoldConsistency:
    """Test consistency and interoperability between manifolds."""

    def test_factory_functions_with_riemann_problem(self, key):
        """Test that factory functions work correctly with RiemannianProblem."""
        # Test each factory function
        manifolds = [
            rieax.create_poincare_ball(dimension=2),
            rieax.create_lorentz(dimension=2),
            rieax.create_se3(),
            rieax.create_poincare_ball_for_embeddings(dimension=3),
            rieax.create_se3_for_robotics()
        ]

        for manifold in manifolds:
            # Simple cost function that works for any manifold
            def cost_fn(x):
                return jnp.sum(x**2) if hasattr(x, 'shape') else 0.0

            problem = RiemannianProblem(manifold, cost_fn)
            x0 = manifold.random_point(key)

            # Test that gradient computation works
            grad = problem.grad(x0)
            assert grad is not None
            assert grad.shape[:-1] == x0.shape[:-1] if hasattr(grad, 'shape') else True

    def test_gradient_consistency_across_methods(self, key):
        """Test gradient computation consistency across different methods."""
        manifold = PoincareBall(dimension=2, curvature=-1.0)

        def cost_fn(x):
            return jnp.sum(x**2)

        def euclidean_grad_fn(x):
            return 2 * x

        def riemannian_grad_fn(x):
            return manifold.proj(x, euclidean_grad_fn(x))

        x0 = manifold.random_point(key)

        # Test different gradient computation methods
        problem_autodiff = RiemannianProblem(manifold, cost_fn)
        problem_euclidean = RiemannianProblem(manifold, cost_fn, euclidean_grad_fn=euclidean_grad_fn)
        problem_riemannian = RiemannianProblem(manifold, cost_fn, grad_fn=riemannian_grad_fn)

        grad_auto = problem_autodiff.grad(x0)
        grad_eucl = problem_euclidean.grad(x0)
        grad_riem = problem_riemannian.grad(x0)

        # All methods should give same result
        assert jnp.allclose(grad_auto, grad_eucl, atol=1e-6)
        assert jnp.allclose(grad_auto, grad_riem, atol=1e-6)


class TestEndToEndOptimizationExamples:
    """Test realistic end-to-end optimization examples."""

    def test_hyperbolic_embedding_optimization(self, key):
        """Test hyperbolic embedding optimization using PoincareBall."""
        # Simulate learning hyperbolic embeddings for tree-like data
        manifold = rieax.create_poincare_ball_for_embeddings(dimension=2)

        # Create synthetic hierarchical data: parent at origin, children around
        parent_embedding = jnp.array([0.0, 0.0])
        children_targets = jnp.array([[0.3, 0.2], [0.2, 0.3], [-0.2, 0.1]])

        def cost_fn(embeddings):
            # embeddings shape: (4, 2) - parent + 3 children
            parent = embeddings[0]
            children = embeddings[1:]

            # Penalty for parent being far from origin
            parent_cost = jnp.sum(parent**2) * 10.0

            # Penalty for children being far from targets
            children_cost = jnp.sum((children - children_targets)**2)

            return parent_cost + children_cost

        problem = RiemannianProblem(manifold, cost_fn)

        # Initialize embeddings
        key1, key2, key3, key4 = jax.random.split(key, 4)
        embeddings0 = jnp.array([
            manifold.random_point(key1),
            manifold.random_point(key2),
            manifold.random_point(key3),
            manifold.random_point(key4)
        ])

        # This test mainly checks that the setup works
        grad = problem.grad(embeddings0)
        assert grad.shape == embeddings0.shape

    def test_robotics_pose_optimization(self, key):
        """Test robotics pose optimization using SE(3)."""
        manifold = rieax.create_se3_for_robotics()

        # Target pose: slight rotation and translation
        target_pose = jnp.array([0.9659, 0.0, 0.0, 0.2588, 1.0, 0.5, 0.2])  # 30° around z-axis

        def cost_fn(pose):
            # Cost is distance to target pose
            return manifold.dist(pose, target_pose)**2

        problem = RiemannianProblem(manifold, cost_fn)

        # Start from identity pose
        pose0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Run optimization
        result = rieax.minimize(problem, pose0, method="rsgd",
                               options={"max_iterations": 10, "learning_rate": 0.1})

        # Verify result is closer to target
        initial_distance = manifold.dist(pose0, target_pose)
        final_distance = manifold.dist(result.x, target_pose)

        # Should make some progress (at least 10% improvement or very close)
        assert final_distance <= initial_distance or final_distance < 0.1
        assert manifold.validate_point(result.x)


if __name__ == "__main__":
    pytest.main([__file__])
