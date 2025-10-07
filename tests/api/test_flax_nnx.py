"""Tests for Flax NNX manifold-constrained modules."""

import jax
import jax.numpy as jnp
import pytest

try:
    from flax import nnx
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

from riemannax.api.flax_nnx import (
    ConstraintViolation,
    ManifoldConstrainedLinear,
    ManifoldConstrainedModule,
    create_manifold_linear,
)
from riemannax.manifolds import Sphere, Stiefel

pytestmark = pytest.mark.skipif(not FLAX_AVAILABLE, reason="Flax not installed")


class TestManifoldConstrainedModule:
    """Test suite for ManifoldConstrainedModule base class."""

    def test_module_inherits_from_nnx_module(self):
        """Test that ManifoldConstrainedModule inherits from nnx.Module."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)

        # Act
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        # Assert
        assert isinstance(module, nnx.Module)

    def test_module_has_constraint_variables(self):
        """Test that module tracks constraint violations using custom Variable types."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)

        # Act
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        # Assert
        assert hasattr(module, 'constraint_violations')
        assert isinstance(module.constraint_violations, nnx.Variable)

    def test_module_initializes_parameters_on_manifold(self):
        """Test that parameters are initialized to satisfy manifold constraints."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)

        # Act
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        # Assert
        params = module.params.value
        # For sphere, check unit norm
        assert jnp.allclose(jnp.linalg.norm(params), 1.0, atol=1e-5)

    def test_module_projects_parameters_to_manifold(self):
        """Test that project_params() enforces manifold constraints using mutable state."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        # Violate constraint
        module.params.value = jnp.array([2.0, 0.0, 0.0, 0.0])

        # Act
        module.project_params()

        # Assert
        params = module.params.value
        assert jnp.allclose(jnp.linalg.norm(params), 1.0, atol=1e-5)


class TestManifoldConstrainedLinear:
    """Test suite for ManifoldConstrainedLinear layer."""

    def test_linear_layer_forward_pass(self):
        """Test that linear layer performs forward pass correctly."""
        # Arrange
        key = jax.random.PRNGKey(0)
        in_features, out_features = 5, 3
        manifold = Stiefel(n=in_features, p=out_features)

        layer = ManifoldConstrainedLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=manifold,
            rngs=nnx.Rngs(key)
        )

        x = jax.random.normal(jax.random.PRNGKey(1), (2, in_features))

        # Act
        output = layer(x)

        # Assert
        assert output.shape == (2, out_features)

    def test_linear_layer_maintains_orthogonality(self):
        """Test that Stiefel-constrained weights remain orthogonal."""
        # Arrange
        key = jax.random.PRNGKey(0)
        in_features, out_features = 5, 3
        manifold = Stiefel(n=in_features, p=out_features)

        layer = ManifoldConstrainedLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=manifold,
            rngs=nnx.Rngs(key)
        )

        # Act
        weights = layer.weight.value

        # Assert - W^T W should be identity for orthogonal matrices
        gram = weights.T @ weights
        expected = jnp.eye(out_features, dtype=weights.dtype)
        assert jnp.allclose(gram, expected, atol=1e-5)

    def test_linear_layer_tracks_constraint_violations(self):
        """Test that constraint violations are tracked when projecting parameters."""
        # Arrange
        key = jax.random.PRNGKey(0)
        in_features, out_features = 5, 3
        manifold = Stiefel(n=in_features, p=out_features)

        layer = ManifoldConstrainedLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=manifold,
            rngs=nnx.Rngs(key)
        )

        # Violate constraint
        layer.weight.value = jax.random.normal(jax.random.PRNGKey(2), (in_features, out_features))

        # Act - Project back to manifold
        layer.project_params()

        # Assert
        assert float(layer.constraint_violations.value) > 0


class TestConstraintViolationTracking:
    """Test suite for constraint violation tracking."""

    def test_constraint_violation_variable_type(self):
        """Test that ConstraintViolation is a custom Variable type."""
        # Arrange & Act
        violation = ConstraintViolation(jnp.array(0.0))

        # Assert
        assert isinstance(violation, nnx.Variable)

    def test_constraint_violation_increments(self):
        """Test that constraint violations can be incremented."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        initial_count = float(module.constraint_violations.value)

        # Act - violate constraint multiple times
        for i in range(3):
            k_i = jax.random.fold_in(key, i + 1)
            module.params.value = jax.random.normal(k_i, (4,))
            module.project_params()

        # Assert
        assert float(module.constraint_violations.value) > initial_count


class TestNNXCheckpointing:
    """Test suite for Flax NNX checkpointing compatibility."""

    def test_module_state_serialization(self):
        """Test that module state including constraints can be serialized."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        # Act - Extract state
        state = nnx.state(module)

        # Assert
        assert state is not None
        assert 'params' in state or any('param' in str(k).lower() for k in state)

    def test_module_state_deserialization(self):
        """Test that module state can be restored from checkpoint."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Sphere(n=3)
        module1 = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(key)
        )

        # Save state
        state = nnx.state(module1)
        original_params = module1.params.value.copy()

        # Create new module and restore state
        module2 = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(4,),
            rngs=nnx.Rngs(jax.random.PRNGKey(1))
        )

        # Act
        nnx.update(module2, state)

        # Assert
        assert jnp.allclose(module2.params.value, original_params)


class TestManifoldShapeValidation:
    """Test suite for param_shape validation in ManifoldConstrainedModule."""

    def test_module_accepts_correct_stiefel_shape(self):
        """Test that ManifoldConstrainedModule accepts correct Stiefel param_shape."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Stiefel(n=5, p=3)

        # Act - Use manifold's intrinsic shape (fixes splatting bug from CodeRabbit review)
        module = ManifoldConstrainedModule(
            manifold=manifold,
            param_shape=(5, 3),  # Matches Stiefel(n=5, p=3)
            rngs=nnx.Rngs(key)
        )

        # Assert
        assert module.params.value.shape == (5, 3)
        assert manifold.validate_point(module.params.value)


class TestFactoryFunctionEdgeCases:
    """Test suite for create_manifold_linear factory function edge cases."""

    def test_factory_rejects_stiefel_with_in_lt_out(self):
        """Test that factory raises ValueError when in_features < out_features for Stiefel."""
        # Arrange
        key = jax.random.PRNGKey(0)

        # Act & Assert - Stiefel requires n >= p
        with pytest.raises(ValueError, match="Stiefel requires in_features >= out_features"):
            create_manifold_linear(
                in_features=2, out_features=3, manifold_type="stiefel", rngs=nnx.Rngs(key)
            )

    def test_factory_rejects_sphere_for_linear_weights(self):
        """Test that factory raises NotImplementedError for sphere manifold."""
        # Arrange
        key = jax.random.PRNGKey(0)

        # Act & Assert - Sphere produces 1D vectors, not 2D matrices
        with pytest.raises(NotImplementedError, match="ManifoldConstrainedLinear requires a matrix-shaped manifold"):
            create_manifold_linear(
                in_features=4, out_features=2, manifold_type="sphere", rngs=nnx.Rngs(key)
            )

    def test_factory_rejects_unsupported_manifold_type(self):
        """Test that factory raises ValueError for unsupported manifold types."""
        # Arrange
        key = jax.random.PRNGKey(0)

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported manifold type"):
            create_manifold_linear(
                in_features=4, out_features=2, manifold_type="unsupported", rngs=nnx.Rngs(key)
            )


class TestTrainingWithConstraints:
    """Test suite for training neural networks with manifold constraints."""

    def test_gradient_updates_maintain_constraints(self):
        """Test that gradient updates followed by projection maintain constraints."""
        # Arrange
        key = jax.random.PRNGKey(0)
        in_features, out_features = 5, 3
        manifold = Stiefel(n=in_features, p=out_features)

        layer = ManifoldConstrainedLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=manifold,
            rngs=nnx.Rngs(key)
        )

        def loss_fn(layer, x, y):
            pred = layer(x)
            return jnp.mean((pred - y) ** 2)

        x = jax.random.normal(jax.random.PRNGKey(1), (2, in_features))
        y = jax.random.normal(jax.random.PRNGKey(2), (2, out_features))

        # Act - Perform gradient update
        grads = nnx.grad(loss_fn)(layer, x, y)

        # Simple gradient descent update
        layer.weight.value = layer.weight.value - 0.01 * grads.weight.value

        # Project back to manifold
        layer.project_params()

        # Assert - Should still be orthogonal
        weights = layer.weight.value
        gram = weights.T @ weights
        expected = jnp.eye(out_features, dtype=weights.dtype)
        assert jnp.allclose(gram, expected, atol=1e-4)

    def test_jit_compilation_with_constraints(self):
        """Test that modules with constraints can be JIT compiled."""
        # Arrange
        key = jax.random.PRNGKey(0)
        in_features, out_features = 5, 3
        manifold = Stiefel(n=in_features, p=out_features)

        layer = ManifoldConstrainedLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=manifold,
            rngs=nnx.Rngs(key)
        )

        x = jax.random.normal(jax.random.PRNGKey(1), (2, in_features))

        # Act - JIT compile forward pass using NNX's jit
        @nnx.jit
        def forward(layer, x):
            return layer(x)

        output = forward(layer, x)

        # Assert
        assert output.shape == (2, out_features)
