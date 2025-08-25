"""Test module to verify JIT decorator is properly applied to manifold methods."""

import inspect

import jax.numpy as jnp

from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.stiefel import Stiefel


class TestJITDecoratorApplication:
    """Test that JIT decorators are properly applied to concrete manifold methods."""

    def test_sphere_methods_have_jit_decorator(self):
        """Test that Sphere manifold methods are decorated with @jit_optimized."""
        sphere = Sphere()

        # Core mathematical methods that should be JIT-compiled
        core_methods = ["proj", "exp", "log", "retr", "transp", "inner", "dist"]

        for method_name in core_methods:
            method = getattr(sphere, method_name)

            # Check if method has been JIT-decorated (has _original_func attribute)
            assert hasattr(method, "_original_func"), (
                f"Method {method_name} should have JIT decorator with _original_func attribute"
            )
            assert hasattr(method, "_static_args"), (
                f"Method {method_name} should have JIT decorator with _static_args attribute"
            )

    def test_grassmann_methods_have_jit_decorator(self):
        """Test that Grassmann manifold methods are decorated with @jit_optimized."""
        grassmann = Grassmann(n=5, p=3)

        core_methods = ["proj", "exp", "log", "retr", "transp", "inner", "dist"]

        for method_name in core_methods:
            method = getattr(grassmann, method_name)
            assert hasattr(method, "_original_func"), f"Method {method_name} should have JIT decorator"

    def test_stiefel_methods_have_jit_decorator(self):
        """Test that Stiefel manifold methods are decorated with @jit_optimized."""
        stiefel = Stiefel(n=5, p=3)

        core_methods = ["proj", "exp", "log", "retr", "transp", "inner", "dist"]

        for method_name in core_methods:
            method = getattr(stiefel, method_name)
            assert hasattr(method, "_original_func"), f"Method {method_name} should have JIT decorator"

    def test_spd_methods_have_jit_decorator(self):
        """Test that SPD manifold methods are decorated with @jit_optimized."""
        spd = SymmetricPositiveDefinite(n=3)

        # SPD doesn't have a retr method (uses base class NotImplementedError)
        core_methods = ["proj", "exp", "log", "transp", "inner", "dist"]

        for method_name in core_methods:
            method = getattr(spd, method_name)
            assert hasattr(method, "_original_func"), f"Method {method_name} should have JIT decorator"

    def test_so_methods_have_jit_decorator(self):
        """Test that SO manifold methods are decorated with @jit_optimized."""
        so = SpecialOrthogonal(n=3)

        # SO doesn't have a retr method (uses base class NotImplementedError)
        core_methods = ["proj", "exp", "log", "transp", "inner", "dist"]

        for method_name in core_methods:
            method = getattr(so, method_name)
            assert hasattr(method, "_original_func"), f"Method {method_name} should have JIT decorator"

    def test_jit_decorated_methods_work_correctly(self):
        """Test that JIT-decorated methods produce correct results."""
        import jax

        sphere = Sphere()
        key = jax.random.PRNGKey(42)

        # Generate test data
        x = sphere.random_point(key)
        key, subkey = jax.random.split(key)
        v = sphere.random_tangent(subkey, x)

        # Test that JIT-decorated methods work
        proj_result = sphere.proj(x, v)
        exp_result = sphere.exp(x, v)
        inner_result = sphere.inner(x, v, v)

        # Check results are valid arrays
        assert isinstance(proj_result, jnp.ndarray)
        assert isinstance(exp_result, jnp.ndarray)
        assert isinstance(inner_result, jnp.ndarray)

        # Check shapes are correct
        assert proj_result.shape == v.shape
        assert exp_result.shape == x.shape
        assert inner_result.shape == ()

    def test_static_args_properly_configured(self):
        """Test that static arguments are properly configured for methods that need them."""
        # Methods that might benefit from static args (like dimension-related parameters)
        grassmann = Grassmann(n=5, p=3)

        # Check that methods have static_args attribute
        method = grassmann.proj
        assert hasattr(method, "_static_args")
        assert isinstance(method._static_args, tuple)

    def test_import_jit_decorator_in_manifolds(self):
        """Test that manifold modules import the jit_optimized decorator."""
        # Check that sphere module imports jit_optimized
        from riemannax.manifolds import sphere

        source = inspect.getsource(sphere)

        assert "jit_optimized" in source, "sphere.py should import jit_optimized decorator"
        assert "from ..core.jit_decorator import jit_optimized" in source, (
            "sphere.py should import jit_optimized from core.jit_decorator"
        )

    def test_performance_methods_not_jit_decorated(self):
        """Test that non-computational methods are not JIT-decorated."""
        sphere = Sphere()

        # Methods that should NOT be JIT-decorated
        non_jit_methods = ["validate_point", "validate_tangent", "dimension", "ambient_dimension"]

        for method_name in non_jit_methods:
            method = getattr(sphere, method_name)

            # These should NOT have JIT decorator attributes
            assert not hasattr(method, "_original_func"), f"Method {method_name} should NOT have JIT decorator"

    def test_random_methods_not_jit_decorated(self):
        """Test that random generation methods are not JIT-decorated."""
        sphere = Sphere()

        # Random methods that should NOT be JIT-decorated (they use PRNG keys)
        random_methods = ["random_point", "random_tangent"]

        for method_name in random_methods:
            method = getattr(sphere, method_name)

            # These should NOT have JIT decorator attributes
            assert not hasattr(method, "_original_func"), f"Method {method_name} should NOT have JIT decorator"
