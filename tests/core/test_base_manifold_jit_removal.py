"""Test module to verify JIT logic is properly removed from BaseManifold."""

import pytest

from riemannax.manifolds.base import Manifold


class TestBaseManifoldJITRemoval:
    """Test that JIT-related attributes and methods are removed from BaseManifold."""

    def test_jit_attributes_removed(self):
        """Test that JIT-related attributes are not present in BaseManifold."""
        manifold = Manifold()

        # These JIT-related attributes should not exist
        assert not hasattr(manifold, '_jit_compiled_methods'), "_jit_compiled_methods should be removed"
        assert not hasattr(manifold, '_safe_jit_wrapper'), "_safe_jit_wrapper should be removed"
        assert not hasattr(manifold, '_jit_enabled'), "_jit_enabled should be removed"
        assert not hasattr(manifold, '_jit_initialized'), "_jit_initialized should be removed"
        assert not hasattr(manifold, '_performance_tracking'), "_performance_tracking should be removed"

    def test_jit_methods_removed(self):
        """Test that JIT-related methods are not present in BaseManifold."""
        manifold = Manifold()

        # These JIT-related methods should not exist
        assert not hasattr(manifold, '_compile_core_methods'), "_compile_core_methods should be removed"
        assert not hasattr(manifold, '_call_jit_method'), "_call_jit_method should be removed"
        assert not hasattr(manifold, 'clear_jit_cache'), "clear_jit_cache should be removed"
        assert not hasattr(manifold, '_reset_jit_cache'), "_reset_jit_cache should be removed"
        assert not hasattr(manifold, 'enable_performance_tracking'), "enable_performance_tracking should be removed"
        assert not hasattr(manifold, 'get_performance_report'), "get_performance_report should be removed"

    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods properly raise NotImplementedError."""
        manifold = Manifold()

        # Create dummy arrays for testing
        import jax
        import jax.numpy as jnp
        x = jnp.array([1.0, 0.0])
        y = jnp.array([0.0, 1.0])
        v = jnp.array([0.0, 1.0])
        key = jax.random.PRNGKey(42)

        # All abstract methods should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            manifold.proj(x, v)

        with pytest.raises(NotImplementedError):
            manifold.exp(x, v)

        with pytest.raises(NotImplementedError):
            manifold.log(x, y)

        with pytest.raises(NotImplementedError):
            manifold.retr(x, v)

        with pytest.raises(NotImplementedError):
            manifold.transp(x, y, v)

        with pytest.raises(NotImplementedError):
            manifold.inner(x, v, v)

        with pytest.raises(NotImplementedError):
            manifold.dist(x, y)

        with pytest.raises(NotImplementedError):
            manifold.random_point(key)

        with pytest.raises(NotImplementedError):
            manifold.random_tangent(key, x)

    def test_safe_jit_import_removed(self):
        """Test that SafeJITWrapper import is removed from base.py."""
        import inspect
        from riemannax.manifolds import base

        # Check the source code doesn't contain SafeJITWrapper import
        source = inspect.getsource(base)
        assert 'SafeJITWrapper' not in source, "SafeJITWrapper import should be removed"
        assert 'safe_jit' not in source, "safe_jit imports should be removed"

    def test_manifold_initialization_clean(self):
        """Test that manifold initialization is clean without JIT setup."""
        manifold = Manifold()

        # Verify initialization doesn't create JIT-related state
        manifold_dict = manifold.__dict__
        jit_related_keys = [key for key in manifold_dict.keys() if 'jit' in key.lower()]

        assert len(jit_related_keys) == 0, f"Found JIT-related attributes: {jit_related_keys}"

    def test_docstring_updated(self):
        """Test that class docstring no longer mentions JIT optimization."""
        docstring = Manifold.__doc__ or ""

        # Docstring should not mention JIT optimization
        assert 'JIT optimization' not in docstring, "Docstring should not mention JIT optimization"
        assert 'jit' not in docstring.lower(), "Docstring should not mention jit"
