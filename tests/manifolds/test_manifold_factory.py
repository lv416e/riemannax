"""Test manifold factory pattern for dynamic manifold construction.

This module tests the factory functions that provide a convenient interface
for creating manifold instances with dimension validation and clear error messages.

Following TDD methodology:
- RED phase: Tests fail because factory functions don't exist yet
- GREEN phase: Implement factory functions to make tests pass
- REFACTOR phase: Optimize and clean up the implementation
"""

import jax
import pytest

from riemannax.manifolds import create_grassmann, create_so, create_spd, create_sphere, create_stiefel
from riemannax.manifolds.base import DimensionError
from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.stiefel import Stiefel


class TestManifoldFactory:
    """Test factory pattern for manifold construction."""

    def test_create_sphere_function_exists(self):
        """Test that create_sphere factory function exists."""
        assert callable(create_sphere), "create_sphere should be a callable factory function"

    def test_create_sphere_with_valid_dimensions(self):
        """Test create_sphere with valid dimensions."""
        # Test various valid dimensions
        for n in [1, 2, 3, 5, 10, 50]:
            sphere = create_sphere(n)
            assert isinstance(sphere, Sphere), f"create_sphere({n}) should return Sphere instance"
            assert sphere.dimension == n, f"Sphere dimension should be {n}"
            assert sphere.ambient_dimension == n + 1, f"Ambient dimension should be {n + 1}"

    def test_create_sphere_default_dimension(self):
        """Test create_sphere with default dimension."""
        sphere = create_sphere()
        assert isinstance(sphere, Sphere)
        assert sphere.dimension == 2  # Default dimension
        assert sphere.ambient_dimension == 3

    def test_create_sphere_dimension_validation(self):
        """Test create_sphere dimension validation."""
        # Test invalid dimensions
        invalid_dims = [0, -1, -5]

        for n in invalid_dims:
            with pytest.raises((ValueError, DimensionError)) as exc_info:
                create_sphere(n)

            # Check error message quality
            error_msg = str(exc_info.value).lower()
            assert "dimension" in error_msg
            assert "positive" in error_msg or "greater" in error_msg

    def test_create_sphere_type_validation(self):
        """Test create_sphere with invalid types."""
        invalid_types = [1.5, "3", None, [2], {"n": 3}]

        for invalid_input in invalid_types:
            with pytest.raises((TypeError, ValueError)):
                create_sphere(invalid_input)

    def test_create_grassmann_function_exists(self):
        """Test that create_grassmann factory function exists."""
        assert callable(create_grassmann), "create_grassmann should be a callable factory function"

    def test_create_grassmann_with_valid_dimensions(self):
        """Test create_grassmann with valid dimensions."""
        # Test various valid (p, n) combinations where p < n
        test_cases = [(2, 5), (3, 8), (1, 4), (5, 10)]

        for p, n in test_cases:
            grassmann = create_grassmann(p, n)
            assert isinstance(grassmann, Grassmann), f"create_grassmann({p}, {n}) should return Grassmann"
            # Note: Grassmann dimension is p * (n - p)
            expected_dim = p * (n - p)
            assert grassmann.dimension == expected_dim

    def test_create_grassmann_dimension_validation(self):
        """Test create_grassmann dimension validation."""
        # Test invalid dimensions
        invalid_cases = [
            (0, 5),  # p must be positive
            (5, 0),  # n must be positive
            (5, 5),  # p must be < n
            (6, 5),  # p must be < n
            (-1, 5),  # negative p
            (2, -3),  # negative n
        ]

        for p, n in invalid_cases:
            with pytest.raises((ValueError, DimensionError)):
                create_grassmann(p, n)

    def test_create_stiefel_function_exists(self):
        """Test that create_stiefel factory function exists."""
        assert callable(create_stiefel), "create_stiefel should be a callable factory function"

    def test_create_stiefel_with_valid_dimensions(self):
        """Test create_stiefel with valid dimensions."""
        # Test various valid (p, n) combinations where p <= n
        test_cases = [(2, 5), (3, 8), (1, 4), (5, 5), (3, 10)]

        for p, n in test_cases:
            stiefel = create_stiefel(p, n)
            assert isinstance(stiefel, Stiefel), f"create_stiefel({p}, {n}) should return Stiefel"

    def test_create_so_function_exists(self):
        """Test that create_so factory function exists."""
        assert callable(create_so), "create_so should be a callable factory function"

    def test_create_so_with_valid_dimensions(self):
        """Test create_so with valid dimensions."""
        # Test various valid dimensions for SO(n)
        for n in [2, 3, 4, 5, 10]:
            so = create_so(n)
            assert isinstance(so, SpecialOrthogonal), f"create_so({n}) should return SpecialOrthogonal"

    def test_create_so_dimension_validation(self):
        """Test create_so dimension validation."""
        # Test invalid dimensions
        invalid_dims = [0, 1, -1, -5]  # SO(n) needs n >= 2

        for n in invalid_dims:
            with pytest.raises((ValueError, DimensionError)):
                create_so(n)

    def test_create_spd_function_exists(self):
        """Test that create_spd factory function exists."""
        assert callable(create_spd), "create_spd should be a callable factory function"

    def test_create_spd_with_valid_dimensions(self):
        """Test create_spd with valid dimensions."""
        # Test various valid dimensions for SPD(n)
        for n in [2, 3, 4, 5, 10]:
            spd = create_spd(n)
            assert isinstance(spd, SymmetricPositiveDefinite), f"create_spd({n}) should return SPD"

    def test_create_spd_dimension_validation(self):
        """Test create_spd dimension validation."""
        # Test invalid dimensions
        invalid_dims = [0, 1, -1, -5]  # SPD needs n >= 2

        for n in invalid_dims:
            with pytest.raises((ValueError, DimensionError)):
                create_spd(n)

    def test_factory_functions_work_with_operations(self):
        """Test that manifolds created by factories work with operations."""
        key = jax.random.PRNGKey(42)

        # Test sphere operations
        sphere = create_sphere(3)
        x = sphere.random_point(key)
        key, subkey = jax.random.split(key)
        v = sphere.random_tangent(subkey, x)

        # Should work without errors
        proj_result = sphere.proj(x, v)
        exp_result = sphere.exp(x, v)
        inner_result = sphere.inner(x, v, v)

        assert proj_result.shape == v.shape
        assert exp_result.shape == x.shape
        assert inner_result.ndim == 0

    def test_factory_vs_direct_construction_equivalence(self):
        """Test that factory functions produce equivalent results to direct construction."""
        # Sphere
        factory_sphere = create_sphere(5)
        direct_sphere = Sphere(n=5)

        assert type(factory_sphere) == type(direct_sphere)
        assert factory_sphere.dimension == direct_sphere.dimension
        assert factory_sphere.ambient_dimension == direct_sphere.ambient_dimension

        # Grassmann
        factory_grassmann = create_grassmann(3, 7)
        direct_grassmann = Grassmann(p=3, n=7)

        assert type(factory_grassmann) == type(direct_grassmann)
        assert factory_grassmann.dimension == direct_grassmann.dimension

    def test_factory_error_messages_are_informative(self):
        """Test that factory functions provide clear, informative error messages."""
        # Test sphere error message
        try:
            create_sphere(-2)
            raise AssertionError("Should have raised an error")
        except (ValueError, DimensionError) as e:
            error_msg = str(e)
            assert "sphere" in error_msg.lower() or "dimension" in error_msg.lower()
            assert "positive" in error_msg.lower() or "-2" in error_msg

        # Test grassmann error message
        try:
            create_grassmann(5, 3)  # p >= n
            raise AssertionError("Should have raised an error")
        except (ValueError, DimensionError) as e:
            error_msg = str(e)
            assert "grassmann" in error_msg.lower() or "p" in error_msg.lower() or "dimension" in error_msg.lower()

    def test_all_factory_functions_exported(self):
        """Test that all factory functions are properly exported from the module."""
        from riemannax.manifolds import create_grassmann, create_so, create_spd, create_sphere, create_stiefel

        # Should not raise ImportError
        assert callable(create_sphere)
        assert callable(create_grassmann)
        assert callable(create_stiefel)
        assert callable(create_so)
        assert callable(create_spd)

    def test_factory_docstrings_exist(self):
        """Test that factory functions have proper docstrings."""
        factories = [create_sphere, create_grassmann, create_stiefel, create_so, create_spd]

        for factory in factories:
            assert factory.__doc__ is not None, f"{factory.__name__} should have a docstring"
            assert len(factory.__doc__.strip()) > 10, f"{factory.__name__} docstring should be descriptive"
