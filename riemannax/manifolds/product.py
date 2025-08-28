"""Implementation of product manifolds M = M₁ x M₂ x ... x Mₖ.

This module provides the ProductManifold class for composing multiple manifolds
into a single product manifold, enabling optimization on composite spaces where
different components may belong to different manifolds.
"""

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .base import Manifold


class ProductManifold(Manifold):
    """Product manifold M = M₁ x M₂ x ... x Mₖ.

    A product manifold combines multiple manifolds into a single manifold where
    points are tuples of points from each component manifold, and operations
    are performed component-wise.

    For manifolds M₁, M₂, ..., Mₖ with dimensions d₁, d₂, ..., dₖ:
    - The product manifold M = M₁ x M₂ x ... x Mₖ has dimension Σᵢ dᵢ
    - Points are represented as flattened arrays combining all components
    - Operations are delegated to component manifolds after splitting
    """

    def __init__(self, manifolds: tuple[Manifold, ...]):
        """Initialize product manifold from component manifolds.

        Args:
            manifolds: Tuple of component manifolds to combine.

        Raises:
            ValueError: If manifolds tuple is empty.
            TypeError: If any element in manifolds is not a Manifold instance.
        """
        if not manifolds:
            raise ValueError("Product manifold requires at least one component manifold")

        for i, manifold in enumerate(manifolds):
            if not isinstance(manifold, Manifold):
                raise TypeError(f"Component {i} is not a Manifold instance: {type(manifold)}")

        self.manifolds = manifolds

        # Compute dimensions
        self._dimension = sum(manifold.dimension for manifold in self.manifolds)
        self._ambient_dimension = sum(manifold.ambient_dimension for manifold in self.manifolds)

        # Store component dimensions and shapes for efficient splitting/combining
        self._component_dimensions = [manifold.dimension for manifold in self.manifolds]
        self._component_ambient_dimensions = [manifold.ambient_dimension for manifold in self.manifolds]

        # Store original component shapes for proper reshaping
        # Generate a dummy point from each manifold to determine its natural shape
        self._component_shapes = []
        dummy_key = jr.PRNGKey(0)  # Temporary key for shape determination
        for i, manifold in enumerate(self.manifolds):
            subkey = jr.fold_in(dummy_key, i)
            dummy_point = manifold.random_point(subkey)
            self._component_shapes.append(dummy_point.shape)

    def _split_point(self, x: Array) -> tuple[Array, ...]:
        """Split a product manifold point into component manifold points.

        Args:
            x: Point on the product manifold (flattened array).

        Returns:
            Tuple of points on component manifolds.
        """
        components = []
        start_idx = 0

        for i, _manifold in enumerate(self.manifolds):
            ambient_dim = self._component_ambient_dimensions[i]
            end_idx = start_idx + ambient_dim

            # Extract component data
            component_data = x[start_idx:end_idx]

            # Reshape to original component shape
            original_shape = self._component_shapes[i]
            reshaped_component = component_data.reshape(original_shape)
            components.append(reshaped_component)

            start_idx = end_idx

        return tuple(components)

    def _combine_points(self, components: tuple[Array, ...]) -> Array:
        """Combine component manifold points into a product manifold point.

        Args:
            components: Tuple of points on component manifolds.

        Returns:
            Point on the product manifold (flattened array).
        """
        if len(components) != len(self.manifolds):
            raise ValueError(f"Expected {len(self.manifolds)} components, got {len(components)}")

        # Flatten each component and concatenate
        flattened_components = []
        for component in components:
            flattened_components.append(component.flatten())

        return jnp.concatenate(flattened_components)

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the product manifold (sum of component dimensions)."""
        return self._dimension

    @property
    def ambient_dimension(self) -> int:
        """Ambient dimension of the product manifold (sum of component ambient dimensions)."""
        return self._ambient_dimension

    def exp(self, x: Array, v: Array) -> Array:
        """Apply component-wise exponential map on the product manifold.

        For product manifold M = M₁ x M₂ x ... x Mₖ:
        exp_x(v) = (exp_x₁(v₁), exp_x₂(v₂), ..., exp_xₖ(vₖ))

        Args:
            x: Point on the product manifold (flattened array).
            v: Tangent vector at x (flattened array).

        Returns:
            Point on the product manifold reached by the exponential map.
        """
        # Split into component points and tangent vectors
        x_components = self._split_point(x)
        v_components = self._split_point(v)

        # Apply exponential map component-wise
        exp_components = []
        for i, manifold in enumerate(self.manifolds):
            exp_comp = manifold.exp(x_components[i], v_components[i])
            exp_components.append(exp_comp)

        # Combine results back into product manifold point
        return self._combine_points(tuple(exp_components))

    def log(self, x: Array, y: Array) -> Array:
        """Apply component-wise logarithmic map on the product manifold.

        For product manifold M = M₁ x M₂ x ... x Mₖ:
        log_x(y) = (log_x₁(y₁), log_x₂(y₂), ..., log_xₖ(yₖ))

        Args:
            x: Starting point on the product manifold (flattened array).
            y: Target point on the product manifold (flattened array).

        Returns:
            Tangent vector at x pointing toward y.
        """
        # Split into component points
        x_components = self._split_point(x)
        y_components = self._split_point(y)

        # Apply logarithmic map component-wise
        log_components = []
        for i, manifold in enumerate(self.manifolds):
            log_comp = manifold.log(x_components[i], y_components[i])
            log_components.append(log_comp)

        # Combine results back into product manifold tangent vector
        return self._combine_points(tuple(log_components))

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Compute inner product as sum of component inner products.

        For product manifold M = M₁ x M₂ x ... x Mₖ:
        <u, v>_x = Σᵢ <uᵢ, vᵢ>_xᵢ

        Args:
            x: Point on the product manifold (flattened array).
            u: First tangent vector at x (flattened array).
            v: Second tangent vector at x (flattened array).

        Returns:
            Inner product value (scalar).
        """
        # Split into component points and tangent vectors
        x_components = self._split_point(x)
        u_components = self._split_point(u)
        v_components = self._split_point(v)

        # Compute sum of component inner products
        total_inner = jnp.array(0.0)
        for i, manifold in enumerate(self.manifolds):
            comp_inner = manifold.inner(x_components[i], u_components[i], v_components[i])
            total_inner = total_inner + comp_inner

        return jnp.asarray(total_inner)

    def proj(self, x: Array, v: Array) -> Array:
        """Apply component-wise projection onto tangent spaces.

        For product manifold M = M₁ x M₂ x ... x Mₖ:
        proj_x(v) = (proj_x₁(v₁), proj_x₂(v₂), ..., proj_xₖ(vₖ))

        Args:
            x: Point on the product manifold (flattened array).
            v: Vector in ambient space (flattened array).

        Returns:
            Projection of v onto the tangent space at x.
        """
        # Split into component points and vectors
        x_components = self._split_point(x)
        v_components = self._split_point(v)

        # Apply projection component-wise
        proj_components = []
        for i, manifold in enumerate(self.manifolds):
            proj_comp = manifold.proj(x_components[i], v_components[i])
            proj_components.append(proj_comp)

        # Combine results back into product manifold tangent vector
        return self._combine_points(tuple(proj_components))

    def dist(self, x: Array, y: Array) -> Array:
        """Compute distance as Euclidean composition of component distances.

        For product manifold M = M₁ x M₂ x ... x Mₖ:
        d(x, y) = √(Σᵢ dᵢ(xᵢ, yᵢ)²)

        Args:
            x: First point on the product manifold (flattened array).
            y: Second point on the product manifold (flattened array).

        Returns:
            Geodesic distance between x and y.
        """
        # Split into component points
        x_components = self._split_point(x)
        y_components = self._split_point(y)

        # Compute sum of squared component distances
        total_dist_sq = jnp.array(0.0)
        for i, manifold in enumerate(self.manifolds):
            comp_dist = manifold.dist(x_components[i], y_components[i])
            total_dist_sq = total_dist_sq + comp_dist**2

        return jnp.sqrt(total_dist_sq)

    def random_point(self, key: PRNGKeyArray, *shape: int) -> Array:
        """Generate random point(s) on the product manifold.

        For product manifold M = M₁ x M₂ x ... x Mₖ, generates random points
        by independently sampling from each component manifold and combining
        the results into a flattened product manifold representation.

        Args:
            key: JAX PRNG key for random number generation.
            *shape: Shape of the output array of points. If empty, returns single point.

        Returns:
            Random point(s) on the product manifold as flattened array(s).
            Shape: (*shape, ambient_dimension) if shape provided,
                   (ambient_dimension,) otherwise.
        """
        # Handle batch generation
        if shape:
            # For batch generation, use vmap
            keys = jr.split(key, shape[0])

            def single_random_point(single_key):
                return self._generate_single_random_point(single_key)

            if len(shape) == 1:
                # Simple batch case: (batch_size,)
                import jax

                return jax.vmap(single_random_point)(keys)
            else:
                # Multi-dimensional batch case - need to reshape
                total_batch_size = 1
                for s in shape:
                    total_batch_size *= s

                flat_keys = jr.split(key, total_batch_size)

                import jax

                flat_points = jax.vmap(single_random_point)(flat_keys)

                # Reshape to desired batch shape
                target_shape = (*shape, self.ambient_dimension)
                return flat_points.reshape(target_shape)
        else:
            # Single point generation
            return self._generate_single_random_point(key)

    def _generate_single_random_point(self, key: PRNGKeyArray) -> Array:
        """Generate a single random point on the product manifold.

        Args:
            key: JAX PRNG key.

        Returns:
            Single random point as flattened array.
        """
        # Split key for each component manifold
        subkeys = jr.split(key, len(self.manifolds))

        # Generate random point on each component manifold independently
        component_points = []
        for manifold, subkey in zip(self.manifolds, subkeys, strict=False):
            component_point = manifold.random_point(subkey)
            component_points.append(component_point)

        # Combine component points into product manifold point
        return self._combine_points(tuple(component_points))

    def random_tangent(self, key: PRNGKeyArray, x: Array, *shape: int) -> Array:
        """Generate random tangent vector(s) at point x on the product manifold.

        For product manifold M = M₁ x M₂ x ... x Mₖ, generates random tangent
        vectors by independently sampling tangent vectors from each component
        manifold's tangent space at the corresponding component point.

        Args:
            key: JAX PRNG key for random number generation.
            x: Base point on the product manifold (flattened array).
            *shape: Shape of the output array of tangent vectors. If empty, returns single vector.

        Returns:
            Random tangent vector(s) at x as flattened array(s).
            Shape: (*shape, ambient_dimension) if shape provided,
                   (ambient_dimension,) otherwise.
        """
        # Handle batch generation
        if shape:
            # For batch generation, use vmap
            keys = jr.split(key, shape[0])

            def single_random_tangent(single_key):
                return self._generate_single_random_tangent(single_key, x)

            if len(shape) == 1:
                # Simple batch case: (batch_size,)
                import jax

                return jax.vmap(single_random_tangent)(keys)
            else:
                # Multi-dimensional batch case - need to reshape
                total_batch_size = 1
                for s in shape:
                    total_batch_size *= s

                flat_keys = jr.split(key, total_batch_size)

                import jax

                flat_tangents = jax.vmap(single_random_tangent)(flat_keys)

                # Reshape to desired batch shape
                target_shape = (*shape, self.ambient_dimension)
                return flat_tangents.reshape(target_shape)
        else:
            # Single tangent vector generation
            return self._generate_single_random_tangent(key, x)

    def _generate_single_random_tangent(self, key: PRNGKeyArray, x: Array) -> Array:
        """Generate a single random tangent vector at point x.

        Args:
            key: JAX PRNG key.
            x: Base point on the product manifold.

        Returns:
            Single random tangent vector as flattened array.
        """
        # Split base point into component points
        x_components = self._split_point(x)

        # Split key for each component manifold
        subkeys = jr.split(key, len(self.manifolds))

        # Generate random tangent vector on each component manifold independently
        component_tangents = []
        for manifold, x_comp, subkey in zip(self.manifolds, x_components, subkeys, strict=False):
            component_tangent = manifold.random_tangent(subkey, x_comp)
            component_tangents.append(component_tangent)

        # Combine component tangent vectors into product manifold tangent vector
        return self._combine_points(tuple(component_tangents))

    def __repr__(self) -> str:
        """String representation of the product manifold."""
        component_reprs = [repr(manifold) for manifold in self.manifolds]
        return f"ProductManifold({' x '.join(component_reprs)})"
