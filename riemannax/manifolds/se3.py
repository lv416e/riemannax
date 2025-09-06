"""SE(3) Special Euclidean Group manifold implementation.

This module implements the SE(3) manifold, which represents rigid transformations
in 3D space combining rotations and translations. Points are parameterized using
unit quaternions (qw, qx, qy, qz) for rotation and (x, y, z) for translation.

SE(3) = SO(3) ⋉ R³ represents the semidirect product of 3D rotations and translations.
The quaternion follows the wxyz convention where w is the scalar (real) part.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from riemannax.manifolds.base import Manifold, ManifoldPoint, TangentVector


class SE3(Manifold):
    """SE(3) Special Euclidean Group manifold.
    
    Represents rigid transformations in 3D space as the semidirect product
    SE(3) = SO(3) ⋉ R³. Points are parameterized using unit quaternions for 
    rotation and 3D vectors for translation: (qw, qx, qy, qz, x, y, z).
    
    The quaternion follows the convention (w, x, y, z) where w is the real part.
    """
    
    def __init__(self, atol: float = 1e-8) -> None:
        """Initialize SE(3) manifold with quaternion + translation parameterization.
        
        Args:
            atol: Absolute tolerance for numerical validation and operations.
        """
        super().__init__()
        self.atol = atol
        
    @property
    def dimension(self) -> int:
        """Intrinsic dimension of SE(3) manifold."""
        return 6  # 3 for rotation (SO(3)) + 3 for translation
        
    @property 
    def ambient_dimension(self) -> int:
        """Dimension of ambient representation space."""
        return 7  # 4 for quaternion + 3 for translation
        
    def _quaternion_normalize(self, q: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
        """Normalize quaternion with numerical stability.
        
        Ensures quaternion has unit norm while handling edge cases robustly.
        For zero quaternions, defaults to identity quaternion (1, 0, 0, 0).
        
        Args:
            q: Quaternion array of shape (..., 4) in (w, x, y, z) format.
            
        Returns:
            Normalized quaternion with unit norm.
        """
        norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
        
        # Prevent division by zero with safe normalization threshold
        safe_norm = jnp.maximum(norm, 1e-12)
        normalized = q / safe_norm
        
        # Replace invalid results with identity quaternion
        is_degenerate = norm < 1e-12
        identity_q = jnp.array([1.0, 0.0, 0.0, 0.0])
        
        # Handle batch case efficiently
        if q.ndim > 1:
            identity_q = jnp.broadcast_to(identity_q, q.shape)
            
        result = jnp.where(is_degenerate, identity_q, normalized)
        return result
        
    def random_point(self, key: PRNGKeyArray, *shape: int) -> Float[Array, "... 7"]:
        """Generate random SE(3) transform(s).
        
        Generates uniformly distributed random rotations using quaternion representation
        and normally distributed translations.
        
        Args:
            key: JAX PRNG key for random generation.
            *shape: Additional shape dimensions for batch generation.
            
        Returns:
            Random SE(3) transform(s) with shape (*shape, 7) where each transform
            is parameterized as (qw, qx, qy, qz, x, y, z).
        """
        # Split key for quaternion and translation generation
        key_q, key_t = jax.random.split(key)
        
        if shape:
            # Batch generation
            batch_shape = shape
            
            # Generate random quaternions from normal distribution and normalize
            quaternions = jax.random.normal(key_q, batch_shape + (4,))
            quaternions = self._quaternion_normalize(quaternions)
            
            # Generate random translations
            translations = jax.random.normal(key_t, batch_shape + (3,))
            
            # Combine quaternion and translation efficiently
            points = jnp.concatenate([quaternions, translations], axis=-1)
        else:
            # Single point generation - more efficient path
            quaternion = jax.random.normal(key_q, (4,))
            quaternion = self._quaternion_normalize(quaternion)
            
            translation = jax.random.normal(key_t, (3,))
            
            points = jnp.concatenate([quaternion, translation])
            
        return points
            
    def validate_point(self, x: Float[Array, "... 7"], atol: float = None) -> bool | Array:
        """Validate that x is a valid SE(3) transform.
        
        Checks that the quaternion part has unit norm, which is the primary
        constraint for SE(3) representations. Translation part is unconstrained.
        
        Args:
            x: Point to validate with shape (..., 7).
            atol: Absolute tolerance for validation. Uses self.atol if None.
            
        Returns:
            True if x is valid SE(3) transform, False otherwise.
            For batched input, returns boolean array.
        """
        if atol is None:
            atol = self.atol
            
        # Check shape constraint
        if x.shape[-1] != 7:
            return False
            
        # Extract quaternion part (first 4 components) 
        quaternion = x[..., :4]
        
        # Validate quaternion normalization
        q_norm = jnp.linalg.norm(quaternion, axis=-1)
        is_normalized = jnp.abs(q_norm - 1.0) <= atol
        
        # Return result compatible with JAX transformations
        try:
            # Handle both scalar and batch cases
            return bool(jnp.all(is_normalized)) if quaternion.ndim > 1 else bool(is_normalized)
        except TypeError:
            # In JAX traced context, return array directly
            return jnp.all(is_normalized) if quaternion.ndim > 1 else is_normalized
            
    def __repr__(self) -> str:
        """String representation of SE(3) manifold."""
        return f"SE3(atol={self.atol})"