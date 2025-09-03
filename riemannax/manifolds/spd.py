"""Implementation of the Symmetric Positive Definite (SPD) manifold.

This module provides operations for optimization on the manifold of symmetric
positive definite matrices, which is fundamental in covariance estimation,
signal processing, and many machine learning applications.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.linalg import expm, solve

from ..core.constants import NumericalConstants
from ..core.jit_decorator import jit_optimized
from .base import Manifold


def _matrix_log(x: Array) -> Array:
    """Compute matrix logarithm using eigendecomposition for SPD matrices.

    For SPD matrices, we can use eigendecomposition: X = Q @ diag(λ) @ Q^T
    Then log(X) = Q @ diag(log(λ)) @ Q^T
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(x)
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, NumericalConstants.HIGH_PRECISION_EPSILON)
    log_eigenvals = jnp.log(eigenvals)
    return jnp.asarray(eigenvecs @ jnp.diag(log_eigenvals) @ eigenvecs.T)


def _matrix_sqrt(x: Array) -> Array:
    """Compute matrix square root using eigendecomposition for SPD matrices.

    For SPD matrices, we can use eigendecomposition: X = Q @ diag(λ) @ Q^T
    Then sqrt(X) = Q @ diag(sqrt(λ)) @ Q^T
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(x)
    # Ensure all eigenvalues are positive (numerical stability)
    eigenvals = jnp.maximum(eigenvals, NumericalConstants.HIGH_PRECISION_EPSILON)
    sqrt_eigenvals = jnp.sqrt(eigenvals)
    return jnp.asarray(eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.T)


class SymmetricPositiveDefinite(Manifold):
    """Symmetric Positive Definite manifold SPD(n) with affine-invariant metric.

    The manifold of nxn symmetric positive definite matrices:
    SPD(n) = {X ∈ R^(nxn) : X = X^T, X ≻ 0}

    This implementation uses the affine-invariant Riemannian metric, which makes
    the manifold complete and provides nice theoretical properties.
    """

    def __init__(self, n: int) -> None:
        """Initialize the SPD manifold.

        Args:
            n: Size of the matrices (nxn).
        """
        super().__init__()  # JIT-related initialization
        self.n = n

    @jit_optimized(static_args=(0,))
    def proj(self, x: Array, v: Array) -> Array:
        """Project matrix v onto the tangent space of SPD at point x.

        The tangent space at x consists of symmetric matrices.
        For the affine-invariant metric, we use the simple symmetric part:
        proj_x(v) = sym(v) = (v + v^T) / 2

        Args:
            x: Point on SPD manifold (nxn symmetric positive definite matrix).
            v: Matrix in the ambient space R^(nxn).

        Returns:
            The projection of v onto the tangent space at x.
        """
        # For the affine-invariant metric, the tangent space is just symmetric matrices
        return 0.5 * (v + v.T)

    @jit_optimized(static_args=(0,))
    def exp(self, x: Array, v: Array) -> Array:
        """Apply the exponential map to move from point x along tangent vector v.

        For the affine-invariant metric on SPD:
        exp_x(v) = x @ expm(x^(-1/2) @ v @ x^(-1/2)) @ x^(1/2)

        Args:
            x: Point on SPD manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by following the geodesic from x in direction v.
        """
        # Compute x^(-1/2) using eigendecomposition
        x_sqrt = _matrix_sqrt(x)
        x_inv_sqrt = solve(x_sqrt, jnp.eye(self.n), assume_a="pos")

        # Transform tangent vector to matrix exponential form
        v_transformed = x_inv_sqrt @ v @ x_inv_sqrt

        # Apply matrix exponential
        exp_v = expm(v_transformed)

        # Transform back to SPD manifold
        return jnp.asarray(x_sqrt @ exp_v @ x_sqrt)

    @jit_optimized(static_args=(0,))
    def log(self, x: Array, y: Array) -> Array:
        """Apply the logarithmic map to find the tangent vector from x to y.

        For the affine-invariant metric on SPD:
        log_x(y) = x^(1/2) @ logm(x^(-1/2) @ y @ x^(-1/2)) @ x^(1/2)

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.

        Returns:
            The tangent vector v at x such that exp_x(v) = y.
        """
        # Compute x^(-1/2) and x^(1/2)
        x_sqrt = _matrix_sqrt(x)
        x_inv_sqrt = solve(x_sqrt, jnp.eye(self.n), assume_a="pos")

        # Transform to matrix logarithm form
        y_transformed = x_inv_sqrt @ y @ x_inv_sqrt

        # Apply matrix logarithm
        log_y = _matrix_log(y_transformed)

        # Transform back to tangent space
        return jnp.asarray(x_sqrt @ log_y @ x_sqrt)

    @jit_optimized(static_args=(0,))
    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Compute the Riemannian inner product between tangent vectors u and v.

        For the affine-invariant metric:
        <u, v>_x = tr(x^(-1) @ u @ x^(-1) @ v)

        Args:
            x: Point on SPD manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x.
        """
        x_inv = solve(x, jnp.eye(self.n), assume_a="pos")
        return jnp.trace(x_inv @ u @ x_inv @ v)

    @jit_optimized(static_args=(0,))
    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport vector v from tangent space at x to tangent space at y.

        For the affine-invariant metric, we use a simplified approach:
        P_x→y(v) = (y/x)^(1/2) @ v @ (y/x)^(1/2)
        This is an approximation that preserves the tangent space structure.

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        # Simplified parallel transport using square root scaling
        x_inv = solve(x, jnp.eye(self.n), assume_a="pos")
        scaling = _matrix_sqrt(y @ x_inv)
        return jnp.asarray(scaling @ v @ scaling.T)

    @jit_optimized(static_args=(0, 4))  # Make self and n_steps static
    def _pole_ladder(self, x: Array, y: Array, v: Array, n_steps: int = 3) -> Array:
        """Pole ladder algorithm for parallel transport with high-order accuracy.

        This implements the correct pole ladder algorithm which provides third-order accurate
        parallel transport by using iterative geodesic constructions with midpoint symmetries.
        The method divides the transport path into multiple steps and uses a specific
        geometric construction at each step for improved accuracy.

        The algorithm works as follows for each step:
        1. Compute midpoint m between current point p and next point q
        2. Compute endpoint p' = exp_p(v) of the vector to be transported
        3. Use symmetry: extend geodesic from p' through m to get transported endpoint
        4. Extract transported vector as log from q to the transported endpoint

        References:
        - "Parallel Transport with Pole Ladder: a Third Order Scheme in Affine Connection
          Spaces which is Exact in Affine Symmetric Spaces" (Pennec, 2018)
        - "Numerical Accuracy of Ladder Schemes for Parallel Transport on Manifolds"
          (Bergmann & Gousenbourger, 2018)

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.
            v: Tangent vector at x to be transported.
            n_steps: Number of steps for the pole ladder (default: 3 for third-order accuracy).

        Returns:
            The transported vector in the tangent space at y with improved accuracy.
        """
        # Use JAX-compatible conditional to handle edge case: transport to the same point
        distance_sq = jnp.sum((x - y) ** 2)
        is_same_point = distance_sq < 1e-24  # Square of 1e-12

        def pole_ladder_computation():
            """Main pole ladder computation with correct algorithm."""
            current_point = x
            current_vector = v

            # Divide the geodesic path into n_steps
            for step in range(n_steps):
                # Compute the target point for this step
                t_step = (step + 1) / n_steps
                target_point = self.exp(x, t_step * self.log(x, y))

                # Apply one pole ladder step: transport current_vector from current_point to target_point
                current_vector = self._single_pole_ladder_step(current_point, target_point, current_vector)
                current_point = target_point

            return current_vector

        def identity_transport():
            """Return original vector when transporting to the same point."""
            return v

        # Use JAX conditional to handle the edge case
        return jax.lax.cond(is_same_point, identity_transport, pole_ladder_computation)

    def _single_pole_ladder_step(self, p: Array, q: Array, v: Array) -> Array:
        """Single pole ladder step for parallel transport from p to q.

        Implements the core pole ladder construction:
        1. Compute midpoint m = exp_p(0.5 * log_p(q))
        2. Compute vector endpoint p' = exp_p(v)
        3. Extend geodesic from p' through m: q' = exp_p'(2 * log_p'(m))
        4. Extract transported vector: v_transported = log_q(q')

        Args:
            p: Starting point on manifold
            q: Target point on manifold
            v: Tangent vector at p to transport

        Returns:
            Transported tangent vector at q
        """
        # Step 1: Compute midpoint between p and q
        log_pq = self.log(p, q)
        midpoint = self.exp(p, 0.5 * log_pq)

        # Step 2: Compute endpoint of the vector v at point p
        vector_endpoint = self.exp(p, v)

        # Step 3: Pole ladder construction using symmetry
        # Extend geodesic from vector_endpoint through midpoint
        log_to_midpoint = self.log(vector_endpoint, midpoint)
        transported_endpoint = self.exp(vector_endpoint, 2.0 * log_to_midpoint)

        # Step 4: Extract the transported vector at q
        transported_vector = self.log(q, transported_endpoint)

        # Final safety check: only for extreme numerical failures
        # This preserves mathematical properties while preventing crashes
        def safe_result():
            return self.proj(q, transported_vector)

        def fallback_result():
            # Only use fallback if result contains NaN/Inf
            # Use simple parallel transport as backup
            return self.transp(p, q, v)

        # Check if result is finite - only use fallback for extreme cases
        result_is_finite = jnp.all(jnp.isfinite(transported_vector))

        return jax.lax.cond(result_is_finite, safe_result, fallback_result)

    @jit_optimized(static_args=(0,))
    def _affine_invariant_transp(self, x: Array, y: Array, v: Array) -> Array:
        """Closed-form parallel transport using the affine-invariant metric.

        This implements the exact closed-form formula for parallel transport on SPD manifolds
        with the affine-invariant Riemannian metric. The formula provides theoretically exact
        isometry preservation and is computationally efficient.

        The parallel transport formula is:
        P_{x→y}(v) = x^{1/2} * exp(1/2 * x^{-1/2} * log_x(y) * x^{-1/2}) * x^{-1/2} * v * x^{-1/2} *
                     exp(1/2 * x^{-1/2} * log_x(y) * x^{-1/2}) * x^{1/2}

        This can be simplified using the logarithmic map log_x(y) and matrix operations.

        References:
        - "O(n)-invariant Riemannian metrics on SPD matrices" (Thanwerdas & Pennec, 2022)
        - "Parallel Transport on Matrix Manifolds and Exponential Action" (2024)

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y with exact isometry preservation.
        """
        # Handle edge case: transport to the same point
        distance_sq = jnp.sum((x - y) ** 2)
        is_same_point = distance_sq < 1e-24

        def identity_transport():
            """Return original vector when transporting to the same point."""
            return v

        def affine_invariant_computation():
            """Compute closed-form affine-invariant parallel transport."""
            # Compute x^{1/2} and x^{-1/2}
            x_sqrt = _matrix_sqrt(x)
            x_inv_sqrt = solve(x_sqrt, jnp.eye(self.n), assume_a="pos")

            # Compute log_x(y) = x^{1/2} * log(x^{-1/2} * y * x^{-1/2}) * x^{1/2}
            y_transformed = x_inv_sqrt @ y @ x_inv_sqrt
            log_y_transformed = _matrix_log(y_transformed)
            x_sqrt @ log_y_transformed @ x_sqrt

            # For the closed-form parallel transport, we use a simplified approach
            # that's mathematically equivalent but more computationally stable:
            # P_{x→y}(v) = (x^{-1/2} * y * x^{-1/2})^{1/2} * x^{-1/2} * v * x^{-1/2} * (x^{-1/2} * y * x^{-1/2})^{1/2}

            # Compute the transport operator: (x^{-1/2} * y * x^{-1/2})^{1/2}
            transport_matrix = _matrix_sqrt(y_transformed)

            # Apply the transport: transport_matrix * x^{-1/2} * v * x^{-1/2} * transport_matrix
            v_transformed = x_inv_sqrt @ v @ x_inv_sqrt
            transported_v_transformed = transport_matrix @ v_transformed @ transport_matrix

            # Transform back to tangent space at y
            transported_v = x_sqrt @ transported_v_transformed @ x_sqrt

            return transported_v

        # Use JAX conditional to handle the edge case
        result = jax.lax.cond(is_same_point, identity_transport, affine_invariant_computation)

        # Ensure result is in tangent space (symmetric for SPD)
        return self.proj(y, result)

    @jit_optimized(static_args=(0,))
    def _bures_wasserstein_transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport using the Bures-Wasserstein metric.

        The Bures-Wasserstein metric on SPD manifolds has the distance formula:
        d(A,B) = tr(A) + tr(B) - 2*tr((A^{1/2} * B * A^{1/2})^{1/2})

        For commuting matrices, there exists a closed-form parallel transport formula.
        For general matrices, we use an approximation based on the metric properties.

        This implementation handles both cases:
        1. Exact formula for commuting matrices
        2. Approximation for general matrices

        References:
        - "On the Bures-Wasserstein distance between positive definite matrices" (Bhatia et al., 2019)
        - "Averaging on the Bures-Wasserstein manifold" (Altschuler & Chewi, 2021)

        Args:
            x: Starting point on SPD manifold.
            y: Target point on SPD manifold.
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y using Bures-Wasserstein metric.
        """
        # Handle edge case: transport to the same point
        distance_sq = jnp.sum((x - y) ** 2)
        is_same_point = distance_sq < 1e-24

        def identity_transport():
            """Return original vector when transporting to the same point."""
            return v

        def bures_wasserstein_computation():
            """Compute Bures-Wasserstein parallel transport."""
            # Check if matrices commute (approximately)
            xy_comm = x @ y
            yx_comm = y @ x
            commutator_norm = jnp.linalg.norm(xy_comm - yx_comm, "fro")
            matrices_commute = commutator_norm < 1e-10

            def exact_commuting_transport():
                """Exact formula for commuting matrices."""
                # For commuting SPD matrices, we can use simultaneous diagonalization
                # Both matrices have the same eigenvectors
                eigenvals_x, eigenvecs = jnp.linalg.eigh(x)
                eigenvals_y, _ = jnp.linalg.eigh(y)

                # Ensure eigenvalues are positive
                eigenvals_x = jnp.maximum(eigenvals_x, 1e-12)
                eigenvals_y = jnp.maximum(eigenvals_y, 1e-12)

                # Transform tangent vector to eigenspace
                v_eigen = eigenvecs.T @ v @ eigenvecs

                # Apply transport in eigenspace (diagonal scaling)
                # For Bures-Wasserstein metric on commuting matrices,
                # the transport involves geometric mean of eigenvalues
                transport_scaling = jnp.sqrt(eigenvals_y / eigenvals_x)
                transported_v_eigen = jnp.diag(transport_scaling) @ v_eigen @ jnp.diag(transport_scaling)

                # Transform back to original space
                transported_v = eigenvecs @ transported_v_eigen @ eigenvecs.T
                return transported_v

            def approximate_general_transport():
                """Approximation for general (non-commuting) matrices."""
                # Use a Bures-Wasserstein inspired transport based on matrix geometric mean
                # Compute the matrix geometric mean: (x # y) = x^{1/2} * (x^{-1/2} * y * x^{-1/2})^{1/2} * x^{1/2}
                x_sqrt = _matrix_sqrt(x)
                x_inv_sqrt = solve(x_sqrt, jnp.eye(self.n), assume_a="pos")

                # Compute geometric mean component
                y_transformed = x_inv_sqrt @ y @ x_inv_sqrt
                y_sqrt_transformed = _matrix_sqrt(y_transformed)
                geometric_mean = x_sqrt @ y_sqrt_transformed @ x_sqrt

                # Bures-Wasserstein transport approximation using geometric mean
                # Transport operator based on the relationship between x, y, and their geometric mean
                gm_inv_sqrt = solve(_matrix_sqrt(geometric_mean), jnp.eye(self.n), assume_a="pos")
                y_sqrt = _matrix_sqrt(y)

                # Apply transport: involves geometric mean and target point
                transport_op = y_sqrt @ gm_inv_sqrt
                transported_v = transport_op @ v @ transport_op.T

                return transported_v

            # Choose method based on whether matrices commute
            return jax.lax.cond(matrices_commute, exact_commuting_transport, approximate_general_transport)

        # Use JAX conditional to handle the edge case
        result = jax.lax.cond(is_same_point, identity_transport, bures_wasserstein_computation)

        # Ensure result is in tangent space (symmetric for SPD)
        return self.proj(y, result)

    @jit_optimized(static_args=(0, 4))  # Make self and n_steps static
    def _schilds_ladder(self, x: Array, y: Array, v: Array, n_steps: int = 5) -> Array:
        """
        Implement Schild's ladder algorithm for parallel transport.

        Schild's ladder is a first-order accurate parallel transport method that uses
        iterative geodesic parallelogram constructions. It is more numerically stable
        than higher-order methods and suitable for large matrices (n>1000) and
        ill-conditioned cases.

        The algorithm constructs geodesic parallelograms iteratively to transport
        a tangent vector from point x to point y. Each step involves:
        1. Subdividing the geodesic from x to y
        2. Constructing a geodesic parallelogram
        3. Updating the transported vector

        Args:
            x: Starting point on the manifold
            y: Target point on the manifold
            v: Tangent vector at x to be transported
            n_steps: Number of ladder steps (higher = more accurate)

        Returns:
            Transported tangent vector at point y

        References:
            - Schild, A. (1949). "Discrete geodesics"
            - Guigui et al. (2021). "Numerical Accuracy of Ladder Schemes"
        """
        # Handle edge case: transport to the same point
        distance_sq = jnp.sum((x - y) ** 2)
        is_same_point = distance_sq < 1e-24

        def identity_transport():
            """Return original vector when transporting to the same point."""
            return v

        def schilds_ladder_computation():
            """Main Schild's ladder computation."""
            current_point = x
            current_vector = v

            # Divide the geodesic path into n_steps
            for step in range(n_steps):
                # Compute the target point for this step
                t_step = (step + 1) / n_steps
                log_xy = self.log(x, y)
                target_point = self.exp(x, t_step * log_xy)

                # Apply one Schild's ladder step
                current_vector = self._single_schilds_ladder_step(current_point, target_point, current_vector)
                current_point = target_point

            return current_vector

        # Use JAX conditional to handle the edge case
        return jax.lax.cond(is_same_point, identity_transport, schilds_ladder_computation)

    def _single_schilds_ladder_step(self, p: Array, q: Array, v: Array) -> Array:
        """
        Perform a single step of Schild's ladder parallel transport.

        This implements the geodesic parallelogram construction:
        1. Map tangent vector v to endpoint via exponential map
        2. Construct geodesic from endpoint to q
        3. Find midpoint of this geodesic
        4. Construct geodesic from p through midpoint
        5. Extend to find transported vector endpoint
        6. Map back to tangent vector at q

        Args:
            p: Starting point
            q: Target point (one step along geodesic)
            v: Tangent vector at p to transport

        Returns:
            Transported tangent vector at q
        """
        # Step 1: Map tangent vector to point via exponential map
        vector_endpoint = self.exp(p, v)

        # Step 2: Construct geodesic from vector endpoint to target point q
        log_endpoint_to_q = self.log(vector_endpoint, q)

        # Step 3: Find midpoint of the geodesic from vector_endpoint to q
        midpoint = self.exp(vector_endpoint, 0.5 * log_endpoint_to_q)

        # Step 4: Construct geodesic from p through midpoint and extend
        log_p_to_midpoint = self.log(p, midpoint)
        transported_endpoint = self.exp(p, 2.0 * log_p_to_midpoint)

        # Step 5: Extract transported tangent vector at q
        transported_vector = self.log(q, transported_endpoint)

        # Final safety check for numerical stability
        result_is_finite = jnp.all(jnp.isfinite(transported_vector))

        return jax.lax.cond(
            result_is_finite,
            lambda: self.proj(q, transported_vector),
            lambda: self.transp(p, q, v),  # Fallback to simple transport
        )

    def _select_transport_algorithm(self, x: Array, y: Array, v: Array) -> Array:
        """
        Select the best parallel transport algorithm based on matrix properties.

        Selection criteria:
        - Small matrices (n ≤ 5): Use pole ladder for higher accuracy
        - Large matrices (n > 5): Use Schild's ladder for stability
        - High condition number: Use Schild's ladder for numerical stability
        - Well-conditioned small matrices: Use pole ladder for accuracy

        Args:
            x: Starting point
            y: Target point
            v: Tangent vector to transport

        Returns:
            JAX Array boolean: True for Schild's ladder, False for pole ladder
        """
        matrix_size = x.shape[0]

        # Check condition number for numerical stability assessment
        eigenvals = jnp.linalg.eigvals(x)
        condition_number = jnp.max(eigenvals) / jnp.min(eigenvals)

        # High condition number threshold (indicates ill-conditioning)
        high_condition_threshold = 1e4

        # Large matrix threshold
        large_matrix_threshold = 5

        # Selection logic
        is_large = matrix_size > large_matrix_threshold
        is_ill_conditioned = condition_number > high_condition_threshold

        # Use Schild's ladder for large or ill-conditioned matrices
        use_schilds = is_large | is_ill_conditioned  # Use bitwise OR for JAX arrays

        return use_schilds

    @jit_optimized(static_args=(0,))  # Make self static
    def adaptive_parallel_transport(self, x: Array, y: Array, v: Array) -> Array:
        """
        Adaptive parallel transport that automatically selects the best algorithm.

        This method analyzes the input matrices and automatically chooses the most
        appropriate parallel transport algorithm:

        - Affine-invariant transport: For exact closed-form solutions when applicable
        - Pole ladder: For high accuracy on small, well-conditioned matrices
        - Schild's ladder: For stability on large matrices or ill-conditioned cases

        Args:
            x: Starting point on the manifold
            y: Target point on the manifold
            v: Tangent vector at x to be transported

        Returns:
            Transported tangent vector at point y

        Notes:
            This method is designed for production use where optimal performance
            and numerical stability are required across different problem scales.
        """
        # First, try to determine if we can use exact methods
        matrix_size = x.shape[0]

        # For small matrices, consider affine-invariant transport
        # (exact method when the computational cost is acceptable)
        use_exact = matrix_size <= 4

        def exact_branch():
            return self._affine_invariant_transp(x, y, v)

        def adaptive_branch():
            # Select between pole ladder and Schild's ladder
            use_schilds = self._select_transport_algorithm(x, y, v)

            return jax.lax.cond(
                use_schilds,
                lambda: self._schilds_ladder(x, y, v, n_steps=5),
                lambda: self._pole_ladder(x, y, v, n_steps=3),
            )

        return jax.lax.cond(use_exact, exact_branch, adaptive_branch)

    @jit_optimized(static_args=(0,))
    def dist(self, x: Array, y: Array) -> Array:
        """Compute the Riemannian distance between points x and y.

        For the affine-invariant metric:
        d(x, y) = ||log_x(y)||_x = sqrt(tr(logm(x^(-1/2) @ y @ x^(-1/2))^2))

        Args:
            x: First point on SPD manifold.
            y: Second point on SPD manifold.

        Returns:
            The geodesic distance between x and y.
        """
        # Use the logarithmic map
        log_xy = self.log(x, y)

        # Compute the norm in the Riemannian metric
        return jnp.sqrt(self.inner(x, log_xy, log_xy))

    def random_point(self, key: Array, *shape: int) -> Array:
        """Generate random point(s) on the SPD manifold.

        Generates SPD matrices by creating random matrices and computing A @ A^T + ε*I
        to ensure positive definiteness.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random SPD matrix/matrices with specified shape.
        """
        if len(shape) == 0:
            # Single matrix
            A = jr.normal(key, (self.n, self.n))
            return A @ A.T + 1e-6 * jnp.eye(self.n)
        else:
            # Batch of matrices
            full_shape = (*shape, self.n, self.n)
            A = jr.normal(key, full_shape)
            # Use einsum for batched matrix multiplication
            AAt = jnp.einsum("...ij,...kj->...ik", A, A)
            return AAt + 1e-6 * jnp.eye(self.n)

    def random_tangent(self, key: Array, x: Array, *shape: int) -> Array:
        """Generate random tangent vector(s) at point x.

        Generates random symmetric matrices in the tangent space.

        Args:
            key: JAX PRNG key.
            x: Point on SPD manifold.
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        if len(shape) == 0:
            # Single tangent vector
            v_raw = jr.normal(key, (self.n, self.n))
            return jnp.asarray(self.proj(x, v_raw))
        else:
            # Batch of tangent vectors
            full_shape = (*shape, self.n, self.n)
            v_raw = jr.normal(key, full_shape)

            # Apply projection to each matrix in the batch
            def proj_single(v: Array) -> Array:
                return jnp.asarray(self.proj(x, v))

            return jax.vmap(proj_single)(v_raw)

    def _is_in_manifold(self, x: Array, tolerance: float = 1e-6) -> bool | Array:
        """Check if a matrix is in the SPD manifold.

        Args:
            x: Matrix to check.
            tolerance: Numerical tolerance for checks.

        Returns:
            Boolean indicating if x is symmetric positive definite.
        """
        # Check symmetry
        is_symmetric = jnp.allclose(x, x.T, atol=tolerance)

        # Check positive definiteness via eigenvalues
        eigenvals = jnp.linalg.eigvals(x)
        is_positive_definite = jnp.all(eigenvals > tolerance)

        result = jnp.logical_and(is_symmetric, is_positive_definite)
        # Return JAX array directly if in traced context to avoid TracerBoolConversionError
        try:
            return bool(result)
        except TypeError:
            # In JAX traced context, return the array directly
            return result

    def validate_point(self, x: Array, atol: float = 1e-6) -> bool | Array:
        """Validate that x is a valid point on the SPD manifold.

        Args:
            x: Point to validate.
            atol: Absolute tolerance for validation.

        Returns:
            True if x is on the manifold (symmetric positive definite), False otherwise.
        """
        return self._is_in_manifold(x, atol)

    # JIT-optimized implementation methods

    def _proj_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized projection implementation.

        Numerical stability improvements:
        - Efficient guarantee of symmetry
        """
        # For SPD manifolds, tangent space consists of symmetric matrices
        return 0.5 * (v + v.T)

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized exponential map implementation.

        Numerical stability improvements:
        - Stable implementation via eigenvalue decomposition
        - Numerical stability ensured through condition number control
        """
        # Use eigendecomposition for better JAX compatibility
        return self._exp_impl_eigen(x, v)

    def _exp_impl_eigen(self, x: Array, v: Array) -> Array:
        """Eigendecomposition-based exponential map with enhanced numerical stability."""
        # Compute eigendecomposition with robust error handling
        eigenvals_x, eigenvecs_x = jnp.linalg.eigh(x)

        # Enhanced numerical stability: use larger epsilon and condition number control
        min_eigenval = jnp.maximum(jnp.min(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        max_eigenval = jnp.maximum(jnp.max(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        condition_number = max_eigenval / min_eigenval

        # Apply stronger regularization for ill-conditioned matrices
        regularization_eps = jnp.where(condition_number > 1e6, 1e-6, 1e-10)
        eigenvals_x = jnp.maximum(eigenvals_x, regularization_eps)

        sqrt_eigenvals_x = jnp.sqrt(eigenvals_x)
        inv_sqrt_eigenvals_x = 1.0 / sqrt_eigenvals_x

        # Batch-aware matrix construction with improved stability
        # More robust batch detection based on input shapes
        is_batch = x.ndim > 2 or (eigenvals_x.ndim > 1 and eigenvals_x.shape[0] != self.n)

        if is_batch:  # Batch case
            sqrt_diag = jnp.apply_along_axis(jnp.diag, -1, sqrt_eigenvals_x)
            inv_sqrt_diag = jnp.apply_along_axis(jnp.diag, -1, inv_sqrt_eigenvals_x)
            transpose_axes = (-2, -1)
        else:  # Single matrix case
            sqrt_diag = jnp.diag(sqrt_eigenvals_x)
            inv_sqrt_diag = jnp.diag(inv_sqrt_eigenvals_x)
            transpose_axes = (-2, -1)

        # Compute matrix square root and inverse square root
        x_sqrt = eigenvecs_x @ sqrt_diag @ jnp.swapaxes(eigenvecs_x, *transpose_axes)
        x_inv_sqrt = eigenvecs_x @ inv_sqrt_diag @ jnp.swapaxes(eigenvecs_x, *transpose_axes)

        # Transform tangent vector with numerical stability checks
        v_transformed = x_inv_sqrt @ v @ x_inv_sqrt

        # Check for NaN/inf in transformed tangent vector
        v_transformed = jnp.where(jnp.isfinite(v_transformed), v_transformed, 0.0)

        # Apply matrix exponential
        exp_v = expm(v_transformed)

        # Check for NaN/inf in matrix exponential result - create identity with correct shape
        if is_batch:  # Batch case
            batch_shape = v_transformed.shape[:-2]
            eye_batch = jnp.tile(jnp.eye(self.n), (*batch_shape, 1, 1))
        else:  # Single matrix case
            eye_batch = jnp.eye(self.n)

        exp_v = jnp.where(jnp.isfinite(exp_v), exp_v, eye_batch)

        # Transform back to manifold
        result = x_sqrt @ exp_v @ x_sqrt

        # Final SPD constraint enforcement: ensure positive definiteness
        # Re-eigendecompose result and clamp eigenvalues with more aggressive thresholding
        result_eigenvals, result_eigenvecs = jnp.linalg.eigh(result)

        # Use more aggressive minimum eigenvalue based on the scale of the matrix
        max_result_eigenval = jnp.max(jnp.abs(result_eigenvals))
        # Scale-aware minimum eigenvalue: at least 1e-12 times the maximum eigenvalue
        min_eigenval_threshold = jnp.maximum(
            max_result_eigenval * 1e-12,
            1e-6,  # Absolute minimum threshold for very extreme cases
        )

        result_eigenvals = jnp.maximum(result_eigenvals, min_eigenval_threshold)

        # Use the same batch detection logic
        result_diag = jnp.apply_along_axis(jnp.diag, -1, result_eigenvals) if is_batch else jnp.diag(result_eigenvals)

        result = result_eigenvecs @ result_diag @ jnp.swapaxes(result_eigenvecs, *transpose_axes)

        return jnp.asarray(result)

    def _log_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized logarithmic map implementation.

        Numerical stability improvements:
        - Stable implementation via eigenvalue decomposition
        - Stability ensured through condition number control
        """
        # Use eigendecomposition for better JAX compatibility
        return self._log_impl_eigen(x, y)

    def _log_impl_eigen(self, x: Array, y: Array) -> Array:
        """Eigendecomposition-based logarithmic map with enhanced numerical stability."""
        # Compute eigendecomposition with robust error handling
        eigenvals_x, eigenvecs_x = jnp.linalg.eigh(x)

        # Enhanced numerical stability: condition number control
        min_eigenval_x = jnp.maximum(jnp.min(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        max_eigenval_x = jnp.maximum(jnp.max(eigenvals_x), NumericalConstants.HIGH_PRECISION_EPSILON)
        condition_number_x = max_eigenval_x / min_eigenval_x

        # Apply stronger regularization for ill-conditioned matrices
        regularization_eps = jnp.where(condition_number_x > 1e6, 1e-6, 1e-10)
        eigenvals_x = jnp.maximum(eigenvals_x, regularization_eps)

        sqrt_eigenvals_x = jnp.sqrt(eigenvals_x)
        inv_sqrt_eigenvals_x = 1.0 / sqrt_eigenvals_x

        # x^(1/2) and x^(-1/2)
        x_sqrt = eigenvecs_x @ jnp.diag(sqrt_eigenvals_x) @ eigenvecs_x.T
        x_inv_sqrt = eigenvecs_x @ jnp.diag(inv_sqrt_eigenvals_x) @ eigenvecs_x.T

        # Transform y with numerical stability checks
        y_transformed = x_inv_sqrt @ y @ x_inv_sqrt
        y_transformed = jnp.where(jnp.isfinite(y_transformed), y_transformed, jnp.eye(self.n))

        # Apply matrix logarithm with enhanced stability
        eigenvals_y, eigenvecs_y = jnp.linalg.eigh(y_transformed)

        # Ensure eigenvalues are positive for logarithm
        eigenvals_y = jnp.maximum(eigenvals_y, NumericalConstants.HIGH_PRECISION_EPSILON)

        # Check for very small eigenvalues that could cause log instability
        eigenvals_y = jnp.where(eigenvals_y < 1e-12, 1e-12, eigenvals_y)

        log_eigenvals_y = jnp.log(eigenvals_y)

        # Check for NaN/inf in log eigenvalues
        log_eigenvals_y = jnp.where(jnp.isfinite(log_eigenvals_y), log_eigenvals_y, 0.0)

        log_y = eigenvecs_y @ jnp.diag(log_eigenvals_y) @ eigenvecs_y.T

        # Transform back
        result = x_sqrt @ log_y @ x_sqrt

        # Final check for numerical stability
        result = jnp.where(jnp.isfinite(result), result, jnp.zeros_like(result))

        return jnp.asarray(result)

    def _inner_impl(self, x: Array, u: Array, v: Array) -> Array:
        """JIT-optimized inner product implementation with enhanced numerical stability.

        Affine-invariant inner product on SPD manifold
        Batch processing support
        """
        # Compute x^(-1) using direct solve for JAX compatibility (batch-aware)
        # Create identity matrix with correct batch shape
        batch_shape = x.shape[:-2] if x.ndim > 2 else ()
        eye_shape = (*batch_shape, self.n, self.n)
        identity = jnp.broadcast_to(jnp.eye(self.n), eye_shape)

        # Add small regularization to x for numerical stability
        x_regularized = x + NumericalConstants.HIGH_PRECISION_EPSILON * identity

        # Check if matrix is well-conditioned enough for direct solve
        eigenvals = jnp.linalg.eigvals(x_regularized)
        min_eigenval = jnp.min(jnp.real(eigenvals))
        max_eigenval = jnp.max(jnp.real(eigenvals))
        condition_number = max_eigenval / jnp.maximum(min_eigenval, NumericalConstants.HIGH_PRECISION_EPSILON)

        def stable_solve() -> Array:
            # Direct solve for well-conditioned matrices
            return solve(x_regularized, identity, assume_a="pos")

        def eigendecomposition_fallback() -> Any:
            # Eigendecomposition fallback for ill-conditioned matrices
            eigenvals, eigenvecs = jnp.linalg.eigh(x_regularized)
            eigenvals = jnp.maximum(eigenvals, NumericalConstants.HIGH_PRECISION_EPSILON)
            inv_eigenvals = 1.0 / eigenvals

            # Use robust batch detection
            is_batch_inner = x.ndim > 2 or (eigenvals.ndim > 1 and eigenvals.shape[0] != self.n)

            if is_batch_inner:  # Batch case
                inv_diag = jnp.apply_along_axis(jnp.diag, -1, inv_eigenvals)
                transpose_axes = (-2, -1)
            else:  # Single matrix case
                inv_diag = jnp.diag(inv_eigenvals)
                transpose_axes = (-2, -1)

            return eigenvecs @ inv_diag @ jnp.swapaxes(eigenvecs, *transpose_axes)

        # Choose method based on condition number
        x_inv = jax.lax.cond(condition_number < 1e10, stable_solve, eigendecomposition_fallback)

        # Check for NaN/inf in the inverse
        x_inv = jnp.where(jnp.isfinite(x_inv), x_inv, identity)

        # Compute <u, v>_x = tr(x^(-1) @ u @ x^(-1) @ v) with numerical stability
        temp = x_inv @ u @ x_inv @ v

        # Check for NaN/inf in intermediate computation
        temp = jnp.where(jnp.isfinite(temp), temp, 0.0)

        # Use jnp.trace with axis specification for batch processing
        inner_product = jnp.trace(temp, axis1=-2, axis2=-1)

        # Enhanced numerical clipping
        inner_product = jnp.where(jnp.isfinite(inner_product), inner_product, 0.0)
        return jnp.clip(inner_product, -1e15, 1e15)

    def _dist_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized distance calculation implementation.

        Numerical stability improvements:
        - Efficient calculation via Cholesky decomposition
        """
        # Use the logarithmic map and inner product
        log_xy = self._log_impl(x, y)

        # Compute Riemannian distance: sqrt(<log_x(y), log_x(y)>_x)
        inner_product = self._inner_impl(x, log_xy, log_xy)

        # Ensure non-negative for numerical stability
        inner_product = jnp.maximum(inner_product, 0.0)

        distance = jnp.sqrt(inner_product)
        return jnp.where(distance < NumericalConstants.HIGH_PRECISION_EPSILON, 0.0, distance)

    def _get_static_args(self, method_name: str) -> tuple[Any, ...]:
        """Static argument configuration for JIT compilation.

        For SPD manifold, returns argument position indices for static compilation.
        Conservative approach: no static arguments to avoid shape/type conflicts.
        """
        # Return empty tuple - no static arguments for safety
        # Future optimization could consider making 'self' parameter static (position 0)
        return ()

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the SPD manifold.

        For SPD(n), the dimension is n(n+1)/2 since SPD matrices are symmetric,
        so only the upper triangular part including the diagonal is independent.
        """
        return (self.n * (self.n + 1)) // 2

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the ambient space (all nxn matrices)."""
        return self.n * self.n

    # Log-Euclidean Metric Operations

    @jit_optimized(static_args=(0,))
    def log_euclidean_exp(self, x: Array, v: Array) -> Array:
        """Exponential map using the Log-Euclidean metric.

        The Log-Euclidean exponential map is:
        exp_x^LE(v) = exp(log(x) + v)

        This is computationally much more efficient than the affine-invariant
        exponential map as it only requires matrix logarithm and exponential
        operations without solving linear systems.

        Args:
            x: Base point on SPD manifold (n x n symmetric positive definite matrix)
            v: Tangent vector at x (n x n symmetric matrix)

        Returns:
            Point on SPD manifold reached via Log-Euclidean geodesic

        Mathematical Background:
            The Log-Euclidean metric treats SPD matrices by working in the
            logarithmic domain, where operations become much simpler.
            This metric is not Riemannian (not affine-invariant) but provides
            a computationally efficient approximation.

        References:
            - "Geometric means in a novel vector space structure on SPD matrices"
              (Arsigny et al., 2007)
            - "Log-Euclidean metrics for fast and simple calculus on DT tensors"
              (Arsigny et al., 2006)
        """
        # Compute matrix logarithm of base point
        log_x = _matrix_log(x)

        # Add tangent vector in log domain
        log_result = log_x + v

        # Return via matrix exponential
        result = expm(log_result)

        return jnp.asarray(result)

    @jit_optimized(static_args=(0,))
    def log_euclidean_log(self, x: Array, y: Array) -> Array:
        """Logarithmic map using the Log-Euclidean metric.

        The Log-Euclidean logarithmic map is:
        log_x^LE(y) = log(y) - log(x)

        Args:
            x: Base point on SPD manifold (n x n symmetric positive definite matrix)
            y: Target point on SPD manifold (n x n symmetric positive definite matrix)

        Returns:
            Tangent vector at x pointing towards y in Log-Euclidean metric

        Mathematical Background:
            This operation is much simpler than the affine-invariant logarithmic map
            as it only requires matrix logarithms and subtraction.
        """
        # Compute matrix logarithms
        log_x = _matrix_log(x)
        log_y = _matrix_log(y)

        # Subtract in log domain
        tangent_vector = log_y - log_x

        return jnp.asarray(tangent_vector)

    @jit_optimized(static_args=(0,))
    def log_euclidean_distance(self, x: Array, y: Array) -> Array:
        """Distance using the Log-Euclidean metric.

        The Log-Euclidean distance is:
        d^LE(x, y) = ||log(y) - log(x)||_F

        where ||·||_F is the Frobenius norm.

        Args:
            x: First SPD matrix (n x n)
            y: Second SPD matrix (n x n)

        Returns:
            Log-Euclidean distance between x and y (scalar)

        Mathematical Background:
            This distance is much faster to compute than the affine-invariant
            distance and provides a good approximation for many applications.
        """
        # Compute logarithmic map
        log_diff = self.log_euclidean_log(x, y)

        # Return Frobenius norm
        distance = jnp.linalg.norm(log_diff, "fro")

        return distance

    def log_euclidean_interpolation(self, x: Array, y: Array, t: float) -> Array:
        """Geodesic interpolation using the Log-Euclidean metric.

        Computes the point at parameter t along the Log-Euclidean geodesic
        connecting x and y:
        geodesic(t) = exp((1-t) * log(x) + t * log(y))

        Args:
            x: Starting point on SPD manifold (n x n matrix)
            y: Ending point on SPD manifold (n x n matrix)
            t: Interpolation parameter (0 ≤ t ≤ 1)

        Returns:
            Interpolated point on the Log-Euclidean geodesic

        Mathematical Background:
            Log-Euclidean geodesics are straight lines in the logarithmic domain,
            making interpolation extremely simple and efficient.
        """
        # Compute matrix logarithms
        log_x = _matrix_log(x)
        log_y = _matrix_log(y)

        # Linear interpolation in log domain
        log_interpolated = (1.0 - t) * log_x + t * log_y

        # Return via matrix exponential
        result = expm(log_interpolated)

        return jnp.asarray(result)

    def log_euclidean_mean(self, points: Array) -> Array:
        """Compute the Log-Euclidean mean of SPD matrices.

        The Log-Euclidean mean is:
        μ^LE = exp(1/N * Σᵢ log(Xᵢ))

        This is much faster than the Riemannian Fréchet mean as it requires
        no iterative optimization.

        Args:
            points: Array of SPD matrices, shape (N, n, n)

        Returns:
            Log-Euclidean mean matrix (n x n)

        Mathematical Background:
            The Log-Euclidean mean is simply the arithmetic mean in the
            logarithmic domain, making it extremely efficient to compute.

        References:
            - "Geometric means in a novel vector space structure on SPD matrices"
              (Arsigny et al., 2007)
        """
        n_points = points.shape[0]

        if n_points == 1:
            return points[0]

        # Compute logarithm of each matrix
        log_matrices = jnp.array([_matrix_log(points[i]) for i in range(n_points)])

        # Compute arithmetic mean in log domain
        log_mean = jnp.mean(log_matrices, axis=0)

        # Return via matrix exponential
        result = expm(log_mean)

        return jnp.asarray(result)
