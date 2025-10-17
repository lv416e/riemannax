"""Problem templates for common Riemannian optimization tasks.

This module provides scikit-learn compatible problem templates that solve
specific optimization problems using RiemannAX's manifold optimization capabilities.
"""

import contextlib
import math
import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from riemannax.core.constants import NumericalConstants
from riemannax.manifolds import Sphere, Stiefel, SymmetricPositiveDefinite
from riemannax.manifolds.base import Manifold

try:
    from sklearn.base import BaseEstimator, TransformerMixin

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Mock classes when sklearn not available
    class BaseEstimator:
        """Mock BaseEstimator."""

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Get parameters."""
            return {}

        def set_params(self, **params: Any) -> "BaseEstimator":
            """Set parameters."""
            return self

    class TransformerMixin:
        """Mock TransformerMixin."""

        pass


def _project_onto_manifold(manifold: Manifold, point: Array) -> Array:
    """Project a point onto the manifold.

    Utility function to centralize manifold projection logic for supported manifolds.

    Parameters
    ----------
    manifold : Manifold
        The Riemannian manifold to project onto.
    point : Array
        Point to project (may be off-manifold).

    Returns:
    -------
    projected : Array
        Projected point on the manifold.

    Raises:
    ------
    NotImplementedError
        If the manifold type is not supported.
    """
    # Manifold-specific projections (explicit implementations only)
    if isinstance(manifold, Stiefel):
        # For Stiefel: use QR decomposition to get orthonormal columns
        Q, R = jnp.linalg.qr(point)
        # Ensure positive diagonal elements for uniqueness
        signs = jnp.sign(jnp.diag(R))
        signs = jnp.where(signs == 0, 1, signs)
        # Scale columns directly; avoids forming diag and matmul
        return Q * signs

    elif isinstance(manifold, SymmetricPositiveDefinite):
        # For SPD: symmetrize and correct eigenvalues
        symmetric = (point + point.T) / 2.0
        eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric)
        eigenvalues = jnp.maximum(eigenvalues, NumericalConstants.MEDIUM_PRECISION_EPSILON)  # Ensure positive
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    elif isinstance(manifold, Sphere):
        # For Sphere: normalize to unit norm with zero-norm safeguard
        norm = jnp.linalg.norm(point)
        canonical = jnp.zeros_like(point).at[0].set(1.0)
        return jnp.where(norm > NumericalConstants.EPSILON, point / norm, canonical)

    # No projection method available
    raise NotImplementedError(
        f"Manifold {manifold.__class__.__name__} does not have a known projection method. "
        f"Implement manifold-specific projection. "
        f"Note: proj(x, v) is for tangent space projection (vectors), not manifold projection (points)."
    )


def _validate_spd_batch(X: Array) -> None:
    """Validate a batch of SPD matrices.

    Checks that all matrices in the batch are symmetric and positive definite.

    Parameters
    ----------
    X : Array of shape (n_samples, n, n)
        Batch of matrices to validate.

    Raises:
    ------
    ValueError
        If any matrix is not symmetric or not positive definite, with the index
        of the first failing matrix.
    """
    # Check symmetry (vectorized)
    is_symmetric_per_matrix = jnp.all(
        jnp.isclose(X, jnp.transpose(X, (0, 2, 1)), atol=NumericalConstants.SYMMETRY_TOLERANCE),
        axis=(-2, -1),
    )
    if not bool(jax.device_get(jnp.all(is_symmetric_per_matrix))):
        failed_index = int(jax.device_get(jnp.argmin(is_symmetric_per_matrix)))
        raise ValueError(f"All matrices must be symmetric. Matrix {failed_index} is not symmetric.")

    # Check positive definiteness (vectorized)
    all_eigenvalues = jnp.linalg.eigvalsh(X)
    eps = NumericalConstants.MEDIUM_PRECISION_EPSILON
    is_positive_definite_per_matrix = jnp.all(all_eigenvalues > eps, axis=1)
    if not bool(jax.device_get(jnp.all(is_positive_definite_per_matrix))):
        failed_index = int(jax.device_get(jnp.argmin(is_positive_definite_per_matrix)))
        raise ValueError(f"All matrices must be SPD. Matrix {failed_index} has non-positive eigenvalues.")


class _MatrixCompletionOptState(NamedTuple):
    """Optimization state for MatrixCompletion._optimize().

    Attributes:
    ----------
    U : Array
        Left factor matrix.
    V : Array
        Right factor matrix.
    error : Array
        Current reconstruction error.
    U_prev : Array
        Previous left factor matrix.
    V_prev : Array
        Previous right factor matrix.
    error_prev : Array
        Previous reconstruction error.
    iteration : int
        Current iteration number.
    converged : Array
        Convergence flag (JAX boolean).
    """

    U: Array
    V: Array
    error: Array
    U_prev: Array
    V_prev: Array
    error_prev: Array
    iteration: int
    converged: Array


class MatrixCompletion(BaseEstimator, TransformerMixin):
    """Low-rank matrix completion using gradient descent on low-rank factors.

    Solves the matrix completion problem by finding a low-rank factorization
    X ≈ UV^T that minimizes the reconstruction error on observed entries.
    Uses gradient descent on the low-rank factors U and V (factor-space optimization),
    a common and effective approach that optimizes in the Euclidean factor space.

    The objective function is:
        minimize ||P_Ω(X - UV^T)||_F^2
    where Ω is the set of observed entries indicated by the mask.

    Parameters
    ----------
    rank : int
        Rank of the low-rank factorization. Must be positive and not exceed
        min(m, n) where X is m x n.
    max_iter : int, default=100
        Maximum number of optimization iterations.
    tolerance : float, default=1e-6
        Convergence tolerance for gradient norm.
    learning_rate : float, default=0.1
        Learning rate for gradient descent in the low-rank factor space.
    max_grad_norm : float or None, default=100.0
        Maximum gradient norm for clipping. If None, no clipping is performed.
        Helps prevent gradient explosion during optimization.

    Attributes:
    ----------
    U_ : ndarray of shape (m, rank)
        Left factor of the low-rank decomposition after fitting.
    V_ : ndarray of shape (n, rank)
        Right factor of the low-rank decomposition after fitting.
    n_iter_ : int
        Actual number of iterations performed during fitting.
    reconstruction_error_ : float
        Final reconstruction error on observed entries.

    Examples:
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from riemannax.api.problems import MatrixCompletion
    >>> # Create incomplete low-rank matrix
    >>> key = jax.random.PRNGKey(42)
    >>> m, n, rank = 10, 8, 3
    >>> U_true = jax.random.normal(key, (m, rank))
    >>> V_true = jax.random.normal(jax.random.fold_in(key, 1), (n, rank))
    >>> X_complete = U_true @ V_true.T
    >>> mask = jax.random.bernoulli(jax.random.fold_in(key, 2), 0.7, shape=(m, n))
    >>> X_incomplete = X_complete * mask
    >>> # Fit matrix completion
    >>> mc = MatrixCompletion(rank=rank, max_iter=100)
    >>> X_completed = mc.fit_transform(X_incomplete, mask)
    >>> X_completed.shape
    (10, 8)

    References:
    ----------
    .. [1] Vandereycken, B. (2013). "Low-rank matrix completion by Riemannian
           optimization." SIAM Journal on Optimization, 23(2), 1214-1236.
    .. [2] Mishra, B., Meyer, G., Bach, F., & Sepulchre, R. (2014).
           "Low-rank optimization with trace norm penalty." SIAM Journal on
           Optimization, 23(4), 2124-2149.
    """

    def __init__(
        self,
        rank: int,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        learning_rate: float = 0.1,
        max_grad_norm: float | None = 100.0,
    ):
        """Initialize MatrixCompletion estimator."""
        self.rank = rank
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Validate parameters
        self._validate_parameters()

        # Fitted attributes (set during fit())
        self.U_: Array | None = None
        self.V_: Array | None = None
        self.n_iter_: int | None = None
        self.reconstruction_error_: float | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and sub-estimators.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "rank": self.rank,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
        }

    def _validate_parameters(self) -> None:
        """Validate estimator parameters.

        Raises:
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")

    def set_params(self, **params: Any) -> "MatrixCompletion":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns:
        -------
        self : MatrixCompletion
            Estimator instance.

        Raises:
        ------
        ValueError
            If invalid parameters are provided.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key} for estimator MatrixCompletion")
            setattr(self, key, value)

        # Validate new parameters
        self._validate_parameters()

        # Reset fitted state
        self.U_ = None
        self.V_ = None
        self.n_iter_ = None
        self.reconstruction_error_ = None

        return self

    def fit(self, X: Array, mask: Array, y: Any = None) -> "MatrixCompletion":
        """Fit the matrix completion model.

        Parameters
        ----------
        X : Array of shape (m, n)
            Incomplete matrix with missing entries (can be zero or any value at
            unobserved positions).
        mask : Array of shape (m, n), dtype=bool
            Boolean mask indicating observed entries (True = observed).
        y : Ignored
            Not used, present for sklearn API consistency.

        Returns:
        -------
        self : MatrixCompletion
            Fitted estimator.

        Raises:
        ------
        ValueError
            If X is not 2D, mask shape doesn't match X, or rank exceeds matrix dimensions.
        """
        # Validate inputs
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {X.ndim}D array")

        if X.shape != mask.shape:
            raise ValueError(f"mask must have the same shape as X. Got X.shape={X.shape}, mask.shape={mask.shape}")

        m, n = X.shape
        min_dim = min(m, n)

        if self.rank > min_dim:
            raise ValueError(f"rank (={self.rank}) cannot exceed min(m, n) (={min_dim})")

        # Check sufficient observations
        n_observed = int(jax.device_get(jnp.sum(mask)))
        if n_observed == 0:
            raise ValueError("mask has zero observed entries; cannot fit MatrixCompletion.")
        min_observations = (m + n - self.rank) * self.rank  # Degrees of freedom
        if n_observed < min_observations:
            warnings.warn(
                f"Very few observations ({n_observed}) relative to model complexity "
                f"({min_observations} parameters). Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Initialize factors using SVD of observed entries
        # Replace unobserved entries with column means (simple imputation)
        X_filled = self._initialize_matrix(X, mask)

        # Compute truncated SVD for initialization
        U_init, s_init, Vt_init = jnp.linalg.svd(X_filled, full_matrices=False)
        U_init = U_init[:, : self.rank] * jnp.sqrt(s_init[: self.rank])
        V_init = Vt_init[: self.rank, :].T * jnp.sqrt(s_init[: self.rank])

        # Optimize using Euclidean gradient descent on the factor space
        U, V, n_iter, final_error = self._optimize(X, mask, U_init, V_init)

        # Store fitted attributes
        self.U_ = U
        self.V_ = V
        self.n_iter_ = n_iter
        self.reconstruction_error_ = final_error

        return self

    def _initialize_matrix(self, X: Array, mask: Array) -> Array:
        """Initialize incomplete matrix by imputing missing values.

        Parameters
        ----------
        X : Array of shape (m, n)
            Incomplete matrix.
        mask : Array of shape (m, n)
            Boolean mask of observed entries.

        Returns:
        -------
        X_filled : Array of shape (m, n)
            Matrix with missing values imputed.
        """
        # Compute column means from observed values
        mask_float = mask.astype(X.dtype)
        column_sums = jnp.sum(X * mask_float, axis=0)
        column_counts = jnp.sum(mask_float, axis=0)
        column_means = jnp.where(column_counts > 0, column_sums / column_counts, 0.0)

        # Fill missing values with column means
        X_filled = jnp.where(mask, X, column_means)

        return X_filled

    def _optimize(self, X: Array, mask: Array, U_init: Array, V_init: Array) -> tuple[Array, Array, int, float]:
        """Optimize matrix factorization using Euclidean gradient descent on the low-rank factors.

        Uses jax.lax.while_loop for JIT compatibility.

        Parameters
        ----------
        X : Array of shape (m, n)
            Target matrix with observed entries.
        mask : Array of shape (m, n)
            Boolean mask of observed entries.
        U_init : Array of shape (m, rank)
            Initial left factor.
        V_init : Array of shape (n, rank)
            Initial right factor.

        Returns:
        -------
        U : Array of shape (m, rank)
            Optimized left factor.
        V : Array of shape (n, rank)
            Optimized right factor.
        n_iter : int
            Number of iterations performed.
        final_error : float
            Final reconstruction error.
        """

        def compute_residual_and_error(U_current: Array, V_current: Array) -> tuple[Array, Array]:
            """Compute residual and reconstruction error on observed entries."""
            reconstruction = U_current @ V_current.T
            residual = (reconstruction - X) * mask
            # Mean over observed entries only
            n_observed = jnp.maximum(jnp.sum(mask), 1)
            error = jnp.sum(residual**2) / n_observed
            return residual, error

        def cond_fun(state: _MatrixCompletionOptState) -> Array:
            return (state.iteration < self.max_iter) & ~state.converged

        def body_fun(state: _MatrixCompletionOptState) -> _MatrixCompletionOptState:
            # Compute residual for gradient calculation
            residual, _ = compute_residual_and_error(state.U, state.V)

            # Compute Euclidean gradients
            grad_U = 2.0 * (residual @ state.V)  # Shape: (m, rank)
            grad_V = 2.0 * (residual.T @ state.U)  # Shape: (n, rank)

            # Compute gradient norms for convergence check
            grad_norm = jnp.sqrt(jnp.sum(grad_U**2) + jnp.sum(grad_V**2))
            converged = grad_norm < self.tolerance

            # Gradient clipping (if max_grad_norm is set)
            if self.max_grad_norm is not None:
                clip_factor = jnp.minimum(
                    1.0, self.max_grad_norm / jnp.maximum(grad_norm, NumericalConstants.HIGH_PRECISION_EPSILON)
                )
                grad_U = grad_U * clip_factor
                grad_V = grad_V * clip_factor

            # Update factors directly (simpler and more stable than QR+SVD)
            U_new = state.U - self.learning_rate * grad_U
            V_new = state.V - self.learning_rate * grad_V

            # Compute error of the new state
            _, error_new = compute_residual_and_error(U_new, V_new)
            has_numerical_issue = ~jnp.isfinite(error_new)

            # Stop if converged or if a numerical issue is found
            converged_out = converged | has_numerical_issue

            return _MatrixCompletionOptState(
                U=U_new,
                V=V_new,
                error=error_new,
                U_prev=state.U,
                V_prev=state.V,
                error_prev=state.error,
                iteration=state.iteration + 1,
                converged=converged_out,
            )

        # Initial state: compute initial error
        _, initial_error = compute_residual_and_error(U_init, V_init)
        initial_state = _MatrixCompletionOptState(
            U=U_init,
            V=V_init,
            error=initial_error,
            U_prev=U_init,
            V_prev=V_init,
            error_prev=initial_error,
            iteration=0,
            converged=jnp.array(False),
        )

        # Run optimization loop
        final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

        # Check if the loop terminated due to numerical issues
        has_numerical_issue = ~jnp.isfinite(final_state.error)
        n_iter_int = int(jax.device_get(final_state.iteration))

        if bool(jax.device_get(has_numerical_issue)):
            warnings.warn(
                f"Numerical instability detected at iteration {n_iter_int}. "
                "Returning last valid state. Try reducing learning_rate.",
                UserWarning,
                stacklevel=3,
            )
            # Rollback to the last valid state
            return final_state.U_prev, final_state.V_prev, n_iter_int - 1, float(jax.device_get(final_state.error_prev))

        # Return final state if no issues
        return final_state.U, final_state.V, n_iter_int, float(jax.device_get(final_state.error))

    def transform(self, X: Array) -> Array:
        """Return the completed matrix from the fitted low-rank factors.

        Note: This method does not transform the input `X`. It returns the
        reconstructed matrix based on the factors `U_` and `V_` learned during `fit`.
        The input `X` is only used to validate that its shape is consistent with
        the fitted model; the content of `X` is ignored.

        This behavior differs from typical sklearn transformers but is appropriate
        for matrix completion, where the goal is to recover the complete matrix
        from the fitted factorization, not to transform new data.

        Parameters
        ----------
        X : Array of shape (m, n)
            An array with the expected shape of the completed matrix. The content
            of this array is ignored; only its shape is used for validation.

        Returns:
        -------
        X_completed : Array of shape (m, n)
            The completed matrix reconstruction (U_ @ V_.T) from the fitted factors.

        Raises:
        ------
        ValueError
            If estimator has not been fitted or input shape is inconsistent.
        """
        if self.U_ is None or self.V_ is None:
            raise ValueError("MatrixCompletion is not fitted yet. Call fit() first.")

        # Validate shape consistency
        expected_shape = (self.U_.shape[0], self.V_.shape[0])
        if X.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape} based on fitted dimensions, got {X.shape}")

        # Reconstruct matrix
        X_completed = self.U_ @ self.V_.T

        return X_completed

    def fit_transform(self, X: Array, mask: Array, y: Any = None) -> Array:
        """Fit the model and transform the data.

        Parameters
        ----------
        X : Array of shape (m, n)
            Incomplete matrix.
        mask : Array of shape (m, n)
            Boolean mask of observed entries.
        y : Ignored
            Not used, present for sklearn API consistency.

        Returns:
        -------
        X_completed : Array of shape (m, n)
            Completed matrix.
        """
        self.fit(X, mask, y)
        return self.transform(X)

    def reconstruction_error(self, X: Array, mask: Array) -> float:
        """Compute reconstruction error on observed entries.

        Parameters
        ----------
        X : Array of shape (m, n)
            True matrix values.
        mask : Array of shape (m, n)
            Boolean mask of observed entries.

        Returns:
        -------
        error : float
            Mean squared error on observed entries.

        Raises:
        ------
        ValueError
            If estimator has not been fitted.
        """
        if self.U_ is None or self.V_ is None:
            raise ValueError("MatrixCompletion is not fitted yet. Call fit() first.")

        reconstruction = self.U_ @ self.V_.T
        residual = (reconstruction - X) * mask
        # Mean over observed entries only
        n_observed = jnp.maximum(jnp.sum(mask), 1)
        error = float(jnp.sum(residual**2) / n_observed)

        return error


class _ManifoldPCAMeanState(NamedTuple):
    """Optimization state for ManifoldPCA._compute_riemannian_mean().

    Attributes:
    ----------
    mean : Array
        Current Riemannian mean estimate.
    mean_prev : Array
        Previous Riemannian mean estimate.
    iteration : int
        Current iteration number.
    converged : Array
        Convergence flag (JAX boolean).
    """

    mean: Array
    mean_prev: Array
    iteration: int
    converged: Array


class ManifoldPCA(BaseEstimator, TransformerMixin):
    """Principal Geodesic Analysis (PGA) for dimensionality reduction on manifolds.

    Performs PCA-like dimensionality reduction for data lying on Riemannian manifolds.
    Uses the logarithm map to project data to the tangent space at the intrinsic mean,
    applies standard PCA there, and provides geometric transformations back to the manifold.

    The algorithm follows these steps:
    1. Compute the intrinsic/Riemannian mean on the manifold
    2. Map all data points to tangent space at the mean using log map
    3. Apply standard PCA/SVD in the tangent space
    4. Store principal geodesic directions and explained variance

    Parameters
    ----------
    manifold : Manifold or None, default=None
        The Riemannian manifold on which the data lies. Must be provided either
        at initialization or before calling fit().
    n_components : int
        Number of principal components to keep. Must be positive and not exceed
        the ambient dimension.
    max_iter : int, default=100
        Maximum number of iterations for computing the Riemannian mean.
    tolerance : float, default=1e-6
        Convergence tolerance for Riemannian mean computation.
    mean_step_size : float, default=0.5
        Step size for Riemannian mean updates. Helps avoid overshooting on
        highly curved manifolds. Should be in (0, 1].

    Attributes:
    ----------
    mean_ : ndarray of shape point_shape
        The intrinsic mean on the manifold after fitting (vector for Sphere;
        matrix for Stiefel/SPD, e.g., (n, p) or (n, n)).
    components_ : ndarray of shape (n_components, ambient_dim)
        Principal geodesic directions in the tangent space at the mean.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each principal component, in descending order.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component.

    Examples:
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from riemannax.api.problems import ManifoldPCA
    >>> from riemannax.manifolds import Sphere
    >>> # Create data on unit sphere
    >>> key = jax.random.PRNGKey(42)
    >>> X_raw = jax.random.normal(key, (100, 3))
    >>> X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)
    >>> # Fit manifold PCA
    >>> manifold = Sphere(n=2)
    >>> pca = ManifoldPCA(manifold=manifold, n_components=2)
    >>> X_reduced = pca.fit_transform(X)
    >>> X_reduced.shape
    (100, 2)

    References:
    ----------
    .. [1] Fletcher, P. T., Lu, C., Pizer, S. M., & Joshi, S. (2004).
           "Principal geodesic analysis for the study of nonlinear statistics of shape."
           IEEE transactions on medical imaging, 23(8), 995-1005.
    .. [2] Huckemann, S., Hotz, T., & Munk, A. (2010).
           "Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo
           isometric Lie group actions." Statistica Sinica, 1-58.
    """

    def __init__(
        self,
        manifold: Manifold | None = None,
        n_components: int = 2,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        mean_step_size: float = 0.5,
    ):
        """Initialize ManifoldPCA estimator."""
        self.manifold = manifold
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.mean_step_size = mean_step_size

        # Validate parameters
        self._validate_parameters()

        # Fitted attributes (set during fit())
        self.mean_: Array | None = None
        self.components_: Array | None = None
        self.explained_variance_: Array | None = None
        self.explained_variance_ratio_: Array | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and sub-estimators.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "manifold": self.manifold,
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "mean_step_size": self.mean_step_size,
        }

    def _validate_parameters(self) -> None:
        """Validate estimator parameters.

        Raises:
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive, got {self.n_components}")
        if not (0 < self.mean_step_size <= 1):
            raise ValueError(f"mean_step_size must be in (0, 1], got {self.mean_step_size}")

    def set_params(self, **params: Any) -> "ManifoldPCA":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns:
        -------
        self : ManifoldPCA
            Estimator instance.

        Raises:
        ------
        ValueError
            If invalid parameters are provided.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key} for estimator ManifoldPCA")
            setattr(self, key, value)

        # Validate new parameters
        self._validate_parameters()

        # Reset fitted state
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

        return self

    def _infer_point_shape(self, ambient_dim: int) -> tuple[int, ...]:
        """Infer the point shape on the manifold from ambient dimension.

        Parameters
        ----------
        ambient_dim : int
            Ambient dimension of the flattened representation.

        Returns:
        -------
        point_shape : tuple of int
            Shape of a single point on the manifold.

        Raises:
        ------
        ValueError
            If the ambient dimension is incompatible with the manifold.
        """
        assert self.manifold is not None, "Manifold must be set"

        if isinstance(self.manifold, Sphere):
            # Sphere(n) has points in R^(n+1), so ambient_dim = n+1
            return (ambient_dim,)

        elif isinstance(self.manifold, Stiefel):
            # Stiefel(n, p) has points as (n, p) matrices
            n, p = self.manifold.n, self.manifold.p
            if ambient_dim != n * p:
                raise ValueError(f"For Stiefel({n}, {p}), expected ambient_dim={n * p}, got {ambient_dim}")
            return (n, p)

        elif isinstance(self.manifold, SymmetricPositiveDefinite):
            # SPD(n) has points as (n, n) matrices
            matrix_dim = math.isqrt(int(ambient_dim))
            if matrix_dim * matrix_dim != ambient_dim:
                raise ValueError(
                    f"For SymmetricPositiveDefinite, ambient_dim must be a perfect square, got {ambient_dim}"
                )
            return (matrix_dim, matrix_dim)

        else:
            # For unknown manifolds, assume points are vectors (default behavior)
            return (ambient_dim,)

    def fit(self, X: Array, y: Any = None) -> "ManifoldPCA":
        """Fit the manifold PCA model.

        Parameters
        ----------
        X : Array of shape (n_samples, ambient_dim)
            Training data lying on the manifold. Each row is a point on the manifold.
            For matrix manifolds (Stiefel, SPD), data should be flattened; this method
            will automatically reshape it to the correct tensor format.
        y : Ignored
            Not used, present for sklearn API consistency.

        Returns:
        -------
        self : ManifoldPCA
            Fitted estimator.

        Raises:
        ------
        ValueError
            If manifold is not provided, X is not 2D, n_components exceeds dimension,
            or data points don't lie on the manifold.
        """
        # Validate manifold
        if self.manifold is None:
            raise ValueError("manifold must be provided either at initialization or before calling fit()")

        # Validate input shape
        if X.ndim != 2:
            raise ValueError(f"Expected 2D data matrix, got {X.ndim}D array")

        n_samples, ambient_dim = X.shape

        # Validate minimum sample requirement for covariance estimation
        if n_samples < 2:
            raise ValueError(
                f"ManifoldPCA requires at least 2 samples to estimate covariance; got n_samples={n_samples}"
            )

        # Validate n_components
        if self.n_components > ambient_dim:
            raise ValueError(f"n_components (={self.n_components}) cannot exceed ambient dimension (={ambient_dim})")

        # Warn if n_components > n_samples
        if self.n_components > n_samples:
            warnings.warn(
                f"n_components (={self.n_components}) exceeds n_samples (={n_samples}). Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Infer point shape and reshape data for matrix manifolds
        point_shape = self._infer_point_shape(ambient_dim)
        X_reshaped = X.reshape(n_samples, *point_shape)

        # Validate data lies on manifold
        # Vectorized check for common manifolds for performance
        if isinstance(self.manifold, Sphere):
            norms = jnp.linalg.norm(X_reshaped, axis=1)
            is_unit_norm = jnp.isclose(norms, 1.0, atol=NumericalConstants.VALIDATION_TOLERANCE)
            if not bool(jax.device_get(jnp.all(is_unit_norm))):
                failed_index = int(jax.device_get(jnp.argmin(is_unit_norm)))
                raise ValueError(
                    f"Data points for Sphere manifold must have unit norm. Point {failed_index} fails validation. "
                    "Ensure data is properly projected."
                )

        elif isinstance(self.manifold, Stiefel):
            # For Stiefel(n, p), X_reshaped already has shape (n_samples, n, p)
            # Check X^T X = I (vectorized)
            gram_matrices = jnp.einsum("...ji,...jk->...ik", X_reshaped, X_reshaped)  # X^T X for each sample
            p = self.manifold.p
            identity = jnp.eye(p)
            # Vectorized check: compute boolean per matrix, then find first failure
            is_orthonormal_per_matrix = jnp.all(
                jnp.isclose(gram_matrices, identity, atol=NumericalConstants.VALIDATION_TOLERANCE), axis=(-2, -1)
            )
            if not bool(jax.device_get(jnp.all(is_orthonormal_per_matrix))):
                failed_index = int(jax.device_get(jnp.argmin(is_orthonormal_per_matrix)))
                raise ValueError(
                    f"Data points for Stiefel manifold must have orthonormal columns. Point {failed_index} fails validation. "
                    f"Ensure data is properly projected."
                )

        elif isinstance(self.manifold, SymmetricPositiveDefinite):
            # For SPD(n), X_reshaped already has shape (n_samples, n, n)
            # Validate SPD matrices using helper function
            _validate_spd_batch(X_reshaped)

        else:
            # Try vectorized validation first for better performance with custom JIT-compatible manifolds
            vectorized_validation_succeeded = False
            if hasattr(self.manifold, "validate_point"):
                try:
                    # Attempt vectorized validation using vmap
                    def validate_fn(point):
                        return self.manifold.validate_point(point, atol=NumericalConstants.VALIDATION_TOLERANCE)

                    validation_results = jax.vmap(validate_fn)(X_reshaped)

                    # Check if all points pass validation
                    all_valid = jnp.all(validation_results)
                    if not bool(jax.device_get(all_valid)):
                        # Find first failing point for error message
                        failed_index = int(jax.device_get(jnp.argmin(validation_results)))
                        raise ValueError(
                            f"Data points must lie on the manifold. Point {failed_index} fails validation. "
                            f"Ensure data is properly projected onto the manifold."
                        )
                    vectorized_validation_succeeded = True
                except (TypeError, AttributeError, NotImplementedError):
                    # Vectorized validation not supported, fall back to iterative check
                    warnings.warn(
                        f"Vectorized validation for manifold {self.manifold.__class__.__name__} failed. "
                        "Falling back to a slower iterative validation. For better performance, "
                        "ensure the manifold's `validate_point` method is compatible with `jax.vmap`.",
                        UserWarning,
                        stacklevel=3,
                    )
                    pass

            # Fallback to iterative check if vectorized validation didn't work
            if not vectorized_validation_succeeded:
                for i, point in enumerate(X_reshaped):
                    if not self._check_on_manifold(point):
                        raise ValueError(
                            f"Data points must lie on the manifold. Point {i} fails validation. "
                            f"Ensure data is properly projected onto the manifold."
                        )

        # Step 1: Compute intrinsic mean on the manifold (using tensor-shaped data)
        mean = self._compute_riemannian_mean(X_reshaped)
        self.mean_ = mean

        # Step 2: Map data to tangent space at mean (vectorized)
        # Local variable for type narrowing
        manifold = self.manifold
        tangent_vectors = jax.vmap(manifold.log, in_axes=(None, 0))(mean, X_reshaped)

        # Flatten tangent vectors for covariance computation
        # For Sphere: tangent_vectors is already (n_samples, ambient_dim)
        # For Stiefel/SPD: reshape to (n_samples, ambient_dim)
        tangent_vectors_flat = tangent_vectors.reshape(n_samples, ambient_dim)

        # Step 3: Apply standard PCA in tangent space
        # Center data (should already be centered at origin in tangent space)
        tangent_centered = tangent_vectors_flat - jnp.mean(tangent_vectors_flat, axis=0)

        # Compute covariance matrix
        cov_matrix = (tangent_centered.T @ tangent_centered) / (n_samples - 1)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

        # Sort in descending order
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top n_components
        self.components_ = eigenvectors[:, : self.n_components].T  # Shape: (n_components, ambient_dim)
        explained_variance = eigenvalues[: self.n_components]
        self.explained_variance_ = explained_variance

        # Compute explained variance ratio (guard against division by zero)
        total_variance = jnp.sum(eigenvalues)
        self.explained_variance_ratio_ = jnp.where(
            total_variance > 0,
            explained_variance / total_variance,
            jnp.zeros_like(explained_variance),
        )

        return self

    def _check_on_manifold(self, point: Array, atol: float = NumericalConstants.VALIDATION_TOLERANCE) -> bool:
        """Check if a point lies on the manifold.

        Parameters
        ----------
        point : Array
            Point to check.
        atol : float, default=NumericalConstants.VALIDATION_TOLERANCE
            Absolute tolerance for validation.

        Returns:
        -------
        bool
            True if point is on manifold within tolerance.
        """
        assert self.manifold is not None, "Manifold must be set before calling _check_on_manifold"

        # Use manifold's validate_point method if available
        if hasattr(self.manifold, "validate_point"):
            try:
                result = self.manifold.validate_point(point, atol=atol)
                if isinstance(result, bool):
                    return result
                # Handle JAX scalar array result and bring to host
                result_all = jnp.all(result)
                return bool(jax.device_get(result_all))
            except NotImplementedError:
                # Method exists on base class but is not implemented by subclass
                pass

        # Fallback: warn the user that validation is skipped
        warnings.warn(
            f"Manifold {self.manifold.__class__.__name__} does not have a specific validation check. "
            "Assuming points are on the manifold.",
            UserWarning,
            stacklevel=3,
        )
        return True

    def _compute_riemannian_mean(self, X: Array) -> Array:
        """Compute the intrinsic/Riemannian mean on the manifold.

        Uses iterative gradient descent in the tangent space with a step size
        to improve convergence robustness on curved manifolds.

        Uses jax.lax.while_loop for JIT compatibility.

        Parameters
        ----------
        X : Array of shape (n_samples, *point_shape)
            Data points on the manifold. For vector manifolds like Sphere,
            point_shape is (ambient_dim,). For matrix manifolds like Stiefel(n, p)
            or SPD(n), point_shape is (n, p) or (n, n) respectively.

        Returns:
        -------
        mean : Array of shape point_shape
            The Riemannian mean on the manifold, in the same tensor format as
            individual data points.
        """
        assert self.manifold is not None, "Manifold must be set before calling _compute_riemannian_mean"

        # Initialize mean by projecting the Euclidean mean. This is generally a
        # better starting point than an arbitrary data point.
        # jnp.mean(X, axis=0) preserves the point_shape
        euclidean_mean = jnp.mean(X, axis=0)
        try:
            mean_init = _project_onto_manifold(self.manifold, euclidean_mean)
        except NotImplementedError:
            warnings.warn(
                f"Projection for {self.manifold.__class__.__name__} is not implemented in `_project_onto_manifold`. "
                "Initializing Riemannian mean with the first data point. This may lead to slower convergence.",
                UserWarning,
                stacklevel=3,
            )
            mean_init = X[0]

        # Iterative mean computation using while_loop
        manifold = self.manifold  # Local variable for type narrowing

        def cond_fun(state: _ManifoldPCAMeanState) -> Array:
            return (state.iteration < self.max_iter) & ~state.converged

        def body_fun(state: _ManifoldPCAMeanState) -> _ManifoldPCAMeanState:
            # Compute tangent vectors from current mean to all points (vectorized)
            # manifold.log expects points in their natural tensor shape
            tangent_vectors = jax.vmap(manifold.log, in_axes=(None, 0))(state.mean, X)

            # Compute mean tangent vector (preserves point_shape)
            mean_tangent = jnp.mean(tangent_vectors, axis=0)

            # Compute tangent norm
            tangent_norm = jnp.linalg.norm(mean_tangent)

            # Check for numerical issues (keep as JAX boolean for JIT)
            has_numerical_issue = ~jnp.isfinite(tangent_norm)

            # Check convergence
            converged = tangent_norm < self.tolerance

            # Update mean by exponential map with step size
            # manifold.exp expects points in their natural tensor shape
            mean_new = manifold.exp(state.mean, self.mean_step_size * mean_tangent)

            # Stop if converged or if a numerical issue is found
            converged_out = converged | has_numerical_issue

            return _ManifoldPCAMeanState(
                mean=mean_new,
                mean_prev=state.mean,
                iteration=state.iteration + 1,
                converged=converged_out,
            )

        # Initial state
        initial_state = _ManifoldPCAMeanState(
            mean=mean_init,
            mean_prev=mean_init,
            iteration=0,
            converged=jnp.array(False),
        )

        # Run optimization loop
        final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

        # Check if the loop terminated due to numerical issues
        # We need to check mean_final for numerical issues
        has_numerical_issue = ~jnp.isfinite(jnp.linalg.norm(final_state.mean))

        # Warn about numerical issues (outside the loop for side effects)
        if bool(jax.device_get(has_numerical_issue)):
            warnings.warn(
                f"Numerical instability detected at iteration {int(jax.device_get(final_state.iteration))} while computing Riemannian mean. "
                "Mean computation may have failed. Consider checking data for outliers.",
                UserWarning,
                stacklevel=3,
            )
            # Rollback to the last valid state
            return final_state.mean_prev

        return final_state.mean

    def transform(self, X: Array) -> Array:
        """Project data to the principal component subspace.

        Parameters
        ----------
        X : Array of shape (n_samples, ambient_dim)
            Data points on the manifold to transform. For matrix manifolds,
            data should be flattened; this method will automatically reshape
            it to the correct tensor format.

        Returns:
        -------
        X_transformed : Array of shape (n_samples, n_components)
            Projected coordinates in the principal component subspace.

        Raises:
        ------
        ValueError
            If estimator has not been fitted or input dimension is inconsistent.
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("ManifoldPCA is not fitted yet. Call fit() first.")

        # Validate input is 2D
        if X.ndim != 2:
            raise ValueError(f"Expected 2D data matrix, got {X.ndim}D array")

        n_samples, ambient_dim = X.shape

        # Validate dimensions
        if ambient_dim != self.components_.shape[1]:
            raise ValueError(
                f"Dimension mismatch: expected {self.components_.shape[1]} based on fitted dimensions, got {ambient_dim}"
            )

        # Infer point shape and reshape data for matrix manifolds
        point_shape = self._infer_point_shape(ambient_dim)
        X_reshaped = X.reshape(n_samples, *point_shape)

        # Map to tangent space at mean (vectorized)
        # manifold.log expects points in their natural tensor shape
        # Type narrowing: these are guaranteed non-None after successful fit()
        assert self.manifold is not None, "Manifold must be set after fit()"
        assert self.mean_ is not None, "Mean must be set after fit()"
        manifold = self.manifold
        mean = self.mean_
        tangent_vectors = jax.vmap(manifold.log, in_axes=(None, 0))(mean, X_reshaped)

        # Flatten tangent vectors for projection
        tangent_vectors_flat = tangent_vectors.reshape(n_samples, ambient_dim)

        # Project onto principal components
        X_transformed = tangent_vectors_flat @ self.components_.T

        return X_transformed

    def inverse_transform(self, X_transformed: Array) -> Array:
        """Reconstruct data from principal component subspace.

        Parameters
        ----------
        X_transformed : Array of shape (n_samples, n_components)
            Coordinates in the principal component subspace.

        Returns:
        -------
        X_reconstructed : Array of shape (n_samples, ambient_dim)
            Reconstructed points on the manifold, returned in flattened format
            for consistency with the input format expected by fit and transform.

        Raises:
        ------
        ValueError
            If estimator has not been fitted or input shape is inconsistent.
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("ManifoldPCA is not fitted yet. Call fit() first.")

        # Validate shape
        if X_transformed.shape[1] != self.n_components:
            raise ValueError(f"X_transformed must have {self.n_components} components, got {X_transformed.shape[1]}")

        n_samples = X_transformed.shape[0]
        ambient_dim = self.components_.shape[1]

        # Reconstruct tangent vectors from principal components (flattened format)
        tangent_reconstructed_flat = X_transformed @ self.components_

        # Infer point shape and reshape tangent vectors for manifold operations
        point_shape = self._infer_point_shape(ambient_dim)
        tangent_reconstructed = tangent_reconstructed_flat.reshape(n_samples, *point_shape)

        # Map back to manifold using exponential map (vectorized)
        # manifold.exp expects points in their natural tensor shape
        # Type narrowing: these are guaranteed non-None after successful fit()
        assert self.manifold is not None, "Manifold must be set after fit()"
        assert self.mean_ is not None, "Mean must be set after fit()"
        manifold = self.manifold
        mean = self.mean_
        X_reconstructed_tensor = jax.vmap(manifold.exp, in_axes=(None, 0))(mean, tangent_reconstructed)

        # Project back onto manifold to ensure constraint satisfaction
        # This is necessary for lossy reconstruction (n_components < ambient_dim)
        # Use centralized projection function with vmap for batch operations
        try:
            X_reconstructed_tensor = jax.vmap(lambda point: _project_onto_manifold(manifold, point))(
                X_reconstructed_tensor
            )
        except NotImplementedError:
            # Projection not implemented for this manifold
            warnings.warn(
                f"Manifold {manifold.__class__.__name__} does not support projection. "
                "Reconstructed points may not satisfy manifold constraints exactly.",
                UserWarning,
                stacklevel=2,
            )

        # Flatten back to (n_samples, ambient_dim) for API consistency
        X_reconstructed = X_reconstructed_tensor.reshape(n_samples, ambient_dim)

        return X_reconstructed

    def fit_transform(self, X: Array, y: Any = None) -> Array:
        """Fit the model and transform the data.

        Parameters
        ----------
        X : Array of shape (n_samples, ambient_dim)
            Training data on the manifold.
        y : Ignored
            Not used, present for sklearn API consistency.

        Returns:
        -------
        X_transformed : Array of shape (n_samples, n_components)
            Projected coordinates.
        """
        self.fit(X, y)
        return self.transform(X)


class _RobustCovarianceMedianState(NamedTuple):
    """Optimization state for RobustCovarianceEstimation._compute_geometric_median().

    Attributes:
    ----------
    median : Array
        Current geometric median estimate.
    median_prev : Array
        Previous geometric median estimate.
    iteration : int
        Current iteration number.
    converged : Array
        Convergence flag (JAX boolean).
    """

    median: Array
    median_prev: Array
    iteration: int
    converged: Array


class RobustCovarianceEstimation(BaseEstimator, TransformerMixin):
    """Robust covariance estimation on SPD manifolds using geometric median.

    Estimates robust central tendency and covariance structure for symmetric
    positive definite (SPD) matrices using Riemannian geometry. Uses the geometric
    median instead of the Riemannian mean for robustness to outliers.

    The geometric median is defined as:
        argmin_p Σᵢ dist(p, Xᵢ)
    where dist is the Riemannian distance on the SPD manifold.

    Parameters
    ----------
    metric : str, default="affine_invariant"
        Riemannian metric to use on SPD manifold. Options:
        - "affine_invariant": Affine-invariant metric (default)
        - "log_euclidean": Log-Euclidean metric
    max_iter : int, default=100
        Maximum number of iterations for Weiszfeld algorithm.
    tolerance : float, default=1e-6
        Convergence tolerance for geometric median computation.

    Attributes:
    ----------
    geometric_median_ : ndarray of shape (matrix_dim, matrix_dim)
        The geometric median of SPD matrices after fitting.
    n_iter_ : int
        Actual number of iterations performed during fitting.
    manifold_ : SymmetricPositiveDefinite
        Internal manifold object used for computations.

    Examples:
    --------
    >>> import jax
    >>> from riemannax.api.problems import RobustCovarianceEstimation
    >>> from riemannax.manifolds import SymmetricPositiveDefinite
    >>> # Generate SPD matrices
    >>> manifold = SymmetricPositiveDefinite(n=3)
    >>> X = manifold.random_point(jax.random.PRNGKey(42), (20,))
    >>> # Fit robust covariance estimation
    >>> rce = RobustCovarianceEstimation(max_iter=50)
    >>> rce.fit(X)
    >>> geometric_median = rce.geometric_median_
    >>> geometric_median.shape
    (3, 3)

    References:
    ----------
    .. [1] Fletcher, P. T., Venkatasubramanian, S., & Joshi, S. (2009).
           "The geometric median on Riemannian manifolds with application to
           robust atlas estimation." NeuroImage, 45(1), S143-S152.
    .. [2] Pennec, X. (2006). "Intrinsic statistics on Riemannian manifolds:
           Basic tools for geometric measurements." Journal of Mathematical
           Imaging and Vision, 25(1), 127-154.
    """

    def __init__(
        self,
        metric: str = "affine_invariant",
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ):
        """Initialize RobustCovarianceEstimation estimator."""
        self.metric = metric
        self.max_iter = max_iter
        self.tolerance = tolerance

        # Validate parameters
        self._validate_parameters()

        # Fitted attributes (set during fit())
        self.geometric_median_: Array | None = None
        self.n_iter_: int | None = None
        self.manifold_: SymmetricPositiveDefinite | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and sub-estimators.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "metric": self.metric,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
        }

    def _validate_parameters(self) -> None:
        """Validate estimator parameters.

        Raises:
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.metric not in ["affine_invariant", "log_euclidean"]:
            raise ValueError(f"metric must be 'affine_invariant' or 'log_euclidean', got {self.metric}")

    def set_params(self, **params: Any) -> "RobustCovarianceEstimation":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns:
        -------
        self : RobustCovarianceEstimation
            Estimator instance.

        Raises:
        ------
        ValueError
            If invalid parameters are provided.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key} for estimator RobustCovarianceEstimation")
            setattr(self, key, value)

        # Validate new parameters
        self._validate_parameters()

        # Reset fitted state
        self.geometric_median_ = None
        self.n_iter_ = None
        self.manifold_ = None

        return self

    def _reshape_and_validate_input(
        self, X: Array, expected_matrix_dim: int | None = None, input_name: str = "X"
    ) -> Array:
        """Reshape and validate input arrays for SPD matrices.

        Handles both 2D flattened and 3D stacked input formats, validates dimensions,
        and returns a consistently shaped 3D array.

        Parameters
        ----------
        X : Array
            Input array, either 2D (n_samples, matrix_dim * matrix_dim) flattened
            or 3D (n_samples, matrix_dim, matrix_dim) stacked.
        expected_matrix_dim : int or None, default=None
            Expected matrix dimension. If provided, validates that input dimensions
            match. If None, infers from input shape.
        input_name : str, default="X"
            Name of the input array for error messages.

        Returns:
        -------
        X_reshaped : Array of shape (n_samples, matrix_dim, matrix_dim)
            Reshaped and validated array in 3D stacked format.

        Raises:
        ------
        ValueError
            If input dimensions are invalid or don't match expected_matrix_dim.
        """
        # Handle flattened 2D input
        if X.ndim == 2:
            n_samples, ambient_dim = X.shape
            matrix_dim = math.isqrt(int(ambient_dim))
            if matrix_dim * matrix_dim != ambient_dim:
                raise ValueError(f"For flattened SPD matrices, ambient_dim must be a perfect square, got {ambient_dim}")
            X_reshaped = X.reshape(n_samples, matrix_dim, matrix_dim)
        elif X.ndim == 3:
            X_reshaped = X
            matrix_dim = X_reshaped.shape[1]
        else:
            raise ValueError(f"Expected 2D (flattened) or 3D (stacked) {input_name}, got {X.ndim}D array")

        # Validate dimensions match expected
        if expected_matrix_dim is not None and (
            X_reshaped.shape[1] != expected_matrix_dim or X_reshaped.shape[2] != expected_matrix_dim
        ):
            raise ValueError(
                f"Dimension mismatch: expected ({expected_matrix_dim}, {expected_matrix_dim}) "
                f"based on fitted dimensions, got ({X_reshaped.shape[1]}, {X_reshaped.shape[2]})"
            )

        return X_reshaped

    def fit(self, X: Array, y: Any = None) -> "RobustCovarianceEstimation":
        """Fit the robust covariance estimation model.

        Parameters
        ----------
        X : Array of shape (n_samples, matrix_dim, matrix_dim) or (n_samples, matrix_dim * matrix_dim)
            SPD matrices. Each X[i] is a symmetric positive definite matrix.
            Can be provided as 3D stacked matrices or 2D flattened matrices.
        y : Ignored
            Not used, present for sklearn API consistency.

        Returns:
        -------
        self : RobustCovarianceEstimation
            Fitted estimator.

        Raises:
        ------
        ValueError
            If X is not 2D or 3D, matrices are not square, symmetric, or positive definite.
        """
        # Reshape and validate input
        X = self._reshape_and_validate_input(X)

        _, dim1, dim2 = X.shape

        if dim1 != dim2:
            raise ValueError(f"Expected square matrices, got shape ({dim1}, {dim2})")

        matrix_dim = dim1

        # Validate SPD matrices using helper function
        _validate_spd_batch(X)

        # Create SPD manifold
        self.manifold_ = SymmetricPositiveDefinite(n=matrix_dim)

        # Compute geometric median using Weiszfeld algorithm
        median, n_iter = self._compute_geometric_median(X)

        # Store fitted attributes
        self.geometric_median_ = median
        self.n_iter_ = n_iter

        return self

    def _get_metric_ops(self):
        """Get metric-specific operations for distance, log, and exp.

        Returns:
        -------
        dist_fn : callable
            Distance function for the metric.
        log_fn : callable
            Logarithm map function for the metric.
        exp_fn : callable
            Exponential map function for the metric.
        """
        assert self.manifold_ is not None, "Manifold must be initialized"

        if self.metric == "log_euclidean":
            return (
                self.manifold_.log_euclidean_distance,
                self.manifold_.log_euclidean_log,
                self.manifold_.log_euclidean_exp,
            )
        else:  # affine_invariant
            return (
                self.manifold_.dist,
                self.manifold_.log,
                self.manifold_.exp,
            )

    def _compute_geometric_median(self, X: Array) -> tuple[Array, int]:
        """Compute geometric median using Weiszfeld algorithm.

        The geometric median minimizes the sum of geodesic distances to all data points.

        Uses jax.lax.while_loop for JIT compatibility.

        Parameters
        ----------
        X : Array of shape (n_samples, matrix_dim, matrix_dim)
            SPD matrices.

        Returns:
        -------
        median : Array of shape (matrix_dim, matrix_dim)
            The geometric median on SPD manifold.
        n_iter : int
            Number of iterations performed.
        """
        assert self.manifold_ is not None, "Manifold must be initialized before computing median"

        # Get metric-specific operations once
        dist_fn, log_fn, exp_fn = self._get_metric_ops()

        # Initialize with the Log-Euclidean mean, which is a robust and
        # computationally cheap estimate of the central tendency.
        try:
            # Compute Log-Euclidean mean: expm(mean(logm(X_i)))
            manifold = self.manifold_  # Local variable for mypy
            # All matrices have the same shape, so use first matrix's identity as base point
            base_point = jnp.eye(X.shape[1])
            log_matrices = jax.vmap(manifold.log_euclidean_log, in_axes=(None, 0))(base_point, X)
            log_mean = jnp.mean(log_matrices, axis=0)
            median_init = manifold.log_euclidean_exp(jnp.eye(log_mean.shape[0]), log_mean)
        except (RuntimeError, ValueError):
            warnings.warn(
                "Failed to initialize with Log-Euclidean mean. Falling back to initializing with the first data point.",
                UserWarning,
                stacklevel=3,
            )
            median_init = X[0]

        # Weiszfeld algorithm using while_loop
        epsilon = NumericalConstants.WEISZFELD_EPSILON

        def cond_fun(state: _RobustCovarianceMedianState) -> Array:
            return (state.iteration < self.max_iter) & ~state.converged

        def body_fun(state: _RobustCovarianceMedianState) -> _RobustCovarianceMedianState:
            # Compute distances from current median to all points (vectorized)
            distances = jax.vmap(dist_fn, in_axes=(None, 0))(state.median, X)

            # Check if any distance is (near) zero (exact match found)
            min_idx = jnp.argmin(distances)
            min_dist = distances[min_idx]
            found_exact = min_dist < epsilon

            # If found exact match, use that point
            exact_point = X[min_idx]

            # Compute weights
            weights = 1.0 / (distances + epsilon)
            weights = weights / jnp.sum(weights)

            # Compute weighted sum of log maps (tangent vectors) (vectorized)
            tangent_vectors = jax.vmap(log_fn, in_axes=(None, 0))(state.median, X)

            # Weighted average in tangent space
            mean_tangent = jnp.sum(tangent_vectors * weights[:, None, None], axis=0)

            # Check convergence
            tangent_norm = jnp.linalg.norm(mean_tangent, ord="fro")

            # Check for numerical issues (keep as JAX boolean for JIT)
            has_numerical_issue = ~jnp.isfinite(tangent_norm)

            # Check convergence
            converged = tangent_norm < self.tolerance

            # Update median using exponential map (or use exact point if found)
            median_updated = exp_fn(state.median, mean_tangent)
            median_new = jnp.where(found_exact, exact_point, median_updated)

            # Stop if converged, found exact match, or numerical issue detected
            converged_out = converged | found_exact | has_numerical_issue

            return _RobustCovarianceMedianState(
                median=median_new,
                median_prev=state.median,
                iteration=state.iteration + 1,
                converged=converged_out,
            )

        # Initial state
        initial_state = _RobustCovarianceMedianState(
            median=median_init,
            median_prev=median_init,
            iteration=0,
            converged=jnp.array(False),
        )

        # Run optimization loop
        final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

        # Check if the loop terminated due to numerical issues
        # We need to check median_final for numerical issues
        has_numerical_issue = ~jnp.isfinite(jnp.linalg.norm(final_state.median, ord="fro"))

        # Warn about numerical issues (outside the loop for side effects)
        if bool(jax.device_get(has_numerical_issue)):
            warnings.warn(
                f"Numerical instability detected at iteration {int(jax.device_get(final_state.iteration))} while computing geometric median. "
                "Median computation may have failed. Consider checking data for outliers.",
                UserWarning,
                stacklevel=3,
            )
            # Rollback to the last valid state
            return final_state.median_prev, int(jax.device_get(final_state.iteration)) - 1

        return final_state.median, int(jax.device_get(final_state.iteration))

    def transform(self, X: Array) -> Array:
        """Transform SPD matrices to tangent space at geometric median.

        Parameters
        ----------
        X : Array of shape (n_samples, matrix_dim, matrix_dim) or (n_samples, matrix_dim * matrix_dim)
            SPD matrices to transform. Can be provided as 3D stacked matrices
            or 2D flattened matrices.

        Returns:
        -------
        X_transformed : Array of shape (n_samples, matrix_dim, matrix_dim)
            Tangent vectors at the geometric median.

        Raises:
        ------
        ValueError
            If estimator has not been fitted or input dimension is inconsistent.
        """
        if self.geometric_median_ is None or self.manifold_ is None:
            raise ValueError("RobustCovarianceEstimation is not fitted yet. Call fit() first.")

        matrix_dim = self.geometric_median_.shape[0]

        # Reshape and validate input
        X_reshaped = self._reshape_and_validate_input(X, expected_matrix_dim=matrix_dim, input_name="X")

        # Map to tangent space at geometric median (vectorized)
        assert self.manifold_ is not None  # For mypy
        assert self.geometric_median_ is not None  # For mypy

        # Get metric-specific operations
        _, log_fn, _ = self._get_metric_ops()
        median = self.geometric_median_

        tangent_vectors = jax.vmap(log_fn, in_axes=(None, 0))(median, X_reshaped)

        return tangent_vectors

    def inverse_transform(self, X_tangent: Array) -> Array:
        """Transform tangent vectors back to SPD manifold.

        Parameters
        ----------
        X_tangent : Array of shape (n_samples, matrix_dim, matrix_dim) or (n_samples, matrix_dim * matrix_dim)
            Tangent vectors at the geometric median. Can be provided as 3D stacked matrices
            or 2D flattened matrices.

        Returns:
        -------
        X_spd : Array of shape (n_samples, matrix_dim, matrix_dim)
            SPD matrices on the manifold.

        Raises:
        ------
        ValueError
            If estimator has not been fitted or input dimension is inconsistent.
        """
        if self.geometric_median_ is None or self.manifold_ is None:
            raise ValueError("RobustCovarianceEstimation is not fitted yet. Call fit() first.")

        matrix_dim = self.geometric_median_.shape[0]

        # Reshape and validate input
        X_reshaped = self._reshape_and_validate_input(
            X_tangent, expected_matrix_dim=matrix_dim, input_name="tangent vectors"
        )

        # Map back to manifold using exponential map (vectorized)
        assert self.manifold_ is not None  # For mypy
        assert self.geometric_median_ is not None  # For mypy

        # Get metric-specific operations
        _, _, exp_fn = self._get_metric_ops()
        median = self.geometric_median_

        X_spd = jax.vmap(exp_fn, in_axes=(None, 0))(median, X_reshaped)

        return X_spd

    def fit_transform(self, X: Array, y: Any = None) -> Array:
        """Fit the model and transform the data.

        Parameters
        ----------
        X : Array of shape (n_samples, matrix_dim, matrix_dim)
            Training SPD matrices.
        y : Ignored
            Not used, present for sklearn API consistency.

        Returns:
        -------
        X_transformed : Array of shape (n_samples, matrix_dim, matrix_dim)
            Tangent vectors at the geometric median.
        """
        self.fit(X, y)
        return self.transform(X)


class ManifoldConstrainedParameter:
    """Manifold-constrained parameter for neural networks.

    Wraps neural network parameters to ensure they remain on a Riemannian manifold
    during optimization. Provides methods for projection, Riemannian gradient
    computation, and constrained updates.

    This is useful for neural networks with constrained weights, such as:
    - Orthogonal weight matrices (Stiefel manifold)
    - SPD covariance matrices (SPD manifold)
    - Other geometric constraints

    Parameters
    ----------
    manifold : Manifold
        The Riemannian manifold constraint for this parameter.
    initial_value : Array
        Initial parameter value. Should lie on the manifold or will be projected.

    Attributes:
    ----------
    manifold : Manifold
        The Riemannian manifold for this parameter.
    value : ndarray
        Current parameter value on the manifold.

    Examples:
    --------
    >>> import jax
    >>> from riemannax.api.problems import ManifoldConstrainedParameter
    >>> from riemannax.manifolds import Stiefel
    >>> # Create orthogonal weight matrix
    >>> manifold = Stiefel(n=10, p=5)
    >>> initial = manifold.random_point(jax.random.PRNGKey(42))
    >>> param = ManifoldConstrainedParameter(manifold, initial)
    >>> # Update with gradient while maintaining orthogonality
    >>> gradient = jax.random.normal(jax.random.PRNGKey(1), (10, 5))
    >>> param.value = param.update(gradient, learning_rate=0.01)

    References:
    ----------
    .. [1] Absil, P. A., Mahony, R., & Sepulchre, R. (2009). "Optimization
           algorithms on matrix manifolds." Princeton University Press.
    .. [2] Huang, Z., & Van Gool, L. (2017). "A Riemannian network for SPD
           matrix learning." In AAAI conference on artificial intelligence.
    """

    def __init__(self, manifold: Manifold, initial_value: Array, metric: str = "affine_invariant"):
        """Initialize manifold-constrained parameter.

        Parameters
        ----------
        manifold : Manifold
            Riemannian manifold for this parameter.
        initial_value : Array
            Initial value (will be projected onto manifold).
        metric : str, default="affine_invariant"
            Riemannian metric to use for SPD manifolds. Options:
            - "affine_invariant": Affine-invariant metric (default)
            - "log_euclidean": Log-Euclidean metric
            This parameter is only used for SPD manifolds.
        """
        self.manifold = manifold
        self.metric = metric

        # Validate metric parameter for SPD manifolds
        if isinstance(self.manifold, SymmetricPositiveDefinite) and self.metric not in [
            "affine_invariant",
            "log_euclidean",
        ]:
            raise ValueError(
                f"Unsupported metric '{self.metric}' for SPD manifold. Must be 'affine_invariant' or 'log_euclidean'."
            )

        self._value = self.project(initial_value)

    @property
    def value(self) -> Array:
        """Get current parameter value.

        Returns:
        -------
        value : Array
            Current value on the manifold.
        """
        return self._value

    @value.setter
    def value(self, new_value: Array) -> None:
        """Set parameter value (with automatic projection).

        Parameters
        ----------
        new_value : Array
            New value (will be projected onto manifold).
        """
        self._value = self.project(new_value)

    def project(self, point: Array) -> Array:
        """Project a point onto the manifold.

        Parameters
        ----------
        point : Array
            Point to project (may be off-manifold).

        Returns:
        -------
        projected : Array
            Projected point on the manifold.
        """
        return _project_onto_manifold(self.manifold, point)

    def riemannian_gradient(self, point: Array, euclidean_grad: Array) -> Array:
        """Convert Euclidean gradient to Riemannian gradient.

        Projects the Euclidean gradient onto the tangent space at the given point.

        Parameters
        ----------
        point : Array
            Point on the manifold.
        euclidean_grad : Array
            Euclidean gradient.

        Returns:
        -------
        riemannian_grad : Array
            Riemannian gradient (tangent vector).
        """
        # Check if manifold has egrad2rgrad method
        if hasattr(self.manifold, "egrad2rgrad"):
            return self.manifold.egrad2rgrad(point, euclidean_grad)

        # Special cases for SPD manifolds
        if isinstance(self.manifold, SymmetricPositiveDefinite):
            if self.metric == "affine_invariant":
                # Affine-invariant Riemannian gradient: X @ sym(G_e) @ X
                sym_grad = (euclidean_grad + euclidean_grad.T) / 2.0
                return point @ sym_grad @ point
            elif self.metric == "log_euclidean":
                # For the Log-Euclidean metric, the optimization step is performed in the log-domain.
                # The gradient of the cost function w.r.t. L=logm(X) is required.
                # A common first-order approximation is to use the symmetric part of the Euclidean gradient.
                return (euclidean_grad + euclidean_grad.T) / 2.0

        # Use proj() to project onto tangent space for other cases
        if hasattr(self.manifold, "proj"):
            return self.manifold.proj(point, euclidean_grad)

        # No Riemannian gradient conversion available
        manifold_name = self.manifold.__class__.__name__
        raise NotImplementedError(
            f"Manifold {manifold_name} does not have 'egrad2rgrad' or 'proj' method. "
            f"Cannot convert Euclidean gradient to Riemannian gradient."
        )

    def update(self, gradient: Array, learning_rate: float) -> Array:
        """Perform constrained parameter update.

        Updates the parameter using Riemannian gradient descent with
        retraction to ensure the result stays on the manifold.

        Parameters
        ----------
        gradient : Array
            Euclidean gradient from backpropagation.
        learning_rate : float
            Step size for the update.

        Returns:
        -------
        updated_value : Array
            Updated parameter value on the manifold.

        Raises:
        ------
        ValueError
            If learning_rate is not positive.
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        # Convert to Riemannian gradient
        riemannian_grad = self.riemannian_gradient(self._value, gradient)

        # Compute tangent vector for update
        tangent_step = -learning_rate * riemannian_grad

        # Use a retraction if available; otherwise fall back to exp or project
        updated = None
        if hasattr(self.manifold, "retr"):
            # Explicit try-except for clarity: some manifolds define retr() but raise
            # NotImplementedError (e.g., SPD manifold). Fall back to exponential map.
            # See https://github.com/lv416e/riemannax/issues/32
            with contextlib.suppress(NotImplementedError):
                updated = self.manifold.retr(self._value, tangent_step)

        if updated is None and hasattr(self.manifold, "exp"):
            # Select the correct exponential map based on the metric for SPD manifolds
            exp_fn = self.manifold.exp
            if (
                isinstance(self.manifold, SymmetricPositiveDefinite)
                and self.metric == "log_euclidean"
                and hasattr(self.manifold, "log_euclidean_exp")
            ):
                exp_fn = self.manifold.log_euclidean_exp

            # Try exponential map, with fallback if not implemented
            with contextlib.suppress(NotImplementedError):
                updated = exp_fn(self._value, tangent_step)

        if updated is None:
            updated = self.project(self._value + tangent_step)

        return updated
