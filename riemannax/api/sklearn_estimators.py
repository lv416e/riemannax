"""Scikit-learn estimator interfaces for Riemannian optimization.

This module provides scikit-learn compatible estimators that enable
RiemannAX optimizers and transformers to work seamlessly with sklearn's
ecosystem including pipelines, GridSearchCV, and cross-validation.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

try:
    from sklearn.base import BaseEstimator, TransformerMixin

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Create mock classes for type checking when scikit-learn is not installed
    class BaseEstimator:
        """Mock BaseEstimator for type checking."""

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Get parameters."""
            return {}

        def set_params(self, **params: Any) -> "BaseEstimator":
            """Set parameters."""
            return self

    class TransformerMixin:
        """Mock TransformerMixin for type checking."""

        pass


from riemannax.manifolds.base import Manifold


class RiemannianManifoldEstimator(BaseEstimator):
    """Base class for scikit-learn compatible Riemannian estimators.

    This class implements the BaseEstimator interface, providing get_params()
    and set_params() methods that enable integration with scikit-learn's
    GridSearchCV, pipelines, and cross-validation.

    Args:
        manifold: The Riemannian manifold for optimization/transformation.

    Example:
        >>> manifold = Sphere(n=3)
        >>> estimator = RiemannianManifoldEstimator(manifold=manifold)
        >>> params = estimator.get_params(deep=True)
        >>> estimator.set_params(manifold=Sphere(n=5))
    """

    def __init__(self, manifold: Manifold):
        """Initialize Riemannian estimator."""
        self.manifold = manifold

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, return parameters for sub-estimators.

        Returns:
            Parameter names mapped to their values.
        """
        return {"manifold": self.manifold}

    def set_params(self, **params: Any) -> "RiemannianManifoldEstimator":
        """Set parameters for this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Self with updated parameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RiemannianPCA(TransformerMixin, RiemannianManifoldEstimator):
    """Principal Component Analysis on Riemannian manifolds.

    This transformer performs dimensionality reduction by finding principal
    geodesics on the manifold using Riemannian mean and tangent space PCA.
    Compatible with scikit-learn pipelines.

    Args:
        manifold: The Riemannian manifold.
        n_components: Number of principal components to keep.

    Example:
        >>> manifold = Stiefel(n=10, p=5)
        >>> pca = RiemannianPCA(manifold=manifold, n_components=3)
        >>> X_transformed = pca.fit_transform(X)
    """

    def __init__(self, manifold: Manifold, n_components: int = 2):
        """Initialize Riemannian PCA."""
        super().__init__(manifold=manifold)
        self.n_components = n_components
        self.components_: Array | None = None
        self.mean_: Array | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params["n_components"] = self.n_components
        return params

    def fit(self, X: Array, y: Any = None) -> "RiemannianPCA":
        """Fit Riemannian PCA on the data.

        Args:
            X: Training data of shape (n_samples, *manifold_shape).
            y: Ignored, present for API consistency.

        Returns:
            Self with computed components.
        """
        # Compute Riemannian mean as base point
        mean = self._compute_riemannian_mean(X)
        self.mean_ = mean

        # Project data to tangent space at mean (vectorized for performance)
        log_at_mean = jax.vmap(lambda x: self.manifold.log(mean, x))
        tangent_vectors = log_at_mean(X)

        # Flatten tangent vectors for PCA
        n_samples = tangent_vectors.shape[0]
        if n_samples < 2:
            raise ValueError(f"RiemannianPCA requires at least 2 samples for covariance computation, got {n_samples}.")
        tangent_flat = tangent_vectors.reshape(n_samples, -1)

        # Standard PCA in tangent space
        centered = tangent_flat - jnp.mean(tangent_flat, axis=0, keepdims=True)
        cov = (centered.T @ centered) / (n_samples - 1)

        # Compute principal components
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

        # Sort by descending eigenvalues
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Keep top n_components
        principal_directions = eigenvectors[:, : self.n_components]

        # Reshape back to manifold shape
        manifold_shape = X.shape[1:]
        self.components_ = principal_directions.T.reshape(self.n_components, *manifold_shape)

        return self

    def transform(self, X: Array) -> Array:
        """Transform data to lower-dimensional representation.

        Args:
            X: Data to transform of shape (n_samples, *manifold_shape).

        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        # Project to tangent space at mean (vectorized for performance)
        mean = self.mean_  # Type narrowing for mypy
        log_at_mean = jax.vmap(lambda x: self.manifold.log(mean, x))
        tangent_vectors = log_at_mean(X)

        # Flatten and project onto principal components
        n_samples = tangent_vectors.shape[0]
        tangent_flat = tangent_vectors.reshape(n_samples, -1)

        components_flat = self.components_.reshape(self.n_components, -1)

        return tangent_flat @ components_flat.T

    def _compute_riemannian_mean(
        self, X: Array, learning_rate: float = 0.1, max_iter: int = 50, tol: float = 1e-6
    ) -> Array:
        """Compute Riemannian mean using gradient descent.

        Args:
            X: Data points on the manifold.
            learning_rate: Initial learning rate for gradient descent.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance for gradient norm and step size.

        Returns:
            Riemannian mean point.
        """
        import warnings

        # Initialize at Euclidean mean projected to manifold
        # Compute ambient mean and project it
        euclidean_mean = jnp.mean(X, axis=0)
        # Use exponential map from first point as a simple projection heuristic
        try:
            mean = self.manifold.exp(X[0], self.manifold.log(X[0], euclidean_mean))
        except Exception:
            # Fallback to first point if projection fails
            mean = X[0]

        # Gradient descent to find Frechet mean
        lr = learning_rate
        prev_mean = mean

        def compute_mean_tangent(current_mean: Array) -> Array:
            logs = jax.vmap(lambda x: self.manifold.log(current_mean, x))(X)
            return jnp.mean(logs, axis=0)

        converged = False
        for iteration in range(max_iter):
            # Compute mean of log maps (vectorized for performance)
            mean_tangent = compute_mean_tangent(mean)

            # Update mean using exponential map
            scaled_tangent = jnp.asarray(lr * mean_tangent)
            mean = self.manifold.exp(mean, scaled_tangent)

            # Compute convergence metrics
            grad_norm = jnp.linalg.norm(mean_tangent)
            step_norm = jnp.linalg.norm(scaled_tangent)
            mean_change = jnp.linalg.norm(self.manifold.log(prev_mean, mean) if iteration > 0 else scaled_tangent)

            # Check convergence (any of three criteria)
            if grad_norm < tol or step_norm < tol or mean_change < tol:
                converged = True
                break

            # Adaptive learning rate: reduce if gradient norm increases
            if iteration > 0:
                prev_grad_norm = jnp.linalg.norm(compute_mean_tangent(prev_mean))
                if grad_norm > prev_grad_norm:
                    lr = lr * 0.5  # Backtracking

            prev_mean = mean

        # Warn if not converged
        if not converged:
            warnings.warn(
                f"Riemannian mean computation did not converge after {max_iter} iterations. "
                f"Final gradient norm: {grad_norm:.2e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return mean


class RiemannianOptimizer(RiemannianManifoldEstimator):
    """Scikit-learn compatible Riemannian optimizer.

    This estimator wraps Riemannian optimization in a scikit-learn compatible
    interface, enabling use with GridSearchCV and cross-validation.

    Args:
        manifold: The Riemannian manifold.
        learning_rate: Learning rate for optimization.
        max_iter: Maximum number of iterations.
        method: Optimization method ('sgd' or 'adam').

    Example:
        >>> manifold = Sphere(n=3)
        >>> optimizer = RiemannianOptimizer(
        ...     manifold=manifold,
        ...     learning_rate=0.01,
        ...     max_iter=100
        ... )
        >>> optimizer.fit(X, objective_fn)
        >>> score = optimizer.score(X, objective_fn)
    """

    def __init__(
        self,
        manifold: Manifold,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        method: str = "sgd",
    ):
        """Initialize Riemannian optimizer."""
        super().__init__(manifold=manifold)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.method = method
        self.result_: Array | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "method": self.method,
            }
        )
        return params

    def fit(self, X: Array, y: Callable[[Array], float] | None = None) -> "RiemannianOptimizer":
        """Fit optimizer by minimizing objective function.

        Args:
            X: Initial point(s) for optimization, shape (n_samples, *manifold_shape).
            y: Objective function to minimize. Can be None for compatibility with
                cross_val_score, but must be provided when actually optimizing.

        Returns:
            Self with optimization result stored.
        """
        if y is None:
            # For sklearn compatibility - just store the initial point
            self.result_ = X[0] if X.ndim > 1 else X
            return self

        # Use first point as initial guess
        x0 = X[0] if X.ndim > 1 else X

        # Simple Riemannian gradient descent
        x = x0
        for _ in range(self.max_iter):
            # Compute Euclidean gradient
            grad_fn = jax.grad(y)
            euclidean_grad = grad_fn(x)

            # Project to tangent space (Riemannian gradient)
            riemannian_grad = self.manifold.proj(x, euclidean_grad)

            # Check convergence
            grad_norm = jnp.linalg.norm(riemannian_grad)
            if grad_norm < 1e-6:
                break

            # Update using retraction
            tangent_step = -self.learning_rate * riemannian_grad
            x = self.manifold.retr(x, tangent_step)

        self.result_ = x
        return self

    def score(self, X: Array, y: Callable[[Array], float] | None = None) -> float:
        """Compute negative loss (higher is better for sklearn).

        Args:
            X: Data points (not used, present for API consistency).
            y: Objective function. If None, returns 0.0 for sklearn compatibility.

        Returns:
            Negative objective value at optimized point.
        """
        if self.result_ is None:
            raise ValueError("Optimizer not fitted. Call fit() first.")

        if y is None:
            return 0.0

        return -float(y(self.result_))


def create_riemannian_pipeline(manifold: Manifold, n_components: int = 2, **optimizer_kwargs: Any) -> Any:
    """Create a scikit-learn pipeline with Riemannian components.

    Args:
        manifold: The Riemannian manifold.
        n_components: Number of PCA components.
        **optimizer_kwargs: Additional arguments for RiemannianOptimizer.

    Returns:
        scikit-learn Pipeline with Riemannian transformers/estimators.

    Example:
        >>> manifold = Stiefel(n=10, p=5)
        >>> pipeline = create_riemannian_pipeline(
        ...     manifold=manifold,
        ...     n_components=3,
        ...     learning_rate=0.01
        ... )
        >>> pipeline.fit_transform(X)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for pipeline creation")

    from sklearn.pipeline import Pipeline

    steps = [
        ("pca", RiemannianPCA(manifold=manifold, n_components=n_components)),
    ]

    if optimizer_kwargs:
        steps.append(("optimizer", RiemannianOptimizer(manifold=manifold, **optimizer_kwargs)))

    return Pipeline(steps)
