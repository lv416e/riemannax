"""Scikit-learn estimator interfaces for Riemannian optimization.

This module provides scikit-learn compatible estimators that enable
RiemannAX optimizers and transformers to work seamlessly with sklearn's
ecosystem including pipelines, GridSearchCV, and cross-validation.
"""

import warnings
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
        >>> from riemannax.manifolds import Sphere
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
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from riemannax.manifolds import Stiefel
        >>> manifold = Stiefel(n=10, p=5)
        >>> pca = RiemannianPCA(manifold=manifold, n_components=3)
        >>> # Generate sample data on the manifold
        >>> key = jax.random.PRNGKey(0)
        >>> keys = jax.random.split(key, 20)
        >>> X = jnp.stack([manifold.random_point(k) for k in keys])
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
        D = eigenvectors.shape[0]
        if not (1 <= int(self.n_components) <= int(D)):
            raise ValueError(f"n_components must be in [1, {int(D)}], got {self.n_components}.")
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
        # Initialize at Euclidean mean projected to manifold
        # Compute ambient mean and project it onto the manifold
        euclidean_mean = jnp.mean(X, axis=0)
        # Project Euclidean mean onto manifold using retraction with zero tangent
        try:
            zero_tangent = jnp.zeros_like(euclidean_mean)
            mean = self.manifold.retr(euclidean_mean, zero_tangent)
        except (ValueError, RuntimeError, NotImplementedError) as e:
            # Fallback to first point if projection fails
            warnings.warn(
                f"Failed to project Euclidean mean onto manifold: {e}. Using first data point as initial mean.",
                RuntimeWarning,
                stacklevel=2,
            )
            mean = X[0]

        # Gradient descent to find Frechet mean
        lr = learning_rate
        prev_mean = mean

        def compute_mean_tangent_and_cost(current_mean: Array) -> tuple[Array, float]:
            """Compute mean tangent vector and Fréchet variance (cost function).

            Returns:
                Tuple of (mean_tangent, frechet_variance).
            """
            logs = jax.vmap(lambda x: self.manifold.log(current_mean, x))(X)
            mean_tangent = jnp.mean(logs, axis=0)
            # Fréchet variance = mean of squared distances from mean
            frechet_variance = float(jnp.mean(jnp.sum(logs**2, axis=-1)))
            return mean_tangent, frechet_variance

        converged = False
        prev_cost = float("inf")  # Initialize for backtracking line search
        for iteration in range(max_iter):
            # Compute mean tangent and Fréchet variance (cost function)
            mean_tangent, current_cost = compute_mean_tangent_and_cost(mean)

            # Compute convergence metrics (convert to Python floats for control flow)
            grad_norm = float(jnp.linalg.norm(mean_tangent))

            # Check convergence before taking a step
            if grad_norm < tol:
                converged = True
                break

            # Backtracking line search with Armijo condition
            # Try to find a step size that decreases the cost function
            alpha = lr
            c1 = 1e-4  # Armijo parameter (sufficient decrease)
            beta = 0.5  # Backtracking shrinkage factor
            max_backtracks = 20

            for _ in range(max_backtracks):
                # Try step with current alpha
                candidate_mean = self.manifold.exp(mean, alpha * mean_tangent)
                _, candidate_cost = compute_mean_tangent_and_cost(candidate_mean)

                # Armijo condition: f(x_new) ≤ f(x) - 2*c1*alpha*||mean_tangent||^2
                # Factor of 2 comes from grad F(μ) = -2 * mean_tangent
                if candidate_cost <= current_cost - 2.0 * c1 * alpha * grad_norm**2:
                    break  # Accept step
                alpha *= beta  # Shrink step size
            else:
                # If backtracking fails, use minimum step size
                alpha = max(alpha, 1e-10)

            # Update mean with accepted step size
            scaled_tangent = jnp.asarray(alpha * mean_tangent)
            mean = self.manifold.exp(mean, scaled_tangent)

            # Additional convergence checks
            step_norm = float(jnp.linalg.norm(scaled_tangent))
            mean_change = float(
                jnp.linalg.norm(self.manifold.log(prev_mean, mean) if iteration > 0 else scaled_tangent)
            )

            if step_norm < tol or mean_change < tol:
                converged = True
                break

            # Update learning rate for next iteration (cautious adaptation)
            lr = min(alpha * 1.2, learning_rate) if candidate_cost < prev_cost * 0.9 else alpha

            prev_mean = mean
            prev_cost = candidate_cost  # Use accepted cost

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
        >>> import jax.numpy as jnp
        >>> from riemannax.manifolds import Sphere
        >>> manifold = Sphere(n=3)
        >>> optimizer = RiemannianOptimizer(
        ...     manifold=manifold,
        ...     learning_rate=0.01,
        ...     max_iter=100
        ... )
        >>> target = jnp.array([1.0, 0.0, 0.0, 0.0])
        >>> def objective_fn(x):
        ...     return manifold.dist(x, target) ** 2
        >>> X = jnp.array([[0.0, 1.0, 0.0, 0.0]])
        >>> optimizer.fit(X, objective_fn)  # doctest: +ELLIPSIS
        RiemannianOptimizer(...)
        >>> score = optimizer.score(X, objective_fn)
    """

    def __init__(
        self,
        manifold: Manifold,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        method: str = "sgd",
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Initialize Riemannian optimizer.

        Args:
            manifold: Riemannian manifold for optimization.
            learning_rate: Learning rate for optimization.
            max_iter: Maximum number of iterations.
            method: Optimization method ('sgd' or 'adam').
            b1: Exponential decay rate for first moment (Adam only).
            b2: Exponential decay rate for second moment (Adam only).
            eps: Small constant for numerical stability (Adam only).
        """
        super().__init__(manifold=manifold)
        # Validate method before storing (sklearn clone needs exact parameter match)
        method_lower = method.lower()
        if method_lower not in ["sgd", "adam"]:
            raise ValueError(f"Unsupported method: {method}. Use 'sgd' or 'adam'.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.method = method  # Store original case for sklearn compatibility
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.result_: Array | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "method": self.method,
                "b1": self.b1,
                "b2": self.b2,
                "eps": self.eps,
            }
        )
        return params

    def fit(self, X: Array, y: Callable[[Array], Array] | None = None) -> "RiemannianOptimizer":
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

        # Compute gradient function once outside loop
        grad_fn = jax.grad(y)

        # Riemannian optimization loop
        x = x0
        method_lower = self.method.lower()
        if method_lower == "adam":
            # Initialize Adam state
            m, v = jnp.zeros_like(x), jnp.zeros_like(x)

        for i in range(self.max_iter):
            # Compute Euclidean gradient
            euclidean_grad = grad_fn(x)

            # Project to tangent space (Riemannian gradient)
            riemannian_grad = self.manifold.proj(x, euclidean_grad)

            # Check convergence (convert to Python float for control flow)
            grad_norm = float(jnp.linalg.norm(riemannian_grad))
            if grad_norm < 1e-6:
                break

            # Compute tangent step
            # NOTE: This Adam implementation is similar to RiemannianOptaxAdapter.update()
            # in optax_adapter.py. While there is some code duplication, the implementations
            # are kept separate because:
            # 1. Different state management (loop-local vs NamedTuple)
            # 2. Different API contexts (sklearn vs optax)
            # 3. Optional dependencies (avoiding sklearn<->optax coupling)
            # Future: Consider extracting to a shared internal module if more optimizers are added.
            if method_lower == "adam":
                # Adam update
                m = self.b1 * m + (1 - self.b1) * riemannian_grad
                v = self.b2 * v + (1 - self.b2) * jnp.square(riemannian_grad)
                # Bias correction
                m_hat = m / (1 - self.b1 ** (i + 1))
                v_hat = v / (1 - self.b2 ** (i + 1))
                # Use standard Adam formula for consistency with JAX/Optax ecosystem
                tangent_step = -self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.eps)
            else:  # sgd
                tangent_step = -self.learning_rate * riemannian_grad

            # Update using retraction
            x_new = self.manifold.retr(x, tangent_step)

            # Parallel transport momentum vectors for Adam (critical for correctness)
            if method_lower == "adam":
                m = self.manifold.transp(x, x_new, m)
                v = self.manifold.transp(x, x_new, v)
                # Enforce non-negativity of second moment estimate for numerical stability
                v = jnp.maximum(v, 0.0)

            x = x_new

        self.result_ = x
        return self

    def score(self, X: Array, y: Callable[[Array], Array] | None = None) -> float:
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
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from riemannax.manifolds import Stiefel
        >>> manifold = Stiefel(n=10, p=5)
        >>> pipeline = create_riemannian_pipeline(
        ...     manifold=manifold,
        ...     n_components=3,
        ...     learning_rate=0.01
        ... )
        >>> # Generate sample data on the manifold
        >>> key = jax.random.PRNGKey(0)
        >>> keys = jax.random.split(key, 20)
        >>> X = jnp.stack([manifold.random_point(k) for k in keys])
        >>> pipeline.fit_transform(X)  # doctest: +ELLIPSIS
        Array...
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
