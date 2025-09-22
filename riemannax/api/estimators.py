"""Scikit-learn compatible estimators for Riemannian optimization."""

from collections.abc import Callable
from typing import Any, ClassVar

from jaxtyping import Array
from sklearn.base import BaseEstimator

from ..manifolds import (
    create_grassmann,
    create_so,
    create_spd,
    create_sphere,
    create_stiefel,
)
from ..problems import RiemannianProblem
from ..solvers import minimize


class RiemannianEstimatorMixin:
    """Mixin class providing common functionality for Riemannian estimators."""

    SUPPORTED_MANIFOLDS: ClassVar[dict[str, Any]] = {
        "sphere": create_sphere,
        "stiefel": create_stiefel,
        "grassmann": create_grassmann,
        "so": create_so,
        "spd": create_spd,
    }

    # Type annotations for attributes that will be set by subclasses
    manifold: str
    learning_rate: float
    max_iterations: int
    tolerance: float

    def _validate_manifold(self, manifold: str) -> None:
        """Validate manifold type specification."""
        if manifold not in self.SUPPORTED_MANIFOLDS:
            available = ", ".join(self.SUPPORTED_MANIFOLDS.keys())
            raise ValueError(f"Unsupported manifold type '{manifold}'. Available options: {available}")

    def _validate_parameters(self) -> None:
        """Validate estimator parameters."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    def _is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return hasattr(self, "optimization_result_")

    def _create_manifold(self, initial_point: Array) -> Any:
        """Create manifold instance based on manifold type and initial point."""
        if self.manifold == "sphere":
            n_dim = len(initial_point) - 1  # S^n embedded in R^{n+1}
            return create_sphere(n_dim)
        elif self.manifold == "stiefel":
            p, n = initial_point.shape
            return create_stiefel(p, n)
        elif self.manifold == "grassmann":
            p, n = initial_point.shape
            return create_grassmann(p, n)
        elif self.manifold == "so":
            n = initial_point.shape[0]
            return create_so(n)
        elif self.manifold == "spd":
            n = initial_point.shape[0]
            return create_spd(n)
        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unsupported manifold type: {self.manifold}")


class RiemannianSGD(BaseEstimator, RiemannianEstimatorMixin):
    """Riemannian Stochastic Gradient Descent estimator.

    This estimator provides a scikit-learn compatible interface for Riemannian
    optimization using stochastic gradient descent.

    Parameters
    ----------
    manifold : str, default="sphere"
        Type of manifold for optimization. Supported types:
        "sphere", "stiefel", "grassmann", "so", "spd"
    learning_rate : float, default=0.1
        Learning rate for the optimization algorithm.
    max_iterations : int, default=100
        Maximum number of optimization iterations.
    tolerance : float, default=1e-6
        Tolerance for convergence detection.
    random_state : int or None, default=None
        Random state for reproducible results.

    Attributes:
    ----------
    optimization_result_ : OptimizeResult
        Result of the optimization after fitting.

    Examples:
    --------
    >>> from riemannax.api import RiemannianSGD
    >>> import jax.numpy as jnp
    >>>
    >>> # Define objective function
    >>> def objective(x):
    ...     return jnp.sum((x - jnp.array([0, 0, 1]))**2)
    >>>
    >>> # Create estimator and fit
    >>> estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)
    >>> x0 = jnp.array([1.0, 0.0, 0.0])  # Initial point on unit sphere
    >>> estimator.fit(objective, x0)
    >>> print(f"Optimized point: {estimator.optimization_result_.x}")
    """

    def __init__(
        self,
        manifold: str = "sphere",
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        random_state: int | None = None,
    ):
        """Initialize the RiemannianSGD estimator.

        Parameters
        ----------
        manifold : str, default="sphere"
            Type of manifold for optimization.
        learning_rate : float, default=0.1
            Learning rate for the optimization algorithm.
        max_iterations : int, default=100
            Maximum number of optimization iterations.
        tolerance : float, default=1e-6
            Tolerance for convergence detection.
        random_state : int or None, default=None
            Random state for reproducible results.
        """
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        # Validate parameters
        self._validate_manifold(manifold)
        self._validate_parameters()

    def fit(
        self,
        objective_func: Callable[[Array], float],
        initial_point: Array,
    ) -> "RiemannianSGD":
        """Fit the Riemannian optimizer to minimize the objective function.

        Parameters
        ----------
        objective_func : callable
            Objective function to minimize. Should take an array and return a scalar.
        initial_point : array-like
            Initial point on the manifold for optimization.

        Returns:
        -------
        self : RiemannianSGD
            Returns the estimator instance for method chaining.
        """
        # Create manifold instance
        manifold = self._create_manifold(initial_point)

        # Create problem instance
        problem = RiemannianProblem(
            manifold=manifold,
            cost_fn=objective_func,
        )

        # Set up optimization options
        options = {
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "tolerance": self.tolerance,
        }

        # Run optimization
        self.optimization_result_ = minimize(
            problem=problem,
            x0=initial_point,
            method="rsgd",
            options=options,
        )

        return self


class RiemannianAdam(BaseEstimator, RiemannianEstimatorMixin):
    """Riemannian Adam estimator.

    This estimator provides a scikit-learn compatible interface for Riemannian
    optimization using the Adam algorithm.

    Parameters
    ----------
    manifold : str, default="sphere"
        Type of manifold for optimization. Supported types:
        "sphere", "stiefel", "grassmann", "so", "spd"
    learning_rate : float, default=0.001
        Learning rate for the optimization algorithm.
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    eps : float, default=1e-8
        Small constant for numerical stability.
    max_iterations : int, default=100
        Maximum number of optimization iterations.
    tolerance : float, default=1e-6
        Tolerance for convergence detection.
    random_state : int or None, default=None
        Random state for reproducible results.

    Attributes:
    ----------
    optimization_result_ : OptimizeResult
        Result of the optimization after fitting.

    Examples:
    --------
    >>> from riemannax.api import RiemannianAdam
    >>> import jax.numpy as jnp
    >>>
    >>> # Define objective function
    >>> def objective(x):
    ...     return jnp.sum((x - jnp.array([0, 0, 1]))**2)
    >>>
    >>> # Create estimator and fit
    >>> estimator = RiemannianAdam(manifold="sphere", learning_rate=0.001)
    >>> x0 = jnp.array([1.0, 0.0, 0.0])  # Initial point on unit sphere
    >>> estimator.fit(objective, x0)
    >>> print(f"Optimized point: {estimator.optimization_result_.x}")
    """

    def __init__(
        self,
        manifold: str = "sphere",
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        random_state: int | None = None,
    ):
        """Initialize the RiemannianAdam estimator.

        Parameters
        ----------
        manifold : str, default="sphere"
            Type of manifold for optimization.
        learning_rate : float, default=0.001
            Learning rate for the optimization algorithm.
        beta1 : float, default=0.9
            Exponential decay rate for first moment estimates.
        beta2 : float, default=0.999
            Exponential decay rate for second moment estimates.
        eps : float, default=1e-8
            Small constant for numerical stability.
        max_iterations : int, default=100
            Maximum number of optimization iterations.
        tolerance : float, default=1e-6
            Tolerance for convergence detection.
        random_state : int or None, default=None
            Random state for reproducible results.
        """
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        # Validate parameters
        self._validate_manifold(manifold)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate estimator parameters."""
        super()._validate_parameters()

        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")

        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def fit(
        self,
        objective_func: Callable[[Array], float],
        initial_point: Array,
    ) -> "RiemannianAdam":
        """Fit the Riemannian optimizer to minimize the objective function.

        Parameters
        ----------
        objective_func : callable
            Objective function to minimize. Should take an array and return a scalar.
        initial_point : array-like
            Initial point on the manifold for optimization.

        Returns:
        -------
        self : RiemannianAdam
            Returns the estimator instance for method chaining.
        """
        # Create manifold instance
        manifold = self._create_manifold(initial_point)

        # Create problem instance
        problem = RiemannianProblem(
            manifold=manifold,
            cost_fn=objective_func,
        )

        # Set up optimization options
        options = {
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "tolerance": self.tolerance,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }

        # Run optimization
        self.optimization_result_ = minimize(
            problem=problem,
            x0=initial_point,
            method="radam",
            options=options,
        )

        return self
