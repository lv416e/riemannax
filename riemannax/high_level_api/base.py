"""Base estimator class for scikit-learn compatible Riemannian optimizers.

This module provides the foundation for all high-level Riemannian optimization
estimators with scikit-learn compatible interfaces.

Requirements Coverage:
- R1.1: String manifold specification constructor
- R1.3: get_params()/set_params() methods
- R1.4: Error handling for invalid manifold specifications
- R8.1: Specific exception types with detailed error messages
- R8.2: Constraint violation detection
"""

from typing import Any, ClassVar

from sklearn.base import BaseEstimator

from .exceptions import InvalidManifoldError, ParameterValidationError


class RiemannianEstimator(BaseEstimator):
    """Base class for scikit-learn compatible Riemannian optimizers.

    This class provides the foundation for all high-level Riemannian optimization
    estimators, ensuring consistent interfaces and parameter management across
    different optimization algorithms.

    Parameters
    ----------
    manifold : str, default="sphere"
        Name of the manifold to optimize on. Supported manifolds:
        - "sphere": Unit hypersphere
        - "grassmann": Grassmann manifold
        - "stiefel": Stiefel manifold
        - "spd": Symmetric positive definite matrices
        - "so": Special orthogonal group
    lr : float, default=0.01
        Learning rate for optimization. Must be positive.
    max_iter : int, default=1000
        Maximum number of optimization iterations. Must be positive.
    tol : float, default=1e-6
        Tolerance for convergence. Must be positive.

    Attributes:
    ----------
    manifold : str
        The manifold name specified during construction.
    lr : float
        The learning rate for optimization.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Examples:
    --------
    >>> from riemannax.high_level_api import RiemannianEstimator
    >>> estimator = RiemannianEstimator(manifold="sphere", lr=0.01)
    >>> params = estimator.get_params()
    >>> estimator.set_params(lr=0.02)
    RiemannianEstimator(manifold='sphere', lr=0.02, max_iter=1000, tol=1e-06)
    """

    _SUPPORTED_MANIFOLDS: ClassVar[list[str]] = ["sphere", "grassmann", "stiefel", "spd", "so"]

    def __init__(self, manifold: str = "sphere", lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize RiemannianEstimator.

        Parameters are validated and stored as instance attributes.
        Following scikit-learn conventions, all validation is performed
        during parameter setting.

        Parameters
        ----------
        manifold : str, default="sphere"
            Name of the manifold to optimize on
        lr : float, default=0.01
            Learning rate for optimization
        max_iter : int, default=1000
            Maximum number of optimization iterations
        tol : float, default=1e-6
            Tolerance for convergence
        """
        # Store all parameters as attributes (required for sklearn compatibility)
        # Validation is performed in separate methods
        self.manifold = manifold
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

        # Validate all parameters after storing them
        self._validate_all_parameters()

    def set_params(self, **params) -> "RiemannianEstimator":
        """Set the parameters of this estimator.

        This method follows scikit-learn conventions for parameter setting
        and includes validation of all parameters.

        Parameters
        ----------
        **params : dict
            Estimator parameters to set

        Returns:
        -------
        self : RiemannianEstimator
            The estimator instance

        Raises:
        ------
        InvalidManifoldError
            If manifold parameter is invalid
        ParameterValidationError
            If any other parameter is invalid
        """
        # Use parent class method to handle nested parameters and basic validation
        # This ensures compatibility with sklearn's parameter handling
        super().set_params(**params)

        # Validate all parameters after they've been set
        self._validate_all_parameters()

        return self

    @classmethod
    def get_supported_manifolds(cls) -> list[str]:
        """Get list of supported manifold names.

        Returns:
        -------
        List[str]
            List of supported manifold names
        """
        return cls._SUPPORTED_MANIFOLDS.copy()

    def _validate_all_parameters(self) -> None:
        """Validate all estimator parameters.

        This method centralizes parameter validation to ensure consistency
        and maintainability.

        Raises:
        ------
        InvalidManifoldError
            If manifold parameter is invalid
        ParameterValidationError
            If any other parameter is invalid
        """
        self._validate_manifold(self.manifold)
        self._validate_lr(self.lr)
        self._validate_max_iter(self.max_iter)
        self._validate_tol(self.tol)

    def _validate_manifold(self, manifold: Any) -> None:
        """Validate manifold specification.

        Parameters
        ----------
        manifold : Any
            Manifold specification to validate

        Raises:
        ------
        InvalidManifoldError
            If manifold specification is invalid
        """
        if manifold is None:
            raise InvalidManifoldError(manifold, self._SUPPORTED_MANIFOLDS)

        if not isinstance(manifold, str):
            raise InvalidManifoldError(manifold, self._SUPPORTED_MANIFOLDS)

        if manifold not in self._SUPPORTED_MANIFOLDS:
            raise InvalidManifoldError(manifold, self._SUPPORTED_MANIFOLDS)

    def _validate_lr(self, lr: Any) -> None:
        """Validate learning rate parameter.

        Parameters
        ----------
        lr : Any
            Learning rate to validate

        Raises:
        ------
        ParameterValidationError
            If learning rate is invalid
        """
        if not isinstance(lr, int | float):
            raise ParameterValidationError("lr", lr, "learning rate must be a numeric value (int or float)")

        if lr <= 0:
            raise ParameterValidationError("lr", lr, "learning rate must be positive (> 0)")

    def _validate_max_iter(self, max_iter: Any) -> None:
        """Validate max_iter parameter.

        Parameters
        ----------
        max_iter : Any
            Maximum iterations to validate

        Raises:
        ------
        ParameterValidationError
            If max_iter is invalid
        """
        if not isinstance(max_iter, int):
            raise ParameterValidationError("max_iter", max_iter, "must be an integer")

        if max_iter <= 0:
            raise ParameterValidationError("max_iter", max_iter, "must be positive (> 0)")

    def _validate_tol(self, tol: Any) -> None:
        """Validate tolerance parameter.

        Parameters
        ----------
        tol : Any
            Tolerance to validate

        Raises:
        ------
        ParameterValidationError
            If tolerance is invalid
        """
        if not isinstance(tol, int | float):
            raise ParameterValidationError("tol", tol, "tolerance must be a numeric value (int or float)")

        if tol <= 0:
            raise ParameterValidationError("tol", tol, "tolerance must be positive (> 0)")

    def __repr__(self) -> str:
        """Return string representation of the estimator.

        Returns:
        -------
        str
            String representation including all parameters
        """
        params = self.get_params(deep=False)
        param_strs = [f"{k}={v!r}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(param_strs)})"
