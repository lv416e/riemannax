"""Scikit-learn compatible estimator framework for RiemannAX."""

import abc
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..manifolds.spd import SymmetricPositiveDefinite
from ..manifolds.sphere import Sphere
from ..manifolds.stiefel import Stiefel
from ..optimizers.adam import riemannian_adam
from ..optimizers.sgd import riemannian_gradient_descent
from .detection import ManifoldDetector, ManifoldType
from .errors import ManifoldDetectionError, ParameterValidationError
from .results import ConvergenceStatus, OptimizationResult
from .validation import validate_learning_rate, validate_parameter_type


class _SOManifoldWrapper:
    """Wrapper for SO(n) manifold using Stiefel backend with det=+1 enforcement.

    This wrapper ensures that optimization iterates remain in SO(n) by projecting
    any matrix with det < 0 back to det = +1 by flipping the sign of one column.
    """

    def __init__(self, n: int):
        """Initialize SO(n) wrapper.

        Args:
            n: Dimension of the special orthogonal group SO(n).
        """
        self._stiefel = Stiefel(n=n, p=n)
        self.n = n

    def __getattr__(self, name):
        """Delegate all other methods to underlying Stiefel manifold."""
        return getattr(self._stiefel, name)

    def _ensure_det_positive(self, X: Array) -> Array:
        """Ensure matrix has positive determinant by flipping one column if needed.

        Args:
            X: Orthogonal matrix from O(n). Supports batched (..., n, n) inputs.

        Returns:
            Matrix in SO(n) with det(X) = +1.
        """
        det = jnp.linalg.det(X)
        # Flip the last column only when det < 0; supports batched (..., n, n) inputs.
        sign = jnp.where(det < 0, jnp.array(-1.0, dtype=X.dtype), jnp.array(1.0, dtype=X.dtype))
        return X.at[..., :, -1].set(sign[..., None] * X[..., :, -1])

    def retr(self, X: Array, U: Array) -> Array:
        """Retraction to SO(n) with det=+1 enforcement."""
        Y = self._stiefel.retr(X, U)
        return self._ensure_det_positive(Y)

    def exp(self, X: Array, U: Array) -> Array:
        """Exponential map to SO(n) with det=+1 enforcement."""
        Y = self._stiefel.exp(X, U)
        return self._ensure_det_positive(Y)


class RiemannianEstimator(abc.ABC):
    """Abstract base class for Riemannian optimization estimators.

    This class provides the scikit-learn compatible interface for Riemannian
    optimization, including parameter management, manifold detection, and
    standardized result reporting.
    """

    def __init__(
        self,
        manifold: str = "auto",
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        random_state: int | None = None,
    ):
        """Initialize Riemannian estimator.

        Args:
            manifold: Manifold type ("sphere", "stiefel", "spd", "so", "auto").
            learning_rate: Learning rate for optimization.
            max_iterations: Maximum number of optimization iterations.
            tolerance: Convergence tolerance.
            random_state: Random state for reproducibility.
        """
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        # Validate parameters
        self._validate_parameters()

        # State management
        self._is_fitted = False
        self._optimization_result: OptimizationResult | None = None
        self._detected_manifold_type: ManifoldType | None = None

    def _validate_parameters(self) -> None:
        """Validate estimator parameters."""
        # Validate learning rate
        lr_result = validate_learning_rate(self.learning_rate)
        if not lr_result.is_valid:
            raise ParameterValidationError(
                f"Invalid learning rate: {lr_result.violations[0]}",
                parameter_name="learning_rate",
                expected_type=float,
                received_value=self.learning_rate,
            )

        # Validate manifold type
        valid_manifolds = ["auto", "sphere", "stiefel", "spd", "so"]
        if self.manifold not in valid_manifolds:
            raise ParameterValidationError(
                f"Invalid manifold type '{self.manifold}'. Must be one of {valid_manifolds}",
                parameter_name="manifold",
                expected_type=str,
                received_value=self.manifold,
            )

        # Validate max_iterations
        iter_result = validate_parameter_type(self.max_iterations, int, "max_iterations")
        if not iter_result.is_valid:
            raise ParameterValidationError(
                f"Invalid max_iterations: {iter_result.violations[0]}",
                parameter_name="max_iterations",
                expected_type=int,
                received_value=self.max_iterations,
            )

        if self.max_iterations <= 0:
            raise ParameterValidationError(
                "max_iterations must be positive",
                parameter_name="max_iterations",
                expected_type=int,
                received_value=self.max_iterations,
            )

        # Validate tolerance
        if not isinstance(self.tolerance, (int, float)):  # noqa: UP038
            raise ParameterValidationError(
                "tolerance must be a number",
                parameter_name="tolerance",
                expected_type=float,
                received_value=self.tolerance,
            )
        if self.tolerance <= 0:
            raise ParameterValidationError(
                "tolerance must be positive",
                parameter_name="tolerance",
                expected_type=float,
                received_value=self.tolerance,
            )

        # Validate random_state
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ParameterValidationError(
                "random_state must be an int or None",
                parameter_name="random_state",
                expected_type=int,
                received_value=self.random_state,
            )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, return parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Dictionary of parameter names to their values.
        """
        return {
            "manifold": self.manifold,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "RiemannianEstimator":
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters to set.

        Returns:
            Self (for method chaining).

        Raises:
            ParameterValidationError: If invalid parameters are provided.
        """
        valid_params = set(self.get_params().keys())

        for key, value in params.items():
            if key not in valid_params:
                raise ParameterValidationError(
                    f"Invalid parameter '{key}' for estimator {self.__class__.__name__}",
                    parameter_name=key,
                    received_value=value,
                )
            setattr(self, key, value)

        # Re-validate after setting parameters
        self._validate_parameters()

        # Reset fitted state if parameters changed
        self._is_fitted = False

        return self

    def _detect_or_create_manifold(self, x0: Array) -> Any:
        """Detect or create manifold based on initial point and manifold specification.

        Args:
            x0: Initial point for optimization.

        Returns:
            Manifold instance for optimization.

        Raises:
            ManifoldDetectionError: If manifold detection fails.
        """
        if self.manifold == "auto":
            # Automatic manifold detection
            detection_result = ManifoldDetector.detect_manifold(x0)

            if detection_result.detected_type == ManifoldType.UNKNOWN:
                raise ManifoldDetectionError(
                    f"Could not automatically detect manifold type for input shape {x0.shape}. "
                    f"Errors: {detection_result.validation_errors}",
                    alternatives=[t.value for t in (detection_result.alternatives or [])],
                )

            if not detection_result.constraints_satisfied:
                raise ManifoldDetectionError(
                    f"Input does not satisfy constraints for detected manifold type "
                    f"{detection_result.detected_type.value}. "
                    f"Violations: {detection_result.validation_errors}",
                )

            self._detected_manifold_type = detection_result.detected_type
            manifold_type = detection_result.detected_type.value
        else:
            # Manual manifold specification
            manifold_type = self.manifold
            self._detected_manifold_type = ManifoldType(manifold_type)

            # Validate that initial point satisfies manifold constraints
            validation_result = ManifoldDetector.validate_constraints(x0, self._detected_manifold_type)
            if not validation_result.constraints_satisfied:
                raise ManifoldDetectionError(
                    f"Initial point does not satisfy {manifold_type} manifold constraints. "
                    f"Violations: {validation_result.validation_errors}",
                )

        # Create manifold instance
        return self._create_manifold_instance(manifold_type, x0)

    def _create_manifold_instance(self, manifold_type: str, x0: Array) -> Any:
        """Create manifold instance for given type and initial point.

        Args:
            manifold_type: Type of manifold to create.
            x0: Initial point to determine manifold dimensions.

        Returns:
            Manifold instance.
        """
        if manifold_type == "sphere":
            if x0.ndim != 1:
                raise ManifoldDetectionError(f"Sphere manifold requires 1D vectors, got {x0.ndim}D")
            n = x0.shape[0] - 1
            if n < 1:
                raise ManifoldDetectionError("Sphere S^0 is not supported (require len(x0) >= 2)")
            return Sphere(n=n)
        elif manifold_type == "stiefel":
            if x0.ndim != 2:
                raise ManifoldDetectionError(f"Stiefel manifold requires 2D arrays, got {x0.ndim}D")
            m, n = x0.shape
            return Stiefel(n=m, p=n)
        elif manifold_type == "spd":
            if x0.ndim != 2 or x0.shape[0] != x0.shape[1]:
                raise ManifoldDetectionError(f"SPD manifold requires square matrices, got shape {x0.shape}")
            return SymmetricPositiveDefinite(n=x0.shape[0])
        elif manifold_type == "so":
            if x0.ndim != 2 or x0.shape[0] != x0.shape[1]:
                raise ManifoldDetectionError(f"SO manifold requires square matrices, got shape {x0.shape}")
            # Use dedicated SO(n) wrapper that ensures det=+1 constraint
            return _SOManifoldWrapper(n=x0.shape[0])
        else:
            raise ManifoldDetectionError(f"Unsupported manifold type: {manifold_type}")

    @abc.abstractmethod
    def _create_optimizer(self) -> tuple[Callable, Callable]:
        """Create optimizer init and update functions.

        Returns:
            Tuple of (init_fn, update_fn) for optimization.
        """
        pass

    def fit(self, objective_func: Callable[[Array], float], x0: Array) -> OptimizationResult:
        """Fit the estimator to the optimization problem.

        Args:
            objective_func: Objective function to minimize.
            x0: Initial point on the manifold.

        Returns:
            OptimizationResult with optimization outcome and metadata.
        """
        # Detect/create manifold
        manifold = self._detect_or_create_manifold(x0)

        # Create optimizer
        init_fn, update_fn = self._create_optimizer()

        # Initialize optimizer state
        state = init_fn(x0)

        # Create gradient function
        grad_fn = jax.grad(objective_func)

        # Optimization loop with simple convergence checking
        converged = False
        iteration_count = 0
        grad_norm = float("inf")  # Initialize with infinity
        for _iteration in range(self.max_iterations):
            iteration_count = _iteration + 1  # Track number of iterations performed
            current_x = state.x
            gradient = grad_fn(current_x)

            # Project gradient to tangent space
            riemannian_grad = manifold.proj(current_x, gradient)

            # Update state
            state = update_fn(riemannian_grad, state, manifold)

            # Check convergence (simple gradient norm criterion)
            grad_norm = float(jnp.linalg.norm(riemannian_grad))
            if grad_norm < self.tolerance:
                converged = True
                break

        # Determine convergence status
        convergence_status = ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.MAX_ITERATIONS

        # Compute final objective value
        final_objective = float(objective_func(state.x))

        # Create result
        self._optimization_result = OptimizationResult(
            optimized_params=state.x,
            objective_value=final_objective,
            convergence_status=convergence_status,
            iteration_count=iteration_count,
            metadata={
                "manifold_type": self._detected_manifold_type.value if self._detected_manifold_type else self.manifold,
                "final_gradient_norm": grad_norm,
                "tolerance": self.tolerance,
            },
        )

        self._is_fitted = True

        return self._optimization_result

    def predict(self, X: Array) -> Array:
        """Predict method (not implemented for optimization estimators).

        Args:
            X: Input data.

        Returns:
            Predictions.

        Raises:
            NotImplementedError: Always, as optimization estimators don't predict.
        """
        raise NotImplementedError("Optimization estimators do not implement predict()")

    def score(self, objective_func: Callable[[Array], float], X: Array) -> float:
        """Return the score on the given test data.

        Args:
            objective_func: Objective function to evaluate.
            X: Test input.

        Returns:
            Negative objective value (higher is better for sklearn compatibility).
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before calling score()")

        return -float(objective_func(X))


class RiemannianSGD(RiemannianEstimator):
    """Riemannian Stochastic Gradient Descent estimator.

    Scikit-learn compatible estimator for Riemannian optimization using
    stochastic gradient descent with exponential map or retraction.
    """

    def __init__(
        self,
        manifold: str = "auto",
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        use_retraction: bool = False,
        random_state: int | None = None,
    ):
        """Initialize Riemannian SGD estimator.

        Args:
            manifold: Manifold type ("sphere", "stiefel", "spd", "so", "auto").
            learning_rate: Learning rate for optimization.
            max_iterations: Maximum number of optimization iterations.
            tolerance: Convergence tolerance.
            use_retraction: Whether to use retraction instead of exponential map.
            random_state: Random state for reproducibility.
        """
        self.use_retraction = use_retraction
        super().__init__(manifold, learning_rate, max_iterations, tolerance, random_state)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params["use_retraction"] = self.use_retraction
        return params

    def _create_optimizer(self) -> tuple[Callable, Callable]:
        """Create Riemannian SGD optimizer."""
        return riemannian_gradient_descent(learning_rate=self.learning_rate, use_retraction=self.use_retraction)


class RiemannianAdam(RiemannianEstimator):
    """Riemannian Adam estimator.

    Scikit-learn compatible estimator for Riemannian optimization using
    the Adam algorithm adapted for Riemannian manifolds.
    """

    def __init__(
        self,
        manifold: str = "auto",
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        use_retraction: bool = False,
        random_state: int | None = None,
    ):
        """Initialize Riemannian Adam estimator.

        Args:
            manifold: Manifold type ("sphere", "stiefel", "spd", "so", "auto").
            learning_rate: Learning rate for optimization.
            max_iterations: Maximum number of optimization iterations.
            tolerance: Convergence tolerance.
            beta1: Exponential decay rate for first moment estimates.
            beta2: Exponential decay rate for second moment estimates.
            eps: Small constant for numerical stability.
            use_retraction: Whether to use retraction instead of exponential map.
            random_state: Random state for reproducibility.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_retraction = use_retraction

        super().__init__(manifold, learning_rate, max_iterations, tolerance, random_state)

        # Additional parameter validation for Adam
        self._validate_adam_parameters()

    def _validate_adam_parameters(self) -> None:
        """Validate Adam-specific parameters."""
        if not (0.0 <= self.beta1 < 1.0):
            raise ParameterValidationError(
                f"beta1 must be in [0, 1), got {self.beta1}",
                parameter_name="beta1",
                expected_type=float,
                received_value=self.beta1,
            )

        if not (0.0 <= self.beta2 < 1.0):
            raise ParameterValidationError(
                f"beta2 must be in [0, 1), got {self.beta2}",
                parameter_name="beta2",
                expected_type=float,
                received_value=self.beta2,
            )

        if self.eps <= 0:
            raise ParameterValidationError(
                f"eps must be positive, got {self.eps}",
                parameter_name="eps",
                expected_type=float,
                received_value=self.eps,
            )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
                "use_retraction": self.use_retraction,
            }
        )
        return params

    def set_params(self, **params: Any) -> "RiemannianAdam":
        """Set parameters with Adam-specific validation."""
        super().set_params(**params)
        self._validate_adam_parameters()
        return self

    def _create_optimizer(self) -> tuple[Callable, Callable]:
        """Create Riemannian Adam optimizer."""
        return riemannian_adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            use_retraction=self.use_retraction,
        )
