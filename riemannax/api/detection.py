"""Automatic manifold detection for Riemannian optimization."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from ..manifolds import (
    create_grassmann,
    create_so,
    create_spd,
    create_sphere,
    create_stiefel,
)
from ..problems import RiemannianProblem
from ..solvers import OptimizeResult
from ..solvers import minimize as _minimize

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""

    constraint_type: str
    description: str
    severity: str  # "error", "warning"
    suggestion: str


@dataclass
class ValidationResult:
    """Result of manifold constraint validation."""

    is_valid: bool
    violations: list[ConstraintViolation]
    suggestions: list[str]


@dataclass
class ManifoldCandidate:
    """A potential manifold match with confidence score."""

    manifold_type: str
    confidence: float
    constraints_satisfied: bool
    reasoning: str


class ManifoldDetector:
    """Automatic manifold detection from array structure and constraints.

    This class analyzes the mathematical properties of input arrays to automatically
    detect the most appropriate Riemannian manifold for optimization.

    Parameters
    ----------
    unit_tol : float, default=1e-6
        Tolerance for unit vector detection.
    orthogonal_tol : float, default=1e-6
        Tolerance for orthogonality detection.
    symmetry_tol : float, default=1e-8
        Tolerance for symmetry detection.
    positive_definite_tol : float, default=1e-8
        Tolerance for positive definiteness detection.

    Examples:
    --------
    >>> import jax.numpy as jnp
    >>> from riemannax.api.detection import ManifoldDetector
    >>>
    >>> detector = ManifoldDetector()
    >>>
    >>> # Detect sphere manifold from unit vector
    >>> x = jnp.array([1.0, 0.0, 0.0])
    >>> manifold_type = detector.detect_manifold(x)
    >>> print(f"Detected: {manifold_type}")  # "sphere"
    >>>
    >>> # Detect SPD manifold from positive definite matrix
    >>> X = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    >>> manifold_type = detector.detect_manifold(X)
    >>> print(f"Detected: {manifold_type}")  # "spd"
    """

    def __init__(
        self,
        unit_tol: float = 1e-6,
        orthogonal_tol: float = 1e-6,
        symmetry_tol: float = 1e-8,
        positive_definite_tol: float = 1e-8,
    ):
        """Initialize the manifold detector.

        Parameters
        ----------
        unit_tol : float, default=1e-6
            Tolerance for unit vector detection.
        orthogonal_tol : float, default=1e-6
            Tolerance for orthogonality detection.
        symmetry_tol : float, default=1e-8
            Tolerance for symmetry detection.
        positive_definite_tol : float, default=1e-8
            Tolerance for positive definiteness detection.
        """
        self.unit_tol = unit_tol
        self.orthogonal_tol = orthogonal_tol
        self.symmetry_tol = symmetry_tol
        self.positive_definite_tol = positive_definite_tol

    def detect_manifold(self, x: Array) -> str:
        """Automatically detect manifold type from array structure.

        Analyzes the mathematical properties of the input array to determine
        the most appropriate manifold type for optimization.

        Parameters
        ----------
        x : array-like
            Input array to analyze for manifold detection.

        Returns:
        -------
        str
            Detected manifold type ("sphere", "stiefel", "grassmann", "so", "spd").

        Raises:
        ------
        ValueError
            If no unique manifold can be detected from the input.

        Examples:
        --------
        >>> detector = ManifoldDetector()
        >>> x = jnp.array([0.6, 0.8, 0.0])  # Unit vector
        >>> manifold_type = detector.detect_manifold(x)
        >>> print(manifold_type)  # "sphere"
        """
        x = jnp.asarray(x)

        # Check for unit vector (sphere manifold)
        if x.ndim == 1:
            norm = jnp.linalg.norm(x)
            if abs(norm - 1.0) < self.unit_tol:
                return "sphere"

        # Check for matrices
        elif x.ndim == 2:
            rows, cols = x.shape

            # Check for square matrices
            if rows == cols:
                # Check for symmetric positive definite
                if self._is_symmetric(x) and self._is_positive_definite(x):
                    return "spd"

                # Check for orthogonal matrix with determinant 1 (SO manifold)
                if self._is_orthogonal(x) and abs(jnp.linalg.det(x) - 1.0) < self.unit_tol:
                    return "so"

                # Check for orthogonal matrix with determinant -1 or other (Stiefel)
                if self._is_orthogonal(x):
                    return "stiefel"

            # Check for rectangular orthogonal matrices (Stiefel/Grassmann)
            else:
                if self._is_orthogonal(x):
                    # For now, default to Stiefel for orthogonal matrices
                    # Grassmann manifold distinction would require additional context
                    return "stiefel"

        # If no clear detection, try suggestions
        suggestions = self.suggest_manifold(x)
        if suggestions:
            # If the best suggestion has high confidence and satisfies constraints, use it
            best_suggestion = suggestions[0]
            if best_suggestion.confidence > 0.8 and best_suggestion.constraints_satisfied:
                return best_suggestion.manifold_type

            # Otherwise, raise error with suggestions
            suggestion_text = ", ".join([f"{s.manifold_type} (confidence: {s.confidence:.2f})" for s in suggestions])
            raise ValueError(
                f"Could not automatically detect manifold for array with shape {x.shape}. "
                f"Consider manually specifying manifold type. Suggestions: {suggestion_text}"
            )
        else:
            raise ValueError(
                f"Could not automatically detect manifold for array with shape {x.shape}. "
                f"Array does not satisfy constraints for any supported manifold type. "
                f"Supported types: sphere (unit vectors), stiefel (orthogonal matrices), "
                f"grassmann (subspaces), so (rotation matrices), spd (symmetric positive definite)."
            )

    def validate_constraints(self, x: Array, manifold_type: str) -> ValidationResult:
        """Validate that an array satisfies constraints for a given manifold type.

        Parameters
        ----------
        x : array-like
            Array to validate.
        manifold_type : str
            Manifold type to validate against.

        Returns:
        -------
        ValidationResult
            Validation result with violations and suggestions.

        Examples:
        --------
        >>> detector = ManifoldDetector()
        >>> x = jnp.array([0.6, 0.8, 0.0])
        >>> result = detector.validate_constraints(x, "sphere")
        >>> print(result.is_valid)  # True
        """
        x = jnp.asarray(x)
        violations: list[ConstraintViolation] = []
        suggestions: list[str] = []

        if manifold_type == "sphere":
            # Check if it's a 1D array
            if x.ndim != 1:
                violations.append(
                    ConstraintViolation(
                        constraint_type="dimension",
                        description=f"Sphere manifold requires 1D array, got {x.ndim}D",
                        severity="error",
                        suggestion="Use a 1D array for sphere manifold",
                    )
                )
            else:
                # Check unit norm
                norm = jnp.linalg.norm(x)
                if abs(norm - 1.0) > self.unit_tol:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="unit_norm",
                            description=f"Vector norm is {norm:.6f}, should be 1.0",
                            severity="error",
                            suggestion="Normalize the vector: x = x / jnp.linalg.norm(x)",
                        )
                    )

        elif manifold_type == "stiefel":
            # Check if it's a 2D array
            if x.ndim != 2:
                violations.append(
                    ConstraintViolation(
                        constraint_type="dimension",
                        description=f"Stiefel manifold requires 2D array, got {x.ndim}D",
                        severity="error",
                        suggestion="Use a 2D array for Stiefel manifold",
                    )
                )
            else:
                # Check orthogonality
                if not self._is_orthogonal(x):
                    violations.append(
                        ConstraintViolation(
                            constraint_type="orthogonality",
                            description="Matrix columns are not orthonormal",
                            severity="error",
                            suggestion="Use QR decomposition: Q, _ = jnp.linalg.qr(x)",
                        )
                    )

        elif manifold_type == "spd":
            # Check if it's a square 2D array
            if x.ndim != 2 or x.shape[0] != x.shape[1]:
                violations.append(
                    ConstraintViolation(
                        constraint_type="dimension",
                        description=f"SPD manifold requires square 2D array, got shape {x.shape}",
                        severity="error",
                        suggestion="Use a square matrix for SPD manifold",
                    )
                )
            else:
                # Check symmetry
                if not self._is_symmetric(x):
                    violations.append(
                        ConstraintViolation(
                            constraint_type="symmetry",
                            description="Matrix is not symmetric",
                            severity="error",
                            suggestion="Make matrix symmetric: X = (X + X.T) / 2",
                        )
                    )

                # Check positive definiteness
                if not self._is_positive_definite(x):
                    violations.append(
                        ConstraintViolation(
                            constraint_type="positive_definite",
                            description="Matrix is not positive definite",
                            severity="error",
                            suggestion="Ensure all eigenvalues are positive",
                        )
                    )

        elif manifold_type == "so":
            # Check if it's a square 2D array
            if x.ndim != 2 or x.shape[0] != x.shape[1]:
                violations.append(
                    ConstraintViolation(
                        constraint_type="dimension",
                        description=f"SO manifold requires square 2D array, got shape {x.shape}",
                        severity="error",
                        suggestion="Use a square matrix for SO manifold",
                    )
                )
            else:
                # Check orthogonality
                if not self._is_orthogonal(x):
                    violations.append(
                        ConstraintViolation(
                            constraint_type="orthogonality",
                            description="Matrix is not orthogonal",
                            severity="error",
                            suggestion="Use proper rotation matrix with det=1",
                        )
                    )

                # Check determinant = 1
                det = jnp.linalg.det(x)
                if abs(det - 1.0) > self.unit_tol:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="determinant",
                            description=f"Matrix determinant is {det:.6f}, should be 1.0",
                            severity="error",
                            suggestion="Ensure determinant is 1 for rotation matrices",
                        )
                    )

        is_valid = len(violations) == 0
        return ValidationResult(is_valid=is_valid, violations=violations, suggestions=suggestions)

    def suggest_manifold(self, x: Array) -> list[ManifoldCandidate]:
        """Suggest possible manifold types for ambiguous cases.

        Parameters
        ----------
        x : array-like
            Array to analyze for manifold suggestions.

        Returns:
        -------
        list[ManifoldCandidate]
            List of potential manifold candidates with confidence scores.

        Examples:
        --------
        >>> detector = ManifoldDetector()
        >>> x = jnp.array([0.99, 0.1, 0.1])  # Almost unit vector
        >>> suggestions = detector.suggest_manifold(x)
        >>> for candidate in suggestions:
        ...     print(f"{candidate.manifold_type}: {candidate.confidence:.2f}")
        """
        x = jnp.asarray(x)
        candidates = []

        # Check sphere candidacy
        if x.ndim == 1:
            norm = jnp.linalg.norm(x)
            norm_error = abs(norm - 1.0)
            if norm_error < 0.1:  # Within 10% of unit norm
                confidence = max(0.0, 1.0 - norm_error * 10)
                candidates.append(
                    ManifoldCandidate(
                        manifold_type="sphere",
                        confidence=confidence,
                        constraints_satisfied=norm_error < self.unit_tol,
                        reasoning=f"Vector norm {norm:.4f} is close to 1.0",
                    )
                )

        # Check matrix manifolds
        elif x.ndim == 2:
            rows, cols = x.shape

            # Check Stiefel/SO candidacy
            orthogonality_error = self._orthogonality_error(x)
            if orthogonality_error < 0.1:  # Reasonably orthogonal
                confidence = max(0.0, 1.0 - orthogonality_error * 10)

                if rows == cols:
                    det = jnp.linalg.det(x)
                    if abs(det - 1.0) < 0.1:
                        candidates.append(
                            ManifoldCandidate(
                                manifold_type="so",
                                confidence=confidence * (1.0 - abs(det - 1.0) * 10),
                                constraints_satisfied=orthogonality_error < self.orthogonal_tol
                                and abs(det - 1.0) < self.unit_tol,
                                reasoning=f"Square orthogonal matrix with det≈1 (det={det:.4f})",
                            )
                        )
                    else:
                        candidates.append(
                            ManifoldCandidate(
                                manifold_type="stiefel",
                                confidence=confidence,
                                constraints_satisfied=orthogonality_error < self.orthogonal_tol,
                                reasoning=f"Square orthogonal matrix with det≠1 (det={det:.4f})",
                            )
                        )
                else:
                    candidates.append(
                        ManifoldCandidate(
                            manifold_type="stiefel",
                            confidence=confidence,
                            constraints_satisfied=orthogonality_error < self.orthogonal_tol,
                            reasoning=f"Rectangular matrix with orthogonal columns ({rows}x{cols})",
                        )
                    )

            # Check SPD candidacy (only for square matrices)
            if rows == cols:
                symmetry_error = self._symmetry_error(x)
                if symmetry_error < 0.1:  # Reasonably symmetric
                    try:
                        eigenvals = jnp.linalg.eigvals(x)
                        min_eigenval = jnp.min(eigenvals)
                        if min_eigenval > -0.1:  # Not too negative
                            pd_confidence = max(0.0, min(1.0, min_eigenval * 10 + 1.0))
                            sym_confidence = max(0.0, 1.0 - symmetry_error * 10)
                            confidence = pd_confidence * sym_confidence

                            candidates.append(
                                ManifoldCandidate(
                                    manifold_type="spd",
                                    confidence=confidence,
                                    constraints_satisfied=symmetry_error < self.symmetry_tol
                                    and min_eigenval > self.positive_definite_tol,
                                    reasoning=f"Symmetric matrix with min eigenvalue {min_eigenval:.4f}",
                                )
                            )
                    except Exception:
                        pass  # Skip if eigenvalue computation fails

        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def _is_symmetric(self, x: Array) -> bool:
        """Check if matrix is symmetric within tolerance."""
        return jnp.allclose(x, x.T, atol=self.symmetry_tol)

    def _is_positive_definite(self, x: Array) -> bool:
        """Check if matrix is positive definite."""
        try:
            eigenvals = jnp.linalg.eigvals(x)
            return jnp.all(eigenvals > self.positive_definite_tol)
        except Exception:
            return False

    def _is_orthogonal(self, x: Array) -> bool:
        """Check if matrix has orthonormal rows/columns."""
        rows, cols = x.shape

        if rows <= cols:
            # More columns than rows: check if rows are orthonormal (X @ X.T = I)
            gram = x @ x.T
            identity = jnp.eye(rows)
        else:
            # More rows than columns: check if columns are orthonormal (X.T @ X = I)
            gram = x.T @ x
            identity = jnp.eye(cols)

        return jnp.allclose(gram, identity, atol=self.orthogonal_tol)

    def _orthogonality_error(self, x: Array) -> float:
        """Compute orthogonality error for matrix."""
        rows, cols = x.shape

        if rows <= cols:
            # More columns than rows: check if rows are orthonormal (X @ X.T = I)
            gram = x @ x.T
            identity = jnp.eye(rows)
        else:
            # More rows than columns: check if columns are orthonormal (X.T @ X = I)
            gram = x.T @ x
            identity = jnp.eye(cols)

        return float(jnp.max(jnp.abs(gram - identity)))

    def _symmetry_error(self, x: Array) -> float:
        """Compute symmetry error for matrix."""
        return float(jnp.max(jnp.abs(x - x.T)))


def minimize(
    objective_func: Callable[[Array], float],
    x0: Array,
    method: str = "riemannian_adam",
    options: dict[str, Any] | None = None,
    manifold: str | None = None,
) -> OptimizeResult:
    """Minimize a function with automatic manifold detection.

    This function automatically detects the appropriate Riemannian manifold
    from the initial point structure and performs optimization using the
    specified method.

    Parameters
    ----------
    objective_func : callable
        Objective function to minimize. Should take an array and return a scalar.
    x0 : array-like
        Initial point for optimization.
    method : str, default="riemannian_adam"
        Optimization method to use. Supported methods:
        - "riemannian_adam": Riemannian Adam optimizer
        - "riemannian_sgd": Riemannian gradient descent
        - "riemannian_momentum": Riemannian momentum
    options : dict, optional
        Additional options for the optimizer.
    manifold : str, optional
        Manual manifold specification. If provided, automatic detection is skipped.

    Returns:
    -------
    OptimizeResult
        Optimization result containing solution and metadata.

    Raises:
    ------
    ValueError
        If manifold detection fails or optimization method is unsupported.

    Examples:
    --------
    >>> import jax.numpy as jnp
    >>> from riemannax.api.detection import minimize
    >>>
    >>> # Automatic detection for unit vector (sphere manifold)
    >>> def objective(x):
    ...     return jnp.sum((x - jnp.array([0, 0, 1]))**2)
    >>> x0 = jnp.array([1.0, 0.0, 0.0])
    >>> result = minimize(objective, x0, method="riemannian_adam")
    >>> print(f"Optimized point: {result.x}")
    """
    if options is None:
        options = {}

    x0 = jnp.asarray(x0)

    # Automatic manifold detection if not provided
    if manifold is None:
        detector = ManifoldDetector()
        manifold = detector.detect_manifold(x0)
        logger.info(f"Selected manifold: {manifold}")

    # Create manifold instance
    manifold_instance: Any
    if manifold == "sphere":
        n_dim = len(x0) - 1  # S^n embedded in R^{n+1}
        manifold_instance = create_sphere(n_dim)
    elif manifold == "stiefel":
        n, p = x0.shape  # n rows, p columns for St(p,n)
        manifold_instance = create_stiefel(p, n)
    elif manifold == "grassmann":
        n, p = x0.shape  # n rows, p columns for Gr(p,n)
        manifold_instance = create_grassmann(p, n)
    elif manifold == "so":
        n = x0.shape[0]
        manifold_instance = create_so(n)
    elif manifold == "spd":
        n = x0.shape[0]
        manifold_instance = create_spd(n)
    else:
        raise ValueError(f"Unsupported manifold type: {manifold}")

    # Create problem instance
    problem = RiemannianProblem(
        manifold=manifold_instance,
        cost_fn=objective_func,
    )

    # Map method names
    method_map = {
        "riemannian_adam": "radam",
        "riemannian_sgd": "rsgd",
        "riemannian_momentum": "rmom",
    }

    if method not in method_map:
        available_methods = ", ".join(method_map.keys())
        raise ValueError(f"Unsupported optimization method '{method}'. Available methods: {available_methods}")

    # Run optimization
    return _minimize(
        problem=problem,
        x0=x0,
        method=method_map[method],
        options=options,
    )
