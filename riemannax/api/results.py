"""Optimization result standardization for RiemannAX high-level APIs."""

import dataclasses
from enum import Enum
from typing import Any

from jaxtyping import Array


class ConvergenceStatus(Enum):
    """Enumeration of possible optimization convergence statuses."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations_reached"
    FAILED = "failed"
    STOPPED = "stopped_by_user"


@dataclasses.dataclass
class OptimizationResult:
    """Standardized optimization result with enhanced metadata and compatibility.

    This class extends the functionality of the basic OptimizeResult to provide
    more detailed information about optimization outcomes while maintaining
    backward compatibility with existing code.

    Attributes:
        optimized_params: Final optimized parameters on the manifold.
        objective_value: Final objective function value.
        convergence_status: Detailed convergence status information.
        iteration_count: Number of optimization iterations performed.
        metadata: Additional optimization metadata and diagnostics.
    """

    optimized_params: Array
    objective_value: float
    convergence_status: ConvergenceStatus
    iteration_count: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize default metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def x(self) -> Array:
        """Alias for optimized_params (backward compatibility)."""
        return self.optimized_params

    @property
    def fun(self) -> float:
        """Alias for objective_value (backward compatibility)."""
        return self.objective_value

    @property
    def success(self) -> bool:
        """Whether optimization was successful (backward compatibility)."""
        return self.convergence_status == ConvergenceStatus.CONVERGED

    @property
    def niter(self) -> int:
        """Alias for iteration_count (backward compatibility)."""
        return self.iteration_count

    @property
    def message(self) -> str:
        """Descriptive message about optimization outcome (backward compatibility)."""
        status_messages = {
            ConvergenceStatus.CONVERGED: "Optimization terminated successfully.",
            ConvergenceStatus.MAX_ITERATIONS: "Maximum number of iterations reached.",
            ConvergenceStatus.FAILED: "Optimization failed to converge.",
            ConvergenceStatus.STOPPED: "Optimization stopped by user.",
        }
        return status_messages.get(self.convergence_status, "Unknown convergence status.")
