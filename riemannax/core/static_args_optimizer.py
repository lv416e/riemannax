"""Static argument optimization system for RiemannAX manifolds.

This module provides intelligent optimization of static_argnums configurations
for JAX JIT compilation across all manifold operations.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

import jax

logger = logging.getLogger(__name__)


class StaticArgsOptimizer:
    """Optimizer for static argument configurations in manifold operations."""

    def __init__(self) -> None:
        """Initialize the static argument optimizer."""
        self._optimization_cache: dict[str, Any] = {}
        self._analysis_results: dict[str, Any] = {}

    def analyze_function_signature(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Analyze function signature to understand argument patterns.

        Args:
            func: Function to analyze

        Returns:
            Dictionary containing signature analysis
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        analysis: dict[str, Any] = {
            "total_params": len(params),
            "param_names": [p.name for p in params],
            "param_kinds": [p.kind.name for p in params],
            "has_self": len(params) > 0 and params[0].name == "self",
            "data_param_positions": [],
            "potential_static_positions": [],
        }

        # Identify data parameters (typically x, v, y, etc.)
        data_param_names = {"x", "v", "y", "u", "point", "tangent", "vector"}
        for i, param in enumerate(params):
            if param.name.lower() in data_param_names:
                analysis["data_param_positions"].append(i)
            elif param.name == "self" and i == 0:
                # 'self' parameter is potentially static for manifold methods
                analysis["potential_static_positions"].append(i)

        return analysis

    def recommend_static_args(self, func: Callable[..., Any], manifold: Any = None) -> tuple[int, ...]:
        """Recommend optimal static argument configuration.

        Args:
            func: Function to optimize
            manifold: Optional manifold instance for context

        Returns:
            Tuple of recommended static argument positions
        """
        analysis = self.analyze_function_signature(func)

        # For manifold methods, we generally don't want to make data static
        # But we might consider making 'self' static in certain cases
        recommended_positions: list[int] = []

        # Strategy 1: For unbound methods (functions), don't use static args by default
        # This is safest and avoids the dimension value confusion
        if not analysis["has_self"]:
            return ()

        # Strategy 2: For bound methods with 'self', consider making 'self' static
        # only if the manifold parameters are truly constant
        if analysis["has_self"] and manifold is not None and self._has_constant_manifold_parameters(manifold):
            # Check if manifold has constant parameters that could benefit from static treatment
            # Making 'self' static can help with dimension-dependent optimizations
            # but we need to be careful about when this is beneficial
            pass  # For now, be conservative and don't make 'self' static

        return tuple(recommended_positions)

    def _has_constant_manifold_parameters(self, manifold: Any) -> bool:
        """Check if manifold has parameters that are constant during optimization.

        Args:
            manifold: Manifold instance

        Returns:
            True if manifold parameters are constant
        """
        # Parameters like dimension (n, p) are typically constant
        # but making them static requires careful consideration
        constant_params = []

        if hasattr(manifold, "n") and isinstance(manifold.n, int):
            constant_params.append("n")
        if hasattr(manifold, "p") and isinstance(manifold.p, int):
            constant_params.append("p")

        # For now, return False to be conservative
        # This can be enabled after more thorough testing
        return len(constant_params) > 0 and False  # Disabled for safety

    def optimize_manifold_static_args(self, manifold: Any) -> dict[str, tuple[int, ...]]:
        """Optimize static arguments for all methods of a manifold.

        Args:
            manifold: Manifold instance to optimize

        Returns:
            Dictionary mapping method names to optimal static argument configurations
        """
        optimization_results = {}

        # Standard manifold methods to optimize
        methods_to_optimize = [
            "proj",
            "exp",
            "log",
            "retr",
            "transp",
            "inner",
            "dist",
            "random_point",
            "random_tangent",
        ]

        for method_name in methods_to_optimize:
            if hasattr(manifold, method_name):
                method = getattr(manifold, method_name)
                if callable(method):
                    optimal_config = self.recommend_static_args(method, manifold)
                    optimization_results[method_name] = optimal_config

                    logger.debug(f"Optimized {type(manifold).__name__}.{method_name}: static_argnums={optimal_config}")

        return optimization_results

    def benchmark_static_args_configuration(
        self, func: Callable[..., Any], args: tuple[Any, ...], static_configs: list[tuple[int, ...]], num_runs: int = 10
    ) -> dict[tuple[int, ...], dict[str, Any]]:
        """Benchmark different static argument configurations.

        Args:
            func: Function to benchmark
            args: Function arguments
            static_configs: List of static argument configurations to test
            num_runs: Number of benchmark runs per configuration

        Returns:
            Dictionary mapping static configs to performance metrics
        """
        import statistics
        import time

        from .jit_manager import JITManager

        results: dict[tuple[int, ...], dict[str, Any]] = {}

        for config in static_configs:
            JITManager.clear_cache()

            try:
                # Compile with this configuration
                jit_func = jax.jit(func) if len(config) == 0 else jax.jit(func, static_argnums=config)

                # Measure compilation time
                compile_start = time.perf_counter()
                jit_func(*args)  # Trigger compilation
                compile_time = time.perf_counter() - compile_start

                # Measure execution times
                exec_times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    jit_func(*args)
                    end = time.perf_counter()
                    exec_times.append(end - start)

                results[config] = {
                    "compile_time": compile_time,
                    "mean_exec_time": statistics.mean(exec_times),
                    "std_exec_time": statistics.stdev(exec_times) if len(exec_times) > 1 else 0.0,
                    "min_exec_time": min(exec_times),
                    "max_exec_time": max(exec_times),
                }

            except Exception as e:
                results[config] = {
                    "compile_time": float("inf"),
                    "mean_exec_time": float("inf"),
                    "std_exec_time": 0.0,
                    "min_exec_time": float("inf"),
                    "max_exec_time": float("inf"),
                    "error_message": str(e)
                }
                logger.warning(f"Static config {config} failed: {e}")

        return results

    def find_optimal_static_args(self, func: Callable[..., Any], args: tuple[Any, ...], max_static_args: int = 2) -> tuple[int, ...]:
        """Find optimal static argument configuration through systematic search.

        Args:
            func: Function to optimize
            args: Representative function arguments
            max_static_args: Maximum number of static arguments to consider

        Returns:
            Optimal static argument configuration
        """
        analysis = self.analyze_function_signature(func)
        num_params = analysis["total_params"]

        # Generate candidate configurations
        candidates: list[tuple[int, ...]] = [()]  # No static args as baseline

        # Don't make data parameters static
        data_positions = set(analysis["data_param_positions"])

        # Consider individual argument positions (except data positions)
        for i in range(num_params):
            if i not in data_positions and i < max_static_args:
                candidates.append((i,))

        # Consider combinations of non-data positions
        if max_static_args > 1:
            for i in range(num_params):
                for j in range(i + 1, num_params):
                    if i not in data_positions and j not in data_positions and len((i, j)) <= max_static_args:
                        candidates.append((i, j))

        # Benchmark all candidates
        try:
            benchmark_results = self.benchmark_static_args_configuration(func, args, candidates, num_runs=5)

            # Find the configuration with best performance
            valid_results = {
                k: v for k, v in benchmark_results.items() if "error" not in v and v["mean_exec_time"] != float("inf")
            }

            if not valid_results:
                logger.warning("No valid static arg configurations found")
                return ()

            # Choose configuration with lowest mean execution time
            optimal_config = min(valid_results.keys(), key=lambda k: valid_results[k]["mean_exec_time"])

            logger.info(
                f"Optimal static_argnums={optimal_config} "
                f"(exec_time: {valid_results[optimal_config]['mean_exec_time']:.6f}s)"
            )

            return optimal_config

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return ()

    def create_optimized_manifold_wrapper(self, manifold: Any) -> Any:
        """Create a wrapper with optimized static argument configurations.

        Args:
            manifold: Original manifold instance

        Returns:
            Optimized manifold wrapper
        """

        class OptimizedManifoldWrapper:
            def __init__(self, original_manifold: Any):
                self._original = original_manifold
                self._optimized_configs: dict[str, tuple[int, ...]] = {}

                # Pre-compute optimal configurations for key methods
                optimizer = StaticArgsOptimizer()
                self._optimized_configs = optimizer.optimize_manifold_static_args(original_manifold)

            def __getattr__(self, name: str) -> Any:
                # Delegate to original manifold
                attr = getattr(self._original, name)

                # If it's an optimized method, wrap it
                if name in self._optimized_configs and callable(attr):
                    static_args = self._optimized_configs[name]
                    if len(static_args) > 0:
                        # Apply static argument optimization
                        return jax.jit(attr, static_argnums=static_args)

                return attr

            def _get_static_args(self, method_name: str) -> tuple[int, ...]:
                """Return optimized static arguments for the given method."""
                return self._optimized_configs.get(method_name, ())

        return OptimizedManifoldWrapper(manifold)

    def validate_static_args_configuration(self, func: Callable[..., Any], static_argnums: tuple[int, ...]) -> bool:
        """Validate that a static argument configuration is valid for a function.

        Args:
            func: Function to validate
            static_argnums: Static argument configuration

        Returns:
            True if configuration is valid
        """
        try:
            sig = inspect.signature(func)
            num_params = len(sig.parameters)

            # Check if all static argument indices are within bounds
            return all(not (argnum >= num_params or argnum < -num_params) for argnum in static_argnums)

        except Exception:
            return False

    def generate_optimization_report(self, manifold: Any) -> str:
        """Generate a detailed optimization report for a manifold.

        Args:
            manifold: Manifold to analyze

        Returns:
            Formatted optimization report
        """
        report = f"# Static Arguments Optimization Report: {type(manifold).__name__}\n\n"

        # Get current configurations
        methods = ["proj", "exp", "log", "inner", "dist"]

        report += "## Current Configuration\n"
        for method_name in methods:
            if hasattr(manifold, method_name) and hasattr(manifold, "_get_static_args"):
                current_config = manifold._get_static_args(method_name)
                report += f"- **{method_name}**: {current_config}\n"

        # Get optimization recommendations
        optimized_configs = self.optimize_manifold_static_args(manifold)

        report += "\n## Recommended Configuration\n"
        for method_name, config in optimized_configs.items():
            report += f"- **{method_name}**: {config}\n"

        # Add manifold-specific information
        report += "\n## Manifold Properties\n"
        if hasattr(manifold, "n"):
            report += f"- Dimension n: {manifold.n}\n"
        if hasattr(manifold, "p"):
            report += f"- Dimension p: {manifold.p}\n"

        report += "\n## Notes\n"
        report += "- Empty tuple () means no static arguments (safest default)\n"
        report += "- Static arguments should be used carefully to avoid shape/type conflicts\n"
        report += "- Performance benefits may vary depending on use case\n"

        return report
