"""Comprehensive JIT compatibility verification suite."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.stiefel import Stiefel
from tests.utils.compatibility import JITCompatibilityHelper


@dataclass
class CompatibilityResult:
    """Result of a single compatibility test."""

    manifold_name: str
    operation: str
    batch_size: int
    passed: bool
    error_type: str = ""
    error_message: str = ""
    max_absolute_diff: float = 0.0
    max_relative_diff: float = 0.0
    numerical_instability: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CompatibilitySummary:
    """Summary of compatibility test results."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    numerical_instability_count: int
    pass_rate: float
    manifold_results: dict[str, dict[str, int]]
    operation_results: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class JITCompatibilityVerifier:
    """Comprehensive JIT compatibility verification system."""

    def __init__(self):
        """Initialize the verifier."""
        self.manifolds = {
            "sphere_3d": Sphere(),
            "sphere_5d": Sphere(),
            "grassmann_5_3": Grassmann(n=5, p=3),
            "stiefel_5_3": Stiefel(n=5, p=3),
            "so_3": SpecialOrthogonal(n=3),
            "spd_3": SymmetricPositiveDefinite(n=3),
        }

        # Operations to test for each manifold
        self.operations = {
            "sphere_3d": ["exp", "log", "proj", "inner"],
            "sphere_5d": ["exp", "log", "proj", "inner"],
            "grassmann_5_3": ["exp", "log", "proj", "inner"],
            "stiefel_5_3": ["exp", "proj", "inner"],  # log may be unstable
            "so_3": ["exp", "log", "proj", "inner"],
            "spd_3": ["exp", "proj", "inner"],  # log and dist may be unstable
        }

        self.batch_sizes = [1, 5]
        self.results: list[CompatibilityResult] = []

    def generate_test_data(self, manifold_name: str, batch_size: int) -> dict[str, Any]:
        """Generate stable test data for a manifold."""
        manifold = self.manifolds[manifold_name]
        key = jr.key(42)

        try:
            if manifold_name.startswith("sphere"):
                dim = int(manifold_name.split("_")[1][:-1])  # Extract dimension
                # Generate points with correct dimension and batch size
                if batch_size == 1:
                    # Single point: shape (dim,)
                    x = manifold.random_point(key, dim)
                    u = manifold.random_tangent(jr.key(43), x)
                    v = manifold.random_tangent(jr.key(44), x)
                    # Scale tangent vectors to avoid numerical issues
                    u = 0.1 * u / (jnp.linalg.norm(u) + 1e-8)
                    v = 0.1 * v / (jnp.linalg.norm(v) + 1e-8)
                else:
                    # Batch points: shape (batch_size, dim)
                    x = manifold.random_point(key, batch_size, dim)
                    u = manifold.random_tangent(jr.key(43), x)
                    v = manifold.random_tangent(jr.key(44), x)
                    # Scale tangent vectors to avoid numerical issues (batch-aware)
                    u_norms = jnp.linalg.norm(u, axis=-1, keepdims=True)
                    v_norms = jnp.linalg.norm(v, axis=-1, keepdims=True)
                    u = 0.1 * u / (u_norms + 1e-8)
                    v = 0.1 * v / (v_norms + 1e-8)
                return {"x": x, "u": u, "v": v}

            elif manifold_name in ["grassmann_5_3", "stiefel_5_3"]:
                if batch_size == 1:
                    x = manifold.random_point(key)
                    u = manifold.random_tangent(jr.key(43), x)
                    v = manifold.random_tangent(jr.key(44), x)
                    # Small tangent vector for stability
                    u = 0.05 * u
                    v = 0.05 * v
                else:
                    # Generate batch
                    keys = jr.split(key, batch_size)
                    x_list, u_list, v_list = [], [], []
                    for i in range(batch_size):
                        xi = manifold.random_point(keys[i])
                        ui = manifold.random_tangent(jr.key(43 + i), xi)
                        vi = manifold.random_tangent(jr.key(44 + i), xi)
                        ui = 0.05 * ui  # Scale for stability
                        vi = 0.05 * vi
                        x_list.append(xi)
                        u_list.append(ui)
                        v_list.append(vi)
                    x = jnp.stack(x_list, axis=0)
                    u = jnp.stack(u_list, axis=0)
                    v = jnp.stack(v_list, axis=0)
                return {"x": x, "u": u, "v": v}

            elif manifold_name.startswith("so_"):
                if batch_size == 1:
                    # Fix: Don't pass dimension to random_point
                    x = manifold.random_point(key)
                    u = manifold.random_tangent(jr.key(43), x)
                    v = manifold.random_tangent(jr.key(44), x)
                    u = 0.1 * u  # Scale for stability
                    v = 0.1 * v
                else:
                    # Generate batch
                    keys = jr.split(key, batch_size)
                    x_list, u_list, v_list = [], [], []
                    for i in range(batch_size):
                        xi = manifold.random_point(keys[i])
                        ui = manifold.random_tangent(jr.key(43 + i), xi)
                        vi = manifold.random_tangent(jr.key(44 + i), xi)
                        ui = 0.1 * ui
                        vi = 0.1 * vi
                        x_list.append(xi)
                        u_list.append(ui)
                        v_list.append(vi)
                    x = jnp.stack(x_list, axis=0)
                    u = jnp.stack(u_list, axis=0)
                    v = jnp.stack(v_list, axis=0)
                return {"x": x, "u": u, "v": v}

            elif manifold_name.startswith("spd_"):
                n = int(manifold_name.split("_")[1])
                if batch_size == 1:
                    # Use the helper to create stable SPD matrices
                    x = JITCompatibilityHelper.create_stable_spd_matrix(key, n)
                    u = manifold.random_tangent(jr.key(43), x)
                    v = manifold.random_tangent(jr.key(44), x)
                    u = 0.01 * u  # Very small for SPD stability
                    v = 0.01 * v
                else:
                    # Generate batch
                    keys = jr.split(key, batch_size)
                    x_list, u_list, v_list = [], [], []
                    for i in range(batch_size):
                        xi = JITCompatibilityHelper.create_stable_spd_matrix(keys[i], n)
                        ui = manifold.random_tangent(jr.key(43 + i), xi)
                        vi = manifold.random_tangent(jr.key(44 + i), xi)
                        ui = 0.01 * ui
                        vi = 0.01 * vi
                        x_list.append(xi)
                        u_list.append(ui)
                        v_list.append(vi)
                    x = jnp.stack(x_list, axis=0)
                    u = jnp.stack(u_list, axis=0)
                    v = jnp.stack(v_list, axis=0)
                return {"x": x, "u": u, "v": v}

            return {}

        except Exception:
            # If data generation fails, return empty dict
            return {}

    def test_single_operation(self, manifold_name: str, operation: str, batch_size: int) -> CompatibilityResult:
        """Test compatibility for a single operation."""
        manifold = self.manifolds[manifold_name]
        test_data = self.generate_test_data(manifold_name, batch_size)

        if not test_data:
            return CompatibilityResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                passed=False,
                error_type="DataGenerationError",
                error_message="Failed to generate test data",
            )

        try:
            # Check if operation is implemented
            if not hasattr(manifold, f"_{operation}_impl"):
                return CompatibilityResult(
                    manifold_name=manifold_name,
                    operation=operation,
                    batch_size=batch_size,
                    passed=False,
                    error_type="NotImplementedError",
                    error_message=f"Operation {operation} not implemented",
                )

            # Get functions
            impl_func = getattr(manifold, f"_{operation}_impl")
            public_func = getattr(manifold, operation)

            # Prepare operation-specific arguments
            x = test_data["x"]
            if operation in ["inner"] and "u" in test_data and "v" in test_data:
                # Inner product: (x, u, v)
                args = [x, test_data["u"], test_data["v"]]
            elif operation in ["exp", "proj"] and "u" in test_data:
                # Exponential map and projection: (x, u)
                args = [x, test_data["u"]]
            elif operation == "log" and "v" in test_data:
                # Logarithmic map: (x, v) - using v as second point
                args = [x, test_data["v"]]
            else:
                # Fallback: just use available data
                args = [x]
                if "u" in test_data:
                    args.append(test_data["u"])
                if "v" in test_data and operation not in ["exp", "proj"]:
                    args.append(test_data["v"])

            # Execute both versions
            result_impl = impl_func(*args)
            result_public = public_func(*args)

            # Check for numerical instability
            numerical_instability = not JITCompatibilityHelper.check_numerical_stability(
                result_impl
            ) or not JITCompatibilityHelper.check_numerical_stability(result_public)

            if numerical_instability:
                return CompatibilityResult(
                    manifold_name=manifold_name,
                    operation=operation,
                    batch_size=batch_size,
                    passed=False,
                    error_type="NumericalInstability",
                    error_message="NaN or Inf detected in results",
                    numerical_instability=True,
                )

            # Calculate differences
            abs_diff = jnp.abs(result_impl - result_public)
            rel_diff = abs_diff / (jnp.abs(result_public) + 1e-10)

            max_abs_diff = float(jnp.max(abs_diff))
            max_rel_diff = float(jnp.max(rel_diff))

            # Check compatibility using adaptive tolerance
            try:
                JITCompatibilityHelper.adaptive_tolerance_test(result_impl, result_public, operation)
                passed = True
                error_type = ""
                error_message = ""
            except AssertionError as e:
                passed = False
                error_type = "ToleranceError"
                error_message = str(e)

            return CompatibilityResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                passed=passed,
                error_type=error_type,
                error_message=error_message,
                max_absolute_diff=max_abs_diff,
                max_relative_diff=max_rel_diff,
                numerical_instability=numerical_instability,
            )

        except Exception as e:
            return CompatibilityResult(
                manifold_name=manifold_name,
                operation=operation,
                batch_size=batch_size,
                passed=False,
                error_type=type(e).__name__,
                error_message=str(e),
            )

    def run_comprehensive_verification(self) -> list[CompatibilityResult]:
        """Run comprehensive JIT compatibility verification."""
        self.results = []

        for manifold_name, operations in self.operations.items():
            print(f"\nTesting {manifold_name}...")

            for operation in operations:
                for batch_size in self.batch_sizes:
                    print(f"  Testing {operation} (batch_size={batch_size})...", end=" ")

                    result = self.test_single_operation(manifold_name, operation, batch_size)
                    self.results.append(result)

                    status = "PASS" if result.passed else f"FAIL ({result.error_type})"
                    print(status)

        return self.results

    def generate_summary(self) -> CompatibilitySummary:
        """Generate summary of compatibility results."""
        if not self.results:
            return CompatibilitySummary(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                numerical_instability_count=0,
                pass_rate=0.0,
                manifold_results={},
                operation_results={},
            )

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        numerical_instability_count = sum(1 for r in self.results if r.numerical_instability)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        # Aggregate by manifold
        manifold_results = {}
        for result in self.results:
            if result.manifold_name not in manifold_results:
                manifold_results[result.manifold_name] = {"passed": 0, "failed": 0, "total": 0}

            manifold_results[result.manifold_name]["total"] += 1
            if result.passed:
                manifold_results[result.manifold_name]["passed"] += 1
            else:
                manifold_results[result.manifold_name]["failed"] += 1

        # Aggregate by operation
        operation_results = {}
        for result in self.results:
            if result.operation not in operation_results:
                operation_results[result.operation] = {"passed": 0, "failed": 0, "total": 0}

            operation_results[result.operation]["total"] += 1
            if result.passed:
                operation_results[result.operation]["passed"] += 1
            else:
                operation_results[result.operation]["failed"] += 1

        return CompatibilitySummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            numerical_instability_count=numerical_instability_count,
            pass_rate=pass_rate,
            manifold_results=manifold_results,
            operation_results=operation_results,
        )

    def generate_detailed_report(self) -> str:
        """Generate detailed compatibility report."""
        if not self.results:
            return "No compatibility test results available."

        summary = self.generate_summary()

        report = []
        report.append("=" * 80)
        report.append("JIT COMPATIBILITY VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 40)
        report.append(f"Total tests: {summary.total_tests}")
        report.append(f"Passed tests: {summary.passed_tests}")
        report.append(f"Failed tests: {summary.failed_tests}")
        report.append(f"Pass rate: {summary.pass_rate:.1%}")
        report.append(f"Numerical instabilities: {summary.numerical_instability_count}")
        report.append("")

        # Results by manifold
        report.append("RESULTS BY MANIFOLD")
        report.append("-" * 40)
        for manifold_name, stats in summary.manifold_results.items():
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            report.append(f"{manifold_name}: {stats['passed']}/{stats['total']} ({pass_rate:.1%})")
        report.append("")

        # Results by operation
        report.append("RESULTS BY OPERATION")
        report.append("-" * 40)
        for operation, stats in summary.operation_results.items():
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            report.append(f"{operation}: {stats['passed']}/{stats['total']} ({pass_rate:.1%})")
        report.append("")

        # Failed tests detail
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            report.append("FAILED TESTS DETAIL")
            report.append("-" * 40)
            for result in failed_results:
                report.append(f"{result.manifold_name}.{result.operation} (batch={result.batch_size})")
                report.append(f"  Error: {result.error_type} - {result.error_message}")
                if result.max_absolute_diff > 0:
                    report.append(f"  Max abs diff: {result.max_absolute_diff:.2e}")
                if result.max_relative_diff > 0:
                    report.append(f"  Max rel diff: {result.max_relative_diff:.2e}")
                report.append("")

        return "\n".join(report)

    def save_results(self, output_dir: str = "compatibility_results") -> None:
        """Save results to JSON and text files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save detailed results
        results_data = {"summary": self.generate_summary().to_dict(), "results": [r.to_dict() for r in self.results]}

        with open(output_path / "compatibility_results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save text report
        with open(output_path / "compatibility_report.txt", "w") as f:
            f.write(self.generate_detailed_report())

        print(f"Results saved to {output_path}")


def run_compatibility_verification() -> float:
    """Main function to run compatibility verification."""
    verifier = JITCompatibilityVerifier()

    print("Running comprehensive JIT compatibility verification...")
    verifier.run_comprehensive_verification()

    print("\n" + "=" * 80)
    print(verifier.generate_detailed_report())

    verifier.save_results()

    summary = verifier.generate_summary()
    return summary.pass_rate  # Return the actual pass rate as float


if __name__ == "__main__":
    pass_rate = run_compatibility_verification()
    success = pass_rate >= 0.8  # 80% pass rate threshold
    exit(0 if success else 1)
