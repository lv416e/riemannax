"""Command-line interface for RiemannAX performance benchmarking."""

import argparse
import sys
from pathlib import Path

from .performance_benchmark import PerformanceBenchmark, run_quick_benchmark


def parse_list_argument(arg_string: str) -> list[str]:
    """Parse comma-separated string into list."""
    if not arg_string:
        return []
    return [item.strip() for item in arg_string.split(",")]


def parse_int_list_argument(arg_string: str) -> list[int]:
    """Parse comma-separated string of integers into list."""
    if not arg_string:
        return []
    return [int(item.strip()) for item in arg_string.split(",")]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RiemannAX Performance Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with default settings
  python -m benchmarks.cli --quick

  # Full benchmark of specific manifolds
  python -m benchmarks.cli --manifolds sphere_3d,grassmann_5_3 --batch-sizes 1,10,100

  # Benchmark specific operations
  python -m benchmarks.cli --operations exp,proj --output-dir ./benchmark_results

  # Load and display previous results
  python -m benchmarks.cli --load-results ./benchmark_results/benchmark_results.json
        """,
    )

    # Main command options
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Run quick benchmark with small subset of configurations"
    )

    parser.add_argument(
        "--manifolds",
        "-m",
        type=str,
        help="Comma-separated list of manifolds to benchmark (e.g., sphere_3d,grassmann_5_3)",
    )

    parser.add_argument(
        "--operations", "-o", type=str, help="Comma-separated list of operations to benchmark (e.g., exp,proj,inner)"
    )

    parser.add_argument(
        "--batch-sizes", "-b", type=str, help="Comma-separated list of batch sizes to test (e.g., 1,10,100)"
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results (default: benchmark_results)",
    )

    parser.add_argument(
        "--save-results", action="store_true", default=True, help="Save benchmark results to JSON file (default: True)"
    )

    parser.add_argument("--no-save", action="store_true", help="Do not save benchmark results")

    parser.add_argument("--load-results", type=str, help="Load and display results from existing JSON file")

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results without running new benchmarks",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # Available options info
    parser.add_argument("--list-manifolds", action="store_true", help="List available manifolds and exit")

    parser.add_argument("--list-operations", action="store_true", help="List available operations by manifold and exit")

    args = parser.parse_args()

    # Handle info commands
    if args.list_manifolds:
        benchmark = PerformanceBenchmark()
        print("Available manifolds:")
        for manifold in sorted(benchmark.manifolds.keys()):
            print(f"  - {manifold}")
        sys.exit(0)

    if args.list_operations:
        benchmark = PerformanceBenchmark()
        print("Available operations by manifold:")
        for manifold in sorted(benchmark.operations.keys()):
            ops = benchmark.operations[manifold]
            print(f"  {manifold}: {', '.join(ops)}")
        sys.exit(0)

    # Handle load results command
    if args.load_results:
        try:
            # Create temporary benchmark to load results
            benchmark = PerformanceBenchmark()
            benchmark.output_dir = Path(args.load_results).parent
            benchmark.load_results(Path(args.load_results).name)

            print("Loaded benchmark results:")
            report = benchmark.generate_detailed_report()
            print(report)

        except Exception as e:
            print(f"Error loading results: {e}")
            sys.exit(1)

        sys.exit(0)

    # Parse arguments
    manifolds = parse_list_argument(args.manifolds) if args.manifolds else None
    operations = parse_list_argument(args.operations) if args.operations else None
    batch_sizes = parse_int_list_argument(args.batch_sizes) if args.batch_sizes else None

    # Quick benchmark mode
    if args.quick:
        if args.verbose:
            print("Running quick benchmark...")

        report = run_quick_benchmark(manifolds=manifolds, batch_sizes=batch_sizes)
        print(report)
        sys.exit(0)

    # Full benchmark mode
    try:
        # Initialize benchmark system
        if args.verbose:
            print(f"Initializing benchmark system with output directory: {args.output_dir}")

        output_dir = args.output_dir if not args.no_save else None
        benchmark = PerformanceBenchmark(output_dir=output_dir)

        # Handle report-only mode
        if args.report_only:
            if not benchmark.results:
                benchmark.load_results()

            if benchmark.results:
                report = benchmark.generate_detailed_report()
                print(report)
            else:
                print("No results found. Please run benchmark first or specify --load-results.")
                sys.exit(1)
            sys.exit(0)

        # Run comprehensive benchmark
        if args.verbose:
            print("Running comprehensive benchmark...")
            if manifolds:
                print(f"Manifolds: {', '.join(manifolds)}")
            if operations:
                print(f"Operations: {', '.join(operations)}")
            if batch_sizes:
                print(f"Batch sizes: {batch_sizes}")

        results = benchmark.run_comprehensive_benchmark(
            manifolds=manifolds, operations=operations, batch_sizes=batch_sizes
        )

        # Generate and display report
        report = benchmark.generate_detailed_report(results)
        print("\n" + report)

        # Save results
        if args.save_results and not args.no_save:
            benchmark.save_results()
            if benchmark.output_dir:
                print(f"\nResults saved to: {benchmark.output_dir / 'benchmark_results.json'}")
            else:
                print("\nResults saved to current directory")

        # Print summary
        summary = benchmark.generate_benchmark_summary(results)
        print("\nBenchmark completed successfully!")
        print(f"Total benchmarks run: {summary.total_benchmarks}")
        print(f"Average speedup: {summary.avg_speedup:.2f}x")

        if summary.avg_speedup >= 2.0:
            print("🚀 Excellent performance improvement!")
        elif summary.avg_speedup >= 1.5:
            print("✅ Good performance improvement!")
        elif summary.avg_speedup >= 1.0:
            print("📊 Performance improvement detected!")
        else:
            print("⚠️  Limited performance improvement - review JIT implementation")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"Error running benchmark: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def benchmark_manifold_subset():
    """Benchmark a specific subset of manifolds - useful for development."""
    print("Running targeted manifold benchmark...")

    benchmark = PerformanceBenchmark(output_dir="targeted_benchmark")

    # Focus on core manifolds with key operations
    results = benchmark.run_comprehensive_benchmark(
        manifolds=["sphere_3d", "grassmann_5_3", "so_3"], operations=["exp", "proj", "inner"], batch_sizes=[1, 10, 50]
    )

    report = benchmark.generate_detailed_report(results)
    print(report)

    benchmark.save_results("targeted_benchmark_results.json")
    return results


def profile_compilation_overhead():
    """Profile JIT compilation overhead across different operations."""
    print("Profiling JIT compilation overhead...")

    benchmark = PerformanceBenchmark(output_dir="compilation_profile")

    # Focus on compilation time analysis
    results = benchmark.run_comprehensive_benchmark(
        manifolds=["sphere_3d", "grassmann_5_3"],
        operations=["exp", "log", "proj"],
        batch_sizes=[1, 100],  # Small and large batch for compilation analysis
    )

    # Generate compilation-focused report
    print("\nCOMPILATION OVERHEAD ANALYSIS")
    print("=" * 50)

    for result in results:
        if result.compilation_time_ms and result.jit_time_ms > 0:
            overhead_ratio = result.compilation_time_ms / result.jit_time_ms
            print(
                f"{result.manifold_name}.{result.operation} (batch={result.batch_size}): "
                f"Compilation: {result.compilation_time_ms:.2f}ms, "
                f"Execution: {result.jit_time_ms:.2f}ms, "
                f"Overhead Ratio: {overhead_ratio:.2f}x"
            )

    return results


if __name__ == "__main__":
    main()
