"""RiemannAX core module for JIT optimization and performance management."""

from .batch_ops import BatchJITOptimizer, batch_compile, get_batch_optimizer, vectorize
from .cholesky_engine import CholeskyEngine
from .device_manager import DeviceManager
from .geodesic_connection import GeodesicConnection
from .jit_manager import JITManager
from .numerical_stability import NumericalStabilityLayer
from .performance import PerformanceMonitor
from .safe_jit import SafeJITWrapper
from .spd_algorithm_selector import SPDAlgorithmSelector

__all__ = [
    "BatchJITOptimizer",
    "CholeskyEngine",
    "DeviceManager",
    "GeodesicConnection",
    "JITManager",
    "NumericalStabilityLayer",
    "PerformanceMonitor",
    "SPDAlgorithmSelector",
    "SafeJITWrapper",
    "batch_compile",
    "get_batch_optimizer",
    "vectorize",
]
