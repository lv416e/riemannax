"""RiemannAX core module for JIT optimization and performance management."""

from .batch_ops import BatchJITOptimizer, batch_compile, get_batch_optimizer, vectorize
from .device_manager import DeviceManager
from .jit_manager import JITManager
from .performance import PerformanceMonitor
from .safe_jit import SafeJITWrapper

__all__ = [
    "BatchJITOptimizer",
    "DeviceManager",
    "JITManager",
    "PerformanceMonitor",
    "SafeJITWrapper",
    "batch_compile",
    "get_batch_optimizer",
    "vectorize",
]
