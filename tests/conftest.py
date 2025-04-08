"""Test configuration for riemannax.

This module contains fixtures and configuration for pytest.
"""

import os
import sys

import pytest

# Add the parent directory to the path so we can import riemannax
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
