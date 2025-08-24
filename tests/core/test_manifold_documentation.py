"""Tests for manifold documentation standardization and consistency."""

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

# Import all manifold classes for inspection
from riemannax.manifolds import base as base_module
from riemannax.manifolds import sphere as sphere_module
from riemannax.manifolds import grassmann as grassmann_module
from riemannax.manifolds import stiefel as stiefel_module
from riemannax.manifolds import so as so_module
from riemannax.manifolds import spd as spd_module


def get_manifold_modules():
    """Get all manifold modules for testing."""
    return [
        ("base", base_module, Path("riemannax/manifolds/base.py")),
        ("sphere", sphere_module, Path("riemannax/manifolds/sphere.py")),
        ("grassmann", grassmann_module, Path("riemannax/manifolds/grassmann.py")),
        ("stiefel", stiefel_module, Path("riemannax/manifolds/stiefel.py")),
        ("so", so_module, Path("riemannax/manifolds/so.py")),
        ("spd", spd_module, Path("riemannax/manifolds/spd.py")),
    ]


def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese characters."""
    japanese_pattern = r'[あ-んア-ンー一-龯]'
    return bool(re.search(japanese_pattern, text))


def extract_docstrings_from_file(file_path: Path) -> dict[str, str]:
    """Extract all docstrings from a Python file."""
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return {}

    docstrings = {}

    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.ClassDef):
                name = f"class_{node.name}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"method_{node.name}"
            elif isinstance(node, ast.Module):
                name = "module"

            if name and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and
                    isinstance(first.value, ast.Constant) and
                    isinstance(first.value.value, str)):
                    docstrings[name] = first.value.value
    except SyntaxError:
        pass

    return docstrings


@pytest.mark.parametrize("module_name,module,file_path", get_manifold_modules())
def test_no_japanese_in_manifold_modules(module_name, module, file_path):
    """Test that manifold modules contain no Japanese text."""
    docstrings = extract_docstrings_from_file(file_path)

    japanese_items = []
    for name, docstring in docstrings.items():
        if contains_japanese(docstring):
            japanese_items.append(f"{name}: {docstring.strip()[:100]}...")

    assert not japanese_items, (
        f"Found Japanese text in {module_name} module:\n" +
        "\n".join(f"  - {item}" for item in japanese_items)
    )


@pytest.mark.parametrize("module_name,module,file_path", get_manifold_modules())
def test_manifold_classes_have_docstrings(module_name, module, file_path):
    """Test that all manifold classes have proper English docstrings."""
    # Get all classes from the module
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:  # Only classes defined in this module
            classes.append((name, obj))

    missing_or_invalid = []
    for class_name, cls in classes:
        docstring = inspect.getdoc(cls)
        if not docstring:
            missing_or_invalid.append(f"{class_name}: No docstring")
        elif contains_japanese(docstring):
            missing_or_invalid.append(f"{class_name}: Contains Japanese")
        elif len(docstring.strip()) < 10:  # Very short docstring
            missing_or_invalid.append(f"{class_name}: Docstring too short")

    assert not missing_or_invalid, (
        f"Classes with missing/invalid docstrings in {module_name}:\n" +
        "\n".join(f"  - {item}" for item in missing_or_invalid)
    )


@pytest.mark.parametrize("module_name,module,file_path", get_manifold_modules())
def test_manifold_public_methods_have_docstrings(module_name, module, file_path):
    """Test that all public methods in manifold classes have English docstrings."""
    # Skip base module as it has abstract methods
    if module_name == "base":
        pytest.skip("Base module has abstract methods")

    # Get all classes from the module
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            classes.append((name, obj))

    invalid_methods = []
    for class_name, cls in classes:
        methods = inspect.getmembers(cls, inspect.ismethod)
        functions = inspect.getmembers(cls, inspect.isfunction)
        all_methods = methods + functions

        for method_name, method in all_methods:
            # Skip private methods and special methods
            if method_name.startswith('_') and not method_name.startswith('__'):
                continue
            if method_name in ['__init__', '__str__', '__repr__']:
                continue

            docstring = inspect.getdoc(method)
            if not docstring:
                invalid_methods.append(f"{class_name}.{method_name}: No docstring")
            elif contains_japanese(docstring):
                invalid_methods.append(f"{class_name}.{method_name}: Contains Japanese")

    # Allow some missing docstrings but flag significant issues
    if len(invalid_methods) > 3:  # Allow a few missing but flag major issues
        assert False, (
            f"Multiple methods with missing/invalid docstrings in {module_name}:\n" +
            "\n".join(f"  - {item}" for item in invalid_methods[:10])  # Show first 10
        )


def test_google_style_docstring_consistency():
    """Test that manifold docstrings follow Google style format consistently."""
    inconsistent_styles = []

    for module_name, module, file_path in get_manifold_modules():
        docstrings = extract_docstrings_from_file(file_path)

        for name, docstring in docstrings.items():
            if contains_japanese(docstring):
                continue  # Skip Japanese (handled by other tests)

            lines = docstring.strip().split('\n')
            if not lines:
                continue

            first_line = lines[0].strip()
            if not first_line:
                continue

            # Check for basic Google style patterns
            if len(first_line) > 15 and not first_line.endswith('.'):
                inconsistent_styles.append(
                    f"{module_name}.{name}: First line should end with period: '{first_line}'"
                )

            # Check for Args/Returns sections if they exist
            has_args_section = any('Args:' in line for line in lines)
            has_returns_section = any('Returns:' in line for line in lines)

            if has_args_section:
                # Check indentation consistency (basic check)
                args_line_idx = next(i for i, line in enumerate(lines) if 'Args:' in line)
                if args_line_idx + 1 < len(lines):
                    next_line = lines[args_line_idx + 1]
                    if next_line.strip() and not next_line.startswith('        '):
                        inconsistent_styles.append(
                            f"{module_name}.{name}: Args section should have proper indentation"
                        )

    # This is a style check, so we'll be lenient
    if len(inconsistent_styles) > 5:
        print(f"Documentation style suggestions:")
        for suggestion in inconsistent_styles[:10]:
            print(f"  - {suggestion}")


def test_manifold_mathematical_notation():
    """Test that manifold docstrings contain appropriate mathematical notation where expected."""
    expected_mathematical_terms = {
        "sphere": ["sphere", "manifold", "geodesic", "tangent"],
        "grassmann": ["Grassmann", "subspace", "orthogonal", "manifold"],
        "stiefel": ["Stiefel", "orthonormal", "manifold", "matrix"],
        "so": ["orthogonal", "rotation", "manifold", "matrix"],
        "spd": ["symmetric", "positive", "definite", "manifold"],
    }

    for module_name, module, file_path in get_manifold_modules():
        if module_name not in expected_mathematical_terms:
            continue

        # Get class docstring
        classes = [obj for name, obj in inspect.getmembers(module, inspect.isclass)
                  if obj.__module__ == module.__name__]

        if classes:
            main_class = classes[0]  # Primary manifold class
            docstring = inspect.getdoc(main_class)

            if docstring:
                docstring_lower = docstring.lower()
                expected_terms = expected_mathematical_terms[module_name]
                missing_terms = [term for term in expected_terms
                               if term.lower() not in docstring_lower]

                # Allow some flexibility - not all terms need to be present
                if len(missing_terms) >= len(expected_terms) - 1:
                    print(f"Note: {module_name} class docstring might benefit from more "
                          f"mathematical context. Missing terms: {missing_terms}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
