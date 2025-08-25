"""Tests for internationalization compliance - ensuring no Japanese comments remain in production code."""

import ast
import re
from pathlib import Path

import pytest


def extract_comments_and_docstrings(file_path: Path) -> list[str]:
    """Extract all comments and docstrings from a Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    comments_and_docstrings = []

    # Extract single-line and multi-line comments using regex
    comment_pattern = r"#.*"
    comments = re.findall(comment_pattern, content)
    comments_and_docstrings.extend(comments)

    # Extract docstrings using AST
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef | ast.Module) and (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                comments_and_docstrings.append(node.body[0].value.value)
    except SyntaxError:
        pass  # Skip files with syntax errors

    return comments_and_docstrings


def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese characters (hiragana, katakana, kanji)."""
    japanese_pattern = r"[あ-んア-ンー一-龯]"
    return bool(re.search(japanese_pattern, text))


def test_no_japanese_comments_in_performance_module():
    """Test that performance.py module contains no Japanese comments after internationalization."""
    performance_file = Path("riemannax/core/performance.py")

    # Extract all comments and docstrings
    comments_and_docstrings = extract_comments_and_docstrings(performance_file)

    # Check for Japanese characters
    japanese_items = []
    for item in comments_and_docstrings:
        if contains_japanese(item):
            japanese_items.append(item.strip())

    # Assert no Japanese characters found
    assert not japanese_items, f"Found Japanese text in {performance_file}:\n" + "\n".join(
        f"  - {item}" for item in japanese_items
    )


def test_performance_module_has_english_docstrings():
    """Test that performance.py module has proper English docstrings following Google style."""
    performance_file = Path("riemannax/core/performance.py")

    try:
        with open(performance_file, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        pytest.skip(f"File {performance_file} not found")

    # Parse the AST to check docstrings
    tree = ast.parse(content)

    # Check class docstrings
    classes_with_docstrings = []
    classes_without_docstrings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value
                if not contains_japanese(docstring):
                    classes_with_docstrings.append(node.name)
                else:
                    classes_without_docstrings.append(node.name)
            else:
                classes_without_docstrings.append(node.name)

    # All classes should have English docstrings
    assert not classes_without_docstrings, f"Classes without proper English docstrings: {classes_without_docstrings}"

    # Should have at least the expected classes
    expected_classes = ["OperationMetrics", "PerformanceMonitor"]
    for expected_class in expected_classes:
        assert expected_class in classes_with_docstrings, (
            f"Expected class {expected_class} not found with English docstring"
        )


def test_performance_module_method_docstrings():
    """Test that all methods in performance.py have English docstrings."""
    performance_file = Path("riemannax/core/performance.py")

    try:
        with open(performance_file, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        pytest.skip(f"File {performance_file} not found")

    tree = ast.parse(content)

    methods_without_english_docstrings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Skip private methods and dunder methods for now
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue

            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value
                if contains_japanese(docstring):
                    methods_without_english_docstrings.append(node.name)
            else:
                # Methods without docstrings
                if not node.name.startswith("_"):  # Only check public methods
                    methods_without_english_docstrings.append(node.name)

    assert not methods_without_english_docstrings, (
        f"Methods with Japanese or missing docstrings: {methods_without_english_docstrings}"
    )


def test_google_style_docstring_format():
    """Test that docstrings follow Google style format."""
    performance_file = Path("riemannax/core/performance.py")

    try:
        with open(performance_file, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        pytest.skip(f"File {performance_file} not found")

    tree = ast.parse(content)

    invalid_docstrings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

            # Skip if it contains Japanese (will be caught by other tests)
            if contains_japanese(docstring):
                continue

            # Check for basic Google style patterns
            # Should have proper capitalization and end with period for descriptions
            lines = docstring.strip().split("\n")
            if lines:
                first_line = lines[0].strip()
                if first_line and not first_line.endswith("."):
                    # Allow exceptions for very short descriptions
                    if len(first_line) > 10:
                        invalid_docstrings.append(f"{node.name}: '{first_line}' should end with period")

    # This is a soft check - we want good style but not overly strict
    if invalid_docstrings:
        print(f"Docstring style suggestions for {performance_file}:")
        for suggestion in invalid_docstrings:
            print(f"  - {suggestion}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
