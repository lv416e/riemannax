.PHONY: test lint clean

# Default target
all: lint test

# Run tests
test:
	python -m pytest

# Run tests with coverage
coverage:
	python -m pytest --cov=riemannax --cov-report=term --cov-report=html

# Run linting
lint:
	flake8 riemannax
	isort --check-only --profile black riemannax
	black --check riemannax

# Format code
format:
	isort --profile black riemannax tests
	black riemannax tests

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type d -name "*.pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +