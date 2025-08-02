.PHONY: install dev-install test lint format type-check clean docs build publish help
.PHONY: docker-build docker-run security-check performance-test docker-compose-up docker-compose-down

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_IMAGE := llm-tab-cleaner
DOCKER_TAG := latest
DOCKER_REGISTRY := ghcr.io/danieleschmidt
VERSION := $(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# DEVELOPMENT ENVIRONMENT
# =============================================================================

install: ## Install package
	pip install -e .

dev-install: ## Install package with development dependencies
	pip install -e ".[dev,test]"
	pre-commit install

setup-dev: dev-install ## Complete development environment setup
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works"

# =============================================================================
# TESTING
# =============================================================================

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-performance: ## Run performance tests
	pytest tests/performance/ -v --benchmark-only

test-all: ## Run all tests including slow ones
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html -m "not requires_llm"

test-ci: ## Run tests suitable for CI
	pytest tests/ -x --cov=src --cov-report=xml --cov-report=term

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linting
	ruff check src tests
	black --check src tests

format: ## Format code
	black src tests
	ruff check --fix src tests
	isort src tests

type-check: ## Run type checking
	mypy src

security-check: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json || true
	pip-audit --desc --format=json --output=pip-audit.json || true
	safety check --json --output=safety-report.json || true

quality: lint type-check security-check ## Run all quality checks

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-clean: ## Clean documentation build
	rm -rf site/

# =============================================================================
# BUILD AND PACKAGING
# =============================================================================

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

publish: build ## Publish to PyPI
	python -m twine upload dist/*

publish-test: build ## Publish to test PyPI
	python -m twine upload --repository testpypi dist/*

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--target production \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VCS_REF=$(shell git rev-parse HEAD) \
		.

docker-build-dev: ## Build Docker image for development
	docker build -t $(DOCKER_IMAGE):dev \
		--target development \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VCS_REF=$(shell git rev-parse HEAD) \
		.

docker-build-spark: ## Build Docker image with Spark
	docker build -t $(DOCKER_IMAGE):spark \
		--target spark \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VCS_REF=$(shell git rev-parse HEAD) \
		.

docker-run: ## Run Docker container
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run Docker container in development mode
	docker run -it --rm \
		-v $(PWD):/app \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):dev

docker-push: docker-build ## Push Docker image to registry
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

docker-compose-logs: ## Show docker-compose logs
	docker-compose logs -f

# =============================================================================
# CI/CD TARGETS
# =============================================================================

ci: quality test-ci ## Run all CI checks

ci-build: docker-build ## Build for CI
	@echo "Built Docker image $(DOCKER_IMAGE):$(DOCKER_TAG)"

ci-test: ## Run tests in CI environment
	pytest tests/ -x --cov=src --cov-report=xml --cov-report=term -m "not requires_llm and not slow"

ci-security: security-check ## Run security checks for CI
	@echo "Security checks completed"

# =============================================================================
# RELEASE OPERATIONS
# =============================================================================

version: ## Show current version
	@echo $(VERSION)

tag-release: ## Tag current version for release
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

release: ci-build docker-push publish ## Full release process
	@echo "Released version $(VERSION)"

# =============================================================================
# UTILITY TARGETS
# =============================================================================

requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml --output-file requirements.txt
	pip-compile pyproject.toml --extra dev --output-file requirements-dev.txt

install-hooks: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

check-deps: ## Check for dependency updates
	pip list --outdated

benchmark: ## Run benchmarks
	pytest tests/performance/ --benchmark-only --benchmark-compare

profile: ## Profile the application
	python -m cProfile -o profile.stats -m llm_tab_cleaner.cli --help
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# =============================================================================
# MONITORING AND OBSERVABILITY
# =============================================================================

metrics: ## Generate metrics report
	@echo "Generating metrics report..."
	@find src -name "*.py" | xargs wc -l | tail -1
	@echo "Test coverage:"
	@pytest --cov=src --cov-report=term-missing tests/ | grep TOTAL

health-check: ## Run health check
	python -c "from llm_tab_cleaner import TableCleaner; print('✓ Health check passed')"

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

notebook: ## Start Jupyter notebook
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

shell: ## Start interactive Python shell with project loaded
	python -c "from llm_tab_cleaner import *; print('LLM Tab Cleaner loaded')" && python

demo: ## Run demo cleaning example
	python examples/demo.py || echo "Create examples/demo.py to run demo"