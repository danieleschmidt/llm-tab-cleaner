# Contributing to LLM Tab Cleaner

We welcome contributions to the LLM Tab Cleaner project! This document provides guidelines for contributing code, documentation, and other improvements.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Local Development

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-tab-cleaner.git
   cd llm-tab-cleaner
   ```

2. **Set up development environment**
   ```bash
   make dev-install
   ```

3. **Run tests to verify setup**
   ```bash
   make test
   ```

## Development Workflow

### Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Follow the existing code style

3. **Run quality checks**
   ```bash
   make ci  # Runs linting, type checking, and tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### Pull Request Process

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request**
   - Use a clear, descriptive title
   - Include a detailed description of changes
   - Reference any related issues
   - Ensure all CI checks pass

3. **Code Review**
   - Address reviewer feedback
   - Keep the PR updated with main branch
   - Maintain a clean commit history

## Code Quality Standards

### Code Style

- **Formatting**: We use [Black](https://black.readthedocs.io/) for code formatting
- **Linting**: We use [Ruff](https://github.com/charliermarsh/ruff) for linting
- **Type Hints**: All public functions should have type hints
- **Docstrings**: Use Google-style docstrings for all public functions

### Testing

- **Coverage**: Aim for >90% test coverage
- **Test Types**: Include unit tests, integration tests, and property-based tests
- **Test Structure**: Use descriptive test names and organize tests logically

### Documentation

- **Code Comments**: Focus on *why*, not *what*
- **API Documentation**: Keep docstrings up-to-date
- **Examples**: Include usage examples in docstrings

## Project Structure

```
llm-tab-cleaner/
├── src/llm_tab_cleaner/     # Main package
│   ├── __init__.py          # Package exports
│   ├── core.py              # Core cleaning functionality
│   ├── cleaning_rule.py     # Custom rules
│   ├── confidence.py        # Confidence calibration
│   ├── incremental.py       # Incremental processing
│   └── cli.py              # Command line interface
├── tests/                   # Test suite
├── docs/                    # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # Project overview
```

## Areas for Contribution

### High Priority

- **Additional LLM Providers**: Support for more LLM APIs
- **Streaming Data Support**: Real-time cleaning capabilities
- **Performance Optimization**: Faster processing algorithms
- **Documentation**: Tutorial improvements and examples

### Medium Priority

- **Multi-language Support**: Cleaning text in different languages
- **Privacy Features**: Data anonymization and PII handling
- **Integration Plugins**: More ETL framework integrations
- **Monitoring Tools**: Better observability features

### Getting Started Ideas

- **Bug Reports**: Help identify and fix issues
- **Example Notebooks**: Create Jupyter notebooks with use cases
- **Unit Tests**: Improve test coverage
- **Documentation**: Fix typos, improve clarity

## Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and inclusive in all interactions.

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community discussion
- **Documentation**: Check the [docs](https://llm-tab-cleaner.readthedocs.io) first

### Recognition

Contributors are recognized in:
- README acknowledgments
- Release notes
- Hall of Fame (for significant contributions)

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Publish to PyPI

## License

By contributing, you agree that your contributions will be licensed under the MIT License.