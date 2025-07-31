# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive GitHub Actions workflow templates (CI/CD, Security, Release)
- CODEOWNERS file for automated review assignments
- Advanced security scanning (Bandit, Safety, Trivy, TruffleHog)
- SBOM (Software Bill of Materials) generation
- Release automation with signed packages and trusted publishing
- Development container configuration
- Performance monitoring and observability setup
- Enhanced testing infrastructure with multi-version support
- Comprehensive security configurations and vulnerability management

### Changed
- Enhanced security posture with automated scanning and vulnerability detection
- Improved CI/CD pipeline with parallel testing and comprehensive checks
- Updated pre-commit hooks and development tooling configurations

### Deprecated

### Removed

### Fixed

### Security
- Added comprehensive security scanning with Bandit, Safety, and Trivy
- Implemented SBOM generation for supply chain security
- Added secret detection with TruffleHog OSS
- Enhanced container security scanning
- Automated security vulnerability reporting to GitHub Security tab
- Improved dependency management with grouped Dependabot updates

## [0.1.0] - 2025-01-XX

### Added
- Initial release of LLM Tab Cleaner
- Core table cleaning functionality with LLM integration
- Support for multiple LLM providers (OpenAI, Anthropic, local)
- Confidence-based cleaning with audit trails
- Custom cleaning rules and rule sets
- Incremental cleaning capabilities
- Multi-engine support (Pandas, Spark, DuckDB)
- Command-line interface
- Comprehensive test suite
- Development tooling (Black, Ruff, MyPy, pre-commit)
- Documentation and examples

### Features
- **Data Quality Issues Handled**:
  - Missing values with inference
  - Format inconsistencies
  - Duplicate record detection
  - Outlier correction
  - Schema violation fixes
  - Referential integrity repairs

- **Processing Engines**:
  - Pandas DataFrame support
  - Apache Spark distributed processing
  - DuckDB local processing
  - Apache Arrow Flight streaming

- **LLM Integration**:
  - OpenAI GPT models
  - Anthropic Claude models
  - Local model support
  - Confidence calibration
  - Retry logic and caching

- **Production Features**:
  - Audit logging with JSON-patch format
  - Incremental processing
  - Rollback capabilities
  - Performance monitoring
  - Cost optimization

[Unreleased]: https://github.com/terragonlabs/llm-tab-cleaner/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/terragonlabs/llm-tab-cleaner/releases/tag/v0.1.0