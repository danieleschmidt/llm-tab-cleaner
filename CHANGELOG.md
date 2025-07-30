# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive SDLC tooling and automation
- Security vulnerability reporting process
- GitHub issue and pull request templates
- Community guidelines and code of conduct
- Development container support
- Enhanced testing and coverage configuration

### Changed
- Enhanced development workflow documentation
- Improved project structure and organization

### Deprecated

### Removed

### Fixed

### Security
- Added security policy and vulnerability reporting process
- Enhanced dependency management with Dependabot

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