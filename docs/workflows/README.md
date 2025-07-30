# GitHub Actions Workflow Documentation

This directory contains documentation for the required GitHub Actions workflows. **Note: Actual workflow files must be created manually in `.github/workflows/` directory.**

## Required Workflows

### 1. CI/CD Pipeline (`ci.yml`)

**Purpose**: Continuous integration and testing for all pull requests and main branch pushes.

**Triggers**:
- Pull requests to `main` branch
- Pushes to `main` branch
- Manual dispatch

**Jobs**:
1. **Code Quality** (Python 3.9, 3.10, 3.11, 3.12)
   - Checkout code
   - Setup Python
   - Install dependencies: `pip install -e ".[dev,test]"`
   - Run linting: `make lint`
   - Run type checking: `make type-check`
   - Run tests with coverage: `make test`
   - Upload coverage to Codecov

2. **Security Scanning**
   - Dependency vulnerability scanning with `pip-audit`
   - Code security analysis with `bandit`
   - Secret detection with `detect-secrets`

3. **Documentation Build**
   - Build documentation with MkDocs
   - Deploy to GitHub Pages (on main branch)

**Required Secrets**:
- `CODECOV_TOKEN`: For coverage reporting
- `PYPI_API_TOKEN`: For package deployment

### 2. Release Pipeline (`release.yml`)

**Purpose**: Automated package building and publishing for releases.

**Triggers**:
- Release published
- Tag pushed matching `v*.*.*`

**Jobs**:
1. **Build Package**
   - Build source and wheel distributions
   - Run security scans on built package
   - Store artifacts

2. **Publish to PyPI**
   - Deploy to TestPyPI first
   - Run integration tests against TestPyPI package
   - Deploy to production PyPI on success

3. **Create Release Assets**
   - Generate release notes from CHANGELOG.md
   - Upload built packages to GitHub release

### 3. Security Scanning (`security.yml`)

**Purpose**: Regular security scanning and dependency updates.

**Triggers**:
- Daily schedule (3 AM UTC)
- Manual dispatch
- Pull requests (lightweight scan)

**Jobs**:
1. **Dependency Scanning**
   - `pip-audit` for known vulnerabilities
   - `safety` check for security issues
   - Generate security report

2. **Code Analysis**
   - `bandit` for security issues in code
   - `semgrep` for additional security patterns
   - SAST (Static Application Security Testing)

3. **Container Scanning** (if Docker images exist)
   - Scan container images for vulnerabilities
   - Check base image security

### 4. Performance Testing (`performance.yml`)

**Purpose**: Monitor performance regressions and resource usage.

**Triggers**:
- Pull requests (on performance-critical files)
- Weekly schedule
- Manual dispatch

**Jobs**:
1. **Benchmark Tests**
   - Run performance benchmarks
   - Compare against baseline
   - Generate performance report

2. **Memory Profiling**
   - Profile memory usage
   - Check for memory leaks
   - Resource utilization analysis

## Workflow Configuration Examples

### Environment Variables
```yaml
env:
  PYTHON_VERSION: "3.9"
  CACHE_VERSION: "1"
  PIP_CACHE_DIR: ~/.cache/pip
```

### Common Steps

#### Python Setup
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}
    cache: 'pip'
    cache-dependency-path: 'pyproject.toml'
```

#### Dependency Installation
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e ".[dev,test]"
```

#### Test Execution
```yaml
- name: Run tests
  run: |
    pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
```

## Branch Protection Rules

Configure these branch protection rules for `main` branch:

1. **Required status checks**:
   - `CI / Code Quality (3.9)`
   - `CI / Code Quality (3.10)`  
   - `CI / Code Quality (3.11)`
   - `CI / Code Quality (3.12)`
   - `CI / Security Scanning`

2. **Additional restrictions**:
   - Require branches to be up to date
   - Require review from code owners
   - Dismiss stale reviews when new commits are pushed
   - Require signed commits (recommended)

## Security Considerations

### Secrets Management
- Store API tokens in GitHub Secrets
- Use least-privilege access principles
- Rotate secrets regularly
- Never log secret values

### Third-party Actions
- Pin actions to specific SHA commits
- Regularly update action versions
- Review action security advisories
- Use official actions when possible

### Permissions
```yaml
permissions:
  contents: read
  security-events: write
  checks: write
  pull-requests: write
```

## Monitoring and Alerting

### Workflow Notifications
- Set up Slack/Discord webhooks for failures
- Email notifications for security issues
- Status badges in README

### Metrics to Track
- Build success rate
- Test coverage trends
- Security vulnerability count
- Performance regression alerts
- Dependency update frequency

## Manual Setup Instructions

1. Create `.github/workflows/` directory
2. Add workflow YAML files based on documentation above
3. Configure branch protection rules
4. Set up required secrets in repository settings
5. Test workflows with draft pull requests
6. Monitor initial runs and adjust as needed

## Integration with External Services

### Codecov
- Sign up at codecov.io
- Add repository
- Configure `codecov.yml` for coverage requirements

### Security Scanning Services
- Snyk for vulnerability scanning
- CodeQL for security analysis
- Dependabot for dependency updates (already configured)

## Troubleshooting

### Common Issues
- **Tests failing in CI but passing locally**: Check Python version differences, environment variables
- **Dependency conflicts**: Pin dependency versions, use dependency groups
- **Permission denied**: Check GitHub token permissions, secrets configuration
- **Workflow not triggering**: Verify trigger conditions, branch names, file paths

### Debug Steps
1. Check workflow logs in GitHub Actions tab
2. Verify secret configuration
3. Test locally with `act` (GitHub Actions local runner)
4. Check branch protection rules
5. Validate YAML syntax