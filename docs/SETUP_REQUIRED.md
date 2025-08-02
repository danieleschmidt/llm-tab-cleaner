# Manual Setup Requirements

This document outlines the manual setup steps required after implementing the SDLC checkpoints. Due to GitHub App permission limitations, some configurations must be applied manually.

## ⚠️ Required Manual Actions

### 1. GitHub Workflows Creation

**Priority: HIGH** - Copy workflow templates to enable CI/CD

```bash
# Copy workflow templates to actual GitHub Actions directory
mkdir -p .github/workflows/
cp docs/workflows/ci.yml.template .github/workflows/ci.yml
cp docs/workflows/release.yml.template .github/workflows/release.yml
cp docs/workflows/security.yml.template .github/workflows/security.yml
```

### 2. Repository Secrets Configuration

**Priority: HIGH** - Add these secrets in repository settings

Navigate to: Repository Settings → Secrets and variables → Actions

#### Required Secrets:
```
PYPI_API_TOKEN              # For PyPI package publishing
TEST_PYPI_API_TOKEN         # For TestPyPI staging
CODECOV_TOKEN               # For coverage reporting (get from codecov.io)
```

#### Optional Secrets for Enhanced Functionality:
```
ANTHROPIC_API_KEY_TEST      # For integration testing with real LLM
OPENAI_API_KEY_TEST         # For integration testing with real LLM
SLACK_WEBHOOK_URL           # For build notifications
SENTRY_DSN                  # For error tracking
```

### 3. Repository Environments Setup

**Priority: MEDIUM** - Configure deployment environments

1. Go to Repository Settings → Environments
2. Create environments:
   - `pypi` (Production PyPI deployment)
   - `test-pypi` (TestPyPI staging)

3. Configure environment protection rules:
   - **pypi**: Require admin approval for deployments
   - **test-pypi**: No restrictions needed

### 4. Branch Protection Rules

**Priority: HIGH** - Protect main branch

Navigate to: Repository Settings → Branches → Add rule for `main`

#### Required Settings:
- ✅ Require a pull request before merging
- ✅ Require approvals (minimum 1)
- ✅ Dismiss stale reviews when new commits are pushed
- ✅ Require review from code owners
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require signed commits (recommended)

#### Required Status Checks:
```
CI / Code Quality (3.9)
CI / Code Quality (3.10)
CI / Code Quality (3.11)
CI / Code Quality (3.12)
CI / Security Scanning
CI / Documentation Build
```

### 5. Third-Party Service Integration

#### Codecov Setup (for coverage reporting)
1. Sign up at [codecov.io](https://codecov.io)
2. Add your repository
3. Copy the upload token to `CODECOV_TOKEN` secret
4. Optional: Configure `codecov.yml` for custom settings

#### PyPI/TestPyPI Setup (for package publishing)
1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Generate API tokens for both services
3. Add tokens to repository secrets
4. Optional: Set up trusted publishing (OIDC) for enhanced security

### 6. Repository Settings Configuration

**Priority: MEDIUM** - Optimize repository configuration

#### General Settings:
- Enable "Automatically delete head branches" after PR merge
- Disable "Allow merge commits" (prefer squash and merge)
- Enable "Allow squash merging"
- Disable "Allow rebase merging"

#### Security Settings:
- Enable Dependabot alerts
- Enable Dependabot security updates
- Enable Secret scanning alerts
- Enable Push protection for secrets

### 7. CODEOWNERS File

**Priority: MEDIUM** - Already implemented but verify

The `.github/CODEOWNERS` file should exist with appropriate reviewers assigned. Update as needed for your team structure.

### 8. Issue and PR Templates

**Priority: LOW** - Enhance GitHub collaboration

Templates are already in place in `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md`. Verify they meet your needs.

---

## 🚀 Verification Steps

### After completing manual setup:

1. **Test CI Pipeline**:
   ```bash
   # Create a test branch and PR
   git checkout -b test-ci-setup
   echo "# Test CI" >> TEST.md
   git add TEST.md
   git commit -m "test: verify CI pipeline setup"
   git push -u origin test-ci-setup
   # Create PR and verify all checks pass
   ```

2. **Test Security Scanning**:
   ```bash
   # Trigger security workflow manually
   # Go to Actions → Security Scanning → Run workflow
   ```

3. **Test Release Process** (optional):
   ```bash
   # Create a test release
   git tag v0.1.0-test
   git push origin v0.1.0-test
   # Verify release workflow triggers
   ```

4. **Verify Branch Protection**:
   - Try to push directly to main (should be blocked)
   - Verify PR requires reviews and status checks

---

## 📋 Setup Checklist

### GitHub Actions & CI/CD
- [ ] Copy workflow templates to `.github/workflows/`
- [ ] Configure repository secrets
- [ ] Set up deployment environments
- [ ] Test CI pipeline with test PR
- [ ] Verify all status checks pass

### Security & Compliance
- [ ] Configure branch protection rules
- [ ] Enable Dependabot alerts and updates
- [ ] Enable secret scanning
- [ ] Test security workflows
- [ ] Verify CODEOWNERS assignments

### External Integrations
- [ ] Set up Codecov integration
- [ ] Configure PyPI publishing
- [ ] Test package publishing to TestPyPI
- [ ] Set up notification channels (Slack/Discord)

### Documentation & Community
- [ ] Verify issue and PR templates
- [ ] Update README badges with actual status
- [ ] Test documentation builds
- [ ] Review contributor guidelines

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Workflow Not Triggering
- Check file paths: workflows must be in `.github/workflows/`
- Verify YAML syntax with online validator
- Check trigger conditions (branch names, file paths)
- Ensure workflows have proper permissions

#### Status Checks Not Required
- Workflows must run at least once before they appear in branch protection settings
- Create a test PR to trigger workflows first
- Check status check names match exactly

#### Secret Access Issues
- Verify secret names match exactly (case-sensitive)
- Check that secrets are set at repository level, not organization
- For forked repositories, secrets may need special configuration

#### Permission Denied Errors
- Check GitHub token permissions in workflow files
- Verify repository settings allow Actions to create releases/comments
- Ensure branch protection rules aren't blocking the action

### Getting Help

1. **GitHub Documentation**: [GitHub Actions Docs](https://docs.github.com/en/actions)
2. **Community Support**: [GitHub Community Forum](https://github.community/)
3. **Project Issues**: Create an issue in this repository for project-specific problems

---

## 📈 Success Metrics

After completing setup, you should see:

- ✅ All CI checks passing on new PRs
- ✅ Coverage reports appearing on PRs
- ✅ Security scans running automatically
- ✅ Documentation building and deploying
- ✅ Branch protection preventing direct pushes to main
- ✅ Automated dependency updates from Dependabot

---

*Last Updated: January 31, 2025*  
*Review Date: April 2025*