#!/usr/bin/env python3
"""Build automation script for LLM Tab Cleaner."""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BuildManager:
    """Comprehensive build manager for LLM Tab Cleaner."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_info = self._get_build_info()
        
    def _get_build_info(self) -> Dict[str, str]:
        """Get build information from git and environment."""
        try:
            # Get git information
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            # Check if working directory is clean
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            is_dirty = bool(git_status)
            
        except subprocess.CalledProcessError:
            git_commit = "unknown"
            git_branch = "unknown"
            is_dirty = True
        
        # Get version from pyproject.toml
        try:
            import tomllib
            with open(self.project_root / "pyproject.toml", "rb") as f:
                pyproject = tomllib.load(f)
            version = pyproject["project"]["version"]
        except Exception:
            version = "unknown"
        
        return {
            "version": version,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "is_dirty": is_dirty,
            "build_date": datetime.utcnow().isoformat() + "Z",
            "build_user": os.getenv("USER", "unknown"),
        }
    
    def build_docker_image(
        self,
        target: str = "production",
        tag: Optional[str] = None,
        push: bool = False,
        registry: Optional[str] = None,
        **kwargs
    ) -> int:
        """Build Docker image with specified target."""
        if tag is None:
            if target == "production":
                tag = f"llm-tab-cleaner:{self.build_info['version']}"
            else:
                tag = f"llm-tab-cleaner:{target}"
        
        build_args = [
            f"BUILD_DATE={self.build_info['build_date']}",
            f"VERSION={self.build_info['version']}",
            f"VCS_REF={self.build_info['git_commit']}",
        ]
        
        cmd = [
            "docker", "build",
            "--target", target,
            "--tag", tag,
            *[f"--build-arg={arg}" for arg in build_args],
        ]
        
        if kwargs.get('no_cache'):
            cmd.append("--no-cache")
        
        if kwargs.get('pull'):
            cmd.append("--pull")
        
        cmd.append(".")
        
        print(f"Building Docker image: {tag}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0 and push:
            if registry:
                registry_tag = f"{registry}/{tag}"
                subprocess.run(["docker", "tag", tag, registry_tag])
                subprocess.run(["docker", "push", registry_tag])
            else:
                subprocess.run(["docker", "push", tag])
        
        return result.returncode
    
    def build_wheel(self, **kwargs) -> int:
        """Build Python wheel package."""
        print("Building Python wheel...")
        
        # Clean previous builds
        subprocess.run(["python", "-m", "build", "--clean"])
        
        cmd = ["python", "-m", "build"]
        
        if kwargs.get('wheel_only'):
            cmd.append("--wheel")
        
        if kwargs.get('sdist_only'):
            cmd.append("--sdist")
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Python package built successfully")
            dist_dir = self.project_root / "dist"
            for file in dist_dir.glob("*"):
                print(f"   📦 {file.name}")
        else:
            print("❌ Python package build failed")
        
        return result.returncode
    
    def run_quality_checks(self, **kwargs) -> int:
        """Run comprehensive quality checks."""
        print("Running quality checks...")
        
        checks = [
            ("Linting", ["ruff", "check", "src", "tests"]),
            ("Formatting", ["black", "--check", "src", "tests"]),
            ("Type checking", ["mypy", "src"]),
            ("Security scanning", ["bandit", "-r", "src/"]),
        ]
        
        if kwargs.get('fix'):
            checks[0] = ("Linting (with fixes)", ["ruff", "check", "--fix", "src", "tests"])
            checks[1] = ("Formatting (with fixes)", ["black", "src", "tests"])
        
        failed_checks = []
        
        for check_name, check_cmd in checks:
            print(f"\n🔍 {check_name}...")
            result = subprocess.run(check_cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                print(f"✅ {check_name} passed")
            else:
                print(f"❌ {check_name} failed")
                failed_checks.append(check_name)
                
                if kwargs.get('fail_fast'):
                    break
        
        if failed_checks:
            print(f"\n💥 {len(failed_checks)} quality check(s) failed: {', '.join(failed_checks)}")
            return 1
        else:
            print("\n🎉 All quality checks passed!")
            return 0
    
    def run_tests(self, **kwargs) -> int:
        """Run test suite."""
        print("Running tests...")
        
        cmd = ["python", "-m", "pytest"]
        
        if kwargs.get('coverage'):
            cmd.extend([
                "--cov=src/llm_tab_cleaner",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=xml"
            ])
        
        if kwargs.get('parallel'):
            cmd.extend(["-n", "auto"])
        
        if kwargs.get('verbose'):
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        if kwargs.get('fast'):
            cmd.extend(["-m", "not slow"])
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ All tests passed")
        else:
            print("❌ Some tests failed")
        
        return result.returncode
    
    def generate_sbom(self, output_format: str = "json", **kwargs) -> int:
        """Generate Software Bill of Materials (SBOM)."""
        print("Generating SBOM...")
        
        try:
            # Install cyclonedx-bom if not available
            subprocess.run([
                "pip", "install", "cyclonedx-bom"
            ], check=True, capture_output=True)
            
            output_file = f"sbom.{output_format}"
            cmd = [
                "cyclonedx-py",
                "-o", output_file,
                "-f", output_format
            ]
            
            result = subprocess.run(cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                print(f"✅ SBOM generated: {output_file}")
            else:
                print("❌ SBOM generation failed")
            
            return result.returncode
            
        except subprocess.CalledProcessError:
            print("❌ Failed to install cyclonedx-bom")
            return 1
    
    def security_scan(self, **kwargs) -> int:
        """Run comprehensive security scans."""
        print("Running security scans...")
        
        scans = [
            ("Dependency vulnerabilities", ["pip-audit", "--format=json", "--output=security-audit.json"]),
            ("Code security issues", ["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"]),
            ("Secrets detection", ["detect-secrets", "scan", "--all-files"]),
        ]
        
        failed_scans = []
        
        for scan_name, scan_cmd in scans:
            print(f"\n🔒 {scan_name}...")
            
            try:
                result = subprocess.run(scan_cmd, cwd=self.project_root, capture_output=True)
                
                if result.returncode == 0:
                    print(f"✅ {scan_name} - no issues found")
                else:
                    print(f"⚠️  {scan_name} - issues found, check reports")
                    failed_scans.append(scan_name)
                    
            except FileNotFoundError:
                print(f"⚠️  {scan_name} - tool not installed, skipping")
        
        if failed_scans and not kwargs.get('ignore_security_issues'):
            print(f"\n⚠️  Security issues found in: {', '.join(failed_scans)}")
            return 1
        else:
            print("\n🔒 Security scans completed")
            return 0
    
    def full_build(self, **kwargs) -> int:
        """Run complete build pipeline."""
        print("🚀 Starting full build pipeline...")
        print(f"Version: {self.build_info['version']}")
        print(f"Commit: {self.build_info['git_commit']}")
        print(f"Branch: {self.build_info['git_branch']}")
        print(f"Build date: {self.build_info['build_date']}")
        
        if self.build_info['is_dirty']:
            print("⚠️  Working directory has uncommitted changes")
        
        # Build steps
        steps = [
            ("Quality checks", self.run_quality_checks),
            ("Tests", self.run_tests),
            ("Security scans", self.security_scan),
            ("Python package", self.build_wheel),
            ("Docker image", lambda **kw: self.build_docker_image("production", **kw)),
            ("SBOM generation", self.generate_sbom),
        ]
        
        if kwargs.get('skip_quality'):
            steps = steps[1:]
        
        if kwargs.get('skip_tests'):
            steps = [s for s in steps if s[0] != "Tests"]
        
        if kwargs.get('skip_security'):
            steps = [s for s in steps if s[0] != "Security scans"]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            print(f"\n{'=' * 60}")
            print(f"Step: {step_name}")
            print('=' * 60)
            
            result = step_func(**kwargs)
            
            if result == 0:
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
                failed_steps.append(step_name)
                
                if kwargs.get('fail_fast'):
                    break
        
        # Build summary
        print(f"\n{'=' * 60}")
        print("BUILD SUMMARY")
        print('=' * 60)
        
        if failed_steps:
            print(f"❌ Build failed. Failed steps: {', '.join(failed_steps)}")
            return 1
        else:
            print("✅ Build completed successfully!")
            
            # Print build artifacts
            artifacts = []
            
            dist_dir = self.project_root / "dist"
            if dist_dir.exists():
                artifacts.extend([f"📦 {f.name}" for f in dist_dir.glob("*")])
            
            if (self.project_root / "sbom.json").exists():
                artifacts.append("📋 sbom.json")
            
            if artifacts:
                print("\nBuild artifacts:")
                for artifact in artifacts:
                    print(f"   {artifact}")
            
            return 0


def main():
    """Main entry point for build script."""
    parser = argparse.ArgumentParser(
        description="Build automation for LLM Tab Cleaner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build.py full --coverage --parallel
  python scripts/build.py docker --target=spark --push
  python scripts/build.py wheel --upload
  python scripts/build.py quality --fix
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'full', 'docker', 'wheel', 'quality', 'tests', 
            'security', 'sbom'
        ],
        help='Build command to execute'
    )
    
    # Common options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    
    # Full build options
    parser.add_argument('--skip-quality', action='store_true', help='Skip quality checks')
    parser.add_argument('--skip-tests', action='store_true', help='Skip tests')
    parser.add_argument('--skip-security', action='store_true', help='Skip security scans')
    
    # Docker options
    parser.add_argument('--target', default='production', help='Docker build target')
    parser.add_argument('--tag', help='Docker image tag')
    parser.add_argument('--push', action='store_true', help='Push Docker image')
    parser.add_argument('--registry', help='Docker registry')
    parser.add_argument('--no-cache', action='store_true', help='Build without cache')
    parser.add_argument('--pull', action='store_true', help='Pull base image')
    
    # Quality options
    parser.add_argument('--fix', action='store_true', help='Auto-fix issues where possible')
    
    # Test options
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    
    # Security options
    parser.add_argument('--ignore-security-issues', action='store_true', 
                       help='Continue build even with security issues')
    
    # Wheel options
    parser.add_argument('--wheel-only', action='store_true', help='Build wheel only')
    parser.add_argument('--sdist-only', action='store_true', help='Build source dist only')
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Create build manager
    builder = BuildManager(project_root)
    
    # Map commands to methods
    command_map = {
        'full': builder.full_build,
        'docker': builder.build_docker_image,
        'wheel': builder.build_wheel,
        'quality': builder.run_quality_checks,
        'tests': builder.run_tests,
        'security': builder.security_scan,
        'sbom': builder.generate_sbom,
    }
    
    # Execute command
    build_func = command_map[args.command]
    kwargs = {k: v for k, v in vars(args).items() if k != 'command' and v is not None}
    
    result = build_func(**kwargs)
    
    sys.exit(result)


if __name__ == "__main__":
    main()