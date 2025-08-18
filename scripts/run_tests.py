#!/usr/bin/env python3
"""Comprehensive test runner for LLM Tab Cleaner."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


class TestRunner:
    """Comprehensive test runner with multiple test suites and configurations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        
    def run_unit_tests(self, **kwargs) -> int:
        """Run unit tests."""
        cmd = [
            "pytest",
            str(self.test_dir / "unit"),
            "-m", "unit",
            "-v",
            "--tb=short"
        ]
        
        if kwargs.get('coverage'):
            cmd.extend([
                "--cov=src/llm_tab_cleaner",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
            
        if kwargs.get('parallel'):
            cmd.extend(["-n", "auto"])
            
        return subprocess.run(cmd).returncode
    
    def run_integration_tests(self, **kwargs) -> int:
        """Run integration tests."""
        cmd = [
            "pytest",
            str(self.test_dir / "integration"),
            "-m", "integration",
            "-v",
            "--tb=short"
        ]
        
        if kwargs.get('slow'):
            cmd.extend(["-m", "not slow"])
        else:
            cmd.append("--slow")
            
        return subprocess.run(cmd).returncode
    
    def run_e2e_tests(self, **kwargs) -> int:
        """Run end-to-end tests."""
        cmd = [
            "pytest",
            str(self.test_dir / "e2e"),
            "-m", "e2e",
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure for e2e
        ]
        
        return subprocess.run(cmd).returncode
    
    def run_performance_tests(self, **kwargs) -> int:
        """Run performance and benchmark tests."""
        cmd = [
            "pytest",
            str(self.test_dir / "performance"),
            "-m", "benchmark",
            "-v",
            "--benchmark-only",
            "--durations=0"
        ]
        
        if kwargs.get('benchmark_save'):
            cmd.extend(["--benchmark-save", kwargs['benchmark_save']])
            
        if kwargs.get('benchmark_compare'):
            cmd.extend(["--benchmark-compare", kwargs['benchmark_compare']])
            
        return subprocess.run(cmd).returncode
    
    def run_security_tests(self, **kwargs) -> int:
        """Run security-related tests."""
        cmd = [
            "pytest",
            str(self.test_dir),
            "-m", "security",
            "-v",
            "--tb=short"
        ]
        
        return subprocess.run(cmd).returncode
    
    def run_all_tests(self, **kwargs) -> int:
        """Run all test suites."""
        test_suites = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Security Tests", self.run_security_tests),
        ]
        
        if kwargs.get('include_e2e'):
            test_suites.append(("End-to-End Tests", self.run_e2e_tests))
        
        results = {}
        
        for suite_name, test_func in test_suites:
            print(f"\n{'=' * 60}")
            print(f"Running {suite_name}")
            print('=' * 60)
            
            result = test_func(**kwargs)
            results[suite_name] = result
            
            if result != 0 and kwargs.get('fail_fast'):
                print(f"\n❌ {suite_name} failed. Stopping due to --fail-fast.")
                break
        
        # Print summary
        print(f"\n{'=' * 60}")
        print("TEST SUMMARY")
        print('=' * 60)
        
        total_failures = 0
        for suite_name, result in results.items():
            status = "✅ PASS" if result == 0 else "❌ FAIL"
            print(f"{suite_name:<25} {status}")
            if result != 0:
                total_failures += 1
        
        if total_failures == 0:
            print("\n🎉 All test suites passed!")
        else:
            print(f"\n💥 {total_failures} test suite(s) failed.")
        
        return total_failures
    
    def run_quick_tests(self, **kwargs) -> int:
        """Run a quick subset of tests for fast feedback."""
        cmd = [
            "pytest",
            str(self.test_dir),
            "-m", "not slow and not requires_llm",
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
            "--maxfail=3"  # Stop after 3 failures
        ]
        
        return subprocess.run(cmd).returncode
    
    def run_ci_tests(self, **kwargs) -> int:
        """Run tests optimized for CI/CD environments."""
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "--tb=short",
            "--maxfail=5",
            "--cov=src/llm_tab_cleaner",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=80",
            "--junitxml=junit.xml",
            "-m", "not requires_llm and not requires_gpu"
        ]
        
        if kwargs.get('parallel'):
            cmd.extend(["-n", "auto"])
        
        return subprocess.run(cmd).returncode
    
    def run_regression_tests(self, **kwargs) -> int:
        """Run regression tests for known bugs."""
        cmd = [
            "pytest",
            str(self.test_dir),
            "-m", "regression",
            "-v",
            "--tb=long"
        ]
        
        return subprocess.run(cmd).returncode
    
    def check_test_coverage(self, **kwargs) -> int:
        """Check test coverage and generate reports."""
        cmd = [
            "pytest",
            str(self.test_dir),
            "--cov=src/llm_tab_cleaner",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=80",
            "-q"  # Quiet output, focus on coverage
        ]
        
        result = subprocess.run(cmd).returncode
        
        if result == 0:
            print("✅ Coverage requirements met!")
            print("📊 Coverage report generated in htmlcov/")
        else:
            print("❌ Coverage requirements not met!")
            
        return result


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for LLM Tab Cleaner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py unit --coverage
  python scripts/run_tests.py all --parallel --fail-fast
  python scripts/run_tests.py performance --benchmark-save baseline
  python scripts/run_tests.py ci --parallel
  python scripts/run_tests.py quick
        """
    )
    
    parser.add_argument(
        'suite',
        choices=[
            'unit', 'integration', 'e2e', 'performance', 'security',
            'all', 'quick', 'ci', 'regression', 'coverage'
        ],
        help='Test suite to run'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first test suite failure'
    )
    
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Include slow tests'
    )
    
    parser.add_argument(
        '--include-e2e',
        action='store_true',
        help='Include E2E tests in "all" suite'
    )
    
    parser.add_argument(
        '--benchmark-save',
        help='Save benchmark results with given name'
    )
    
    parser.add_argument(
        '--benchmark-compare',
        help='Compare with saved benchmark results'
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Create test runner
    runner = TestRunner(project_root)
    
    # Map suite names to methods
    suite_map = {
        'unit': runner.run_unit_tests,
        'integration': runner.run_integration_tests,
        'e2e': runner.run_e2e_tests,
        'performance': runner.run_performance_tests,
        'security': runner.run_security_tests,
        'all': runner.run_all_tests,
        'quick': runner.run_quick_tests,
        'ci': runner.run_ci_tests,
        'regression': runner.run_regression_tests,
        'coverage': runner.check_test_coverage,
    }
    
    # Run the selected test suite
    test_func = suite_map[args.suite]
    kwargs = vars(args)
    del kwargs['suite']  # Remove suite from kwargs
    
    result = test_func(**kwargs)
    
    sys.exit(result)


if __name__ == "__main__":
    main()