#!/usr/bin/env python3
"""Security and Performance Validator - Quality Gates Implementation.

This script implements comprehensive security scanning and performance validation
without requiring external tools like bandit or ruff.

Author: Terry (Terragon Labs)
"""

import re
import ast
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityScanner:
    """Security vulnerability scanner."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
                r'cursor\.execute\s*\([^)]*%[^)]*\)',
                r'\.format\s*\([^)]*\)\s*["\'][^"\']*SELECT',
            ],
            'command_injection': [
                r'os\.system\s*\([^)]*\+',
                r'subprocess\.[^(]+\([^)]*shell\s*=\s*True',
                r'eval\s*\([^)]*input',
                r'exec\s*\([^)]*input',
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\./[^)]*\)',
                r'file\s*\([^)]*\.\./[^)]*\)',
            ],
            'unsafe_deserialization': [
                r'pickle\.loads?\s*\(',
                r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader',
                r'marshal\.loads?\s*\(',
            ]
        }
        
        self.security_imports = {
            'dangerous': ['pickle', 'marshal', 'shelve'],
            'requires_care': ['subprocess', 'os', 'eval', 'exec', 'compile'],
            'crypto_weak': ['md5', 'sha1']
        }
    
    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single file for security vulnerabilities."""
        try:
            content = file_path.read_text(encoding='utf-8')
            vulnerabilities = []
            
            # Check for vulnerability patterns
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': vuln_type,
                            'severity': self._get_severity(vuln_type),
                            'line': line_num,
                            'code': match.group(0),
                            'description': self._get_description(vuln_type)
                        })
            
            # Check for dangerous imports
            import_vulns = self._check_imports(content, file_path)
            vulnerabilities.extend(import_vulns)
            
            return {
                'file': str(file_path),
                'vulnerabilities': vulnerabilities,
                'total_vulns': len(vulnerabilities),
                'critical_vulns': len([v for v in vulnerabilities if v['severity'] == 'critical']),
                'high_vulns': len([v for v in vulnerabilities if v['severity'] == 'high']),
                'medium_vulns': len([v for v in vulnerabilities if v['severity'] == 'medium']),
                'low_vulns': len([v for v in vulnerabilities if v['severity'] == 'low'])
            }
            
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            return {
                'file': str(file_path),
                'error': str(e),
                'vulnerabilities': [],
                'total_vulns': 0,
                'critical_vulns': 0,
                'high_vulns': 0,
                'medium_vulns': 0,
                'low_vulns': 0
            }
    
    def _check_imports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for dangerous imports."""
        vulnerabilities = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._check_import_name(alias.name, node.lineno, vulnerabilities)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._check_import_name(node.module, node.lineno, vulnerabilities)
                        
        except SyntaxError:
            # Try regex fallback for imports
            import_patterns = [
                r'^import\s+([^\s#]+)',
                r'^from\s+([^\s#]+)\s+import'
            ]
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                for pattern in import_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        self._check_import_name(match.group(1), i + 1, vulnerabilities)
        
        return vulnerabilities
    
    def _check_import_name(self, import_name: str, line_num: int, vulnerabilities: List[Dict]):
        """Check if an import is potentially dangerous."""
        if import_name in self.security_imports['dangerous']:
            vulnerabilities.append({
                'type': 'dangerous_import',
                'severity': 'high',
                'line': line_num,
                'code': f'import {import_name}',
                'description': f'Dangerous import: {import_name} can be unsafe if not used carefully'
            })
        elif import_name in self.security_imports['requires_care']:
            vulnerabilities.append({
                'type': 'requires_care_import',
                'severity': 'medium',
                'line': line_num,
                'code': f'import {import_name}',
                'description': f'Import requires careful usage: {import_name}'
            })
        elif import_name in self.security_imports['crypto_weak']:
            vulnerabilities.append({
                'type': 'weak_crypto',
                'severity': 'medium',
                'line': line_num,
                'code': f'import {import_name}',
                'description': f'Weak cryptographic algorithm: {import_name}'
            })
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'sql_injection': 'critical',
            'command_injection': 'critical',
            'path_traversal': 'high',
            'unsafe_deserialization': 'high',
            'dangerous_import': 'high',
            'requires_care_import': 'medium',
            'weak_crypto': 'medium'
        }
        return severity_map.get(vuln_type, 'low')
    
    def _get_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            'hardcoded_secrets': 'Hardcoded secrets in source code',
            'sql_injection': 'Potential SQL injection vulnerability',
            'command_injection': 'Potential command injection vulnerability',
            'path_traversal': 'Potential path traversal vulnerability',
            'unsafe_deserialization': 'Unsafe deserialization that could lead to code execution',
            'dangerous_import': 'Import of potentially dangerous module',
            'requires_care_import': 'Import of module that requires careful usage',
            'weak_crypto': 'Use of weak cryptographic algorithm'
        }
        return descriptions.get(vuln_type, 'Unknown vulnerability type')
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan all Python files in a directory."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'directory': str(directory),
            'files_scanned': 0,
            'total_vulnerabilities': 0,
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 0,
            'medium_vulnerabilities': 0,
            'low_vulnerabilities': 0,
            'file_results': []
        }
        
        python_files = list(directory.rglob("*.py"))
        
        for py_file in python_files:
            file_result = self.scan_file(py_file)
            results['file_results'].append(file_result)
            results['files_scanned'] += 1
            
            if 'error' not in file_result:
                results['total_vulnerabilities'] += file_result['total_vulns']
                results['critical_vulnerabilities'] += file_result['critical_vulns']
                results['high_vulnerabilities'] += file_result['high_vulns']
                results['medium_vulnerabilities'] += file_result['medium_vulns']
                results['low_vulnerabilities'] += file_result['low_vulns']
        
        # Calculate security score
        results['security_score'] = self._calculate_security_score(results)
        
        return results
    
    def _calculate_security_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall security score."""
        total_files = results['files_scanned']
        if total_files == 0:
            return 100.0
        
        # Penalty points for vulnerabilities
        penalty = (
            results['critical_vulnerabilities'] * 20 +
            results['high_vulnerabilities'] * 10 +
            results['medium_vulnerabilities'] * 5 +
            results['low_vulnerabilities'] * 1
        )
        
        # Score starts at 100 and decreases with vulnerabilities
        score = max(0, 100 - penalty)
        
        # Bonus for having no critical/high vulnerabilities
        if results['critical_vulnerabilities'] == 0 and results['high_vulnerabilities'] == 0:
            score = min(100, score + 10)
        
        return score


class PerformanceValidator:
    """Performance validation and benchmarking."""
    
    def __init__(self):
        self.benchmarks = {}
        self.performance_thresholds = {
            'import_time': 3.0,      # seconds
            'memory_usage': 200,      # MB
            'cpu_usage': 80,         # percent
            'startup_time': 5.0      # seconds
        }
    
    def measure_import_performance(self, module_path: str) -> Dict[str, Any]:
        """Measure module import performance."""
        import sys
        import importlib
        
        # Clear module from cache if it exists
        module_name = module_path.replace('/', '.').replace('.py', '')
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        start_time = time.time()
        try:
            importlib.import_module(module_name)
            import_time = time.time() - start_time
            
            return {
                'module': module_name,
                'import_time': import_time,
                'status': 'success',
                'meets_threshold': import_time <= self.performance_thresholds['import_time']
            }
            
        except Exception as e:
            return {
                'module': module_name,
                'import_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e),
                'meets_threshold': False
            }
    
    def measure_memory_usage(self) -> Dict[str, Any]:
        """Measure current memory usage."""
        if not PSUTIL_AVAILABLE:
            return {
                'status': 'skipped',
                'error': 'psutil not available',
                'meets_threshold': True,  # Assume passing if can't measure
                'memory_usage_mb': 0
            }
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_mb = memory_info.rss / 1024 / 1024
            
            return {
                'memory_usage_mb': memory_mb,
                'memory_usage_bytes': memory_info.rss,
                'meets_threshold': memory_mb <= self.performance_thresholds['memory_usage'],
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'meets_threshold': False
            }
    
    def measure_cpu_usage(self, duration: float = 5.0) -> Dict[str, Any]:
        """Measure CPU usage over a duration."""
        if not PSUTIL_AVAILABLE:
            return {
                'status': 'skipped',
                'error': 'psutil not available',
                'meets_threshold': True,  # Assume passing if can't measure
                'cpu_usage_percent': 0
            }
        
        try:
            # Initial CPU reading
            cpu_start = psutil.cpu_percent(interval=None)
            time.sleep(duration)
            cpu_end = psutil.cpu_percent(interval=None)
            
            avg_cpu = (cpu_start + cpu_end) / 2
            
            return {
                'cpu_usage_percent': avg_cpu,
                'measurement_duration': duration,
                'meets_threshold': avg_cpu <= self.performance_thresholds['cpu_usage'],
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'meets_threshold': False
            }
    
    def benchmark_code_execution(self, code_func, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark code execution time."""
        execution_times = []
        
        try:
            for _ in range(iterations):
                start_time = time.perf_counter()
                code_func()
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                return {
                    'iterations': iterations,
                    'average_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'total_time': sum(execution_times),
                    'status': 'success'
                }
            else:
                return {'status': 'failed', 'error': 'No execution times recorded'}
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'iterations': len(execution_times)
            }
    
    def validate_performance(self, src_directory: Path) -> Dict[str, Any]:
        """Validate overall performance."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'directory': str(src_directory),
            'import_performance': [],
            'memory_performance': {},
            'cpu_performance': {},
            'overall_score': 0.0
        }
        
        # Test import performance for key modules
        key_modules = [
            'llm_tab_cleaner.core',
            'llm_tab_cleaner.autonomous_production_system',
            'llm_tab_cleaner.enhanced_reliability_system',
            'llm_tab_cleaner.hyperscale_performance_optimizer'
        ]
        
        for module in key_modules:
            import_result = self.measure_import_performance(module)
            results['import_performance'].append(import_result)
        
        # Measure memory usage
        results['memory_performance'] = self.measure_memory_usage()
        
        # Measure CPU usage
        results['cpu_performance'] = self.measure_cpu_usage(2.0)  # 2 second sample
        
        # Calculate overall performance score
        results['overall_score'] = self._calculate_performance_score(results)
        
        return results
    
    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        score_components = []
        
        # Import performance score
        import_scores = []
        for import_result in results['import_performance']:
            if import_result['status'] == 'success':
                if import_result['meets_threshold']:
                    import_scores.append(100)
                else:
                    # Penalty based on how much over threshold
                    import_time = import_result['import_time']
                    threshold = self.performance_thresholds['import_time']
                    penalty = min(50, (import_time - threshold) / threshold * 50)
                    import_scores.append(max(50, 100 - penalty))
            else:
                import_scores.append(0)
        
        if import_scores:
            score_components.append(sum(import_scores) / len(import_scores))
        
        # Memory performance score
        memory_result = results['memory_performance']
        if memory_result['status'] == 'success':
            if memory_result['meets_threshold']:
                score_components.append(100)
            else:
                memory_mb = memory_result['memory_usage_mb']
                threshold = self.performance_thresholds['memory_usage']
                penalty = min(50, (memory_mb - threshold) / threshold * 50)
                score_components.append(max(50, 100 - penalty))
        else:
            score_components.append(50)
        
        # CPU performance score
        cpu_result = results['cpu_performance']
        if cpu_result['status'] == 'success':
            if cpu_result['meets_threshold']:
                score_components.append(100)
            else:
                cpu_percent = cpu_result['cpu_usage_percent']
                threshold = self.performance_thresholds['cpu_usage']
                penalty = min(50, (cpu_percent - threshold) / threshold * 50)
                score_components.append(max(50, 100 - penalty))
        else:
            score_components.append(50)
        
        # Overall score is average of components
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 50.0


def main():
    """Main validation execution."""
    print("🔒 Security and Performance Validation")
    print("=" * 60)
    
    src_path = Path("src/llm_tab_cleaner")
    
    # Security scanning
    print("🔍 Running security scan...")
    security_scanner = SecurityScanner()
    security_results = security_scanner.scan_directory(src_path)
    
    # Performance validation
    print("⚡ Running performance validation...")
    performance_validator = PerformanceValidator()
    performance_results = performance_validator.validate_performance(src_path)
    
    # Save results
    with open("security_scan_results.json", 'w') as f:
        json.dump(security_results, f, indent=2, default=str)
    
    with open("performance_validation_results.json", 'w') as f:
        json.dump(performance_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"🔒 Security Results:")
    print(f"  Files Scanned: {security_results['files_scanned']}")
    print(f"  Critical Vulnerabilities: {security_results['critical_vulnerabilities']}")
    print(f"  High Vulnerabilities: {security_results['high_vulnerabilities']}")
    print(f"  Medium Vulnerabilities: {security_results['medium_vulnerabilities']}")
    print(f"  Low Vulnerabilities: {security_results['low_vulnerabilities']}")
    print(f"  Security Score: {security_results['security_score']:.1f}/100")
    
    print(f"\n⚡ Performance Results:")
    print(f"  Import Tests: {len(performance_results['import_performance'])}")
    memory_mb = performance_results['memory_performance'].get('memory_usage_mb', 0)
    cpu_percent = performance_results['cpu_performance'].get('cpu_usage_percent', 0)
    print(f"  Memory Usage: {memory_mb:.1f} MB")
    print(f"  CPU Usage: {cpu_percent:.1f}%")
    print(f"  Performance Score: {performance_results['overall_score']:.1f}/100")
    
    # Overall assessment
    overall_security_pass = security_results['security_score'] >= 70
    overall_performance_pass = performance_results['overall_score'] >= 70
    
    print(f"\n🎯 Overall Assessment:")
    security_status = "✅ PASS" if overall_security_pass else "❌ FAIL"
    performance_status = "✅ PASS" if overall_performance_pass else "❌ FAIL"
    
    print(f"  Security: {security_status}")
    print(f"  Performance: {performance_status}")
    
    print("=" * 60)
    if overall_security_pass and overall_performance_pass:
        print("🎉 SECURITY AND PERFORMANCE VALIDATION PASSED!")
    else:
        print("⚠️  Some validations failed. Review results before deployment.")
    print("=" * 60)
    
    return 0 if (overall_security_pass and overall_performance_pass) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())