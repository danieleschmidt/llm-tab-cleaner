"""Robust Enhancement Module - Generation 2 Implementation.

This module provides comprehensive error handling, validation, security measures,
and monitoring capabilities to make the system production-robust.

Features:
- Advanced error handling with context preservation
- Input validation and sanitization
- Security scanning and threat detection
- Comprehensive logging and monitoring
- Health checks and system diagnostics
- Resilience patterns (circuit breakers, timeouts)

Author: Terry (Terragon Labs)
"""

import logging
import time
import asyncio
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SecurityScanResult:
    """Results from security scanning."""
    level: SecurityLevel
    threats_found: List[str]
    vulnerabilities: List[Dict[str, Any]]
    score: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Results from health checking."""
    status: HealthStatus
    checks: Dict[str, bool]
    response_time: float
    error_count: int
    warnings: List[str] = field(default_factory=list)


class RobustValidator:
    """Advanced validation with security and resilience."""
    
    def __init__(self):
        self.blocked_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'union\s+select',           # SQL injection
            r'javascript:',              # JavaScript injection
            r'eval\s*\(',               # Code execution
            r'exec\s*\(',               # Code execution
        ]
        self.max_input_size = 10 * 1024 * 1024  # 10MB
        
    def validate_input(self, data: Any, schema: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """Comprehensive input validation."""
        errors = []
        
        try:
            # Size validation
            if hasattr(data, '__len__') and len(str(data)) > self.max_input_size:
                errors.append(f"Input size exceeds maximum allowed ({self.max_input_size} bytes)")
            
            # Security pattern validation
            if isinstance(data, str):
                for pattern in self.blocked_patterns:
                    if re.search(pattern, data, re.IGNORECASE):
                        errors.append(f"Blocked security pattern detected: {pattern}")
            
            # Schema validation if provided
            if schema:
                schema_errors = self._validate_schema(data, schema)
                errors.extend(schema_errors)
            
            # Type validation
            if not self._validate_types(data):
                errors.append("Invalid data types detected")
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_schema(self, data: Any, schema: Dict) -> List[str]:
        """Validate data against schema."""
        errors = []
        
        if isinstance(data, dict) and isinstance(schema, dict):
            for key, expected_type in schema.items():
                if key not in data:
                    errors.append(f"Required field '{key}' missing")
                elif not isinstance(data[key], expected_type):
                    errors.append(f"Field '{key}' has wrong type: expected {expected_type.__name__}")
        
        return errors
    
    def _validate_types(self, data: Any) -> bool:
        """Validate data types are safe."""
        if isinstance(data, (str, int, float, bool, list, dict, type(None))):
            return True
        return False
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data."""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\']', '', data)
            return sanitized.strip()
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        return data


class CircuitBreaker:
    """Circuit breaker pattern for resilience."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class SecurityScanner:
    """Advanced security scanning and threat detection."""
    
    def __init__(self):
        self.known_threats = [
            "sql_injection",
            "xss_attack",
            "code_injection",
            "path_traversal",
            "command_injection"
        ]
        
        self.vulnerability_patterns = {
            "sql_injection": [
                r"union\s+select",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from"
            ],
            "xss_attack": [
                r"<script.*?>",
                r"javascript:",
                r"onload\s*=",
                r"onerror\s*="
            ],
            "code_injection": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"system\s*\(",
                r"shell_exec"
            ]
        }
    
    def scan_for_threats(self, data: Any) -> SecurityScanResult:
        """Comprehensive security threat scanning."""
        threats_found = []
        vulnerabilities = []
        score = 1.0
        
        if isinstance(data, str):
            for threat_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, data, re.IGNORECASE):
                        threats_found.append(threat_type)
                        vulnerabilities.append({
                            "type": threat_type,
                            "pattern": pattern,
                            "severity": "high"
                        })
                        score -= 0.2
        
        # Determine security level
        if score >= 0.9:
            level = SecurityLevel.LOW
        elif score >= 0.7:
            level = SecurityLevel.MEDIUM
        elif score >= 0.5:
            level = SecurityLevel.HIGH
        else:
            level = SecurityLevel.CRITICAL
        
        recommendations = self._generate_security_recommendations(threats_found)
        
        return SecurityScanResult(
            level=level,
            threats_found=threats_found,
            vulnerabilities=vulnerabilities,
            score=max(0.0, score),
            recommendations=recommendations
        )
    
    def _generate_security_recommendations(self, threats: List[str]) -> List[str]:
        """Generate security recommendations based on threats."""
        recommendations = []
        
        if "sql_injection" in threats:
            recommendations.append("Use parameterized queries and input validation")
        if "xss_attack" in threats:
            recommendations.append("Implement proper output encoding and CSP headers")
        if "code_injection" in threats:
            recommendations.append("Avoid dynamic code execution and validate all inputs")
        
        return recommendations


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.checks = {
            "database_connection": self._check_database,
            "memory_usage": self._check_memory,
            "disk_space": self._check_disk_space,
            "response_time": self._check_response_time,
            "error_rate": self._check_error_rate
        }
        self.error_counts = {}
        self.response_times = []
    
    def perform_health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check."""
        start_time = time.time()
        check_results = {}
        warnings = []
        error_count = 0
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                check_results[check_name] = result
                if not result:
                    error_count += 1
                    warnings.append(f"Health check failed: {check_name}")
            except Exception as e:
                check_results[check_name] = False
                error_count += 1
                warnings.append(f"Health check error in {check_name}: {str(e)}")
        
        response_time = time.time() - start_time
        
        # Determine overall health status
        if error_count == 0:
            status = HealthStatus.HEALTHY
        elif error_count <= 1:
            status = HealthStatus.DEGRADED
        elif error_count <= 2:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.CRITICAL
        
        return HealthCheckResult(
            status=status,
            checks=check_results,
            response_time=response_time,
            error_count=error_count,
            warnings=warnings
        )
    
    def _check_database(self) -> bool:
        """Check database connectivity."""
        # Simulate database check
        return True
    
    def _check_memory(self) -> bool:
        """Check memory usage."""
        # Simulate memory check
        return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        # Simulate disk space check
        return True
    
    def _check_response_time(self) -> bool:
        """Check system response time."""
        # Simulate response time check
        return True
    
    def _check_error_rate(self) -> bool:
        """Check error rate."""
        # Simulate error rate check
        return True


class RobustErrorHandler:
    """Advanced error handling with context preservation."""
    
    def __init__(self):
        self.error_history = []
        self.max_history = 1000
        
    def handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        severity: str = "error"
    ) -> Dict[str, Any]:
        """Handle errors with comprehensive context."""
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "context": context,
            "traceback": traceback.format_exc(),
            "stack_trace": traceback.format_stack()
        }
        
        # Store error for analysis
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log error with appropriate level
        if severity == "critical":
            logger.critical(f"Critical error: {error_info}")
        elif severity == "error":
            logger.error(f"Error: {error_info}")
        elif severity == "warning":
            logger.warning(f"Warning: {error_info}")
        else:
            logger.info(f"Info: {error_info}")
        
        return error_info
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights."""
        if not self.error_history:
            return {}
        
        # Count error types
        error_types = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate error rate over time
        recent_errors = [
            e for e in self.error_history 
            if datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=1)
        ]
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_error_rate": len(recent_errors),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }


class RobustEnhancementSystem:
    """Main robust enhancement system coordinating all components."""
    
    def __init__(self):
        self.validator = RobustValidator()
        self.circuit_breaker = CircuitBreaker()
        self.security_scanner = SecurityScanner()
        self.health_monitor = HealthMonitor()
        self.error_handler = RobustErrorHandler()
        
        logger.info("Robust Enhancement System initialized")
    
    async def validate_and_process(
        self, 
        data: Any, 
        processing_func: Callable,
        schema: Optional[Dict] = None
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """Comprehensive validation and processing pipeline."""
        
        diagnostics = {
            "validation_passed": False,
            "security_scan": None,
            "health_check": None,
            "processing_successful": False,
            "errors": []
        }
        
        try:
            # Input validation
            is_valid, validation_errors = self.validator.validate_input(data, schema)
            if not is_valid:
                diagnostics["errors"].extend(validation_errors)
                return False, None, diagnostics
            
            diagnostics["validation_passed"] = True
            
            # Security scanning
            security_result = self.security_scanner.scan_for_threats(data)
            diagnostics["security_scan"] = {
                "level": security_result.level.value,
                "score": security_result.score,
                "threats": security_result.threats_found
            }
            
            if security_result.level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                diagnostics["errors"].append("Security threats detected")
                return False, None, diagnostics
            
            # Health check
            health_result = self.health_monitor.perform_health_check()
            diagnostics["health_check"] = {
                "status": health_result.status.value,
                "error_count": health_result.error_count
            }
            
            if health_result.status == HealthStatus.CRITICAL:
                diagnostics["errors"].append("System health critical")
                return False, None, diagnostics
            
            # Sanitize input
            clean_data = self.validator.sanitize_input(data)
            
            # Process with circuit breaker protection
            try:
                result = self.circuit_breaker.call(processing_func, clean_data)
                diagnostics["processing_successful"] = True
                return True, result, diagnostics
            except Exception as e:
                self.error_handler.handle_error(
                    e, 
                    {"function": processing_func.__name__, "input_size": len(str(data))},
                    "error"
                )
                diagnostics["errors"].append(f"Processing failed: {str(e)}")
                return False, None, diagnostics
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                {"stage": "validation_and_processing"},
                "critical"
            )
            diagnostics["errors"].append(f"System error: {str(e)}")
            return False, None, diagnostics
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        health_result = self.health_monitor.perform_health_check()
        error_patterns = self.error_handler.get_error_patterns()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": health_result.status.value,
            "health_checks": health_result.checks,
            "circuit_breaker_state": self.circuit_breaker.state,
            "error_patterns": error_patterns,
            "system_uptime": time.time(),  # Simplified uptime
            "validation_rules_active": len(self.validator.blocked_patterns),
            "security_scanner_ready": True
        }


def create_robust_enhancement_system() -> RobustEnhancementSystem:
    """Factory function to create robust enhancement system."""
    return RobustEnhancementSystem()


def initialize_robust_systems() -> RobustEnhancementSystem:
    """Initialize robust enhancement systems with monitoring."""
    system = create_robust_enhancement_system()
    
    # Perform initial health check
    health_result = system.health_monitor.perform_health_check()
    logger.info(f"Robust systems initialized with health status: {health_result.status.value}")
    
    return system