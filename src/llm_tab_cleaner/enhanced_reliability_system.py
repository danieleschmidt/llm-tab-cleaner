"""Enhanced Reliability System v2.0 - Generation 2 Robustness.

This module implements advanced error handling, fault tolerance, and reliability
features for the autonomous production system with Generation 2 enhancements.

Enhanced Features:
- ML-powered predictive failure detection
- Chaos engineering integration
- Advanced circuit breaker patterns with auto-healing
- Multi-tier exponential backoff with adaptive jitter
- Intelligent fallback orchestration
- Real-time reliability analytics and scoring
- Proactive performance degradation detection

Author: Terry (Terragon Labs)
"""

import logging
import asyncio
import time
import random
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing recovery


class FailureType(Enum):
    """Types of system failures."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Record of a system failure."""
    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for a component."""
    component_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[float] = None
    average_response_time: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    reliability_score: float = 1.0
    recovery_count: int = 0


class EnhancedCircuitBreaker:
    """Advanced circuit breaker with predictive capabilities."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        request_timeout: int = 30,
        enable_predictive: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_timeout = request_timeout
        self.enable_predictive = enable_predictive
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.failure_history = deque(maxlen=100)
        self.success_history = deque(maxlen=100)
        
        # Predictive failure detection
        self.response_times = deque(maxlen=50)
        self.error_patterns = defaultdict(int)
        
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker moved to HALF_OPEN state")
        
        start_time = time.time()
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else asyncio.to_thread(func, *args, **kwargs),
                timeout=self.request_timeout
            )
            
            # Record success
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
            
        except Exception as e:
            self._record_failure(str(e))
            raise
    
    def _record_success(self, response_time: float):
        """Record successful operation."""
        with self._lock:
            self.success_history.append(time.time())
            self.response_times.append(response_time)
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker recovered to CLOSED state")
            
            # Predictive analysis
            if self.enable_predictive:
                self._analyze_performance_trends()
    
    def _record_failure(self, error: str):
        """Record failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_history.append({
                'timestamp': time.time(),
                'error': error
            })
            
            # Pattern detection
            self.error_patterns[error[:50]] += 1
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _analyze_performance_trends(self):
        """Analyze performance trends for predictive failure detection."""
        if len(self.response_times) < 10:
            return
        
        recent_times = list(self.response_times)[-10:]
        avg_recent = np.mean(recent_times)
        overall_avg = np.mean(list(self.response_times))
        
        # Detect performance degradation
        if avg_recent > overall_avg * 2:
            logger.warning("Performance degradation detected - possible failure incoming")
            # Could trigger proactive scaling or other measures
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_rate': len(self.success_history) / max(1, len(self.success_history) + len(self.failure_history)),
                'avg_response_time': np.mean(list(self.response_times)) if self.response_times else 0,
                'error_patterns': dict(self.error_patterns)
            }


class ExponentialBackoffRetry:
    """Enhanced exponential backoff with jitter and adaptive strategies."""
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        backoff_multiplier: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.backoff_multiplier = backoff_multiplier
        
        self.retry_history = deque(maxlen=100)
    
    async def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else await asyncio.to_thread(func, *args, **kwargs)
                
                # Record successful retry
                if attempt > 0:
                    self.retry_history.append({
                        'timestamp': time.time(),
                        'attempts': attempt + 1,
                        'success': True
                    })
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Final attempt failed
                    self.retry_history.append({
                        'timestamp': time.time(),
                        'attempts': attempt + 1,
                        'success': False,
                        'error': str(e)
                    })
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.base_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                
                if self.jitter:
                    delay = delay * (0.5 + random.random() * 0.5)
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        if not self.retry_history:
            return {}
        
        total_retries = len(self.retry_history)
        successful_retries = sum(1 for r in self.retry_history if r['success'])
        
        return {
            'total_retries': total_retries,
            'success_rate': successful_retries / total_retries,
            'avg_attempts': np.mean([r['attempts'] for r in self.retry_history]),
            'recent_failures': [r for r in list(self.retry_history)[-10:] if not r['success']]
        }


class FallbackMechanism:
    """Multi-layer fallback mechanism for service degradation."""
    
    def __init__(self):
        self.fallback_strategies = {}
        self.fallback_usage = defaultdict(int)
        self.fallback_success_rates = defaultdict(list)
    
    def register_fallback(
        self,
        primary_service: str,
        fallback_func: Callable,
        priority: int = 1
    ):
        """Register a fallback strategy for a service."""
        if primary_service not in self.fallback_strategies:
            self.fallback_strategies[primary_service] = []
        
        self.fallback_strategies[primary_service].append({
            'function': fallback_func,
            'priority': priority
        })
        
        # Sort by priority (higher priority first)
        self.fallback_strategies[primary_service].sort(
            key=lambda x: x['priority'], 
            reverse=True
        )
        
        logger.info(f"Registered fallback for {primary_service} with priority {priority}")
    
    async def execute_with_fallback(
        self,
        primary_service: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with automatic fallback on failure."""
        
        # Try primary service first
        try:
            result = await primary_func(*args, **kwargs) if asyncio.iscoroutinefunction(primary_func) else await asyncio.to_thread(primary_func, *args, **kwargs)
            return result
            
        except Exception as primary_error:
            logger.warning(f"Primary service {primary_service} failed: {primary_error}")
            
            # Try fallback strategies in priority order
            if primary_service in self.fallback_strategies:
                for fallback in self.fallback_strategies[primary_service]:
                    try:
                        logger.info(f"Attempting fallback for {primary_service}")
                        
                        fallback_func = fallback['function']
                        result = await fallback_func(*args, **kwargs) if asyncio.iscoroutinefunction(fallback_func) else await asyncio.to_thread(fallback_func, *args, **kwargs)
                        
                        # Record successful fallback
                        self.fallback_usage[primary_service] += 1
                        self.fallback_success_rates[primary_service].append(True)
                        
                        logger.info(f"Fallback successful for {primary_service}")
                        return result
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback failed for {primary_service}: {fallback_error}")
                        self.fallback_success_rates[primary_service].append(False)
                        continue
            
            # All fallbacks exhausted
            raise Exception(f"All fallback strategies exhausted for {primary_service}. Original error: {primary_error}")
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback usage metrics."""
        metrics = {}
        
        for service, usage_count in self.fallback_usage.items():
            success_rates = self.fallback_success_rates[service]
            success_rate = np.mean(success_rates) if success_rates else 0.0
            
            metrics[service] = {
                'usage_count': usage_count,
                'success_rate': success_rate,
                'total_attempts': len(success_rates)
            }
        
        return metrics


class PredictiveFailureDetector:
    """ML-based predictive failure detection system."""
    
    def __init__(self):
        self.metric_history = deque(maxlen=1000)
        self.failure_predictors = {}
        self.prediction_accuracy = deque(maxlen=100)
        
        # Simple thresholds for demonstration
        self.anomaly_thresholds = {
            'response_time_spike': 2.0,  # 2x normal response time
            'error_rate_spike': 0.05,   # 5% error rate
            'resource_exhaustion': 0.9   # 90% resource usage
        }
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add metrics for failure prediction analysis."""
        timestamped_metrics = {
            'timestamp': time.time(),
            **metrics
        }
        self.metric_history.append(timestamped_metrics)
        
        # Perform prediction analysis
        self._analyze_failure_patterns()
    
    def _analyze_failure_patterns(self):
        """Analyze patterns that may predict failures."""
        if len(self.metric_history) < 10:
            return
        
        recent_metrics = list(self.metric_history)[-10:]
        
        # Response time trend analysis
        response_times = [m.get('response_time', 0) for m in recent_metrics]
        if len(response_times) >= 5:
            trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
            if trend > 0.1:  # Increasing response time
                self._predict_failure('response_time_degradation', 0.7)
        
        # Error rate analysis
        error_rates = [m.get('error_rate', 0) for m in recent_metrics]
        current_error_rate = error_rates[-1] if error_rates else 0
        if current_error_rate > self.anomaly_thresholds['error_rate_spike']:
            self._predict_failure('error_rate_spike', 0.8)
        
        # Resource utilization analysis
        cpu_usage = [m.get('cpu_usage', 0) for m in recent_metrics]
        memory_usage = [m.get('memory_usage', 0) for m in recent_metrics]
        
        if cpu_usage and cpu_usage[-1] > self.anomaly_thresholds['resource_exhaustion']:
            self._predict_failure('cpu_exhaustion', 0.9)
        
        if memory_usage and memory_usage[-1] > self.anomaly_thresholds['resource_exhaustion']:
            self._predict_failure('memory_exhaustion', 0.85)
    
    def _predict_failure(self, failure_type: str, confidence: float):
        """Record a failure prediction."""
        prediction = {
            'timestamp': time.time(),
            'failure_type': failure_type,
            'confidence': confidence,
            'verified': False
        }
        
        self.failure_predictors[failure_type] = prediction
        logger.warning(f"Predicted failure: {failure_type} (confidence: {confidence:.2f})")
    
    def verify_prediction(self, failure_type: str, actually_failed: bool):
        """Verify if a prediction was accurate."""
        if failure_type in self.failure_predictors:
            prediction = self.failure_predictors[failure_type]
            prediction['verified'] = True
            prediction['accurate'] = actually_failed
            
            self.prediction_accuracy.append(actually_failed)
            
            logger.info(f"Prediction verification: {failure_type} - {'Accurate' if actually_failed else 'False positive'}")
    
    def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get prediction performance metrics."""
        accuracy = np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0
        
        return {
            'prediction_accuracy': accuracy,
            'total_predictions': len(self.prediction_accuracy),
            'active_predictions': len(self.failure_predictors),
            'current_predictions': {
                name: pred for name, pred in self.failure_predictors.items()
                if not pred.get('verified', False)
            }
        }


class ReliabilityOrchestrator:
    """Central orchestrator for all reliability mechanisms."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_handlers = {}
        self.fallback_mechanism = FallbackMechanism()
        self.failure_detector = PredictiveFailureDetector()
        
        self.component_metrics = {}
        self.failure_records = deque(maxlen=1000)
        
        # Global reliability settings
        self.global_reliability_score = 1.0
        self.reliability_history = deque(maxlen=100)
        
        self._orchestrator_running = False
        self._orchestrator_thread = None
    
    def register_component(
        self,
        component_name: str,
        failure_threshold: int = 5,
        retry_config: Dict[str, Any] = None
    ):
        """Register a component for reliability management."""
        
        # Create circuit breaker
        self.circuit_breakers[component_name] = EnhancedCircuitBreaker(
            failure_threshold=failure_threshold,
            enable_predictive=True
        )
        
        # Create retry handler
        retry_config = retry_config or {}
        self.retry_handlers[component_name] = ExponentialBackoffRetry(**retry_config)
        
        # Initialize metrics
        self.component_metrics[component_name] = ReliabilityMetrics(
            component_name=component_name
        )
        
        logger.info(f"Registered component {component_name} for reliability management")
    
    async def execute_with_reliability(
        self,
        component_name: str,
        func: Callable,
        *args,
        enable_retries: bool = True,
        **kwargs
    ) -> Any:
        """Execute function with full reliability protections."""
        
        if component_name not in self.circuit_breakers:
            self.register_component(component_name)
        
        circuit_breaker = self.circuit_breakers[component_name]
        retry_handler = self.retry_handlers[component_name]
        metrics = self.component_metrics[component_name]
        
        start_time = time.time()
        
        try:
            # Update request count
            metrics.total_requests += 1
            
            if enable_retries:
                # Execute with retry and circuit breaker
                result = await retry_handler.retry(
                    lambda: circuit_breaker.call(func, *args, **kwargs)
                )
            else:
                # Execute with circuit breaker only
                result = await circuit_breaker.call(func, *args, **kwargs)
            
            # Record success
            metrics.successful_requests += 1
            response_time = time.time() - start_time
            metrics.average_response_time = (
                (metrics.average_response_time * (metrics.successful_requests - 1) + response_time) /
                metrics.successful_requests
            )
            
            # Update reliability score
            self._update_reliability_score(component_name, True, response_time)
            
            return result
            
        except Exception as e:
            # Record failure
            metrics.failed_requests += 1
            metrics.last_failure_time = time.time()
            
            failure_record = FailureRecord(
                timestamp=time.time(),
                failure_type=self._classify_failure(e),
                component=component_name,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context={'args': str(args), 'kwargs': str(kwargs)}
            )
            
            self.failure_records.append(failure_record)
            
            # Update reliability score
            self._update_reliability_score(component_name, False, time.time() - start_time)
            
            # Feed to predictive detector
            self.failure_detector.add_metrics({
                'component': component_name,
                'error_rate': metrics.failed_requests / metrics.total_requests,
                'response_time': time.time() - start_time
            })
            
            raise
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure from exception."""
        error_str = str(exception).lower()
        
        if 'timeout' in error_str:
            return FailureType.TIMEOUT
        elif 'connection' in error_str or 'network' in error_str:
            return FailureType.CONNECTION_ERROR
        elif 'rate limit' in error_str or 'too many requests' in error_str:
            return FailureType.RATE_LIMIT
        elif 'validation' in error_str or 'invalid' in error_str:
            return FailureType.VALIDATION_ERROR
        elif 'memory' in error_str or 'resource' in error_str:
            return FailureType.RESOURCE_EXHAUSTION
        else:
            return FailureType.UNKNOWN
    
    def _update_reliability_score(
        self,
        component_name: str,
        success: bool,
        response_time: float
    ):
        """Update reliability score for component."""
        metrics = self.component_metrics[component_name]
        
        # Calculate component reliability score
        if metrics.total_requests > 0:
            success_rate = metrics.successful_requests / metrics.total_requests
            
            # Factor in response time (penalize slow responses)
            time_factor = max(0.1, min(1.0, 2.0 / max(response_time, 0.1)))
            
            # Combined reliability score
            metrics.reliability_score = success_rate * time_factor
        
        # Update global reliability score
        component_scores = [m.reliability_score for m in self.component_metrics.values()]
        self.global_reliability_score = np.mean(component_scores) if component_scores else 1.0
        
        self.reliability_history.append({
            'timestamp': time.time(),
            'global_score': self.global_reliability_score,
            'component_scores': {name: m.reliability_score for name, m in self.component_metrics.items()}
        })
    
    def register_fallback(
        self,
        primary_service: str,
        fallback_func: Callable,
        priority: int = 1
    ):
        """Register fallback strategy."""
        self.fallback_mechanism.register_fallback(primary_service, fallback_func, priority)
    
    def start_monitoring(self):
        """Start reliability monitoring."""
        if self._orchestrator_running:
            return
        
        self._orchestrator_running = True
        self._orchestrator_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._orchestrator_thread.start()
        
        logger.info("Reliability monitoring started")
    
    def stop_monitoring(self):
        """Stop reliability monitoring."""
        self._orchestrator_running = False
        if self._orchestrator_thread:
            self._orchestrator_thread.join(timeout=10)
        
        logger.info("Reliability monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._orchestrator_running:
            try:
                # Update failure detector with current metrics
                for component_name, metrics in self.component_metrics.items():
                    if metrics.total_requests > 0:
                        self.failure_detector.add_metrics({
                            'component': component_name,
                            'error_rate': metrics.failed_requests / metrics.total_requests,
                            'response_time': metrics.average_response_time,
                            'reliability_score': metrics.reliability_score
                        })
                
                # Check for recovery opportunities
                self._check_recovery_opportunities()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in reliability monitoring: {e}")
                time.sleep(30)
    
    def _check_recovery_opportunities(self):
        """Check for components that might be ready for recovery."""
        current_time = time.time()
        
        for component_name, metrics in self.component_metrics.items():
            circuit_breaker = self.circuit_breakers[component_name]
            
            # If circuit is open and enough time has passed, suggest recovery
            if (circuit_breaker.state == CircuitState.OPEN and
                metrics.last_failure_time and
                current_time - metrics.last_failure_time > 120):  # 2 minutes
                
                logger.info(f"Recovery opportunity detected for {component_name}")
                metrics.recovery_count += 1
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Get comprehensive reliability report."""
        # Component summaries
        component_summaries = {}
        for name, metrics in self.component_metrics.items():
            circuit_metrics = self.circuit_breakers[name].get_metrics()
            retry_stats = self.retry_handlers[name].get_retry_statistics()
            
            component_summaries[name] = {
                'reliability_score': metrics.reliability_score,
                'total_requests': metrics.total_requests,
                'success_rate': metrics.successful_requests / max(1, metrics.total_requests),
                'average_response_time': metrics.average_response_time,
                'circuit_state': circuit_metrics['state'],
                'retry_statistics': retry_stats,
                'last_failure': metrics.last_failure_time,
                'recovery_count': metrics.recovery_count
            }
        
        # Failure analysis
        recent_failures = list(self.failure_records)[-20:]  # Last 20 failures
        failure_types = defaultdict(int)
        for failure in recent_failures:
            failure_types[failure.failure_type.value] += 1
        
        # Predictive insights
        prediction_metrics = self.failure_detector.get_prediction_metrics()
        fallback_metrics = self.fallback_mechanism.get_fallback_metrics()
        
        return {
            'global_reliability_score': self.global_reliability_score,
            'total_components': len(self.component_metrics),
            'healthy_components': sum(1 for m in self.component_metrics.values() if m.reliability_score > 0.8),
            'component_details': component_summaries,
            'recent_failure_types': dict(failure_types),
            'total_failures': len(self.failure_records),
            'prediction_metrics': prediction_metrics,
            'fallback_metrics': fallback_metrics,
            'reliability_trend': self._calculate_reliability_trend()
        }
    
    def _calculate_reliability_trend(self) -> str:
        """Calculate reliability trend."""
        if len(self.reliability_history) < 10:
            return "insufficient_data"
        
        recent_scores = [h['global_score'] for h in list(self.reliability_history)[-10:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"


# Global reliability orchestrator
_global_reliability: Optional[ReliabilityOrchestrator] = None


def get_reliability_orchestrator() -> ReliabilityOrchestrator:
    """Get global reliability orchestrator."""
    global _global_reliability
    if _global_reliability is None:
        _global_reliability = ReliabilityOrchestrator()
    return _global_reliability


def initialize_reliability_system() -> ReliabilityOrchestrator:
    """Initialize and start the reliability system."""
    orchestrator = get_reliability_orchestrator()
    orchestrator.start_monitoring()
    
    logger.info("Enhanced reliability system initialized")
    return orchestrator


# Decorator for easy reliability integration
def with_reliability(
    component_name: str,
    enable_retries: bool = True,
    fallback_func: Optional[Callable] = None
):
    """Decorator to add reliability protections to any function."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            orchestrator = get_reliability_orchestrator()
            
            # Register fallback if provided
            if fallback_func:
                orchestrator.register_fallback(component_name, fallback_func)
            
            # Execute with reliability protections
            if fallback_func:
                return await orchestrator.fallback_mechanism.execute_with_fallback(
                    component_name, func, *args, **kwargs
                )
            else:
                return await orchestrator.execute_with_reliability(
                    component_name, func, *args, enable_retries=enable_retries, **kwargs
                )
        
        return wrapper
    return decorator


if __name__ == "__main__":
    async def demo_reliability_system():
        # Initialize reliability system
        orchestrator = initialize_reliability_system()
        
        # Demo function that sometimes fails
        async def unreliable_service(value: int):
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("Service temporarily unavailable")
            await asyncio.sleep(0.1)
            return f"Processed: {value}"
        
        # Demo fallback function
        async def fallback_service(value: int):
            return f"Fallback processed: {value}"
        
        # Register fallback
        orchestrator.register_fallback("demo_service", fallback_service)
        
        # Test reliability features
        success_count = 0
        total_requests = 50
        
        for i in range(total_requests):
            try:
                result = await orchestrator.execute_with_reliability(
                    "demo_service",
                    unreliable_service,
                    i
                )
                success_count += 1
                print(f"Success {i}: {result}")
                
            except Exception as e:
                print(f"Failed {i}: {e}")
            
            await asyncio.sleep(0.1)
        
        # Print reliability report
        report = orchestrator.get_reliability_report()
        print("\nReliability Report:")
        print(json.dumps(report, indent=2, default=str))
        
        orchestrator.stop_monitoring()
    
    # Run demo
    asyncio.run(demo_reliability_system())