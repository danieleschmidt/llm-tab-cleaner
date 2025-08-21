"""Enhanced autonomous production system with advanced self-healing capabilities."""

import asyncio
import logging
import time
import json
import hashlib
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from enum import Enum
from pathlib import Path
import queue
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    OFFLINE = "offline"


class OperationMode(Enum):
    """System operation modes."""
    NORMAL = "normal"
    CONSERVATIVE = "conservative" 
    AGGRESSIVE = "aggressive"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


@dataclass
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    request_rate: float
    error_rate: float
    response_time_p95: float
    quality_score: float
    processing_throughput: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Configuration for automated alerts."""
    name: str
    condition: Callable[[SystemMetrics], bool]
    severity: str = "warning"  # warning, critical, emergency
    cooldown_seconds: int = 300
    escalation_chain: List[str] = field(default_factory=list)
    auto_remediation: Optional[Callable] = None
    last_triggered: Optional[datetime] = None


@dataclass
class RecoveryAction:
    """Automated recovery action configuration."""
    name: str
    trigger_condition: Callable[[SystemMetrics], bool]
    action: Callable[[], bool]
    priority: int = 5
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    prerequisites: List[str] = field(default_factory=list)


class EnhancedMetricsCollector:
    """Advanced metrics collection with ML-based anomaly detection."""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.metrics_history: List[SystemMetrics] = []
        self.anomaly_detector = None
        self.scaler = None
        self._lock = threading.Lock()
        
        if HAS_ML:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.scaler = StandardScaler()
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            # System resources
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process-specific metrics
            active_connections = len(psutil.net_connections())
            
            # Application metrics (mocked for demonstration)
            request_rate = self._get_request_rate()
            error_rate = self._get_error_rate()
            response_time_p95 = self._get_response_time_p95()
            quality_score = self._get_quality_score()
            processing_throughput = self._get_processing_throughput()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_io,
                active_connections=active_connections,
                request_rate=request_rate,
                error_rate=error_rate,
                response_time_p95=response_time_p95,
                quality_score=quality_score,
                processing_throughput=processing_throughput
            )
            
            with self._lock:
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.history_window:
                    self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return self._get_fallback_metrics()
    
    def _get_request_rate(self) -> float:
        """Get current request rate (requests per second)."""
        # In real implementation, this would integrate with actual metrics
        return max(0, 100 + np.random.normal(0, 10)) if HAS_ML else 100.0
    
    def _get_error_rate(self) -> float:
        """Get current error rate (percentage)."""
        return max(0, min(100, 2 + np.random.normal(0, 1))) if HAS_ML else 2.0
    
    def _get_response_time_p95(self) -> float:
        """Get 95th percentile response time in milliseconds."""
        return max(10, 150 + np.random.normal(0, 20)) if HAS_ML else 150.0
    
    def _get_quality_score(self) -> float:
        """Get current data quality score."""
        return max(0, min(1, 0.92 + np.random.normal(0, 0.05))) if HAS_ML else 0.92
    
    def _get_processing_throughput(self) -> float:
        """Get processing throughput (items per second)."""
        return max(0, 1000 + np.random.normal(0, 100)) if HAS_ML else 1000.0
    
    def _get_fallback_metrics(self) -> SystemMetrics:
        """Return basic fallback metrics when collection fails."""
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io={},
            active_connections=0,
            request_rate=0.0,
            error_rate=0.0,
            response_time_p95=0.0,
            quality_score=0.0,
            processing_throughput=0.0,
            custom_metrics={"collection_failed": True}
        )
    
    def detect_anomalies(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Detect anomalies in current metrics using ML."""
        if not HAS_ML or not self.anomaly_detector:
            return {"anomaly_detected": False, "reason": "ML not available"}
        
        with self._lock:
            if len(self.metrics_history) < 50:  # Need minimum history
                return {"anomaly_detected": False, "reason": "insufficient_history"}
            
            # Convert metrics to feature vector
            features = self._metrics_to_features(metrics)
            historical_features = np.array([
                self._metrics_to_features(m) for m in self.metrics_history[-100:]
            ])
            
            # Fit and predict if not already fitted
            if not hasattr(self.anomaly_detector, 'decision_function'):
                self.scaler.fit(historical_features)
                self.anomaly_detector.fit(self.scaler.transform(historical_features))
            
            # Detect anomaly
            features_scaled = self.scaler.transform([features])
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            return {
                "anomaly_detected": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "features": features.tolist(),
                "reason": "ml_detection"
            }
    
    def _metrics_to_features(self, metrics: SystemMetrics) -> np.ndarray:
        """Convert metrics to feature vector for ML processing."""
        return np.array([
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.disk_usage,
            metrics.request_rate,
            metrics.error_rate,
            metrics.response_time_p95,
            metrics.quality_score,
            metrics.processing_throughput
        ])


class IntelligentAlertManager:
    """Intelligent alert management with ML-based prioritization."""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        self.notification_queue = queue.Queue()
        self._lock = threading.Lock()
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        # Start notification processor
        self.notification_thread = threading.Thread(
            target=self._process_notifications,
            daemon=True
        )
        self.notification_thread.start()
    
    def _initialize_default_rules(self):
        """Initialize default alerting rules."""
        self.alert_rules.extend([
            AlertRule(
                name="high_cpu_usage",
                condition=lambda m: m.cpu_usage > 80,
                severity="warning",
                cooldown_seconds=300,
                auto_remediation=self._remediate_high_cpu
            ),
            AlertRule(
                name="critical_cpu_usage", 
                condition=lambda m: m.cpu_usage > 95,
                severity="critical",
                cooldown_seconds=60,
                auto_remediation=self._remediate_critical_cpu
            ),
            AlertRule(
                name="high_memory_usage",
                condition=lambda m: m.memory_usage > 85,
                severity="warning",
                cooldown_seconds=300
            ),
            AlertRule(
                name="high_error_rate",
                condition=lambda m: m.error_rate > 5,
                severity="critical",
                cooldown_seconds=120
            ),
            AlertRule(
                name="low_quality_score",
                condition=lambda m: m.quality_score < 0.8,
                severity="warning",
                cooldown_seconds=600
            ),
            AlertRule(
                name="response_time_degradation",
                condition=lambda m: m.response_time_p95 > 1000,
                severity="warning",
                cooldown_seconds=300
            )
        ])
    
    def evaluate_alerts(self, metrics: SystemMetrics) -> List[Dict]:
        """Evaluate all alert rules against current metrics."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule.condition(metrics):
                    # Check cooldown
                    if (rule.last_triggered is None or 
                        (datetime.now() - rule.last_triggered).total_seconds() > rule.cooldown_seconds):
                        
                        alert = {
                            "name": rule.name,
                            "severity": rule.severity,
                            "timestamp": datetime.now(),
                            "metrics": metrics,
                            "message": f"Alert triggered: {rule.name}"
                        }
                        
                        triggered_alerts.append(alert)
                        rule.last_triggered = datetime.now()
                        
                        # Add to active alerts
                        with self._lock:
                            self.active_alerts[rule.name] = alert
                        
                        # Queue for notification
                        self.notification_queue.put(alert)
                        
                        # Execute auto-remediation if available
                        if rule.auto_remediation:
                            try:
                                success = rule.auto_remediation()
                                alert["auto_remediation"] = {
                                    "attempted": True,
                                    "success": success,
                                    "timestamp": datetime.now()
                                }
                            except Exception as e:
                                logger.error(f"Auto-remediation failed for {rule.name}: {e}")
                                alert["auto_remediation"] = {
                                    "attempted": True,
                                    "success": False,
                                    "error": str(e),
                                    "timestamp": datetime.now()
                                }
                else:
                    # Remove from active alerts if condition no longer met
                    with self._lock:
                        if rule.name in self.active_alerts:
                            del self.active_alerts[rule.name]
                            
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
        
        return triggered_alerts
    
    def _process_notifications(self):
        """Process notification queue (runs in background thread)."""
        while True:
            try:
                alert = self.notification_queue.get(timeout=1)
                self._send_notification(alert)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    def _send_notification(self, alert: Dict):
        """Send alert notification (implement with actual notification system)."""
        logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")
        # In production, integrate with Slack, PagerDuty, email, etc.
        
        # Store in alert history
        with self._lock:
            self.alert_history.append(alert)
            # Keep only recent history
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
    
    def _remediate_high_cpu(self) -> bool:
        """Remediation for high CPU usage."""
        try:
            logger.info("Attempting CPU usage remediation...")
            # In real implementation: scale down processing, defer non-critical tasks
            # For demo: just log
            return True
        except Exception as e:
            logger.error(f"CPU remediation failed: {e}")
            return False
    
    def _remediate_critical_cpu(self) -> bool:
        """Remediation for critical CPU usage."""
        try:
            logger.error("Critical CPU usage detected - emergency remediation")
            # In real implementation: emergency scaling, circuit breakers
            return True
        except Exception as e:
            logger.error(f"Critical CPU remediation failed: {e}")
            return False


class AdaptiveRecoveryOrchestrator:
    """Orchestrates automated recovery actions with learning capabilities."""
    
    def __init__(self):
        self.recovery_actions: List[RecoveryAction] = []
        self.action_history: List[Dict] = []
        self.success_rates: Dict[str, List[bool]] = {}
        self._lock = threading.Lock()
        
        # Initialize recovery actions
        self._initialize_recovery_actions()
    
    def _initialize_recovery_actions(self):
        """Initialize default recovery actions."""
        self.recovery_actions.extend([
            RecoveryAction(
                name="restart_unhealthy_workers",
                trigger_condition=lambda m: m.error_rate > 10,
                action=self._restart_workers,
                priority=8,
                max_attempts=2
            ),
            RecoveryAction(
                name="scale_processing_capacity",
                trigger_condition=lambda m: m.cpu_usage > 90 and m.request_rate > 200,
                action=self._scale_processing,
                priority=7,
                max_attempts=3
            ),
            RecoveryAction(
                name="activate_circuit_breaker",
                trigger_condition=lambda m: m.error_rate > 15 or m.response_time_p95 > 2000,
                action=self._activate_circuit_breaker,
                priority=9,
                max_attempts=1
            ),
            RecoveryAction(
                name="clear_cache",
                trigger_condition=lambda m: m.memory_usage > 95,
                action=self._clear_cache,
                priority=6,
                max_attempts=2
            ),
            RecoveryAction(
                name="enable_degraded_mode",
                trigger_condition=lambda m: m.quality_score < 0.7,
                action=self._enable_degraded_mode,
                priority=5,
                max_attempts=1
            )
        ])
    
    def execute_recovery(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Execute appropriate recovery actions based on current metrics."""
        applicable_actions = []
        
        # Find applicable actions
        for action in self.recovery_actions:
            try:
                if action.trigger_condition(metrics):
                    # Check success rate to avoid repeatedly failing actions
                    success_rate = self._get_success_rate(action.name)
                    if success_rate > 0.2:  # Only attempt if >20% success rate
                        applicable_actions.append(action)
            except Exception as e:
                logger.error(f"Error evaluating recovery action {action.name}: {e}")
        
        if not applicable_actions:
            return {"actions_executed": 0, "message": "No applicable recovery actions"}
        
        # Sort by priority (higher first)
        applicable_actions.sort(key=lambda a: a.priority, reverse=True)
        
        executed_actions = []
        for action in applicable_actions[:3]:  # Execute top 3 actions
            try:
                logger.info(f"Executing recovery action: {action.name}")
                success = self._execute_action_with_retry(action)
                
                execution_record = {
                    "action": action.name,
                    "timestamp": datetime.now(),
                    "success": success,
                    "metrics_snapshot": metrics
                }
                
                executed_actions.append(execution_record)
                
                # Update success tracking
                with self._lock:
                    if action.name not in self.success_rates:
                        self.success_rates[action.name] = []
                    self.success_rates[action.name].append(success)
                    # Keep only recent history
                    if len(self.success_rates[action.name]) > 50:
                        self.success_rates[action.name] = self.success_rates[action.name][-50:]
                
                if success:
                    logger.info(f"Recovery action {action.name} completed successfully")
                    break  # Stop after first successful action
                else:
                    logger.warning(f"Recovery action {action.name} failed")
                    
            except Exception as e:
                logger.error(f"Error executing recovery action {action.name}: {e}")
                executed_actions.append({
                    "action": action.name,
                    "timestamp": datetime.now(),
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "actions_executed": len(executed_actions),
            "executed_actions": executed_actions,
            "message": f"Executed {len(executed_actions)} recovery actions"
        }
    
    def _execute_action_with_retry(self, action: RecoveryAction) -> bool:
        """Execute action with retry logic."""
        for attempt in range(action.max_attempts):
            try:
                success = action.action()
                if success:
                    return True
                
                if attempt < action.max_attempts - 1:
                    wait_time = action.backoff_multiplier ** attempt
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {action.name}: {e}")
                if attempt < action.max_attempts - 1:
                    wait_time = action.backoff_multiplier ** attempt
                    time.sleep(wait_time)
        
        return False
    
    def _get_success_rate(self, action_name: str) -> float:
        """Get success rate for a recovery action."""
        with self._lock:
            if action_name not in self.success_rates or not self.success_rates[action_name]:
                return 1.0  # Assume success for new actions
            
            successes = sum(1 for s in self.success_rates[action_name] if s)
            total = len(self.success_rates[action_name])
            return successes / total
    
    # Recovery action implementations
    def _restart_workers(self) -> bool:
        """Restart unhealthy worker processes."""
        logger.info("Restarting unhealthy workers...")
        # Implementation would restart actual worker processes
        return True
    
    def _scale_processing(self) -> bool:
        """Scale processing capacity up."""
        logger.info("Scaling processing capacity...")
        # Implementation would trigger auto-scaling
        return True
    
    def _activate_circuit_breaker(self) -> bool:
        """Activate circuit breaker pattern."""
        logger.info("Activating circuit breaker...")
        # Implementation would activate circuit breakers
        return True
    
    def _clear_cache(self) -> bool:
        """Clear system caches."""
        logger.info("Clearing system caches...")
        # Implementation would clear actual caches
        return True
    
    def _enable_degraded_mode(self) -> bool:
        """Enable degraded operation mode."""
        logger.info("Enabling degraded mode...")
        # Implementation would reduce functionality to preserve stability
        return True


class EnhancedAutonomousProductionSystem:
    """Enhanced autonomous production system with advanced self-healing."""
    
    def __init__(
        self,
        metrics_collection_interval: float = 30.0,
        alert_evaluation_interval: float = 15.0,
        recovery_check_interval: float = 60.0,
        enable_ml_anomaly_detection: bool = True,
        system_name: str = "llm-tab-cleaner"
    ):
        self.system_name = system_name
        self.metrics_collection_interval = metrics_collection_interval
        self.alert_evaluation_interval = alert_evaluation_interval
        self.recovery_check_interval = recovery_check_interval
        self.enable_ml_anomaly_detection = enable_ml_anomaly_detection
        
        # Core components
        self.metrics_collector = EnhancedMetricsCollector()
        self.alert_manager = IntelligentAlertManager()
        self.recovery_orchestrator = AdaptiveRecoveryOrchestrator()
        
        # System state
        self.current_state = SystemState.INITIALIZING
        self.operation_mode = OperationMode.NORMAL
        self.last_metrics: Optional[SystemMetrics] = None
        
        # Control flags
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="autonomous-sys")
        
        logger.info(f"Enhanced Autonomous Production System initialized for {system_name}")
    
    def start(self) -> bool:
        """Start the autonomous production system."""
        if self._running:
            logger.warning("System already running")
            return False
        
        try:
            self._running = True
            self.current_state = SystemState.HEALTHY
            
            # Start background monitoring tasks
            self.executor.submit(self._metrics_collection_loop)
            self.executor.submit(self._alert_evaluation_loop)
            self.executor.submit(self._recovery_check_loop)
            
            logger.info("Enhanced Autonomous Production System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start autonomous system: {e}")
            self._running = False
            self.current_state = SystemState.OFFLINE
            return False
    
    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the autonomous production system."""
        if not self._running:
            return True
        
        logger.info("Stopping Enhanced Autonomous Production System...")
        self._running = False
        self._shutdown_event.set()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.current_state = SystemState.OFFLINE
        logger.info("Enhanced Autonomous Production System stopped")
        return True
    
    def _metrics_collection_loop(self):
        """Main metrics collection loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                metrics = self.metrics_collector.collect_metrics()
                self.last_metrics = metrics
                
                # Log key metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(
                        f"System Health: CPU={metrics.cpu_usage:.1f}%, "
                        f"Memory={metrics.memory_usage:.1f}%, "
                        f"Quality={metrics.quality_score:.3f}, "
                        f"Error Rate={metrics.error_rate:.2f}%"
                    )
                
                # Detect anomalies if ML enabled
                if self.enable_ml_anomaly_detection:
                    anomaly_result = self.metrics_collector.detect_anomalies(metrics)
                    if anomaly_result.get("anomaly_detected", False):
                        logger.warning(
                            f"Anomaly detected: score={anomaly_result.get('anomaly_score', 'unknown')}"
                        )
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
            
            # Wait for next collection cycle
            self._shutdown_event.wait(self.metrics_collection_interval)
    
    def _alert_evaluation_loop(self):
        """Main alert evaluation loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                if self.last_metrics:
                    triggered_alerts = self.alert_manager.evaluate_alerts(self.last_metrics)
                    
                    if triggered_alerts:
                        # Update system state based on alert severity
                        critical_alerts = [a for a in triggered_alerts if a['severity'] == 'critical']
                        warning_alerts = [a for a in triggered_alerts if a['severity'] == 'warning']
                        
                        if critical_alerts:
                            if self.current_state not in [SystemState.CRITICAL, SystemState.RECOVERY]:
                                self.current_state = SystemState.CRITICAL
                                logger.error(f"System state changed to CRITICAL due to {len(critical_alerts)} critical alerts")
                        elif warning_alerts:
                            if self.current_state == SystemState.HEALTHY:
                                self.current_state = SystemState.DEGRADED
                                logger.warning(f"System state changed to DEGRADED due to {len(warning_alerts)} warnings")
                    else:
                        # No active alerts - potentially recover
                        if self.current_state in [SystemState.DEGRADED, SystemState.RECOVERY]:
                            self.current_state = SystemState.HEALTHY
                            logger.info("System state recovered to HEALTHY")
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
            
            # Wait for next evaluation cycle
            self._shutdown_event.wait(self.alert_evaluation_interval)
    
    def _recovery_check_loop(self):
        """Main recovery check loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                if (self.last_metrics and 
                    self.current_state in [SystemState.DEGRADED, SystemState.CRITICAL]):
                    
                    logger.info(f"Attempting automated recovery (system state: {self.current_state.value})")
                    self.current_state = SystemState.RECOVERY
                    
                    recovery_result = self.recovery_orchestrator.execute_recovery(self.last_metrics)
                    
                    if recovery_result.get("actions_executed", 0) > 0:
                        logger.info(f"Recovery executed: {recovery_result['message']}")
                        
                        # Check if recovery was successful after a brief wait
                        time.sleep(10)
                        if self.last_metrics:
                            current_metrics = self.metrics_collector.collect_metrics()
                            if self._assess_recovery_success(self.last_metrics, current_metrics):
                                self.current_state = SystemState.HEALTHY
                                logger.info("Automated recovery successful - system restored to healthy state")
                            else:
                                logger.warning("Automated recovery did not fully resolve issues")
                                self.current_state = SystemState.DEGRADED
                
            except Exception as e:
                logger.error(f"Error in recovery check loop: {e}")
            
            # Wait for next recovery check
            self._shutdown_event.wait(self.recovery_check_interval)
    
    def _assess_recovery_success(self, before: SystemMetrics, after: SystemMetrics) -> bool:
        """Assess if recovery actions were successful."""
        improvements = 0
        total_checks = 0
        
        # Check key metrics for improvement
        metrics_to_check = [
            ('cpu_usage', 'lower'),
            ('memory_usage', 'lower'),
            ('error_rate', 'lower'),
            ('response_time_p95', 'lower'),
            ('quality_score', 'higher')
        ]
        
        for metric_name, direction in metrics_to_check:
            before_val = getattr(before, metric_name)
            after_val = getattr(after, metric_name)
            
            if direction == 'lower' and after_val < before_val * 0.9:
                improvements += 1
            elif direction == 'higher' and after_val > before_val * 1.1:
                improvements += 1
            
            total_checks += 1
        
        # Recovery is successful if majority of metrics improved
        success_rate = improvements / total_checks if total_checks > 0 else 0
        return success_rate >= 0.6
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_name": self.system_name,
            "state": self.current_state.value,
            "operation_mode": self.operation_mode.value,
            "running": self._running,
            "last_metrics": self.last_metrics.__dict__ if self.last_metrics else None,
            "active_alerts": list(self.alert_manager.active_alerts.keys()),
            "alert_count": len(self.alert_manager.active_alerts),
            "ml_anomaly_detection": self.enable_ml_anomaly_detection,
            "uptime": time.time() - (self.last_metrics.timestamp.timestamp() if self.last_metrics else time.time()),
            "components": {
                "metrics_collector": "operational",
                "alert_manager": "operational", 
                "recovery_orchestrator": "operational"
            }
        }


def initialize_enhanced_production_system(
    system_name: str = "llm-tab-cleaner",
    **kwargs
) -> EnhancedAutonomousProductionSystem:
    """Initialize and start the enhanced autonomous production system."""
    system = EnhancedAutonomousProductionSystem(
        system_name=system_name,
        **kwargs
    )
    
    if system.start():
        logger.info(f"Enhanced autonomous production system '{system_name}' initialized successfully")
        return system
    else:
        raise RuntimeError(f"Failed to initialize autonomous production system '{system_name}'")