"""Monitoring, health checks, and observability for data cleaning operations."""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages metrics for monitoring."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_points_per_metric: Maximum data points to keep per metric
        """
        self.max_points = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric (monotonically increasing)."""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            self.metrics[key].append(MetricPoint(
                timestamp=datetime.utcnow(),
                value=self.counters[key],
                labels=labels or {}
            ))
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric (can go up or down)."""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            self.metrics[key].append(MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            ))
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.metrics[key].append(MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            ))
    
    def get_metric_summary(self, name: str, labels: Dict[str, str] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            if key not in self.metrics or not self.metrics[key]:
                return {"error": "No data available"}
            
            points = list(self.metrics[key])
            values = [p.value for p in points]
            
            if not values:
                return {"error": "No values available"}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "latest": values[-1],
                "latest_timestamp": points[-1].timestamp.isoformat(),
                "oldest_timestamp": points[0].timestamp.isoformat()
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        all_metrics = {}
        
        with self._lock:
            for key in self.metrics:
                metric_name = key.split("|")[0]  # Remove labels from key
                all_metrics[key] = self.get_metric_summary(metric_name)
        
        return all_metrics
    
    def reset_metric(self, name: str, labels: Dict[str, str] = None):
        """Reset a specific metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            if key in self.metrics:
                self.metrics[key].clear()
            if key in self.counters:
                self.counters[key] = 0.0
            if key in self.gauges:
                del self.gauges[key]
    
    def export_metrics(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export metrics in various formats."""
        if format_type == "json":
            return self.get_all_metrics()
        elif format_type == "prometheus":
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for key, points in self.metrics.items():
                if not points:
                    continue
                
                metric_name, labels_str = (key.split("|", 1) + [""])[:2]
                latest_point = points[-1]
                
                # Format metric line
                if labels_str:
                    lines.append(f'{metric_name}{{{labels_str}}} {latest_point.value}')
                else:
                    lines.append(f'{metric_name} {latest_point.value}')
        
        return "\n".join(lines)


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize health checker.
        
        Args:
            metrics_collector: Optional metrics collector for recording health metrics
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status="unhealthy",
                message=f"Health check '{name}' not found",
                timestamp=datetime.utcnow(),
                duration_ms=0
            )
        
        start_time = time.time()
        
        try:
            result = self.health_checks[name]()
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            
            # Record metrics
            self.metrics.record_counter(
                "health_checks_total",
                labels={"check": name, "status": result.status}
            )
            self.metrics.record_histogram(
                "health_check_duration_ms",
                duration_ms,
                labels={"check": name}
            )
            
            self._last_results[name] = result
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                metadata={"error": str(e)}
            )
            
            # Record metrics
            self.metrics.record_counter(
                "health_checks_total",
                labels={"check": name, "status": "unhealthy"}
            )
            self.metrics.record_histogram(
                "health_check_duration_ms",
                duration_ms,
                labels={"check": name}
            )
            
            self._last_results[name] = result
            logger.error(f"Health check '{name}' failed: {e}")
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.health_checks:
            results[name] = self.run_check(name)
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if not results:
            return {
                "status": "unhealthy",
                "message": "No health checks registered",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {}
            }
        
        # Determine overall status
        statuses = [result.status for result in results.values()]
        
        if all(status == "healthy" for status in statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "message": f"System status: {overall_status}",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: {
                "status": result.status,
                "message": result.message,
                "duration_ms": result.duration_ms
            } for name, result in results.items()}
        }


class PerformanceMonitor:
    """Monitors performance metrics and identifies issues."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor."""
        self.metrics = metrics_collector
        self.alert_thresholds = {
            "processing_time_ms": 30000,  # 30 seconds
            "memory_usage_mb": 1000,      # 1 GB
            "error_rate": 0.05,           # 5%
            "queue_depth": 1000
        }
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Check processing time
        processing_time = self.metrics.get_metric_summary("processing_time_ms")
        if (processing_time.get("latest", 0) > self.alert_thresholds["processing_time_ms"]):
            alert = {
                "type": "performance",
                "severity": "warning",
                "metric": "processing_time_ms",
                "message": f"High processing time: {processing_time['latest']:.2f}ms",
                "threshold": self.alert_thresholds["processing_time_ms"],
                "current_value": processing_time["latest"],
                "timestamp": datetime.utcnow().isoformat()
            }
            alerts.append(alert)
            self._trigger_alert("high_processing_time", alert)
        
        # Check memory usage
        memory_usage = self.metrics.get_metric_summary("memory_usage_mb")
        if (memory_usage.get("latest", 0) > self.alert_thresholds["memory_usage_mb"]):
            alert = {
                "type": "performance",
                "severity": "warning",
                "metric": "memory_usage_mb",
                "message": f"High memory usage: {memory_usage['latest']:.2f}MB",
                "threshold": self.alert_thresholds["memory_usage_mb"],
                "current_value": memory_usage["latest"],
                "timestamp": datetime.utcnow().isoformat()
            }
            alerts.append(alert)
            self._trigger_alert("high_memory_usage", alert)
        
        # Check error rate
        total_operations = self.metrics.get_metric_summary("operations_total").get("latest", 0)
        error_operations = self.metrics.get_metric_summary("operations_errors").get("latest", 0)
        
        if total_operations > 0:
            error_rate = error_operations / total_operations
            if error_rate > self.alert_thresholds["error_rate"]:
                alert = {
                    "type": "performance",
                    "severity": "critical",
                    "metric": "error_rate",
                    "message": f"High error rate: {error_rate:.2%}",
                    "threshold": self.alert_thresholds["error_rate"],
                    "current_value": error_rate,
                    "timestamp": datetime.utcnow().isoformat()
                }
                alerts.append(alert)
                self._trigger_alert("high_error_rate", alert)
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "alerts": self.check_performance_alerts()
        }
        
        # Key performance metrics
        key_metrics = [
            "processing_time_ms",
            "memory_usage_mb",
            "operations_total",
            "operations_errors",
            "data_rows_processed",
            "fixes_applied"
        ]
        
        for metric in key_metrics:
            summary["metrics"][metric] = self.metrics.get_metric_summary(metric)
        
        return summary
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


class CleaningMonitor:
    """Monitors data cleaning operations and provides insights."""
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 health_checker: Optional[HealthChecker] = None):
        """Initialize cleaning monitor."""
        self.metrics = metrics_collector or MetricsCollector()
        self.health = health_checker or HealthChecker(self.metrics)
        self.performance = PerformanceMonitor(self.metrics)
        
        # Register default health checks
        self._register_default_health_checks()
    
    def start_operation_monitoring(self, operation_id: str, operation_type: str) -> Dict[str, Any]:
        """Start monitoring a cleaning operation."""
        context = {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage()
        }
        
        # Record operation start
        self.metrics.record_counter(
            "operations_total",
            labels={"operation": operation_type}
        )
        
        logger.info(f"Started monitoring operation: {operation_id}")
        return context
    
    def end_operation_monitoring(self, 
                                context: Dict[str, Any], 
                                success: bool, 
                                rows_processed: int = 0,
                                fixes_applied: int = 0):
        """End monitoring of a cleaning operation."""
        end_time = time.time()
        duration_ms = (end_time - context["start_time"]) * 1000
        end_memory = self._get_memory_usage()
        
        # Record metrics
        self.metrics.record_histogram(
            "processing_time_ms",
            duration_ms,
            labels={"operation": context["operation_type"]}
        )
        
        if context["start_memory"] and end_memory:
            memory_diff = end_memory - context["start_memory"]
            self.metrics.record_histogram(
                "memory_usage_mb",
                memory_diff,
                labels={"operation": context["operation_type"]}
            )
        
        self.metrics.record_counter(
            "data_rows_processed",
            rows_processed,
            labels={"operation": context["operation_type"]}
        )
        
        self.metrics.record_counter(
            "fixes_applied",
            fixes_applied,
            labels={"operation": context["operation_type"]}
        )
        
        if not success:
            self.metrics.record_counter(
                "operations_errors",
                labels={"operation": context["operation_type"]}
            )
        
        logger.info(f"Ended monitoring operation: {context['operation_id']}, "
                   f"Duration: {duration_ms:.2f}ms, Success: {success}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": self.health.get_overall_health(),
            "performance": self.performance.get_performance_summary(),
            "metrics_summary": self.metrics.get_all_metrics()
        }
    
    def export_monitoring_data(self, 
                             output_path: str, 
                             format_type: str = "json"):
        """Export monitoring data to file."""
        dashboard_data = self.get_monitoring_dashboard()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Exported monitoring data to {output_path}")
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        
        def check_metrics_collector():
            """Check metrics collector health."""
            try:
                metric_count = len(self.metrics.metrics)
                return HealthCheckResult(
                    name="metrics_collector",
                    status="healthy" if metric_count >= 0 else "unhealthy",
                    message=f"Metrics collector active with {metric_count} metrics",
                    timestamp=datetime.utcnow(),
                    duration_ms=0,
                    metadata={"metric_count": metric_count}
                )
            except Exception as e:
                return HealthCheckResult(
                    name="metrics_collector",
                    status="unhealthy",
                    message=f"Metrics collector error: {str(e)}",
                    timestamp=datetime.utcnow(),
                    duration_ms=0
                )
        
        def check_memory_usage():
            """Check system memory usage."""
            try:
                memory_mb = self._get_memory_usage()
                if memory_mb is None:
                    status = "degraded"
                    message = "Memory usage unavailable"
                elif memory_mb > 2000:  # 2 GB threshold
                    status = "degraded"
                    message = f"High memory usage: {memory_mb:.2f}MB"
                else:
                    status = "healthy"
                    message = f"Memory usage normal: {memory_mb:.2f}MB"
                
                return HealthCheckResult(
                    name="memory_usage",
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    duration_ms=0,
                    metadata={"memory_mb": memory_mb}
                )
            except Exception as e:
                return HealthCheckResult(
                    name="memory_usage",
                    status="unhealthy",
                    message=f"Memory check error: {str(e)}",
                    timestamp=datetime.utcnow(),
                    duration_ms=0
                )
        
        self.health.register_check("metrics_collector", check_metrics_collector)
        self.health.register_check("memory_usage", check_memory_usage)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
        except Exception:
            return None


# Global monitoring instance for convenience
_global_monitor: Optional[CleaningMonitor] = None


def get_global_monitor() -> CleaningMonitor:
    """Get or create global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CleaningMonitor()
    return _global_monitor


def setup_monitoring(output_dir: Optional[str] = None) -> CleaningMonitor:
    """Setup monitoring with optional persistent output."""
    monitor = CleaningMonitor()
    
    if output_dir:
        # Setup periodic export (this would typically be done with a scheduler)
        def alert_callback(alert_type: str, alert_data: Dict[str, Any]):
            logger.warning(f"ALERT [{alert_type}]: {alert_data}")
        
        monitor.performance.add_alert_callback(alert_callback)
    
    return monitor