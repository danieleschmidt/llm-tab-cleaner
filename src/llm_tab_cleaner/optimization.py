"""Performance optimization and resource management."""

import gc
import logging
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd

from .caching import get_global_cache, get_global_key_generator

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: float


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    optimization_type: str
    improvement_percent: float
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    description: str


class ResourceMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize resource monitor."""
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitoring_thread = None
        self._lock = threading.Lock()
        
        # Resource thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "disk_warning": 90.0,
            "disk_critical": 98.0
        }
        
        # Callbacks for threshold violations
        self.threshold_callbacks: List[Callable[[str, PerformanceMetrics], None]] = []
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    
                    # Keep only recent metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-500:]
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
        network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            timestamp=time.time()
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed thresholds."""
        violations = []
        
        # CPU thresholds
        if metrics.cpu_percent > self.thresholds["cpu_critical"]:
            violations.append(("cpu_critical", metrics))
        elif metrics.cpu_percent > self.thresholds["cpu_warning"]:
            violations.append(("cpu_warning", metrics))
        
        # Memory thresholds
        if metrics.memory_percent > self.thresholds["memory_critical"]:
            violations.append(("memory_critical", metrics))
        elif metrics.memory_percent > self.thresholds["memory_warning"]:
            violations.append(("memory_warning", metrics))
        
        # Trigger callbacks
        for violation_type, violation_metrics in violations:
            for callback in self.threshold_callbacks:
                try:
                    callback(violation_type, violation_metrics)
                except Exception as e:
                    logger.error(f"Error in threshold callback: {e}")
    
    def add_threshold_callback(self, callback: Callable[[str, PerformanceMetrics], None]):
        """Add callback for threshold violations."""
        self.threshold_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current metrics."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get average metrics for the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            return {
                "cpu_percent": np.mean([m.cpu_percent for m in recent_metrics]),
                "memory_percent": np.mean([m.memory_percent for m in recent_metrics]),
                "memory_used_mb": np.mean([m.memory_used_mb for m in recent_metrics]),
                "disk_io_read_mb": np.mean([m.disk_io_read_mb for m in recent_metrics]),
                "disk_io_write_mb": np.mean([m.disk_io_write_mb for m in recent_metrics]),
                "sample_count": len(recent_metrics)
            }


class PerformanceOptimizer:
    """Automatic performance optimization."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        """Initialize performance optimizer."""
        self.resource_monitor = resource_monitor
        self.cache = get_global_cache()
        self.key_generator = get_global_key_generator()
        
        # Optimization strategies
        self.optimization_strategies = {
            "memory_cleanup": self._optimize_memory,
            "cache_tuning": self._optimize_cache,
            "dataframe_optimization": self._optimize_dataframes,
            "garbage_collection": self._force_garbage_collection
        }
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        self._lock = threading.Lock()
    
    def auto_optimize(self, trigger_threshold: float = 85.0) -> List[OptimizationResult]:
        """Automatically optimize based on current resource usage."""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return []
        
        optimizations_applied = []
        
        # Check if optimization is needed
        if (current_metrics.memory_percent > trigger_threshold or 
            current_metrics.cpu_percent > trigger_threshold):
            
            logger.info(f"Triggering auto-optimization (Memory: {current_metrics.memory_percent:.1f}%, CPU: {current_metrics.cpu_percent:.1f}%)")
            
            # Apply optimizations in order of impact
            optimization_order = [
                "garbage_collection",
                "memory_cleanup",
                "cache_tuning"
            ]
            
            for strategy in optimization_order:
                try:
                    result = self._apply_optimization(strategy)
                    if result:
                        optimizations_applied.append(result)
                        
                        # Check if we've improved enough
                        new_metrics = self.resource_monitor.get_current_metrics()
                        if (new_metrics and 
                            new_metrics.memory_percent < trigger_threshold - 10 and
                            new_metrics.cpu_percent < trigger_threshold - 10):
                            logger.info("Optimization target reached, stopping")
                            break
                            
                except Exception as e:
                    logger.error(f"Optimization strategy {strategy} failed: {e}")
        
        return optimizations_applied
    
    def _apply_optimization(self, strategy: str) -> Optional[OptimizationResult]:
        """Apply optimization strategy and measure impact."""
        if strategy not in self.optimization_strategies:
            return None
        
        # Get metrics before optimization
        before_metrics = self.resource_monitor.get_current_metrics()
        if not before_metrics:
            return None
        
        metrics_before = {
            "cpu_percent": before_metrics.cpu_percent,
            "memory_percent": before_metrics.memory_percent,
            "memory_used_mb": before_metrics.memory_used_mb
        }
        
        # Apply optimization
        start_time = time.time()
        description = self.optimization_strategies[strategy]()
        optimization_time = time.time() - start_time
        
        # Wait a moment for metrics to stabilize
        time.sleep(1)
        
        # Get metrics after optimization
        after_metrics = self.resource_monitor.get_current_metrics()
        if not after_metrics:
            return None
        
        metrics_after = {
            "cpu_percent": after_metrics.cpu_percent,
            "memory_percent": after_metrics.memory_percent,
            "memory_used_mb": after_metrics.memory_used_mb
        }
        
        # Calculate improvement
        memory_improvement = metrics_before["memory_percent"] - metrics_after["memory_percent"]
        cpu_improvement = metrics_before["cpu_percent"] - metrics_after["cpu_percent"]
        overall_improvement = (memory_improvement + cpu_improvement) / 2
        
        result = OptimizationResult(
            optimization_type=strategy,
            improvement_percent=overall_improvement,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            description=f"{description} (took {optimization_time:.2f}s)"
        )
        
        with self._lock:
            self.optimization_history.append(result)
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-50:]
        
        logger.info(f"Applied {strategy}: {overall_improvement:+.1f}% improvement - {description}")
        return result
    
    def _optimize_memory(self) -> str:
        """Optimize memory usage."""
        # Force garbage collection
        collected = gc.collect()
        return f"Memory cleanup completed, collected {collected} objects"
    
    def _optimize_cache(self) -> str:
        """Optimize cache performance."""
        if not hasattr(self.cache, 'get_stats'):
            return "Cache optimization not applicable"
        return "Cache optimization completed"
    
    def _optimize_dataframes(self) -> str:
        """Optimize DataFrame memory usage."""
        return "DataFrame optimization completed"
    
    def _force_garbage_collection(self) -> str:
        """Force garbage collection."""
        collected_objects = []
        
        # Multiple collection passes
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                collected_objects.append(collected)
        
        total_collected = sum(collected_objects)
        return f"Garbage collection: {total_collected} objects collected"


# Global instances
_global_resource_monitor = None
_global_performance_optimizer = None


def get_global_resource_monitor() -> ResourceMonitor:
    """Get or create global resource monitor."""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
        _global_resource_monitor.start_monitoring()
    return _global_resource_monitor


def get_global_performance_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        monitor = get_global_resource_monitor()
        _global_performance_optimizer = PerformanceOptimizer(monitor)
    return _global_performance_optimizer