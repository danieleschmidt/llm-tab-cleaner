"""Health monitoring and system diagnostics."""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: str  # "healthy", "unhealthy", "warning"
    message: str
    timestamp: float
    duration: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str
    checks: List[HealthCheck]
    timestamp: float
    uptime: float
    version: str


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.is_running = False
        self.start_time = time.time()
        self._lock = threading.RLock()
        
        # Register default health checks
        self._register_default_checks()
    
    def register_check(self, name: str, check_func: Callable[[], tuple[str, str, Dict]]):
        """Register a health check.
        
        Args:
            name: Unique name for the check
            check_func: Function returning (status, message, metadata)
        """
        with self._lock:
            self.health_checks[name] = check_func
            logger.debug(f"Registered health check: {name}")
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        def memory_check():
            """Check system memory usage."""
            try:
                memory = psutil.virtual_memory()
                usage_percent = memory.percent
                
                if usage_percent > 90:
                    status = "unhealthy"
                    message = f"High memory usage: {usage_percent:.1f}%"
                elif usage_percent > 75:
                    status = "warning"
                    message = f"Elevated memory usage: {usage_percent:.1f}%"
                else:
                    status = "healthy"
                    message = f"Memory usage: {usage_percent:.1f}%"
                
                metadata = {
                    "usage_percent": usage_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
                
                return status, message, metadata
                
            except Exception as e:
                return "unhealthy", f"Memory check failed: {e}", {}
        
        def cpu_check():
            """Check CPU usage."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100 * cpu_count
                
                if cpu_percent > 90:
                    status = "unhealthy"
                    message = f"High CPU usage: {cpu_percent:.1f}%"
                elif cpu_percent > 75:
                    status = "warning"
                    message = f"Elevated CPU usage: {cpu_percent:.1f}%"
                else:
                    status = "healthy"
                    message = f"CPU usage: {cpu_percent:.1f}%"
                
                metadata = {
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_average": load_avg
                }
                
                return status, message, metadata
                
            except Exception as e:
                return "unhealthy", f"CPU check failed: {e}", {}
        
        def disk_check():
            """Check disk usage."""
            try:
                disk = psutil.disk_usage('/')
                usage_percent = (disk.used / disk.total) * 100
                
                if usage_percent > 90:
                    status = "unhealthy"
                    message = f"High disk usage: {usage_percent:.1f}%"
                elif usage_percent > 80:
                    status = "warning"
                    message = f"Elevated disk usage: {usage_percent:.1f}%"
                else:
                    status = "healthy"
                    message = f"Disk usage: {usage_percent:.1f}%"
                
                metadata = {
                    "usage_percent": usage_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
                
                return status, message, metadata
                
            except Exception as e:
                return "unhealthy", f"Disk check failed: {e}", {}
        
        # Register checks
        self.register_check("memory_usage", memory_check)
        self.register_check("cpu_usage", cpu_check)
        self.register_check("disk_usage", disk_check)
    
    async def run_checks(self) -> SystemHealth:
        """Run all health checks."""
        check_results = []
        
        with ThreadPoolExecutor(max_workers=len(self.health_checks)) as executor:
            # Submit all checks
            futures = {}
            for name, check_func in self.health_checks.items():
                future = executor.submit(self._run_single_check, name, check_func)
                futures[future] = name
            
            # Collect results
            for future in futures:
                check_result = await asyncio.wrap_future(future)
                check_results.append(check_result)
                
                # Cache result
                with self._lock:
                    self.last_results[check_result.name] = check_result
        
        # Determine overall status
        overall_status = "healthy"
        for check in check_results:
            if check.status == "unhealthy":
                overall_status = "unhealthy"
                break
            elif check.status == "warning" and overall_status == "healthy":
                overall_status = "warning"
        
        return SystemHealth(
            status=overall_status,
            checks=check_results,
            timestamp=time.time(),
            uptime=time.time() - self.start_time,
            version="1.0"
        )
    
    def _run_single_check(self, name: str, check_func: Callable) -> HealthCheck:
        """Run a single health check with timing."""
        start_time = time.time()
        
        try:
            status, message, metadata = check_func()
            duration = time.time() - start_time
            
            return HealthCheck(
                name=name,
                status=status,
                message=message,
                timestamp=time.time(),
                duration=duration,
                metadata=metadata
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health check {name} failed: {e}")
            
            return HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Check failed: {e}",
                timestamp=time.time(),
                duration=duration,
                metadata={"error": str(e)}
            )
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info(f"Starting health monitoring with {self.check_interval}s interval")
        
        try:
            while self.is_running:
                health_status = await self.run_checks()
                
                # Log health status
                if health_status.status == "unhealthy":
                    logger.error(f"System unhealthy: {len([c for c in health_status.checks if c.status == 'unhealthy'])} failed checks")
                elif health_status.status == "warning":
                    logger.warning(f"System warnings: {len([c for c in health_status.checks if c.status == 'warning'])} warnings")
                else:
                    logger.debug("System healthy")
                
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")
        finally:
            self.is_running = False
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        logger.info("Health monitoring stopped")
    
    def get_cached_health(self) -> Optional[SystemHealth]:
        """Get last cached health status."""
        with self._lock:
            if not self.last_results:
                return None
            
            # Determine overall status from cached results
            overall_status = "healthy"
            for check in self.last_results.values():
                if check.status == "unhealthy":
                    overall_status = "unhealthy"
                    break
                elif check.status == "warning" and overall_status == "healthy":
                    overall_status = "warning"
            
            return SystemHealth(
                status=overall_status,
                checks=list(self.last_results.values()),
                timestamp=time.time(),
                uptime=time.time() - self.start_time,
                version="1.0"
            )


class AlertManager:
    """Manages health alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_handlers: List[Callable[[HealthCheck], None]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "memory_usage": 85.0,
            "cpu_usage": 80.0,
            "disk_usage": 85.0
        }
        self._lock = threading.Lock()
    
    def add_handler(self, handler: Callable[[HealthCheck], None]):
        """Add an alert handler function."""
        with self._lock:
            self.alert_handlers.append(handler)
    
    def process_health_check(self, health_check: HealthCheck):
        """Process a health check and trigger alerts if needed."""
        should_alert = False
        alert_level = "info"
        
        # Check if alert conditions are met
        if health_check.status == "unhealthy":
            should_alert = True
            alert_level = "critical"
        elif health_check.status == "warning":
            # Check specific thresholds
            if health_check.metadata:
                if health_check.name == "memory_usage":
                    if health_check.metadata.get("usage_percent", 0) > self.alert_thresholds["memory_usage"]:
                        should_alert = True
                        alert_level = "warning"
                elif health_check.name == "cpu_usage":
                    if health_check.metadata.get("cpu_percent", 0) > self.alert_thresholds["cpu_usage"]:
                        should_alert = True
                        alert_level = "warning"
                elif health_check.name == "disk_usage":
                    if health_check.metadata.get("usage_percent", 0) > self.alert_thresholds["disk_usage"]:
                        should_alert = True
                        alert_level = "warning"
        
        if should_alert:
            self._trigger_alert(health_check, alert_level)
    
    def _trigger_alert(self, health_check: HealthCheck, level: str):
        """Trigger an alert for a health check."""
        alert_data = {
            "timestamp": time.time(),
            "level": level,
            "check_name": health_check.name,
            "status": health_check.status,
            "message": health_check.message,
            "metadata": health_check.metadata
        }
        
        # Store in history
        with self._lock:
            self.alert_history.append(alert_data)
            
            # Keep only recent alerts
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
            
            # Call all handlers
            for handler in self.alert_handlers:
                try:
                    handler(health_check)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"ALERT [{level}] {health_check.name}: {health_check.message}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts within specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [
                alert for alert in self.alert_history
                if alert["timestamp"] > cutoff_time
            ]


# Global health monitor instance
_global_health_monitor = None
_global_alert_manager = None


def get_global_health_monitor() -> HealthMonitor:
    """Get or create global health monitor."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def get_global_alert_manager() -> AlertManager:
    """Get or create global alert manager."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
        
        # Add default console alert handler
        def console_alert_handler(health_check: HealthCheck):
            """Default console alert handler."""
            print(f"🚨 ALERT: {health_check.name} - {health_check.message}")
        
        _global_alert_manager.add_handler(console_alert_handler)
    
    return _global_alert_manager