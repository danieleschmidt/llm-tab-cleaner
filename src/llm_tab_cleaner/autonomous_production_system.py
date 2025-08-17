"""Autonomous Production System - Final Integration and Deployment.

This module integrates all enhanced components into a complete autonomous
production monitoring and management system for LLM data cleaning.

Components Integrated:
- Enhanced adaptive meta-routing with real-time learning
- Autonomous monitoring and self-healing
- Intelligent auto-scaling and global optimization
- ML-driven quality gates with anomaly detection
- Comprehensive observability and analytics

Author: Terry (Terragon Labs)
"""

import logging
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# Import enhanced components
from .adaptive_meta_routing import (
    MetaLearningRouter, DataCharacteristics, DataCharacteristicsExtractor
)
from .autonomous_monitoring import (
    AutonomousMonitor, SystemAlert, AlertSeverity, initialize_monitoring
)
from .self_healing_coordinator import (
    SelfHealingCoordinator, initialize_self_healing
)
from .intelligent_autoscaling import (
    IntelligentAutoScaler, ResourceType, initialize_autoscaling
)
from .ml_quality_gates import (
    MLQualityGateValidator, QualityGateConfig, initialize_quality_gates
)

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system operational state."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class OperationMode(Enum):
    """System operation modes."""
    AUTONOMOUS = "autonomous"
    SEMI_AUTONOMOUS = "semi_autonomous"
    MANUAL = "manual"
    EMERGENCY = "emergency"


@dataclass
class SystemMetrics:
    """Comprehensive system metrics."""
    timestamp: float
    state: SystemState
    operation_mode: OperationMode
    overall_health_score: float
    quality_score: float
    performance_score: float
    cost_efficiency: float
    throughput: float
    latency_p99: float
    error_rate: float
    active_components: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionConfig:
    """Production system configuration."""
    # Core settings
    enable_autonomous_mode: bool = True
    enable_predictive_features: bool = True
    enable_global_optimization: bool = True
    
    # Monitoring settings
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    alert_escalation_threshold: int = 300
    
    # Auto-scaling settings
    scaling_interval: int = 300
    min_workers: int = 2
    max_workers: int = 50
    
    # Quality gates
    minimum_quality_score: float = 0.8
    enable_ml_quality_prediction: bool = True
    
    # Performance thresholds
    max_latency_p99: float = 5000  # ms
    max_error_rate: float = 0.05
    min_throughput: float = 10  # requests/second


class AutonomousProductionSystem:
    """Complete autonomous production system orchestrator."""
    
    def __init__(self, config: ProductionConfig = None):
        """Initialize autonomous production system.
        
        Args:
            config: Production configuration
        """
        self.config = config or ProductionConfig()
        
        # System state
        self.state = SystemState.STARTING
        self.operation_mode = OperationMode.AUTONOMOUS if self.config.enable_autonomous_mode else OperationMode.MANUAL
        self.startup_time = time.time()
        
        # Core components
        self.meta_router: Optional[MetaLearningRouter] = None
        self.monitor: Optional[AutonomousMonitor] = None
        self.healing_coordinator: Optional[SelfHealingCoordinator] = None
        self.autoscaler: Optional[IntelligentAutoScaler] = None
        self.quality_validator: Optional[MLQualityGateValidator] = None
        
        # System metrics and analytics
        self.metrics_history = deque(maxlen=10000)
        self.performance_baselines = {}
        self.sla_metrics = defaultdict(list)
        
        # Control and coordination
        self._system_running = False
        self._orchestrator_thread = None
        self._metrics_thread = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Decision engine
        self.decision_engine = AutonomousDecisionEngine()
        
        logger.info("Initialized AutonomousProductionSystem")
    
    async def start_system(self):
        """Start the complete autonomous production system."""
        logger.info("Starting autonomous production system...")
        
        try:
            # Initialize all components
            await self._initialize_components()
            
            # Start monitoring and coordination
            self._start_system_threads()
            
            # Set system state to healthy
            self.state = SystemState.HEALTHY
            self._system_running = True
            
            logger.info("Autonomous production system started successfully")
            
            # Initial system assessment
            await self._perform_initial_assessment()
            
        except Exception as e:
            logger.error(f"Failed to start autonomous production system: {e}")
            self.state = SystemState.CRITICAL
            raise
    
    async def stop_system(self):
        """Gracefully stop the autonomous production system."""
        logger.info("Stopping autonomous production system...")
        
        self.state = SystemState.SHUTDOWN
        self._system_running = False
        
        try:
            # Stop all threads
            if self._orchestrator_thread:
                self._orchestrator_thread.join(timeout=10)
            if self._metrics_thread:
                self._metrics_thread.join(timeout=10)
            
            # Stop components
            if self.monitor:
                self.monitor.stop_monitoring()
            if self.healing_coordinator:
                self.healing_coordinator.stop_coordination()
            if self.autoscaler:
                self.autoscaler.stop_autoscaling()
            
            # Shutdown executor
            self._executor.shutdown(wait=True, timeout=30)
            
            logger.info("Autonomous production system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    async def _initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        # Initialize monitoring
        self.monitor = initialize_monitoring(
            check_interval=self.config.health_check_interval,
            enable_auto_recovery=True,
            enable_predictive_alerts=self.config.enable_predictive_features,
            start_immediately=True
        )
        
        # Initialize self-healing
        self.healing_coordinator = initialize_self_healing(
            enable_proactive=self.config.enable_predictive_features,
            enable_learning=True,
            start_immediately=True
        )
        
        # Initialize auto-scaling
        self.autoscaler = initialize_autoscaling(
            enable_predictive=self.config.enable_predictive_features,
            enable_global_opt=self.config.enable_global_optimization,
            start_immediately=True
        )
        
        # Initialize quality gates
        quality_config = QualityGateConfig(
            minimum_overall_score=self.config.minimum_quality_score,
            enable_ml_prediction=self.config.enable_ml_quality_prediction,
            enable_anomaly_detection=True
        )
        self.quality_validator = initialize_quality_gates(quality_config)
        
        # Initialize meta-router
        self.meta_router = MetaLearningRouter(
            llm_providers=["anthropic", "openai", "local"],
            enable_real_time_learning=True,
            enable_predictive_scaling=self.config.enable_predictive_features
        )
        
        # Register components for coordinated healing
        self._register_system_components()
        
        logger.info("All system components initialized successfully")
    
    def _register_system_components(self):
        """Register components for coordinated monitoring and healing."""
        if self.healing_coordinator:
            # Register core components
            self.healing_coordinator.register_component(
                "meta_router",
                dependencies={"monitoring", "quality_gates"}
            )
            self.healing_coordinator.register_component(
                "autoscaler",
                dependencies={"monitoring"}
            )
            self.healing_coordinator.register_component(
                "quality_validator",
                dependencies={"monitoring"}
            )
            self.healing_coordinator.register_component(
                "monitoring_system",
                dependencies=set()
            )
    
    def _start_system_threads(self):
        """Start system orchestration and metrics threads."""
        self._orchestrator_thread = threading.Thread(
            target=self._orchestration_loop, 
            daemon=True
        )
        self._orchestrator_thread.start()
        
        self._metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            daemon=True
        )
        self._metrics_thread.start()
    
    def _orchestration_loop(self):
        """Main orchestration loop for autonomous coordination."""
        logger.info("Started system orchestration loop")
        
        while self._system_running:
            try:
                # Collect system state
                current_metrics = self._collect_system_metrics()
                
                # Update system state based on metrics
                self._update_system_state(current_metrics)
                
                # Make autonomous decisions
                if self.operation_mode == OperationMode.AUTONOMOUS:
                    decisions = self.decision_engine.make_decisions(current_metrics, self)
                    self._execute_autonomous_decisions(decisions)
                
                # Check for escalation conditions
                self._check_escalation_conditions(current_metrics)
                
                time.sleep(30)  # Orchestration interval
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                time.sleep(30)
    
    def _metrics_collection_loop(self):
        """Continuous metrics collection loop."""
        logger.info("Started metrics collection loop")
        
        while self._system_running:
            try:
                # Collect comprehensive metrics
                metrics = self._collect_comprehensive_metrics()
                self.metrics_history.append(metrics)
                
                # Update SLA metrics
                self._update_sla_metrics(metrics)
                
                # Performance baseline updates
                self._update_performance_baselines(metrics)
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.config.metrics_collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # Get component health
        health_score = 1.0
        if self.monitor:
            system_health = self.monitor.get_system_health()
            health_score = self._calculate_health_score(system_health)
        
        # Get quality metrics
        quality_score = 0.8  # Default
        if self.quality_validator:
            analytics = self.quality_validator.get_validation_analytics()
            quality_score = analytics.get('average_score', 0.8)
        
        # Get performance metrics
        performance_score = self._calculate_performance_score()
        
        # Get cost efficiency
        cost_efficiency = self._calculate_cost_efficiency()
        
        # System throughput and latency (simplified)
        throughput = self._estimate_throughput()
        latency_p99 = self._estimate_latency_p99()
        error_rate = self._estimate_error_rate()
        
        return SystemMetrics(
            timestamp=time.time(),
            state=self.state,
            operation_mode=self.operation_mode,
            overall_health_score=health_score,
            quality_score=quality_score,
            performance_score=performance_score,
            cost_efficiency=cost_efficiency,
            throughput=throughput,
            latency_p99=latency_p99,
            error_rate=error_rate,
            active_components=self._count_active_components()
        )
    
    def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            'timestamp': time.time(),
            'system_state': self.state.value,
            'operation_mode': self.operation_mode.value,
            'uptime': time.time() - self.startup_time
        }
        
        # Component-specific metrics
        if self.monitor:
            metrics['monitoring'] = self.monitor.get_analytics()
        
        if self.healing_coordinator:
            metrics['healing'] = self.healing_coordinator.get_coordination_status()
        
        if self.autoscaler:
            metrics['autoscaling'] = self.autoscaler.get_scaling_status()
            metrics['optimization'] = self.autoscaler.get_optimization_analytics()
        
        if self.quality_validator:
            metrics['quality'] = self.quality_validator.get_validation_analytics()
        
        if self.meta_router:
            metrics['routing'] = self.meta_router.get_enhanced_analytics()
        
        return metrics
    
    def _calculate_health_score(self, system_health: Dict[str, Any]) -> float:
        """Calculate overall health score from monitoring data."""
        if not system_health or 'metrics' not in system_health:
            return 0.5
        
        health_scores = []
        for metric_name, metric_data in system_health['metrics'].items():
            if metric_data['status'] == 'healthy':
                health_scores.append(1.0)
            elif metric_data['status'] == 'warning':
                health_scores.append(0.7)
            elif metric_data['status'] == 'critical':
                health_scores.append(0.3)
            else:
                health_scores.append(0.5)
        
        return np.mean(health_scores) if health_scores else 0.5
    
    def _calculate_performance_score(self) -> float:
        """Calculate performance score based on various factors."""
        score_factors = []
        
        # Router performance
        if self.meta_router:
            routing_analytics = self.meta_router.get_enhanced_analytics()
            if 'avg_performance' in routing_analytics:
                score_factors.append(routing_analytics['avg_performance'])
        
        # Auto-scaler efficiency
        if self.autoscaler:
            scaling_status = self.autoscaler.get_scaling_status()
            resource_efficiency = np.mean([
                metrics['efficiency'] for metrics in 
                scaling_status.get('resource_metrics', {}).values()
            ]) if scaling_status.get('resource_metrics') else 0.8
            score_factors.append(resource_efficiency)
        
        return np.mean(score_factors) if score_factors else 0.8
    
    def _calculate_cost_efficiency(self) -> float:
        """Calculate cost efficiency score."""
        if self.autoscaler:
            status = self.autoscaler.get_scaling_status()
            current_cost = status.get('estimated_current_cost', 100)
            
            # Simple efficiency calculation
            # In practice, would compare against baseline or target costs
            efficiency = max(0.3, min(1.0, 100 / max(current_cost, 1)))
            return efficiency
        
        return 0.8  # Default
    
    def _estimate_throughput(self) -> float:
        """Estimate system throughput."""
        # Simplified throughput estimation
        base_throughput = 50.0  # requests/second
        
        if self.meta_router and hasattr(self.meta_router, 'routing_history'):
            recent_requests = len([
                r for r in self.meta_router.routing_history 
                if time.time() - r['timestamp'] < 3600
            ])
            if recent_requests > 0:
                base_throughput = recent_requests / 3600  # Convert to per-second
        
        return base_throughput
    
    def _estimate_latency_p99(self) -> float:
        """Estimate 99th percentile latency."""
        # Simplified latency estimation
        if self.meta_router and hasattr(self.meta_router, 'routing_history'):
            processing_times = [
                r.get('processing_time', 1.0) * 1000  # Convert to ms
                for r in self.meta_router.routing_history[-100:]
            ]
            if processing_times:
                return np.percentile(processing_times, 99)
        
        return 2000.0  # Default 2 seconds
    
    def _estimate_error_rate(self) -> float:
        """Estimate system error rate."""
        # Simplified error rate estimation
        if self.monitor:
            health = self.monitor.get_system_health()
            critical_alerts = health.get('critical_alerts', 0)
            total_operations = max(1, health.get('total_operations', 100))
            return min(0.1, critical_alerts / total_operations)
        
        return 0.01  # Default 1% error rate
    
    def _count_active_components(self) -> int:
        """Count active system components."""
        count = 0
        
        if self.monitor and self.monitor.is_running:
            count += 1
        if self.healing_coordinator and hasattr(self.healing_coordinator, '_coordinator_running') and self.healing_coordinator._coordinator_running:
            count += 1
        if self.autoscaler and hasattr(self.autoscaler, '_scaler_running') and self.autoscaler._scaler_running:
            count += 1
        if self.quality_validator:
            count += 1
        if self.meta_router:
            count += 1
        
        return count
    
    def _update_system_state(self, metrics: SystemMetrics):
        """Update system state based on current metrics."""
        previous_state = self.state
        
        # State transition logic
        if metrics.overall_health_score < 0.3 or metrics.error_rate > 0.1:
            self.state = SystemState.CRITICAL
        elif metrics.overall_health_score < 0.6 or metrics.performance_score < 0.5:
            self.state = SystemState.DEGRADED
        elif metrics.overall_health_score > 0.8 and metrics.performance_score > 0.7:
            self.state = SystemState.HEALTHY
        
        # Log state changes
        if self.state != previous_state:
            logger.warning(f"System state changed: {previous_state.value} -> {self.state.value}")
            
            # Create alert for state change
            if self.monitor:
                self.monitor.create_manual_alert(
                    AlertSeverity.WARNING if self.state in [SystemState.DEGRADED, SystemState.HEALTHY] else AlertSeverity.CRITICAL,
                    "system_orchestrator",
                    f"System state changed to {self.state.value}",
                    ["Review system metrics", "Check component health"]
                )
    
    def _update_sla_metrics(self, metrics: Dict[str, Any]):
        """Update SLA-related metrics."""
        # Track key SLA metrics
        self.sla_metrics['availability'].append(1.0 if self.state in [SystemState.HEALTHY, SystemState.DEGRADED] else 0.0)
        
        if 'throughput' in metrics:
            self.sla_metrics['throughput'].append(metrics['throughput'])
        
        if 'latency_p99' in metrics:
            self.sla_metrics['latency'].append(metrics['latency_p99'])
        
        if 'error_rate' in metrics:
            self.sla_metrics['error_rate'].append(metrics['error_rate'])
        
        # Keep only recent metrics (last 24 hours worth)
        max_samples = 24 * 60  # 24 hours * 60 minutes
        for metric_list in self.sla_metrics.values():
            if len(metric_list) > max_samples:
                metric_list[:] = metric_list[-max_samples:]
    
    def _update_performance_baselines(self, metrics: Dict[str, Any]):
        """Update performance baselines for anomaly detection."""
        # Update rolling baselines
        for key in ['throughput', 'latency_p99', 'error_rate']:
            if key in metrics:
                if key not in self.performance_baselines:
                    self.performance_baselines[key] = deque(maxlen=1000)
                self.performance_baselines[key].append(metrics[key])
    
    def _check_escalation_conditions(self, metrics: SystemMetrics):
        """Check for conditions requiring escalation."""
        escalation_conditions = [
            metrics.overall_health_score < 0.2,
            metrics.error_rate > self.config.max_error_rate * 2,
            metrics.latency_p99 > self.config.max_latency_p99 * 2,
            self.state == SystemState.CRITICAL
        ]
        
        if any(escalation_conditions):
            logger.critical("Escalation conditions detected - switching to emergency mode")
            self.operation_mode = OperationMode.EMERGENCY
            
            # Trigger emergency protocols
            self._trigger_emergency_protocols(metrics)
    
    def _trigger_emergency_protocols(self, metrics: SystemMetrics):
        """Trigger emergency response protocols."""
        logger.critical("Activating emergency response protocols")
        
        try:
            # Emergency healing
            if self.healing_coordinator:
                for component in ["meta_router", "autoscaler", "quality_validator"]:
                    self.healing_coordinator.trigger_emergency_healing(component, "critical")
            
            # Emergency scaling
            if self.autoscaler:
                # Force scale up critical resources
                self.autoscaler.update_resource_metrics(
                    ResourceType.WORKERS, 
                    95.0,  # High utilization to trigger scaling
                    100.0, 
                    cost_per_unit=20.0
                )
            
            # Create critical alert
            if self.monitor:
                self.monitor.create_manual_alert(
                    AlertSeverity.CRITICAL,
                    "emergency_system",
                    f"Emergency protocols activated - System health: {metrics.overall_health_score:.2f}",
                    [
                        "Investigate critical system issues immediately",
                        "Check all component logs",
                        "Consider manual intervention"
                    ]
                )
            
        except Exception as e:
            logger.error(f"Error in emergency protocols: {e}")
    
    def _execute_autonomous_decisions(self, decisions: List[Dict[str, Any]]):
        """Execute autonomous decisions from the decision engine."""
        for decision in decisions:
            try:
                decision_type = decision.get('type')
                
                if decision_type == 'scale_resources':
                    self._execute_scaling_decision(decision)
                elif decision_type == 'optimize_routing':
                    self._execute_routing_optimization(decision)
                elif decision_type == 'adjust_quality_thresholds':
                    self._execute_quality_adjustment(decision)
                elif decision_type == 'trigger_healing':
                    self._execute_healing_decision(decision)
                
                logger.info(f"Executed autonomous decision: {decision_type}")
                
            except Exception as e:
                logger.error(f"Error executing decision {decision}: {e}")
    
    def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute auto-scaling decision."""
        if self.autoscaler:
            resource_type = ResourceType(decision.get('resource_type', 'workers'))
            target_utilization = decision.get('target_utilization', 75.0)
            
            # Update resource metrics to trigger scaling
            self.autoscaler.update_resource_metrics(
                resource_type,
                target_utilization + 10,  # Above target to trigger scaling
                100.0,
                cost_per_unit=decision.get('cost_per_unit', 10.0)
            )
    
    def _execute_routing_optimization(self, decision: Dict[str, Any]):
        """Execute routing optimization decision."""
        if self.meta_router:
            # Update routing configuration based on decision
            optimization_type = decision.get('optimization_type', 'cost')
            
            if optimization_type == 'cost':
                self.meta_router.enable_cost_optimization = True
            elif optimization_type == 'performance':
                self.meta_router.confidence_threshold = decision.get('confidence_threshold', 0.6)
    
    def _execute_quality_adjustment(self, decision: Dict[str, Any]):
        """Execute quality threshold adjustment."""
        if self.quality_validator:
            new_threshold = decision.get('minimum_score', 0.7)
            self.quality_validator.config.minimum_overall_score = new_threshold
    
    def _execute_healing_decision(self, decision: Dict[str, Any]):
        """Execute healing decision."""
        if self.healing_coordinator:
            component = decision.get('component', 'unknown')
            severity = decision.get('severity', 'medium')
            self.healing_coordinator.trigger_emergency_healing(component, severity)
    
    async def _perform_initial_assessment(self):
        """Perform initial system assessment after startup."""
        logger.info("Performing initial system assessment...")
        
        # Wait for components to stabilize
        await asyncio.sleep(30)
        
        # Collect initial metrics
        initial_metrics = self._collect_system_metrics()
        
        # Set initial baselines
        self.performance_baselines['initial_health'] = initial_metrics.overall_health_score
        self.performance_baselines['initial_performance'] = initial_metrics.performance_score
        
        logger.info(f"Initial assessment complete - Health: {initial_metrics.overall_health_score:.2f}, "
                   f"Performance: {initial_metrics.performance_score:.2f}")
    
    # Public interface methods
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.metrics_history:
            return {'status': 'starting', 'components': 0}
        
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        status = {
            'system_state': self.state.value,
            'operation_mode': self.operation_mode.value,
            'uptime': time.time() - self.startup_time,
            'active_components': self._count_active_components(),
            'latest_metrics': latest_metrics.__dict__ if latest_metrics else {},
            'sla_status': self._get_sla_status()
        }
        
        # Add component statuses
        if self.monitor:
            status['monitoring'] = {'running': self.monitor.is_running}
        if self.healing_coordinator:
            status['healing'] = self.healing_coordinator.get_coordination_status()
        if self.autoscaler:
            status['autoscaling'] = self.autoscaler.get_scaling_status()
        
        return status
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data."""
        if not self.metrics_history:
            return {}
        
        # Convert metrics to DataFrame for analysis
        metrics_data = []
        for metric in list(self.metrics_history)[-100:]:  # Last 100 metrics
            metrics_data.append({
                'timestamp': metric.timestamp,
                'health_score': metric.overall_health_score,
                'quality_score': metric.quality_score,
                'performance_score': metric.performance_score,
                'cost_efficiency': metric.cost_efficiency,
                'throughput': metric.throughput,
                'latency_p99': metric.latency_p99,
                'error_rate': metric.error_rate
            })
        
        if not metrics_data:
            return {}
        
        df = pd.DataFrame(metrics_data)
        
        return {
            'time_series': {
                'timestamps': df['timestamp'].tolist(),
                'health_scores': df['health_score'].tolist(),
                'quality_scores': df['quality_score'].tolist(),
                'performance_scores': df['performance_score'].tolist(),
                'throughput': df['throughput'].tolist(),
                'latency_p99': df['latency_p99'].tolist()
            },
            'summary_stats': {
                'avg_health_score': df['health_score'].mean(),
                'avg_quality_score': df['quality_score'].mean(),
                'avg_performance_score': df['performance_score'].mean(),
                'avg_cost_efficiency': df['cost_efficiency'].mean(),
                'p99_latency': df['latency_p99'].quantile(0.99),
                'avg_throughput': df['throughput'].mean(),
                'avg_error_rate': df['error_rate'].mean()
            },
            'sla_compliance': self._get_sla_status(),
            'trends': {
                'health_trend': 'improving' if df['health_score'].iloc[-1] > df['health_score'].iloc[0] else 'declining',
                'performance_trend': 'improving' if df['performance_score'].iloc[-1] > df['performance_score'].iloc[0] else 'declining'
            }
        }
    
    def _get_sla_status(self) -> Dict[str, Any]:
        """Get SLA compliance status."""
        sla_status = {}
        
        # Availability (uptime)
        if 'availability' in self.sla_metrics and self.sla_metrics['availability']:
            availability = np.mean(self.sla_metrics['availability'][-1440:])  # Last 24 hours
            sla_status['availability'] = {
                'current': availability,
                'target': 0.999,  # 99.9% uptime SLA
                'compliant': availability >= 0.999
            }
        
        # Latency
        if 'latency' in self.sla_metrics and self.sla_metrics['latency']:
            p95_latency = np.percentile(self.sla_metrics['latency'][-1440:], 95)
            sla_status['latency_p95'] = {
                'current': p95_latency,
                'target': self.config.max_latency_p99,
                'compliant': p95_latency <= self.config.max_latency_p99
            }
        
        # Throughput
        if 'throughput' in self.sla_metrics and self.sla_metrics['throughput']:
            avg_throughput = np.mean(self.sla_metrics['throughput'][-1440:])
            sla_status['throughput'] = {
                'current': avg_throughput,
                'target': self.config.min_throughput,
                'compliant': avg_throughput >= self.config.min_throughput
            }
        
        # Error rate
        if 'error_rate' in self.sla_metrics and self.sla_metrics['error_rate']:
            avg_error_rate = np.mean(self.sla_metrics['error_rate'][-1440:])
            sla_status['error_rate'] = {
                'current': avg_error_rate,
                'target': self.config.max_error_rate,
                'compliant': avg_error_rate <= self.config.max_error_rate
            }
        
        return sla_status
    
    def force_operation_mode(self, mode: OperationMode):
        """Force system into specific operation mode."""
        previous_mode = self.operation_mode
        self.operation_mode = mode
        
        logger.warning(f"Operation mode changed: {previous_mode.value} -> {mode.value}")
        
        if self.monitor:
            self.monitor.create_manual_alert(
                AlertSeverity.WARNING,
                "system_orchestrator",
                f"Operation mode changed to {mode.value}",
                ["Review system configuration", "Monitor autonomous behavior"]
            )


class AutonomousDecisionEngine:
    """Decision engine for autonomous system management."""
    
    def __init__(self):
        """Initialize decision engine."""
        self.decision_history = deque(maxlen=1000)
        
    def make_decisions(
        self, 
        metrics: SystemMetrics, 
        system: AutonomousProductionSystem
    ) -> List[Dict[str, Any]]:
        """Make autonomous decisions based on current system state."""
        decisions = []
        
        # Health-based decisions
        if metrics.overall_health_score < 0.6:
            decisions.append({
                'type': 'trigger_healing',
                'component': 'all',
                'severity': 'high',
                'reason': f'Low health score: {metrics.overall_health_score:.2f}'
            })
        
        # Performance-based decisions
        if metrics.performance_score < 0.5:
            decisions.append({
                'type': 'scale_resources',
                'resource_type': 'workers',
                'target_utilization': 60.0,
                'reason': f'Low performance score: {metrics.performance_score:.2f}'
            })
        
        # Cost efficiency decisions
        if metrics.cost_efficiency < 0.4:
            decisions.append({
                'type': 'optimize_routing',
                'optimization_type': 'cost',
                'reason': f'Low cost efficiency: {metrics.cost_efficiency:.2f}'
            })
        
        # Quality-based decisions
        if metrics.quality_score < 0.7:
            decisions.append({
                'type': 'adjust_quality_thresholds',
                'minimum_score': 0.6,
                'reason': f'Low quality score: {metrics.quality_score:.2f}'
            })
        
        # Latency-based decisions
        if metrics.latency_p99 > 8000:  # 8 seconds
            decisions.append({
                'type': 'scale_resources',
                'resource_type': 'workers',
                'target_utilization': 50.0,
                'reason': f'High latency: {metrics.latency_p99:.0f}ms'
            })
        
        # Record decisions
        for decision in decisions:
            self.decision_history.append({
                'timestamp': time.time(),
                'decision': decision,
                'metrics_snapshot': metrics
            })
        
        return decisions


# Global system instance
_global_system: Optional[AutonomousProductionSystem] = None


def get_global_system() -> AutonomousProductionSystem:
    """Get global autonomous production system instance."""
    global _global_system
    if _global_system is None:
        _global_system = AutonomousProductionSystem()
    return _global_system


async def initialize_production_system(config: ProductionConfig = None) -> AutonomousProductionSystem:
    """Initialize and start the autonomous production system."""
    global _global_system
    
    _global_system = AutonomousProductionSystem(config)
    await _global_system.start_system()
    
    logger.info("Autonomous production system initialized and started")
    return _global_system


if __name__ == "__main__":
    async def main():
        # Demo autonomous production system
        config = ProductionConfig(
            enable_autonomous_mode=True,
            enable_predictive_features=True,
            health_check_interval=15,
            scaling_interval=60
        )
        
        system = await initialize_production_system(config)
        
        try:
            # Run for demonstration
            await asyncio.sleep(120)
            
            # Print system status
            status = system.get_system_status()
            print("System Status:")
            print(json.dumps(status, indent=2, default=str))
            
            # Print analytics
            analytics = system.get_analytics_dashboard()
            print("\nAnalytics Dashboard:")
            print(json.dumps(analytics, indent=2, default=str))
            
        finally:
            await system.stop_system()
    
    # Run async main
    asyncio.run(main())