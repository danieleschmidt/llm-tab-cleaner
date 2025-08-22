"""Chaos Engineering Module - Generation 2 Robustness Testing.

This module implements chaos engineering capabilities to proactively test system
resilience and identify failure modes before they occur in production.

Features:
- Automated failure injection
- Resilience testing scenarios
- Recovery validation
- System stability analysis
- Intelligent chaos scheduling

Author: Terry (Terragon Labs)
"""

import asyncio
import logging
import random
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class ChaosExperimentType(Enum):
    """Types of chaos experiments."""
    NETWORK_LATENCY = "network_latency"
    SERVICE_FAILURE = "service_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    DEPENDENCY_FAILURE = "dependency_failure"


class ExperimentStatus(Enum):
    """Status of chaos experiments."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""
    name: str
    experiment_type: ChaosExperimentType
    target_component: str
    duration: int  # seconds
    intensity: float  # 0.0 to 1.0
    conditions: Dict[str, Any] = field(default_factory=dict)
    expected_recovery_time: Optional[int] = None
    abort_conditions: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    system_impact: Dict[str, Any] = field(default_factory=dict)
    recovery_time: Optional[float] = None
    lessons_learned: List[str] = field(default_factory=list)
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)


class ChaosInjector:
    """Implements various types of chaos injection."""
    
    def __init__(self):
        self.active_injections = {}
        self.injection_history = []
    
    async def inject_network_latency(
        self, 
        target: str, 
        latency_ms: int, 
        duration: int
    ):
        """Inject network latency for a target component."""
        logger.warning(f"Injecting {latency_ms}ms latency to {target} for {duration}s")
        
        injection_id = f"latency_{target}_{time.time()}"
        self.active_injections[injection_id] = {
            'type': 'network_latency',
            'target': target,
            'latency_ms': latency_ms,
            'start_time': time.time()
        }
        
        # Simulate latency injection (in real implementation, would modify network stack)
        await asyncio.sleep(duration)
        
        # Remove injection
        del self.active_injections[injection_id]
        logger.info(f"Removed latency injection for {target}")
    
    async def inject_service_failure(
        self, 
        target: str, 
        failure_rate: float, 
        duration: int
    ):
        """Inject service failures for a target component."""
        logger.warning(f"Injecting {failure_rate*100}% failure rate to {target} for {duration}s")
        
        injection_id = f"failure_{target}_{time.time()}"
        self.active_injections[injection_id] = {
            'type': 'service_failure',
            'target': target,
            'failure_rate': failure_rate,
            'start_time': time.time()
        }
        
        # Simulate service failure injection
        await asyncio.sleep(duration)
        
        # Remove injection
        del self.active_injections[injection_id]
        logger.info(f"Removed service failure injection for {target}")
    
    async def inject_resource_exhaustion(
        self, 
        target: str, 
        resource_type: str, 
        utilization: float, 
        duration: int
    ):
        """Inject resource exhaustion for a target component."""
        logger.warning(f"Injecting {utilization*100}% {resource_type} utilization to {target} for {duration}s")
        
        injection_id = f"resource_{target}_{time.time()}"
        self.active_injections[injection_id] = {
            'type': 'resource_exhaustion',
            'target': target,
            'resource_type': resource_type,
            'utilization': utilization,
            'start_time': time.time()
        }
        
        # Simulate resource exhaustion
        await asyncio.sleep(duration)
        
        # Remove injection
        del self.active_injections[injection_id]
        logger.info(f"Removed resource exhaustion injection for {target}")
    
    def get_active_injections(self) -> Dict[str, Any]:
        """Get currently active chaos injections."""
        return self.active_injections.copy()
    
    def abort_all_injections(self):
        """Abort all active chaos injections."""
        logger.warning("Aborting all chaos injections")
        self.active_injections.clear()


class ResilienceValidator:
    """Validates system resilience during chaos experiments."""
    
    def __init__(self):
        self.validation_metrics = {}
        self.recovery_patterns = {}
    
    async def validate_system_resilience(
        self, 
        experiment: ChaosExperiment,
        monitoring_callback: Callable = None
    ) -> Dict[str, Any]:
        """Validate system resilience during experiment."""
        
        validation_results = {
            'resilience_score': 0.0,
            'recovery_time': None,
            'impact_severity': 'unknown',
            'critical_failures': [],
            'recovery_successful': False
        }
        
        # Monitor system during experiment
        start_time = time.time()
        max_impact = 0.0
        recovery_detected = False
        
        # Simulate monitoring (in real implementation, would connect to actual metrics)
        monitoring_duration = experiment.duration + 60  # Monitor 60s after experiment
        
        for i in range(monitoring_duration):
            # Simulate getting system metrics
            current_metrics = await self._get_simulated_metrics(experiment, i)
            
            # Calculate impact
            impact = self._calculate_impact(current_metrics, experiment.target_component)
            max_impact = max(max_impact, impact)
            
            # Check for recovery
            if i > experiment.duration and impact < 0.1 and not recovery_detected:
                validation_results['recovery_time'] = i - experiment.duration
                recovery_detected = True
                validation_results['recovery_successful'] = True
                logger.info(f"Recovery detected after {validation_results['recovery_time']}s")
            
            # Check for critical failures
            if impact > 0.8:
                validation_results['critical_failures'].append({
                    'timestamp': start_time + i,
                    'impact': impact,
                    'metrics': current_metrics
                })
            
            if monitoring_callback:
                await monitoring_callback(current_metrics, impact)
            
            await asyncio.sleep(1)
        
        # Calculate final resilience score
        validation_results['resilience_score'] = self._calculate_resilience_score(
            max_impact, validation_results['recovery_time'], len(validation_results['critical_failures'])
        )
        
        # Determine impact severity
        if max_impact < 0.2:
            validation_results['impact_severity'] = 'low'
        elif max_impact < 0.5:
            validation_results['impact_severity'] = 'medium'
        elif max_impact < 0.8:
            validation_results['impact_severity'] = 'high'
        else:
            validation_results['impact_severity'] = 'critical'
        
        return validation_results
    
    async def _get_simulated_metrics(self, experiment: ChaosExperiment, elapsed_time: int) -> Dict[str, float]:
        """Get simulated system metrics during experiment."""
        # Simulate impact based on experiment type and time
        base_impact = 0.0
        
        if elapsed_time <= experiment.duration:
            # During experiment - simulate impact
            progress = elapsed_time / experiment.duration
            
            if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
                base_impact = experiment.intensity * 0.6  # Network issues have moderate impact
            elif experiment.experiment_type == ChaosExperimentType.SERVICE_FAILURE:
                base_impact = experiment.intensity * 0.9  # Service failures have high impact
            elif experiment.experiment_type == ChaosExperimentType.RESOURCE_EXHAUSTION:
                base_impact = experiment.intensity * 0.8  # Resource issues have high impact
            
            # Add some randomness
            base_impact += random.uniform(-0.1, 0.1)
        else:
            # After experiment - simulate recovery
            recovery_time = elapsed_time - experiment.duration
            expected_recovery = experiment.expected_recovery_time or 30
            
            # Exponential recovery
            base_impact = max(0, 0.3 * np.exp(-recovery_time / expected_recovery))
        
        return {
            'error_rate': max(0, min(1, base_impact)),
            'response_time': 100 + (base_impact * 500),  # 100ms base + up to 500ms impact
            'throughput': max(10, 100 * (1 - base_impact)),  # Decrease with impact
            'cpu_usage': 0.3 + (base_impact * 0.4),  # 30% base + up to 40% impact
            'memory_usage': 0.4 + (base_impact * 0.3)  # 40% base + up to 30% impact
        }
    
    def _calculate_impact(self, metrics: Dict[str, float], target: str) -> float:
        """Calculate overall system impact from metrics."""
        # Weighted impact calculation
        error_impact = metrics.get('error_rate', 0) * 0.4
        latency_impact = min(1.0, (metrics.get('response_time', 100) - 100) / 500) * 0.3
        throughput_impact = max(0, (100 - metrics.get('throughput', 100)) / 100) * 0.2
        resource_impact = max(metrics.get('cpu_usage', 0), metrics.get('memory_usage', 0)) * 0.1
        
        return min(1.0, error_impact + latency_impact + throughput_impact + resource_impact)
    
    def _calculate_resilience_score(
        self, 
        max_impact: float, 
        recovery_time: Optional[float], 
        critical_failure_count: int
    ) -> float:
        """Calculate overall resilience score."""
        
        # Base score from impact resistance
        impact_score = max(0, 1.0 - max_impact)
        
        # Recovery score
        if recovery_time is None:
            recovery_score = 0.0  # No recovery detected
        else:
            # Faster recovery = higher score
            recovery_score = max(0, 1.0 - (recovery_time / 120))  # 120s max expected recovery
        
        # Critical failure penalty
        failure_penalty = min(0.5, critical_failure_count * 0.1)
        
        # Combined score
        resilience_score = (impact_score * 0.5 + recovery_score * 0.4) - failure_penalty
        
        return max(0, min(1, resilience_score))


class ChaosOrchestrator:
    """Orchestrates chaos engineering experiments."""
    
    def __init__(self):
        self.chaos_injector = ChaosInjector()
        self.resilience_validator = ResilienceValidator()
        
        self.experiment_queue = []
        self.running_experiments = {}
        self.completed_experiments = []
        
        self.orchestrator_running = False
        self.orchestrator_thread = None
        
        # Safety controls
        self.max_concurrent_experiments = 1
        self.min_stability_period = 300  # 5 minutes between experiments
        self.emergency_abort_conditions = [
            'system_availability < 0.5',
            'critical_component_failure',
            'manual_abort_signal'
        ]
    
    def schedule_experiment(self, experiment: ChaosExperiment, delay: int = 0):
        """Schedule a chaos experiment."""
        scheduled_time = time.time() + delay
        
        self.experiment_queue.append({
            'experiment': experiment,
            'scheduled_time': scheduled_time
        })
        
        logger.info(f"Scheduled chaos experiment '{experiment.name}' for {datetime.fromtimestamp(scheduled_time)}")
    
    async def run_experiment(self, experiment: ChaosExperiment) -> ExperimentResult:
        """Run a single chaos experiment."""
        logger.info(f"Starting chaos experiment: {experiment.name}")
        
        result = ExperimentResult(
            experiment_name=experiment.name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Record baseline metrics
            result.metrics_before = await self._collect_baseline_metrics()
            
            # Start chaos injection
            injection_task = None
            
            if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
                latency_ms = int(experiment.intensity * 1000)  # Convert to ms
                injection_task = asyncio.create_task(
                    self.chaos_injector.inject_network_latency(
                        experiment.target_component, latency_ms, experiment.duration
                    )
                )
            elif experiment.experiment_type == ChaosExperimentType.SERVICE_FAILURE:
                injection_task = asyncio.create_task(
                    self.chaos_injector.inject_service_failure(
                        experiment.target_component, experiment.intensity, experiment.duration
                    )
                )
            elif experiment.experiment_type == ChaosExperimentType.RESOURCE_EXHAUSTION:
                injection_task = asyncio.create_task(
                    self.chaos_injector.inject_resource_exhaustion(
                        experiment.target_component, "cpu", experiment.intensity, experiment.duration
                    )
                )
            
            # Monitor system resilience
            monitoring_callback = lambda metrics, impact: self._log_experiment_progress(
                experiment.name, metrics, impact
            )
            
            validation_task = asyncio.create_task(
                self.resilience_validator.validate_system_resilience(
                    experiment, monitoring_callback
                )
            )
            
            # Wait for both injection and validation to complete
            if injection_task:
                await injection_task
            
            validation_results = await validation_task
            
            # Record final metrics
            result.metrics_after = await self._collect_baseline_metrics()
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.system_impact = validation_results
            result.recovery_time = validation_results.get('recovery_time')
            result.status = ExperimentStatus.COMPLETED
            
            # Generate lessons learned
            result.lessons_learned = self._generate_lessons_learned(experiment, validation_results)
            
            logger.info(f"Chaos experiment '{experiment.name}' completed successfully")
            
        except Exception as e:
            logger.error(f"Chaos experiment '{experiment.name}' failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.system_impact = {'error': str(e)}
            
            # Emergency abort all injections
            self.chaos_injector.abort_all_injections()
        
        self.completed_experiments.append(result)
        return result
    
    async def _collect_baseline_metrics(self) -> Dict[str, float]:
        """Collect baseline system metrics."""
        # Simulate baseline metrics collection
        return {
            'error_rate': random.uniform(0.001, 0.01),
            'response_time': random.uniform(80, 120),
            'throughput': random.uniform(90, 110),
            'cpu_usage': random.uniform(0.2, 0.4),
            'memory_usage': random.uniform(0.3, 0.5),
            'availability': random.uniform(0.99, 1.0)
        }
    
    async def _log_experiment_progress(
        self, 
        experiment_name: str, 
        metrics: Dict[str, float], 
        impact: float
    ):
        """Log experiment progress."""
        if impact > 0.5:  # Log significant impacts
            logger.warning(f"Experiment '{experiment_name}' impact: {impact:.2f} - Metrics: {metrics}")
    
    def _generate_lessons_learned(
        self, 
        experiment: ChaosExperiment, 
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate lessons learned from experiment."""
        lessons = []
        
        resilience_score = validation_results.get('resilience_score', 0)
        recovery_time = validation_results.get('recovery_time')
        impact_severity = validation_results.get('impact_severity', 'unknown')
        
        if resilience_score > 0.8:
            lessons.append(f"System showed excellent resilience to {experiment.experiment_type.value}")
        elif resilience_score > 0.6:
            lessons.append(f"System showed good resilience to {experiment.experiment_type.value}")
        else:
            lessons.append(f"System resilience to {experiment.experiment_type.value} needs improvement")
        
        if recovery_time:
            if recovery_time < 30:
                lessons.append("Fast recovery time indicates good auto-healing capabilities")
            elif recovery_time < 60:
                lessons.append("Moderate recovery time - consider improving auto-healing")
            else:
                lessons.append("Slow recovery time - auto-healing mechanisms need enhancement")
        else:
            lessons.append("No automatic recovery detected - manual intervention may be required")
        
        if impact_severity == 'critical':
            lessons.append("Critical impact detected - review system architecture for single points of failure")
        elif impact_severity == 'high':
            lessons.append("High impact suggests need for better fault isolation")
        
        return lessons
    
    def start_orchestration(self):
        """Start chaos orchestration."""
        if self.orchestrator_running:
            return
        
        self.orchestrator_running = True
        self.orchestrator_thread = threading.Thread(
            target=self._orchestration_loop,
            daemon=True
        )
        self.orchestrator_thread.start()
        
        logger.info("Chaos orchestration started")
    
    def stop_orchestration(self):
        """Stop chaos orchestration."""
        self.orchestrator_running = False
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=10)
        
        # Abort any running experiments
        self.chaos_injector.abort_all_injections()
        
        logger.info("Chaos orchestration stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.orchestrator_running:
            try:
                current_time = time.time()
                
                # Check for scheduled experiments
                ready_experiments = [
                    item for item in self.experiment_queue
                    if item['scheduled_time'] <= current_time
                ]
                
                # Run ready experiments (respecting concurrency limits)
                for item in ready_experiments:
                    if len(self.running_experiments) < self.max_concurrent_experiments:
                        experiment = item['experiment']
                        
                        # Remove from queue
                        self.experiment_queue.remove(item)
                        
                        # Start experiment in background
                        task = asyncio.create_task(self.run_experiment(experiment))
                        self.running_experiments[experiment.name] = task
                        
                        logger.info(f"Started chaos experiment: {experiment.name}")
                
                # Clean up completed experiments
                completed_names = []
                for name, task in self.running_experiments.items():
                    if task.done():
                        completed_names.append(name)
                
                for name in completed_names:
                    del self.running_experiments[name]
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in chaos orchestration: {e}")
                time.sleep(10)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get orchestration status."""
        return {
            'orchestrator_running': self.orchestrator_running,
            'queued_experiments': len(self.experiment_queue),
            'running_experiments': len(self.running_experiments),
            'completed_experiments': len(self.completed_experiments),
            'active_injections': len(self.chaos_injector.get_active_injections()),
            'next_experiment': (
                min(item['scheduled_time'] for item in self.experiment_queue)
                if self.experiment_queue else None
            )
        }
    
    def get_experiment_report(self) -> Dict[str, Any]:
        """Get comprehensive experiment report."""
        if not self.completed_experiments:
            return {'message': 'No completed experiments'}
        
        # Aggregate statistics
        total_experiments = len(self.completed_experiments)
        successful_experiments = sum(
            1 for exp in self.completed_experiments 
            if exp.status == ExperimentStatus.COMPLETED
        )
        
        resilience_scores = [
            exp.system_impact.get('resilience_score', 0)
            for exp in self.completed_experiments
            if exp.system_impact.get('resilience_score') is not None
        ]
        
        avg_resilience_score = np.mean(resilience_scores) if resilience_scores else 0
        
        # Recovery time analysis
        recovery_times = [
            exp.recovery_time for exp in self.completed_experiments
            if exp.recovery_time is not None
        ]
        
        avg_recovery_time = np.mean(recovery_times) if recovery_times else None
        
        # Collect all lessons learned
        all_lessons = []
        for exp in self.completed_experiments:
            all_lessons.extend(exp.lessons_learned)
        
        # Count lesson themes
        lesson_themes = {}
        for lesson in all_lessons:
            # Simple keyword matching for themes
            if 'resilience' in lesson.lower():
                lesson_themes['resilience'] = lesson_themes.get('resilience', 0) + 1
            if 'recovery' in lesson.lower():
                lesson_themes['recovery'] = lesson_themes.get('recovery', 0) + 1
            if 'improvement' in lesson.lower():
                lesson_themes['improvement'] = lesson_themes.get('improvement', 0) + 1
        
        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments,
            'average_resilience_score': avg_resilience_score,
            'average_recovery_time': avg_recovery_time,
            'lesson_themes': lesson_themes,
            'recent_experiments': [
                {
                    'name': exp.experiment_name,
                    'status': exp.status.value,
                    'resilience_score': exp.system_impact.get('resilience_score'),
                    'recovery_time': exp.recovery_time,
                    'lessons_count': len(exp.lessons_learned)
                }
                for exp in self.completed_experiments[-5:]  # Last 5 experiments
            ]
        }


# Predefined experiment templates
def create_network_latency_experiment(target: str, intensity: float = 0.5) -> ChaosExperiment:
    """Create a network latency chaos experiment."""
    return ChaosExperiment(
        name=f"network_latency_{target}",
        experiment_type=ChaosExperimentType.NETWORK_LATENCY,
        target_component=target,
        duration=60,  # 1 minute
        intensity=intensity,
        expected_recovery_time=30,
        abort_conditions=['system_availability < 0.8']
    )


def create_service_failure_experiment(target: str, intensity: float = 0.3) -> ChaosExperiment:
    """Create a service failure chaos experiment."""
    return ChaosExperiment(
        name=f"service_failure_{target}",
        experiment_type=ChaosExperimentType.SERVICE_FAILURE,
        target_component=target,
        duration=30,  # 30 seconds
        intensity=intensity,
        expected_recovery_time=60,
        abort_conditions=['error_rate > 0.5', 'system_availability < 0.7']
    )


def create_resource_exhaustion_experiment(target: str, intensity: float = 0.8) -> ChaosExperiment:
    """Create a resource exhaustion chaos experiment."""
    return ChaosExperiment(
        name=f"resource_exhaustion_{target}",
        experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
        target_component=target,
        duration=45,  # 45 seconds
        intensity=intensity,
        expected_recovery_time=90,
        abort_conditions=['cpu_usage > 0.95', 'system_availability < 0.6']
    )


# Global chaos orchestrator
_global_chaos_orchestrator: Optional[ChaosOrchestrator] = None


def get_chaos_orchestrator() -> ChaosOrchestrator:
    """Get global chaos orchestrator."""
    global _global_chaos_orchestrator
    if _global_chaos_orchestrator is None:
        _global_chaos_orchestrator = ChaosOrchestrator()
    return _global_chaos_orchestrator


def initialize_chaos_engineering() -> ChaosOrchestrator:
    """Initialize chaos engineering system."""
    orchestrator = get_chaos_orchestrator()
    orchestrator.start_orchestration()
    
    logger.info("Chaos engineering system initialized")
    return orchestrator


if __name__ == "__main__":
    async def demo_chaos_engineering():
        # Initialize chaos engineering
        orchestrator = initialize_chaos_engineering()
        
        # Schedule some demo experiments
        experiments = [
            create_network_latency_experiment("llm_provider", 0.4),
            create_service_failure_experiment("data_processor", 0.2),
            create_resource_exhaustion_experiment("quality_validator", 0.7)
        ]
        
        for i, experiment in enumerate(experiments):
            orchestrator.schedule_experiment(experiment, delay=i * 120)  # 2 minutes apart
        
        # Run for demonstration
        await asyncio.sleep(10)  # Let it run for 10 seconds
        
        # Print status
        status = orchestrator.get_orchestration_status()
        print("Chaos Engineering Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Run one experiment immediately for demo
        demo_experiment = create_network_latency_experiment("demo_service", 0.3)
        result = await orchestrator.run_experiment(demo_experiment)
        
        print(f"\nDemo Experiment Result:")
        print(f"Status: {result.status.value}")
        print(f"Resilience Score: {result.system_impact.get('resilience_score', 'N/A')}")
        print(f"Recovery Time: {result.recovery_time}s")
        print(f"Lessons Learned: {len(result.lessons_learned)}")
        
        orchestrator.stop_orchestration()
    
    # Run demo
    import numpy as np
    asyncio.run(demo_chaos_engineering())