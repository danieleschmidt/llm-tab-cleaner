"""Self-Healing Coordination System - Generation 2 Enhancement.

This module coordinates self-healing across the entire LLM data cleaning pipeline,
integrating with the adaptive meta-routing and autonomous monitoring systems.

Key Features:
- Intelligent failure cascade prevention
- Cross-component recovery coordination
- Performance-based healing decisions
- Predictive maintenance triggers
- Automated rollback mechanisms

Author: Terry (Terragon Labs)
"""

import logging
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .autonomous_monitoring import (
    AutonomousMonitor, SystemAlert, AlertSeverity, HealthStatus,
    RecoveryAction, get_global_monitor
)

logger = logging.getLogger(__name__)


class HealingStrategy(Enum):
    """Self-healing strategy types."""
    REACTIVE = "reactive"  # React to failures
    PROACTIVE = "proactive"  # Prevent failures before they occur
    ADAPTIVE = "adaptive"  # Learn and adapt strategies
    COORDINATED = "coordinated"  # Cross-component coordination


class ComponentState(Enum):
    """Component operational states."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    component_id: str
    state: ComponentState
    performance_score: float
    error_rate: float
    last_failure: Optional[float] = None
    recovery_attempts: int = 0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingPlan:
    """Comprehensive healing plan for system recovery."""
    plan_id: str
    target_components: List[str]
    strategy: HealingStrategy
    actions: List[Dict[str, Any]]
    estimated_duration: float
    risk_assessment: Dict[str, float]
    success_probability: float
    rollback_plan: Optional[Dict[str, Any]] = None


class FailureCascadeDetector:
    """Detects and prevents failure cascades across components."""
    
    def __init__(self, window_size: int = 50):
        """Initialize cascade detector.
        
        Args:
            window_size: Size of failure event window for analysis
        """
        self.window_size = window_size
        self.failure_events = deque(maxlen=window_size)
        self.component_dependencies: Dict[str, Set[str]] = {}
        self.cascade_patterns: List[Dict[str, Any]] = []
        
    def add_failure_event(self, component: str, timestamp: float, severity: str):
        """Add a failure event for cascade analysis."""
        event = {
            'component': component,
            'timestamp': timestamp,
            'severity': severity
        }
        self.failure_events.append(event)
        
        # Check for cascade patterns
        self._analyze_cascade_risk()
    
    def register_dependency(self, component: str, depends_on: Set[str]):
        """Register component dependencies."""
        self.component_dependencies[component] = depends_on
    
    def _analyze_cascade_risk(self) -> float:
        """Analyze current cascade risk."""
        if len(self.failure_events) < 3:
            return 0.0
        
        # Look for patterns in recent failures
        recent_events = list(self.failure_events)[-10:]
        
        # Time-based clustering
        time_clusters = []
        current_cluster = [recent_events[0]]
        
        for event in recent_events[1:]:
            if event['timestamp'] - current_cluster[-1]['timestamp'] < 300:  # 5 minutes
                current_cluster.append(event)
            else:
                time_clusters.append(current_cluster)
                current_cluster = [event]
        
        if current_cluster:
            time_clusters.append(current_cluster)
        
        # Calculate cascade risk based on cluster size and dependency overlap
        max_cluster_size = max(len(cluster) for cluster in time_clusters) if time_clusters else 0
        
        # Dependency-based risk
        dependency_risk = 0.0
        for cluster in time_clusters:
            if len(cluster) > 1:
                components = [event['component'] for event in cluster]
                dependency_overlap = self._calculate_dependency_overlap(components)
                dependency_risk = max(dependency_risk, dependency_overlap)
        
        # Combined risk score
        cascade_risk = min(1.0, (max_cluster_size / 5.0) * 0.5 + dependency_risk * 0.5)
        
        return cascade_risk
    
    def _calculate_dependency_overlap(self, components: List[str]) -> float:
        """Calculate dependency overlap between components."""
        if len(components) < 2:
            return 0.0
        
        total_dependencies = 0
        shared_dependencies = 0
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                deps1 = self.component_dependencies.get(comp1, set())
                deps2 = self.component_dependencies.get(comp2, set())
                
                if deps1 or deps2:
                    total_dependencies += 1
                    if deps1.intersection(deps2):
                        shared_dependencies += 1
        
        return shared_dependencies / max(1, total_dependencies)
    
    def get_cascade_risk(self) -> Dict[str, Any]:
        """Get current cascade risk assessment."""
        risk_score = self._analyze_cascade_risk()
        
        return {
            'risk_score': risk_score,
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
            'recent_failures': len([e for e in self.failure_events if time.time() - e['timestamp'] < 3600]),
            'affected_components': len(set(e['component'] for e in self.failure_events)),
            'recommendations': self._get_cascade_prevention_recommendations(risk_score)
        }
    
    def _get_cascade_prevention_recommendations(self, risk_score: float) -> List[str]:
        """Get recommendations for cascade prevention."""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement circuit breakers on critical paths",
                "Increase monitoring frequency",
                "Prepare manual intervention procedures",
                "Consider partial system shutdown to contain failures"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Monitor dependency chains closely",
                "Pre-position recovery resources",
                "Review recent changes for correlation"
            ])
        else:
            recommendations.append("Continue normal monitoring")
        
        return recommendations


class AdaptiveLearningEngine:
    """Learns from healing actions to improve future responses."""
    
    def __init__(self):
        """Initialize adaptive learning engine."""
        self.healing_history: List[Dict[str, Any]] = []
        self.strategy_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.component_patterns: Dict[str, Dict[str, Any]] = {}
        self.learned_rules: List[Dict[str, Any]] = []
        
    def record_healing_outcome(
        self,
        plan: HealingPlan,
        actual_duration: float,
        success: bool,
        performance_impact: float
    ):
        """Record outcome of a healing action for learning."""
        outcome = {
            'timestamp': time.time(),
            'plan_id': plan.plan_id,
            'strategy': plan.strategy.value,
            'components': plan.target_components,
            'estimated_duration': plan.estimated_duration,
            'actual_duration': actual_duration,
            'success': success,
            'performance_impact': performance_impact,
            'success_probability_predicted': plan.success_probability
        }
        
        self.healing_history.append(outcome)
        
        # Update strategy effectiveness
        effectiveness_score = self._calculate_effectiveness(outcome)
        self.strategy_effectiveness[plan.strategy.value].append(effectiveness_score)
        
        # Learn component-specific patterns
        for component in plan.target_components:
            self._update_component_patterns(component, outcome)
        
        # Generate new rules if needed
        self._generate_learned_rules()
    
    def _calculate_effectiveness(self, outcome: Dict[str, Any]) -> float:
        """Calculate effectiveness score for a healing outcome."""
        base_score = 1.0 if outcome['success'] else 0.0
        
        # Adjust for duration accuracy
        duration_ratio = outcome['actual_duration'] / max(outcome['estimated_duration'], 1e-6)
        duration_penalty = abs(1.0 - duration_ratio) * 0.2
        
        # Adjust for performance impact
        performance_penalty = abs(outcome['performance_impact']) * 0.3
        
        # Adjust for prediction accuracy
        prediction_error = abs(outcome['success_probability_predicted'] - base_score)
        prediction_penalty = prediction_error * 0.1
        
        effectiveness = max(0.0, base_score - duration_penalty - performance_penalty - prediction_penalty)
        
        return effectiveness
    
    def _update_component_patterns(self, component: str, outcome: Dict[str, Any]):
        """Update learned patterns for a component."""
        if component not in self.component_patterns:
            self.component_patterns[component] = {
                'failure_patterns': [],
                'successful_strategies': [],
                'typical_recovery_time': [],
                'common_issues': defaultdict(int)
            }
        
        patterns = self.component_patterns[component]
        
        if outcome['success']:
            patterns['successful_strategies'].append(outcome['strategy'])
            patterns['typical_recovery_time'].append(outcome['actual_duration'])
        
        # Keep only recent patterns
        for key in ['successful_strategies', 'typical_recovery_time']:
            if len(patterns[key]) > 50:
                patterns[key] = patterns[key][-50:]
    
    def _generate_learned_rules(self):
        """Generate new learned rules from historical data."""
        if len(self.healing_history) < 20:
            return
        
        # Simple rule generation based on success patterns
        recent_history = self.healing_history[-50:]
        
        # Group by strategy and analyze success rates
        strategy_analysis = defaultdict(list)
        for outcome in recent_history:
            strategy_analysis[outcome['strategy']].append(outcome['success'])
        
        # Generate rules for high-success strategies
        for strategy, successes in strategy_analysis.items():
            if len(successes) >= 5:
                success_rate = sum(successes) / len(successes)
                if success_rate > 0.8:
                    rule = {
                        'type': 'strategy_preference',
                        'condition': f"component_failure",
                        'action': f"prefer_strategy_{strategy}",
                        'confidence': success_rate,
                        'sample_size': len(successes)
                    }
                    
                    # Add rule if not already exists
                    if not any(r['action'] == rule['action'] for r in self.learned_rules):
                        self.learned_rules.append(rule)
        
        # Keep only recent rules
        if len(self.learned_rules) > 20:
            self.learned_rules = sorted(self.learned_rules, key=lambda x: x['confidence'], reverse=True)[:20]
    
    def get_strategy_recommendation(self, component: str, failure_type: str) -> Tuple[HealingStrategy, float]:
        """Get recommended strategy based on learned patterns."""
        # Check learned rules first
        for rule in self.learned_rules:
            if rule['type'] == 'strategy_preference':
                strategy_name = rule['action'].replace('prefer_strategy_', '')
                try:
                    strategy = HealingStrategy(strategy_name)
                    return strategy, rule['confidence']
                except ValueError:
                    continue
        
        # Fall back to component-specific patterns
        if component in self.component_patterns:
            patterns = self.component_patterns[component]
            if patterns['successful_strategies']:
                most_common = max(set(patterns['successful_strategies']), 
                                key=patterns['successful_strategies'].count)
                try:
                    strategy = HealingStrategy(most_common)
                    confidence = patterns['successful_strategies'].count(most_common) / len(patterns['successful_strategies'])
                    return strategy, confidence
                except ValueError:
                    pass
        
        # Default recommendation
        return HealingStrategy.REACTIVE, 0.5
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        return {
            'total_healing_events': len(self.healing_history),
            'strategy_effectiveness': {
                strategy: np.mean(scores) if scores else 0.0
                for strategy, scores in self.strategy_effectiveness.items()
            },
            'learned_rules_count': len(self.learned_rules),
            'components_analyzed': len(self.component_patterns),
            'recent_success_rate': (
                sum(1 for h in self.healing_history[-20:] if h['success']) / 
                max(1, len(self.healing_history[-20:]))
            ) if self.healing_history else 0.0
        }


class SelfHealingCoordinator:
    """Main coordinator for self-healing across the system."""
    
    def __init__(
        self,
        enable_proactive_healing: bool = True,
        enable_learning: bool = True,
        max_concurrent_healings: int = 3
    ):
        """Initialize self-healing coordinator.
        
        Args:
            enable_proactive_healing: Enable proactive healing strategies
            enable_learning: Enable adaptive learning from healing outcomes
            max_concurrent_healings: Maximum concurrent healing operations
        """
        self.enable_proactive_healing = enable_proactive_healing
        self.enable_learning = enable_learning
        self.max_concurrent_healings = max_concurrent_healings
        
        # Core components
        self.monitor = get_global_monitor()
        self.cascade_detector = FailureCascadeDetector()
        self.learning_engine = AdaptiveLearningEngine() if enable_learning else None
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.active_healings: Dict[str, HealingPlan] = {}
        self.healing_queue: List[HealingPlan] = []
        
        # Performance tracking
        self.coordination_metrics = {
            'total_healings': 0,
            'successful_healings': 0,
            'prevented_cascades': 0,
            'average_healing_time': 0.0
        }
        
        # Threading
        self._healing_executor = ThreadPoolExecutor(max_workers=max_concurrent_healings)
        self._coordinator_running = False
        self._coordinator_thread = None
        
        logger.info("Initialized SelfHealingCoordinator")
    
    def register_component(
        self,
        component_id: str,
        dependencies: Set[str] = None,
        health_checker: Callable = None
    ):
        """Register a component for coordinated healing."""
        self.components[component_id] = ComponentHealth(
            component_id=component_id,
            state=ComponentState.ACTIVE,
            performance_score=1.0,
            error_rate=0.0,
            dependencies=dependencies or set()
        )
        
        if dependencies:
            self.cascade_detector.register_dependency(component_id, dependencies)
        
        if health_checker:
            self.monitor.register_health_checker(component_id, health_checker)
        
        logger.info(f"Registered component: {component_id}")
    
    def start_coordination(self):
        """Start the self-healing coordination."""
        if self._coordinator_running:
            logger.warning("Coordination already running")
            return
        
        self._coordinator_running = True
        self._coordinator_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self._coordinator_thread.start()
        
        # Ensure monitoring is running
        if not self.monitor.is_running:
            self.monitor.start_monitoring()
        
        logger.info("Started self-healing coordination")
    
    def stop_coordination(self):
        """Stop the self-healing coordination."""
        self._coordinator_running = False
        if self._coordinator_thread:
            self._coordinator_thread.join(timeout=5)
        
        self._healing_executor.shutdown(wait=True)
        logger.info("Stopped self-healing coordination")
    
    def _coordination_loop(self):
        """Main coordination loop."""
        while self._coordinator_running:
            try:
                # Update component states
                self._update_component_states()
                
                # Check for cascade risks
                self._check_cascade_prevention()
                
                # Process healing queue
                self._process_healing_queue()
                
                # Proactive healing checks
                if self.enable_proactive_healing:
                    self._check_proactive_healing()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(10)
    
    def _update_component_states(self):
        """Update component states based on monitoring data."""
        system_health = self.monitor.get_system_health()
        
        for component_id, component in self.components.items():
            # Update from monitoring metrics
            if component_id in system_health['metrics']:
                metric = system_health['metrics'][component_id]
                
                # Update performance score based on metric status
                if metric['status'] == 'healthy':
                    component.performance_score = min(1.0, component.performance_score + 0.01)
                    component.state = ComponentState.ACTIVE
                elif metric['status'] == 'warning':
                    component.performance_score *= 0.95
                    component.state = ComponentState.DEGRADED
                elif metric['status'] == 'critical':
                    component.performance_score *= 0.8
                    component.state = ComponentState.FAILED
                    component.last_failure = time.time()
                    
                    # Add to cascade detector
                    self.cascade_detector.add_failure_event(
                        component_id, time.time(), metric['status']
                    )
    
    def _check_cascade_prevention(self):
        """Check and prevent potential cascades."""
        cascade_risk = self.cascade_detector.get_cascade_risk()
        
        if cascade_risk['risk_level'] == 'high':
            logger.warning(f"High cascade risk detected: {cascade_risk['risk_score']:.2f}")
            
            # Create preventive healing plan
            plan = self._create_cascade_prevention_plan(cascade_risk)
            if plan:
                self._queue_healing_plan(plan)
                self.coordination_metrics['prevented_cascades'] += 1
    
    def _create_cascade_prevention_plan(self, cascade_risk: Dict[str, Any]) -> Optional[HealingPlan]:
        """Create a plan to prevent cascade failures."""
        affected_components = [
            comp_id for comp_id, comp in self.components.items()
            if comp.state in [ComponentState.FAILED, ComponentState.DEGRADED]
        ]
        
        if not affected_components:
            return None
        
        # Prioritize components with most dependencies
        prioritized_components = sorted(
            affected_components,
            key=lambda c: len(self.components[c].dependencies),
            reverse=True
        )
        
        actions = []
        for component in prioritized_components[:3]:  # Limit to top 3
            actions.append({
                'type': 'stabilize_component',
                'component': component,
                'priority': 'high',
                'timeout': 300
            })
        
        return HealingPlan(
            plan_id=f"cascade_prevention_{int(time.time())}",
            target_components=prioritized_components[:3],
            strategy=HealingStrategy.PROACTIVE,
            actions=actions,
            estimated_duration=600,
            risk_assessment={'cascade_prevention': 0.8},
            success_probability=0.7
        )
    
    def _check_proactive_healing(self):
        """Check for proactive healing opportunities."""
        for component_id, component in self.components.items():
            # Proactive healing based on performance degradation
            if (component.state == ComponentState.DEGRADED and 
                component.performance_score < 0.7):
                
                # Get strategy recommendation from learning engine
                if self.learning_engine:
                    strategy, confidence = self.learning_engine.get_strategy_recommendation(
                        component_id, 'performance_degradation'
                    )
                else:
                    strategy, confidence = HealingStrategy.PROACTIVE, 0.5
                
                # Create proactive healing plan
                plan = HealingPlan(
                    plan_id=f"proactive_{component_id}_{int(time.time())}",
                    target_components=[component_id],
                    strategy=strategy,
                    actions=[{
                        'type': 'performance_optimization',
                        'component': component_id,
                        'priority': 'medium'
                    }],
                    estimated_duration=180,
                    risk_assessment={'performance_impact': 0.2},
                    success_probability=confidence
                )
                
                self._queue_healing_plan(plan)
    
    def _queue_healing_plan(self, plan: HealingPlan):
        """Queue a healing plan for execution."""
        # Check if already healing this component
        for component in plan.target_components:
            if any(component in active_plan.target_components 
                  for active_plan in self.active_healings.values()):
                logger.info(f"Component {component} already being healed, skipping")
                return
        
        self.healing_queue.append(plan)
        logger.info(f"Queued healing plan: {plan.plan_id}")
    
    def _process_healing_queue(self):
        """Process queued healing plans."""
        while (self.healing_queue and 
               len(self.active_healings) < self.max_concurrent_healings):
            
            plan = self.healing_queue.pop(0)
            
            # Execute healing plan
            future = self._healing_executor.submit(self._execute_healing_plan, plan)
            self.active_healings[plan.plan_id] = plan
            
            # Set callback for completion
            future.add_done_callback(lambda f, p=plan: self._on_healing_complete(p, f))
    
    def _execute_healing_plan(self, plan: HealingPlan) -> Dict[str, Any]:
        """Execute a healing plan."""
        start_time = time.time()
        
        logger.info(f"Executing healing plan: {plan.plan_id}")
        
        try:
            # Mark components as recovering
            for component in plan.target_components:
                if component in self.components:
                    self.components[component].state = ComponentState.RECOVERING
                    self.components[component].recovery_attempts += 1
            
            # Execute actions
            results = []
            for action in plan.actions:
                result = self._execute_healing_action(action)
                results.append(result)
            
            # Determine overall success
            success = all(r.get('success', False) for r in results)
            actual_duration = time.time() - start_time
            
            # Update component states based on results
            for component in plan.target_components:
                if component in self.components:
                    if success:
                        self.components[component].state = ComponentState.ACTIVE
                        self.components[component].performance_score = min(1.0, 
                            self.components[component].performance_score + 0.1)
                    else:
                        self.components[component].state = ComponentState.FAILED
            
            return {
                'success': success,
                'duration': actual_duration,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error executing healing plan {plan.plan_id}: {e}")
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def _execute_healing_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single healing action."""
        action_type = action.get('type', 'unknown')
        component = action.get('component', 'unknown')
        
        logger.info(f"Executing healing action: {action_type} on {component}")
        
        # Simulate healing actions (in real implementation, these would be actual operations)
        time.sleep(1)  # Simulate work
        
        # Simple success simulation based on action type
        success_rates = {
            'stabilize_component': 0.85,
            'performance_optimization': 0.75,
            'restart_component': 0.9,
            'scale_resources': 0.8
        }
        
        success_probability = success_rates.get(action_type, 0.7)
        success = np.random.random() < success_probability
        
        return {
            'action_type': action_type,
            'component': component,
            'success': success,
            'timestamp': time.time()
        }
    
    def _on_healing_complete(self, plan: HealingPlan, future):
        """Handle completion of a healing plan."""
        try:
            result = future.result()
            
            # Update metrics
            self.coordination_metrics['total_healings'] += 1
            if result['success']:
                self.coordination_metrics['successful_healings'] += 1
            
            # Update average healing time
            total_time = (self.coordination_metrics['average_healing_time'] * 
                         (self.coordination_metrics['total_healings'] - 1) + 
                         result['duration'])
            self.coordination_metrics['average_healing_time'] = total_time / self.coordination_metrics['total_healings']
            
            # Record outcome for learning
            if self.learning_engine:
                performance_impact = self._calculate_performance_impact(plan, result)
                self.learning_engine.record_healing_outcome(
                    plan, result['duration'], result['success'], performance_impact
                )
            
            logger.info(f"Healing plan {plan.plan_id} completed: {result['success']}")
            
        except Exception as e:
            logger.error(f"Error handling healing completion for {plan.plan_id}: {e}")
        
        finally:
            # Remove from active healings
            if plan.plan_id in self.active_healings:
                del self.active_healings[plan.plan_id]
    
    def _calculate_performance_impact(self, plan: HealingPlan, result: Dict[str, Any]) -> float:
        """Calculate performance impact of healing."""
        # Simplified calculation - in real implementation would measure actual impact
        if result['success']:
            return 0.1  # Positive impact
        else:
            return -0.2  # Negative impact
    
    def trigger_emergency_healing(self, component_id: str, severity: str = "high"):
        """Trigger emergency healing for a specific component."""
        if component_id not in self.components:
            logger.error(f"Unknown component for emergency healing: {component_id}")
            return
        
        plan = HealingPlan(
            plan_id=f"emergency_{component_id}_{int(time.time())}",
            target_components=[component_id],
            strategy=HealingStrategy.REACTIVE,
            actions=[{
                'type': 'emergency_recovery',
                'component': component_id,
                'priority': 'critical',
                'severity': severity
            }],
            estimated_duration=120,
            risk_assessment={'emergency': 1.0},
            success_probability=0.6
        )
        
        # Insert at front of queue for immediate processing
        self.healing_queue.insert(0, plan)
        logger.warning(f"Triggered emergency healing for {component_id}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        cascade_risk = self.cascade_detector.get_cascade_risk()
        
        status = {
            'coordinator_running': self._coordinator_running,
            'components': {
                comp_id: {
                    'state': comp.state.value,
                    'performance_score': comp.performance_score,
                    'error_rate': comp.error_rate,
                    'recovery_attempts': comp.recovery_attempts,
                    'dependencies': list(comp.dependencies)
                }
                for comp_id, comp in self.components.items()
            },
            'active_healings': len(self.active_healings),
            'queued_healings': len(self.healing_queue),
            'cascade_risk': cascade_risk,
            'metrics': self.coordination_metrics.copy()
        }
        
        if self.learning_engine:
            status['learning_summary'] = self.learning_engine.get_learning_summary()
        
        return status


# Global coordinator instance
_global_coordinator: Optional[SelfHealingCoordinator] = None


def get_global_coordinator() -> SelfHealingCoordinator:
    """Get global self-healing coordinator instance."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = SelfHealingCoordinator()
    return _global_coordinator


def initialize_self_healing(
    enable_proactive: bool = True,
    enable_learning: bool = True,
    start_immediately: bool = True
) -> SelfHealingCoordinator:
    """Initialize and optionally start self-healing coordination."""
    global _global_coordinator
    
    _global_coordinator = SelfHealingCoordinator(
        enable_proactive_healing=enable_proactive,
        enable_learning=enable_learning
    )
    
    if start_immediately:
        _global_coordinator.start_coordination()
    
    logger.info("Initialized self-healing coordination system")
    return _global_coordinator


if __name__ == "__main__":
    # Demo self-healing coordination
    coordinator = initialize_self_healing()
    
    # Register some example components
    coordinator.register_component("llm_router", {"monitoring", "cache"})
    coordinator.register_component("data_profiler", {"storage"})
    coordinator.register_component("cleaning_engine", {"llm_router", "data_profiler"})
    
    try:
        # Run for demonstration
        time.sleep(30)
        
        # Trigger emergency healing
        coordinator.trigger_emergency_healing("cleaning_engine")
        
        time.sleep(30)
        
        # Print status
        status = coordinator.get_coordination_status()
        print("Self-Healing Coordination Status:")
        print(json.dumps(status, indent=2, default=str))
        
    finally:
        coordinator.stop_coordination()