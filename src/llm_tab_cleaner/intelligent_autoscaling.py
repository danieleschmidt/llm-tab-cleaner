"""Intelligent Auto-Scaling and Global Optimization - Generation 3 Enhancement.

This module implements advanced auto-scaling with ML-driven predictions and
global optimization across the entire LLM data cleaning infrastructure.

Key Features:
- Predictive auto-scaling using time series forecasting
- Multi-dimensional resource optimization (compute, memory, cost, latency)
- Cross-regional load balancing and optimization
- Intelligent workload distribution
- Dynamic cost optimization with performance constraints
- Global resource allocation strategies

Author: Terry (Terragon Labs)
"""

import logging
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Auto-scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    LLMAPI_CALLS = "llm_api_calls"
    WORKERS = "workers"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    COST = "cost"
    PERFORMANCE = "performance"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    resource_type: ResourceType
    current_usage: float
    capacity: float
    target_utilization: float
    cost_per_unit: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def utilization_ratio(self) -> float:
        """Calculate utilization ratio."""
        return self.current_usage / max(self.capacity, 1e-6)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (higher is better)."""
        target_ratio = self.target_utilization / 100.0
        actual_ratio = self.utilization_ratio
        
        # Efficiency is maximized when actual matches target
        efficiency = 1.0 - abs(actual_ratio - target_ratio)
        return max(0.0, efficiency)


@dataclass
class ScalingRecommendation:
    """Auto-scaling recommendation."""
    resource_type: ResourceType
    direction: ScalingDirection
    magnitude: float  # Scaling factor (e.g., 1.5 for 50% increase)
    confidence: float
    reasoning: str
    estimated_cost_impact: float
    estimated_performance_impact: float
    urgency: str  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalOptimizationPlan:
    """Global optimization plan across regions/providers."""
    plan_id: str
    target_regions: List[str]
    workload_distribution: Dict[str, float]  # region -> percentage
    resource_allocation: Dict[str, Dict[str, float]]  # region -> resource -> amount
    estimated_cost: float
    estimated_performance: float
    implementation_steps: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]


class PredictiveLoadForecaster:
    """ML-based load forecasting for predictive scaling."""
    
    def __init__(self, forecast_horizon: int = 3600):  # 1 hour default
        """Initialize load forecaster.
        
        Args:
            forecast_horizon: Forecast horizon in seconds
        """
        self.forecast_horizon = forecast_horizon
        self.load_history = deque(maxlen=2000)  # Keep 2000 data points
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'load', 'hour_of_day', 'day_of_week', 'month', 
            'load_trend', 'load_volatility', 'cpu_usage', 'memory_usage'
        ]
        
    def add_load_sample(
        self, 
        load: float, 
        cpu_usage: float, 
        memory_usage: float,
        timestamp: Optional[float] = None
    ):
        """Add a load sample for forecasting."""
        if timestamp is None:
            timestamp = time.time()
        
        sample = {
            'timestamp': timestamp,
            'load': load,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        }
        
        self.load_history.append(sample)
        
        # Retrain periodically
        if len(self.load_history) > 50 and len(self.load_history) % 20 == 0:
            self._retrain_model()
    
    def _retrain_model(self):
        """Retrain the forecasting model."""
        if len(self.load_history) < 20:
            return
        
        try:
            # Prepare training data
            df = pd.DataFrame(list(self.load_history))
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create features
            df['hour_of_day'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            
            # Calculate trends and volatility
            df['load_trend'] = df['load'].rolling(window=5, min_periods=1).mean()
            df['load_volatility'] = df['load'].rolling(window=5, min_periods=1).std().fillna(0)
            
            # Prepare features and target
            feature_df = df[self.feature_columns].fillna(0)
            target = df['load'].shift(-1).dropna()  # Predict next load
            features = feature_df[:-1]  # Remove last row to match target length
            
            if len(features) < 10:
                return
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, target)
            self.is_trained = True
            
            logger.info(f"Retrained load forecasting model with {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining load forecasting model: {e}")
    
    def forecast_load(self, steps_ahead: int = 12) -> List[Tuple[float, float]]:
        """Forecast load for specified steps ahead.
        
        Args:
            steps_ahead: Number of time steps to forecast
            
        Returns:
            List of (timestamp, predicted_load) tuples
        """
        if not self.is_trained or len(self.load_history) < 10:
            # Return simple baseline forecast
            current_load = self.load_history[-1]['load'] if self.load_history else 1.0
            return [(time.time() + i * 300, current_load) for i in range(1, steps_ahead + 1)]
        
        try:
            # Get latest data for prediction
            latest = list(self.load_history)[-10:]  # Use last 10 samples
            df = pd.DataFrame(latest)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create features for latest sample
            df['hour_of_day'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['load_trend'] = df['load'].rolling(window=5, min_periods=1).mean()
            df['load_volatility'] = df['load'].rolling(window=5, min_periods=1).std().fillna(0)
            
            forecasts = []
            current_features = df[self.feature_columns].iloc[-1].values.reshape(1, -1)
            
            for i in range(steps_ahead):
                # Scale features
                features_scaled = self.scaler.transform(current_features)
                
                # Predict
                predicted_load = self.model.predict(features_scaled)[0]
                forecast_time = time.time() + (i + 1) * 300  # 5-minute intervals
                
                forecasts.append((forecast_time, max(0, predicted_load)))
                
                # Update features for next prediction (simplified)
                current_features[0][0] = predicted_load  # Update load
                
            return forecasts
            
        except Exception as e:
            logger.error(f"Error forecasting load: {e}")
            # Fallback to baseline
            current_load = self.load_history[-1]['load'] if self.load_history else 1.0
            return [(time.time() + i * 300, current_load) for i in range(1, steps_ahead + 1)]
    
    def get_scaling_signal(self) -> Tuple[ScalingDirection, float]:
        """Get scaling signal based on forecast.
        
        Returns:
            Tuple of (scaling_direction, confidence)
        """
        if not self.is_trained:
            return ScalingDirection.MAINTAIN, 0.0
        
        # Get forecast for next hour
        forecasts = self.forecast_load(12)  # 12 * 5 minutes = 1 hour
        
        if not forecasts:
            return ScalingDirection.MAINTAIN, 0.0
        
        current_load = self.load_history[-1]['load'] if self.load_history else 1.0
        forecast_loads = [f[1] for f in forecasts]
        avg_forecast_load = np.mean(forecast_loads)
        
        # Calculate load change
        load_change_ratio = avg_forecast_load / max(current_load, 1e-6)
        
        # Determine scaling direction
        if load_change_ratio > 1.3:  # 30% increase predicted
            return ScalingDirection.SCALE_UP, min(0.9, (load_change_ratio - 1.0) * 2)
        elif load_change_ratio < 0.7:  # 30% decrease predicted
            return ScalingDirection.SCALE_DOWN, min(0.9, (1.0 - load_change_ratio) * 2)
        else:
            return ScalingDirection.MAINTAIN, 0.5


class GlobalOptimizer:
    """Global optimization across regions and providers."""
    
    def __init__(self):
        """Initialize global optimizer."""
        self.regions = {}  # region_id -> region_info
        self.providers = {}  # provider_id -> provider_info
        self.workload_patterns = defaultdict(list)
        self.cost_models = {}
        self.performance_models = {}
        
    def register_region(
        self, 
        region_id: str, 
        capacity: Dict[str, float],
        cost_multiplier: float = 1.0,
        latency_to_users: float = 50.0  # ms
    ):
        """Register a region for global optimization."""
        self.regions[region_id] = {
            'capacity': capacity,
            'cost_multiplier': cost_multiplier,
            'latency_to_users': latency_to_users,
            'current_load': 0.0,
            'availability_zone': region_id.split('-')[0] if '-' in region_id else 'default'
        }
        
        logger.info(f"Registered region: {region_id}")
    
    def register_provider(
        self,
        provider_id: str,
        cost_per_request: float,
        avg_latency: float,
        rate_limits: Dict[str, float]
    ):
        """Register a provider for global optimization."""
        self.providers[provider_id] = {
            'cost_per_request': cost_per_request,
            'avg_latency': avg_latency,
            'rate_limits': rate_limits,
            'current_usage': 0.0
        }
        
        logger.info(f"Registered provider: {provider_id}")
    
    def optimize_global_allocation(
        self,
        expected_workload: float,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        constraints: Dict[str, Any] = None
    ) -> GlobalOptimizationPlan:
        """Optimize global resource allocation.
        
        Args:
            expected_workload: Expected total workload
            objective: Optimization objective
            constraints: Additional constraints
            
        Returns:
            Global optimization plan
        """
        constraints = constraints or {}
        
        if objective == OptimizationObjective.COST:
            return self._optimize_for_cost(expected_workload, constraints)
        elif objective == OptimizationObjective.PERFORMANCE:
            return self._optimize_for_performance(expected_workload, constraints)
        elif objective == OptimizationObjective.LATENCY:
            return self._optimize_for_latency(expected_workload, constraints)
        else:  # BALANCED
            return self._optimize_balanced(expected_workload, constraints)
    
    def _optimize_for_cost(
        self, 
        workload: float, 
        constraints: Dict[str, Any]
    ) -> GlobalOptimizationPlan:
        """Optimize for minimum cost."""
        # Sort regions by cost efficiency
        sorted_regions = sorted(
            self.regions.items(),
            key=lambda x: x[1]['cost_multiplier']
        )
        
        # Distribute workload to cheapest regions first
        workload_distribution = {}
        resource_allocation = {}
        remaining_workload = workload
        
        for region_id, region_info in sorted_regions:
            if remaining_workload <= 0:
                break
            
            # Calculate how much this region can handle
            region_capacity = min(
                remaining_workload,
                region_info['capacity'].get('total', workload * 0.5)
            )
            
            workload_distribution[region_id] = region_capacity / workload
            resource_allocation[region_id] = {
                'cpu': region_capacity * 2,  # Simplified allocation
                'memory': region_capacity * 4,
                'workers': max(1, int(region_capacity / 10))
            }
            
            remaining_workload -= region_capacity
        
        # Estimate cost
        total_cost = sum(
            allocation['workers'] * 10 * self.regions[region]['cost_multiplier']
            for region, allocation in resource_allocation.items()
        )
        
        return GlobalOptimizationPlan(
            plan_id=f"cost_opt_{int(time.time())}",
            target_regions=list(workload_distribution.keys()),
            workload_distribution=workload_distribution,
            resource_allocation=resource_allocation,
            estimated_cost=total_cost,
            estimated_performance=0.8,  # Simplified
            implementation_steps=[
                {
                    'step': 'allocate_resources',
                    'regions': list(workload_distribution.keys()),
                    'priority': 'high'
                }
            ],
            risk_assessment={'cost_overrun': 0.1, 'performance_degradation': 0.3}
        )
    
    def _optimize_for_performance(
        self, 
        workload: float, 
        constraints: Dict[str, Any]
    ) -> GlobalOptimizationPlan:
        """Optimize for maximum performance."""
        # Distribute workload evenly across high-capacity regions
        high_capacity_regions = {
            region_id: info for region_id, info in self.regions.items()
            if info['capacity'].get('total', 0) > workload * 0.1
        }
        
        if not high_capacity_regions:
            high_capacity_regions = self.regions
        
        # Even distribution for performance
        workload_per_region = workload / len(high_capacity_regions)
        workload_distribution = {
            region_id: workload_per_region / workload
            for region_id in high_capacity_regions.keys()
        }
        
        # Generous resource allocation for performance
        resource_allocation = {}
        for region_id in high_capacity_regions.keys():
            resource_allocation[region_id] = {
                'cpu': workload_per_region * 3,  # Over-provision for performance
                'memory': workload_per_region * 6,
                'workers': max(2, int(workload_per_region / 5))
            }
        
        total_cost = sum(
            allocation['workers'] * 10 * self.regions[region]['cost_multiplier']
            for region, allocation in resource_allocation.items()
        )
        
        return GlobalOptimizationPlan(
            plan_id=f"perf_opt_{int(time.time())}",
            target_regions=list(workload_distribution.keys()),
            workload_distribution=workload_distribution,
            resource_allocation=resource_allocation,
            estimated_cost=total_cost,
            estimated_performance=0.95,
            implementation_steps=[
                {
                    'step': 'scale_up_resources',
                    'regions': list(workload_distribution.keys()),
                    'priority': 'high'
                }
            ],
            risk_assessment={'cost_overrun': 0.5, 'performance_degradation': 0.05}
        )
    
    def _optimize_balanced(
        self, 
        workload: float, 
        constraints: Dict[str, Any]
    ) -> GlobalOptimizationPlan:
        """Optimize for balanced cost and performance."""
        # Score regions based on cost-performance trade-off
        region_scores = {}
        for region_id, region_info in self.regions.items():
            # Simple scoring: inverse of cost multiplier + capacity factor
            capacity_score = min(1.0, region_info['capacity'].get('total', 0) / workload)
            cost_score = 1.0 / region_info['cost_multiplier']
            latency_score = 1.0 / (region_info['latency_to_users'] / 50.0)  # Normalize to 50ms
            
            # Balanced scoring
            region_scores[region_id] = (capacity_score + cost_score + latency_score) / 3
        
        # Distribute based on scores
        total_score = sum(region_scores.values())
        workload_distribution = {
            region_id: score / total_score
            for region_id, score in region_scores.items()
        }
        
        # Balanced resource allocation
        resource_allocation = {}
        for region_id, distribution in workload_distribution.items():
            region_workload = workload * distribution
            resource_allocation[region_id] = {
                'cpu': region_workload * 2.5,
                'memory': region_workload * 5,
                'workers': max(1, int(region_workload / 8))
            }
        
        total_cost = sum(
            allocation['workers'] * 10 * self.regions[region]['cost_multiplier']
            for region, allocation in resource_allocation.items()
        )
        
        return GlobalOptimizationPlan(
            plan_id=f"balanced_opt_{int(time.time())}",
            target_regions=list(workload_distribution.keys()),
            workload_distribution=workload_distribution,
            resource_allocation=resource_allocation,
            estimated_cost=total_cost,
            estimated_performance=0.85,
            implementation_steps=[
                {
                    'step': 'rebalance_workload',
                    'regions': list(workload_distribution.keys()),
                    'priority': 'medium'
                }
            ],
            risk_assessment={'cost_overrun': 0.2, 'performance_degradation': 0.15}
        )


class IntelligentAutoScaler:
    """Main auto-scaling system with ML-driven predictions."""
    
    def __init__(
        self,
        scaling_interval: int = 300,  # 5 minutes
        enable_predictive_scaling: bool = True,
        enable_global_optimization: bool = True
    ):
        """Initialize intelligent auto-scaler.
        
        Args:
            scaling_interval: Seconds between scaling evaluations
            enable_predictive_scaling: Enable ML-based predictive scaling
            enable_global_optimization: Enable global optimization
        """
        self.scaling_interval = scaling_interval
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_global_optimization = enable_global_optimization
        
        # Core components
        self.resource_metrics: Dict[str, ResourceMetrics] = {}
        self.load_forecaster = PredictiveLoadForecaster()
        self.global_optimizer = GlobalOptimizer()
        
        # Scaling state
        self.scaling_history = deque(maxlen=1000)
        self.active_scaling_actions = {}
        self.cooldown_periods = {}  # resource -> last_scaling_time
        self.min_cooldown = 300  # 5 minutes minimum between scalings
        
        # Configuration
        self.scaling_thresholds = {
            ResourceType.CPU: {'scale_up': 80, 'scale_down': 30},
            ResourceType.MEMORY: {'scale_up': 85, 'scale_down': 40},
            ResourceType.WORKERS: {'scale_up': 90, 'scale_down': 20}
        }
        
        # Performance tracking
        self.scaling_effectiveness = defaultdict(list)
        
        # Threading
        self._scaler_running = False
        self._scaler_thread = None
        
        logger.info("Initialized IntelligentAutoScaler")
    
    def start_autoscaling(self):
        """Start the auto-scaling system."""
        if self._scaler_running:
            logger.warning("Auto-scaling already running")
            return
        
        self._scaler_running = True
        self._scaler_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaler_thread.start()
        
        logger.info("Started intelligent auto-scaling")
    
    def stop_autoscaling(self):
        """Stop the auto-scaling system."""
        self._scaler_running = False
        if self._scaler_thread:
            self._scaler_thread.join(timeout=5)
        
        logger.info("Stopped intelligent auto-scaling")
    
    def update_resource_metrics(
        self,
        resource_type: ResourceType,
        current_usage: float,
        capacity: float,
        cost_per_unit: float = 1.0
    ):
        """Update resource metrics for scaling decisions."""
        metrics = ResourceMetrics(
            resource_type=resource_type,
            current_usage=current_usage,
            capacity=capacity,
            target_utilization=70.0,  # Default target
            cost_per_unit=cost_per_unit
        )
        
        self.resource_metrics[resource_type.value] = metrics
        
        # Add to load forecaster if relevant
        if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
            cpu_usage = current_usage if resource_type == ResourceType.CPU else 50.0
            memory_usage = current_usage if resource_type == ResourceType.MEMORY else 60.0
            load = (cpu_usage + memory_usage) / 2  # Simplified load metric
            
            self.load_forecaster.add_load_sample(load, cpu_usage, memory_usage)
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self._scaler_running:
            try:
                # Collect current metrics
                self._collect_system_metrics()
                
                # Generate scaling recommendations
                recommendations = self._generate_scaling_recommendations()
                
                # Execute scaling actions
                for recommendation in recommendations:
                    if self._should_execute_scaling(recommendation):
                        self._execute_scaling_action(recommendation)
                
                # Global optimization
                if self.enable_global_optimization:
                    self._evaluate_global_optimization()
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.scaling_interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_resource_metrics(
                ResourceType.CPU, 
                cpu_percent, 
                100.0, 
                cost_per_unit=0.05
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.update_resource_metrics(
                ResourceType.MEMORY,
                memory.percent,
                100.0,
                cost_per_unit=0.03
            )
            
            # Simulated worker metrics
            current_workers = self.resource_metrics.get('workers', {}).get('current_usage', 5)
            worker_utilization = min(95.0, current_workers * 15 + np.random.normal(0, 5))
            self.update_resource_metrics(
                ResourceType.WORKERS,
                worker_utilization,
                100.0,
                cost_per_unit=10.0
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _generate_scaling_recommendations(self) -> List[ScalingRecommendation]:
        """Generate scaling recommendations based on current state."""
        recommendations = []
        
        for resource_name, metrics in self.resource_metrics.items():
            try:
                resource_type = ResourceType(resource_name)
                
                # Get threshold-based recommendation
                threshold_rec = self._get_threshold_based_recommendation(metrics)
                if threshold_rec:
                    recommendations.append(threshold_rec)
                
                # Get predictive recommendation if enabled
                if self.enable_predictive_scaling and resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                    predictive_rec = self._get_predictive_recommendation(metrics)
                    if predictive_rec:
                        recommendations.append(predictive_rec)
                        
            except ValueError:
                # Unknown resource type
                continue
        
        return recommendations
    
    def _get_threshold_based_recommendation(
        self, 
        metrics: ResourceMetrics
    ) -> Optional[ScalingRecommendation]:
        """Generate threshold-based scaling recommendation."""
        thresholds = self.scaling_thresholds.get(metrics.resource_type, {})
        if not thresholds:
            return None
        
        utilization = metrics.utilization_ratio * 100
        
        if utilization > thresholds['scale_up']:
            # Scale up recommendation
            magnitude = 1.0 + min(0.5, (utilization - thresholds['scale_up']) / 100.0)
            confidence = min(0.9, (utilization - thresholds['scale_up']) / 20.0)
            
            return ScalingRecommendation(
                resource_type=metrics.resource_type,
                direction=ScalingDirection.SCALE_UP,
                magnitude=magnitude,
                confidence=confidence,
                reasoning=f"Utilization {utilization:.1f}% exceeds scale-up threshold {thresholds['scale_up']}%",
                estimated_cost_impact=metrics.cost_per_unit * (magnitude - 1.0),
                estimated_performance_impact=0.2,
                urgency="high" if utilization > 90 else "medium"
            )
            
        elif utilization < thresholds['scale_down']:
            # Scale down recommendation
            magnitude = max(0.5, 1.0 - (thresholds['scale_down'] - utilization) / 100.0)
            confidence = min(0.8, (thresholds['scale_down'] - utilization) / 30.0)
            
            return ScalingRecommendation(
                resource_type=metrics.resource_type,
                direction=ScalingDirection.SCALE_DOWN,
                magnitude=magnitude,
                confidence=confidence,
                reasoning=f"Utilization {utilization:.1f}% below scale-down threshold {thresholds['scale_down']}%",
                estimated_cost_impact=-metrics.cost_per_unit * (1.0 - magnitude),
                estimated_performance_impact=-0.1,
                urgency="low"
            )
        
        return None
    
    def _get_predictive_recommendation(
        self, 
        metrics: ResourceMetrics
    ) -> Optional[ScalingRecommendation]:
        """Generate predictive scaling recommendation."""
        if not self.load_forecaster.is_trained:
            return None
        
        scaling_direction, confidence = self.load_forecaster.get_scaling_signal()
        
        if scaling_direction == ScalingDirection.MAINTAIN or confidence < 0.6:
            return None
        
        if scaling_direction == ScalingDirection.SCALE_UP:
            magnitude = 1.0 + confidence * 0.3  # Up to 30% increase
            cost_impact = metrics.cost_per_unit * (magnitude - 1.0)
            performance_impact = 0.15
        else:  # SCALE_DOWN
            magnitude = 1.0 - confidence * 0.2  # Up to 20% decrease
            cost_impact = -metrics.cost_per_unit * (1.0 - magnitude)
            performance_impact = -0.1
        
        return ScalingRecommendation(
            resource_type=metrics.resource_type,
            direction=scaling_direction,
            magnitude=magnitude,
            confidence=confidence,
            reasoning=f"Predictive model forecasts {scaling_direction.value} with {confidence:.1%} confidence",
            estimated_cost_impact=cost_impact,
            estimated_performance_impact=performance_impact,
            urgency="medium",
            metadata={'type': 'predictive', 'model_confidence': confidence}
        )
    
    def _should_execute_scaling(self, recommendation: ScalingRecommendation) -> bool:
        """Determine if a scaling recommendation should be executed."""
        resource_key = recommendation.resource_type.value
        
        # Check cooldown period
        last_scaling = self.cooldown_periods.get(resource_key, 0)
        if time.time() - last_scaling < self.min_cooldown:
            return False
        
        # Check confidence threshold
        if recommendation.confidence < 0.7:
            return False
        
        # Check if already scaling this resource
        if resource_key in self.active_scaling_actions:
            return False
        
        # Check cost constraints (simplified)
        if abs(recommendation.estimated_cost_impact) > 100.0:  # $100 threshold
            logger.warning(f"Scaling cost impact ${recommendation.estimated_cost_impact:.2f} exceeds threshold")
            return False
        
        return True
    
    def _execute_scaling_action(self, recommendation: ScalingRecommendation):
        """Execute a scaling action."""
        resource_key = recommendation.resource_type.value
        
        logger.info(f"Executing scaling action: {recommendation.direction.value} "
                   f"{recommendation.resource_type.value} by factor {recommendation.magnitude:.2f}")
        
        try:
            # Mark as active
            self.active_scaling_actions[resource_key] = {
                'recommendation': recommendation,
                'start_time': time.time()
            }
            
            # Simulate scaling action (in real implementation, this would call actual scaling APIs)
            self._simulate_scaling_action(recommendation)
            
            # Update cooldown
            self.cooldown_periods[resource_key] = time.time()
            
            # Record scaling event
            self.scaling_history.append({
                'timestamp': time.time(),
                'resource_type': recommendation.resource_type.value,
                'direction': recommendation.direction.value,
                'magnitude': recommendation.magnitude,
                'confidence': recommendation.confidence,
                'cost_impact': recommendation.estimated_cost_impact,
                'reasoning': recommendation.reasoning
            })
            
            logger.info(f"Successfully executed scaling action for {recommendation.resource_type.value}")
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
        
        finally:
            # Remove from active actions
            if resource_key in self.active_scaling_actions:
                del self.active_scaling_actions[resource_key]
    
    def _simulate_scaling_action(self, recommendation: ScalingRecommendation):
        """Simulate scaling action (placeholder for real implementation)."""
        # In real implementation, this would:
        # - Call cloud provider APIs to scale resources
        # - Update container orchestration (K8s, ECS, etc.)
        # - Adjust LLM provider rate limits
        # - Update load balancer configurations
        
        time.sleep(2)  # Simulate scaling time
        
        # Update resource metrics to reflect scaling
        if recommendation.resource_type.value in self.resource_metrics:
            metrics = self.resource_metrics[recommendation.resource_type.value]
            
            if recommendation.direction == ScalingDirection.SCALE_UP:
                metrics.capacity *= recommendation.magnitude
            else:
                metrics.capacity *= recommendation.magnitude
                metrics.current_usage *= recommendation.magnitude  # Assume usage scales proportionally
    
    def _evaluate_global_optimization(self):
        """Evaluate opportunities for global optimization."""
        if not self.global_optimizer.regions:
            return
        
        # Calculate current total workload
        total_workload = sum(
            metrics.current_usage for metrics in self.resource_metrics.values()
        ) / len(self.resource_metrics) if self.resource_metrics else 0
        
        if total_workload < 10:  # Skip optimization for low workload
            return
        
        # Get optimization plan
        try:
            plan = self.global_optimizer.optimize_global_allocation(
                total_workload,
                OptimizationObjective.BALANCED
            )
            
            # Evaluate if plan is worth implementing
            current_cost = self._estimate_current_cost()
            if plan.estimated_cost < current_cost * 0.9:  # 10% cost reduction
                logger.info(f"Global optimization opportunity: "
                           f"${current_cost:.2f} -> ${plan.estimated_cost:.2f}")
                # In real implementation, would queue plan for execution
            
        except Exception as e:
            logger.error(f"Error evaluating global optimization: {e}")
    
    def _estimate_current_cost(self) -> float:
        """Estimate current operational cost."""
        total_cost = 0.0
        for metrics in self.resource_metrics.values():
            total_cost += metrics.current_usage * metrics.cost_per_unit
        return total_cost
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling status."""
        recent_scalings = [
            s for s in self.scaling_history 
            if time.time() - s['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'autoscaling_enabled': self._scaler_running,
            'predictive_scaling_enabled': self.enable_predictive_scaling,
            'global_optimization_enabled': self.enable_global_optimization,
            'resource_metrics': {
                name: {
                    'utilization': metrics.utilization_ratio,
                    'efficiency': metrics.efficiency_score,
                    'cost_per_unit': metrics.cost_per_unit
                }
                for name, metrics in self.resource_metrics.items()
            },
            'active_scaling_actions': len(self.active_scaling_actions),
            'recent_scalings': len(recent_scalings),
            'scaling_directions': {
                direction: len([s for s in recent_scalings if s['direction'] == direction])
                for direction in ['scale_up', 'scale_down']
            },
            'forecasting_status': {
                'model_trained': self.load_forecaster.is_trained,
                'data_points': len(self.load_forecaster.load_history)
            },
            'estimated_current_cost': self._estimate_current_cost(),
            'cooldown_status': {
                resource: max(0, self.min_cooldown - (time.time() - last_time))
                for resource, last_time in self.cooldown_periods.items()
            }
        }
    
    def get_optimization_analytics(self) -> Dict[str, Any]:
        """Get optimization analytics and insights."""
        if not self.scaling_history:
            return {}
        
        df = pd.DataFrame(list(self.scaling_history))
        
        return {
            'total_scaling_events': len(self.scaling_history),
            'scaling_frequency': len(self.scaling_history) / max(1, 
                (time.time() - self.scaling_history[0]['timestamp']) / 3600),  # per hour
            'cost_impact_total': df['cost_impact'].sum(),
            'avg_confidence': df['confidence'].mean(),
            'resource_scaling_distribution': df['resource_type'].value_counts().to_dict(),
            'direction_distribution': df['direction'].value_counts().to_dict(),
            'predictive_vs_reactive': {
                'predictive': len([s for s in self.scaling_history 
                                if 'predictive' in s.get('reasoning', '').lower()]),
                'reactive': len([s for s in self.scaling_history 
                               if 'predictive' not in s.get('reasoning', '').lower()])
            }
        }


# Global auto-scaler instance
_global_autoscaler: Optional[IntelligentAutoScaler] = None


def get_global_autoscaler() -> IntelligentAutoScaler:
    """Get global auto-scaler instance."""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = IntelligentAutoScaler()
    return _global_autoscaler


def initialize_autoscaling(
    enable_predictive: bool = True,
    enable_global_opt: bool = True,
    start_immediately: bool = True
) -> IntelligentAutoScaler:
    """Initialize and optionally start intelligent auto-scaling."""
    global _global_autoscaler
    
    _global_autoscaler = IntelligentAutoScaler(
        enable_predictive_scaling=enable_predictive,
        enable_global_optimization=enable_global_opt
    )
    
    # Register some example regions for global optimization
    _global_autoscaler.global_optimizer.register_region(
        "us-east-1", {"total": 1000}, cost_multiplier=1.0, latency_to_users=30.0
    )
    _global_autoscaler.global_optimizer.register_region(
        "eu-west-1", {"total": 800}, cost_multiplier=1.2, latency_to_users=50.0
    )
    _global_autoscaler.global_optimizer.register_region(
        "ap-southeast-1", {"total": 600}, cost_multiplier=0.8, latency_to_users=80.0
    )
    
    if start_immediately:
        _global_autoscaler.start_autoscaling()
    
    logger.info("Initialized intelligent auto-scaling system")
    return _global_autoscaler


if __name__ == "__main__":
    # Demo intelligent auto-scaling
    autoscaler = initialize_autoscaling()
    
    try:
        # Run for demonstration
        time.sleep(60)
        
        # Print status
        status = autoscaler.get_scaling_status()
        print("Auto-Scaling Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Print analytics
        analytics = autoscaler.get_optimization_analytics()
        print("\nOptimization Analytics:")
        print(json.dumps(analytics, indent=2, default=str))
        
    finally:
        autoscaler.stop_autoscaling()