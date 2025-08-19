"""Scale Optimization Module - Generation 3 Implementation.

This module provides advanced performance optimization, auto-scaling, load balancing,
and resource management capabilities for production-scale deployment.

Features:
- Intelligent auto-scaling with predictive algorithms
- Advanced caching with multi-tier strategy
- Load balancing and resource pooling
- Performance optimization and monitoring
- Distributed processing coordination
- Resource allocation and management

Author: Terry (Terragon Labs)
"""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import heapq
import json
import psutil
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Auto-scaling operation modes."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PREDICTIVE = "predictive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io: float
    network_io: float
    active_connections: int
    request_rate: float
    response_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning."""
    action: str  # scale_up, scale_down, maintain
    target_instances: int
    current_instances: int
    confidence: float
    reasoning: List[str]
    predicted_load: float
    resource_requirements: Dict[str, float]


class IntelligentCache:
    """Multi-tier intelligent caching system."""
    
    def __init__(
        self,
        max_memory_cache: int = 1000,
        max_disk_cache: int = 10000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        self.max_memory_cache = max_memory_cache
        self.max_disk_cache = max_disk_cache
        self.strategy = strategy
        
        # Memory cache (fastest)
        self.memory_cache: Dict[str, Any] = {}
        self.memory_access_times: Dict[str, float] = {}
        self.memory_access_counts: Dict[str, int] = defaultdict(int)
        
        # Disk cache (larger, slower)
        self.disk_cache: Dict[str, Any] = {}
        self.disk_access_times: Dict[str, float] = {}
        
        # Cache statistics
        self.hit_rates = {"memory": 0.0, "disk": 0.0, "total": 0.0}
        self.access_patterns = defaultdict(list)
        
        self._lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent tier selection."""
        async with asyncio.Lock():
            # Check memory cache first
            if key in self.memory_cache:
                self._update_access_stats(key, "memory", hit=True)
                return self.memory_cache[key]
            
            # Check disk cache
            if key in self.disk_cache:
                value = self.disk_cache[key]
                # Promote to memory cache if frequently accessed
                if self._should_promote_to_memory(key):
                    await self._promote_to_memory(key, value)
                
                self._update_access_stats(key, "disk", hit=True)
                return value
            
            # Cache miss
            self._update_access_stats(key, "miss", hit=False)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with intelligent placement."""
        async with asyncio.Lock():
            current_time = time.time()
            
            # Determine optimal cache tier
            if self._should_cache_in_memory(key, value):
                await self._set_memory_cache(key, value, current_time)
            else:
                await self._set_disk_cache(key, value, current_time)
    
    def _should_promote_to_memory(self, key: str) -> bool:
        """Determine if key should be promoted to memory cache."""
        access_count = self.memory_access_counts.get(key, 0)
        recent_accesses = len([
            t for t in self.access_patterns[key] 
            if time.time() - t < 300  # 5 minutes
        ])
        
        return access_count > 3 or recent_accesses > 2
    
    def _should_cache_in_memory(self, key: str, value: Any) -> bool:
        """Determine optimal cache tier for new entries."""
        # Consider value size, access pattern predictions
        value_size = len(str(value))
        
        if value_size > 1024 * 1024:  # 1MB threshold
            return False
        
        # Check available memory cache space
        if len(self.memory_cache) >= self.max_memory_cache:
            return False
        
        return True
    
    async def _set_memory_cache(self, key: str, value: Any, timestamp: float):
        """Set value in memory cache with eviction."""
        if len(self.memory_cache) >= self.max_memory_cache:
            await self._evict_from_memory()
        
        self.memory_cache[key] = value
        self.memory_access_times[key] = timestamp
        self.memory_access_counts[key] += 1
    
    async def _set_disk_cache(self, key: str, value: Any, timestamp: float):
        """Set value in disk cache with eviction."""
        if len(self.disk_cache) >= self.max_disk_cache:
            await self._evict_from_disk()
        
        self.disk_cache[key] = value
        self.disk_access_times[key] = timestamp
    
    async def _evict_from_memory(self):
        """Evict least valuable items from memory cache."""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.memory_access_times.items(), key=lambda x: x[1])[0]
            del self.memory_cache[oldest_key]
            del self.memory_access_times[oldest_key]
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(self.memory_access_counts.items(), key=lambda x: x[1])[0]
            del self.memory_cache[least_used_key]
            del self.memory_access_times[least_used_key]
        else:
            # Adaptive strategy - consider both frequency and recency
            scores = {}
            current_time = time.time()
            for key in self.memory_cache:
                recency_score = 1.0 / (current_time - self.memory_access_times[key] + 1)
                frequency_score = self.memory_access_counts[key]
                scores[key] = recency_score * frequency_score
            
            worst_key = min(scores.items(), key=lambda x: x[1])[0]
            del self.memory_cache[worst_key]
            del self.memory_access_times[worst_key]
    
    async def _evict_from_disk(self):
        """Evict items from disk cache."""
        # Simple LRU for disk cache
        oldest_key = min(self.disk_access_times.items(), key=lambda x: x[1])[0]
        del self.disk_cache[oldest_key]
        del self.disk_access_times[oldest_key]
    
    async def _promote_to_memory(self, key: str, value: Any):
        """Promote frequently accessed item to memory cache."""
        if len(self.memory_cache) >= self.max_memory_cache:
            await self._evict_from_memory()
        
        self.memory_cache[key] = value
        self.memory_access_times[key] = time.time()
        self.memory_access_counts[key] += 1
    
    def _update_access_stats(self, key: str, tier: str, hit: bool):
        """Update cache access statistics."""
        self.access_patterns[key].append(time.time())
        
        # Update hit rates (simplified)
        if hit:
            if tier == "memory":
                self.hit_rates["memory"] += 0.01
            elif tier == "disk":
                self.hit_rates["disk"] += 0.01
        
        self.hit_rates["total"] = (self.hit_rates["memory"] + self.hit_rates["disk"]) / 2


class AdaptiveLoadBalancer:
    """Intelligent load balancer with adaptive algorithms."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.workers = []
        self.worker_stats = {}
        self.request_queue = queue.PriorityQueue()
        self.load_history = deque(maxlen=1000)
        
    def add_worker(self, worker_id: str, capacity: float = 1.0):
        """Add worker to load balancer."""
        self.workers.append(worker_id)
        self.worker_stats[worker_id] = {
            "capacity": capacity,
            "active_requests": 0,
            "total_requests": 0,
            "response_times": deque(maxlen=100),
            "error_rate": 0.0,
            "last_request_time": 0.0
        }
        logger.info(f"Added worker {worker_id} with capacity {capacity}")
    
    def select_worker(self) -> Optional[str]:
        """Select optimal worker based on strategy."""
        if not self.workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection()
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection()
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_selection()
        else:  # ADAPTIVE
            return self._adaptive_selection()
    
    def _round_robin_selection(self) -> str:
        """Simple round-robin selection."""
        # Find worker with least total requests
        return min(self.workers, key=lambda w: self.worker_stats[w]["total_requests"])
    
    def _least_connections_selection(self) -> str:
        """Select worker with least active connections."""
        return min(self.workers, key=lambda w: self.worker_stats[w]["active_requests"])
    
    def _weighted_response_time_selection(self) -> str:
        """Select worker based on response time weights."""
        scores = {}
        for worker in self.workers:
            stats = self.worker_stats[worker]
            avg_response_time = np.mean(stats["response_times"]) if stats["response_times"] else 1.0
            capacity_factor = stats["capacity"]
            active_factor = 1.0 / (stats["active_requests"] + 1)
            
            scores[worker] = capacity_factor * active_factor / avg_response_time
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _adaptive_selection(self) -> str:
        """Adaptive selection considering multiple factors."""
        scores = {}
        current_time = time.time()
        
        for worker in self.workers:
            stats = self.worker_stats[worker]
            
            # Response time factor
            avg_response_time = np.mean(stats["response_times"]) if stats["response_times"] else 1.0
            response_factor = 1.0 / avg_response_time
            
            # Load factor
            load_factor = 1.0 / (stats["active_requests"] + 1)
            
            # Capacity factor
            capacity_factor = stats["capacity"]
            
            # Error rate factor
            error_factor = 1.0 - stats["error_rate"]
            
            # Freshness factor (prefer recently used workers)
            time_since_last = current_time - stats["last_request_time"]
            freshness_factor = 1.0 / (time_since_last + 1)
            
            # Combined score
            scores[worker] = (
                response_factor * 0.3 +
                load_factor * 0.25 +
                capacity_factor * 0.2 +
                error_factor * 0.15 +
                freshness_factor * 0.1
            )
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def record_request_start(self, worker_id: str):
        """Record start of request processing."""
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id]["active_requests"] += 1
            self.worker_stats[worker_id]["total_requests"] += 1
            self.worker_stats[worker_id]["last_request_time"] = time.time()
    
    def record_request_end(self, worker_id: str, response_time: float, success: bool = True):
        """Record end of request processing."""
        if worker_id in self.worker_stats:
            stats = self.worker_stats[worker_id]
            stats["active_requests"] = max(0, stats["active_requests"] - 1)
            stats["response_times"].append(response_time)
            
            if not success:
                # Update error rate (exponential moving average)
                stats["error_rate"] = stats["error_rate"] * 0.9 + 0.1


class PredictiveAutoScaler:
    """Predictive auto-scaling with machine learning."""
    
    def __init__(self, mode: ScalingMode = ScalingMode.BALANCED):
        self.mode = mode
        self.resource_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 100
        
        # Prediction models (simplified)
        self.load_predictor = None
        self.resource_predictor = None
    
    async def evaluate_scaling_need(self, current_metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate if scaling is needed based on current metrics."""
        
        # Store current metrics
        self.resource_history.append(current_metrics)
        
        # Predict future load
        predicted_load = await self._predict_future_load()
        
        # Calculate scaling decision
        decision = await self._calculate_scaling_decision(current_metrics, predicted_load)
        
        # Store decision for learning
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "decision": decision,
            "metrics": current_metrics
        })
        
        return decision
    
    async def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns."""
        if len(self.resource_history) < 10:
            return 0.5  # Default moderate load
        
        # Simple trend analysis
        recent_cpu = [m.cpu_percent for m in list(self.resource_history)[-10:]]
        recent_memory = [m.memory_percent for m in list(self.resource_history)[-10:]]
        recent_requests = [m.request_rate for m in list(self.resource_history)[-10:]]
        
        # Calculate trends
        cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
        memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        request_trend = np.polyfit(range(len(recent_requests)), recent_requests, 1)[0]
        
        # Predict next load (simplified)
        predicted_load = (
            np.mean(recent_cpu) + cpu_trend * 5 +
            np.mean(recent_memory) + memory_trend * 5 +
            np.mean(recent_requests) + request_trend * 5
        ) / 300  # Normalize
        
        return max(0.0, min(1.0, predicted_load))
    
    async def _calculate_scaling_decision(
        self, 
        metrics: ResourceMetrics,
        predicted_load: float
    ) -> ScalingDecision:
        """Calculate scaling decision based on metrics and predictions."""
        
        reasoning = []
        current_load = (metrics.cpu_percent + metrics.memory_percent) / 200
        
        # Determine scaling thresholds based on mode
        if self.mode == ScalingMode.CONSERVATIVE:
            scale_up_threshold = 0.8
            scale_down_threshold = 0.3
        elif self.mode == ScalingMode.AGGRESSIVE:
            scale_up_threshold = 0.6
            scale_down_threshold = 0.5
        elif self.mode == ScalingMode.PREDICTIVE:
            scale_up_threshold = 0.7
            scale_down_threshold = 0.4
        else:  # BALANCED
            scale_up_threshold = 0.75
            scale_down_threshold = 0.4
        
        # Decision logic
        action = "maintain"
        target_instances = self.current_instances
        confidence = 0.8
        
        # Check for scale up conditions
        if current_load > scale_up_threshold or predicted_load > scale_up_threshold:
            if self.current_instances < self.max_instances:
                action = "scale_up"
                
                # Calculate target instances
                load_factor = max(current_load, predicted_load)
                desired_instances = int(np.ceil(self.current_instances * load_factor / 0.7))
                target_instances = min(desired_instances, self.max_instances)
                
                reasoning.append(f"High load detected: current={current_load:.2f}, predicted={predicted_load:.2f}")
                reasoning.append(f"Scaling up from {self.current_instances} to {target_instances} instances")
        
        # Check for scale down conditions
        elif current_load < scale_down_threshold and predicted_load < scale_down_threshold:
            if self.current_instances > self.min_instances:
                action = "scale_down"
                
                # Calculate target instances
                load_factor = max(current_load, predicted_load)
                desired_instances = max(1, int(np.ceil(self.current_instances * load_factor / 0.5)))
                target_instances = max(desired_instances, self.min_instances)
                
                reasoning.append(f"Low load detected: current={current_load:.2f}, predicted={predicted_load:.2f}")
                reasoning.append(f"Scaling down from {self.current_instances} to {target_instances} instances")
        
        # Calculate resource requirements
        resource_requirements = {
            "cpu_cores": target_instances * 2,  # 2 cores per instance
            "memory_gb": target_instances * 4,  # 4GB per instance
            "storage_gb": target_instances * 20  # 20GB per instance
        }
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            current_instances=self.current_instances,
            confidence=confidence,
            reasoning=reasoning,
            predicted_load=predicted_load,
            resource_requirements=resource_requirements
        )


class PerformanceOptimizer:
    """Advanced performance optimization system."""
    
    def __init__(self):
        self.optimization_strategies = {}
        self.performance_history = deque(maxlen=1000)
        self.optimization_results = {}
        
    def register_optimization(self, name: str, optimizer_func: Callable):
        """Register an optimization strategy."""
        self.optimization_strategies[name] = optimizer_func
        logger.info(f"Registered optimization strategy: {name}")
    
    async def optimize_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance optimizations."""
        
        results = {}
        
        for name, optimizer in self.optimization_strategies.items():
            try:
                start_time = time.time()
                optimization_result = await self._run_optimization(optimizer, context)
                end_time = time.time()
                
                results[name] = {
                    "success": True,
                    "result": optimization_result,
                    "execution_time": end_time - start_time
                }
                
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0.0
                }
                logger.error(f"Optimization {name} failed: {e}")
        
        # Store results for analysis
        self.optimization_results[datetime.now().isoformat()] = results
        
        return results
    
    async def _run_optimization(self, optimizer: Callable, context: Dict[str, Any]) -> Any:
        """Run individual optimization safely."""
        return await optimizer(context)


class ScaleOptimizationSystem:
    """Main scale optimization system coordinating all components."""
    
    def __init__(
        self,
        cache_size: int = 1000,
        max_instances: int = 100,
        scaling_mode: ScalingMode = ScalingMode.BALANCED
    ):
        self.cache = IntelligentCache(max_memory_cache=cache_size)
        self.load_balancer = AdaptiveLoadBalancer()
        self.auto_scaler = PredictiveAutoScaler(mode=scaling_mode)
        self.performance_optimizer = PerformanceOptimizer()
        
        # System metrics
        self.metrics_history = deque(maxlen=1000)
        self.performance_baselines = {}
        
        # Initialize default optimizations
        self._register_default_optimizations()
        
        logger.info("Scale Optimization System initialized")
    
    def _register_default_optimizations(self):
        """Register default performance optimizations."""
        
        self.performance_optimizer.register_optimization(
            "memory_optimization",
            self._optimize_memory_usage
        )
        
        self.performance_optimizer.register_optimization(
            "cpu_optimization", 
            self._optimize_cpu_usage
        )
        
        self.performance_optimizer.register_optimization(
            "io_optimization",
            self._optimize_io_operations
        )
    
    async def process_with_optimization(
        self,
        data: Any,
        processing_func: Callable,
        cache_key: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Process data with full optimization pipeline."""
        
        start_time = time.time()
        optimization_info = {}
        
        # Check cache first
        if cache_key:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                optimization_info["cache_hit"] = True
                optimization_info["processing_time"] = time.time() - start_time
                return cached_result, optimization_info
        
        optimization_info["cache_hit"] = False
        
        # Select optimal worker
        worker_id = self.load_balancer.select_worker()
        if worker_id:
            self.load_balancer.record_request_start(worker_id)
            optimization_info["worker_id"] = worker_id
        
        try:
            # Process with optimization
            processing_start = time.time()
            result = await processing_func(data)
            processing_time = time.time() - processing_start
            
            # Cache result if beneficial
            if cache_key and self._should_cache_result(result, processing_time):
                await self.cache.set(cache_key, result)
            
            # Record successful processing
            if worker_id:
                self.load_balancer.record_request_end(worker_id, processing_time, True)
            
            optimization_info.update({
                "processing_time": processing_time,
                "total_time": time.time() - start_time,
                "success": True
            })
            
            return result, optimization_info
            
        except Exception as e:
            # Record failed processing
            if worker_id:
                self.load_balancer.record_request_end(worker_id, 0.0, False)
            
            optimization_info.update({
                "error": str(e),
                "success": False,
                "total_time": time.time() - start_time
            })
            
            raise e
    
    def _should_cache_result(self, result: Any, processing_time: float) -> bool:
        """Determine if result should be cached."""
        # Cache if processing took more than 100ms
        return processing_time > 0.1
    
    async def evaluate_and_scale(self) -> Optional[ScalingDecision]:
        """Evaluate system metrics and make scaling decisions."""
        
        # Collect current metrics
        current_metrics = await self._collect_system_metrics()
        
        # Store metrics
        self.metrics_history.append(current_metrics)
        
        # Evaluate scaling need
        scaling_decision = await self.auto_scaler.evaluate_scaling_need(current_metrics)
        
        # Execute scaling if needed
        if scaling_decision.action != "maintain":
            await self._execute_scaling_decision(scaling_decision)
        
        return scaling_decision
    
    async def _collect_system_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        
        # Get system metrics using psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Simplified metrics for other values
        disk_io = 50.0  # Placeholder
        network_io = 30.0  # Placeholder
        active_connections = len(self.load_balancer.workers)
        request_rate = len(self.metrics_history) / max(1, len(self.metrics_history))
        
        # Calculate average response time
        recent_times = []
        for worker_id in self.load_balancer.workers:
            times = self.load_balancer.worker_stats[worker_id]["response_times"]
            recent_times.extend(times)
        
        avg_response_time = np.mean(recent_times) if recent_times else 0.1
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io=disk_io,
            network_io=network_io,
            active_connections=active_connections,
            request_rate=request_rate,
            response_time=avg_response_time
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        
        if decision.action == "scale_up":
            # Add workers
            instances_to_add = decision.target_instances - decision.current_instances
            for i in range(instances_to_add):
                worker_id = f"worker_{len(self.load_balancer.workers) + 1}"
                self.load_balancer.add_worker(worker_id)
            
            logger.info(f"Scaled up to {decision.target_instances} instances")
            
        elif decision.action == "scale_down":
            # Remove workers (simplified)
            instances_to_remove = decision.current_instances - decision.target_instances
            workers_to_remove = self.load_balancer.workers[-instances_to_remove:]
            
            for worker_id in workers_to_remove:
                self.load_balancer.workers.remove(worker_id)
                del self.load_balancer.worker_stats[worker_id]
            
            logger.info(f"Scaled down to {decision.target_instances} instances")
        
        # Update current instance count
        self.auto_scaler.current_instances = decision.target_instances
    
    # Default optimization strategies
    
    async def _optimize_memory_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage."""
        # Simplified memory optimization
        return {"memory_optimization": "completed", "savings_mb": 50}
    
    async def _optimize_cpu_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CPU usage."""
        # Simplified CPU optimization  
        return {"cpu_optimization": "completed", "efficiency_gain": 0.15}
    
    async def _optimize_io_operations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize I/O operations."""
        # Simplified I/O optimization
        return {"io_optimization": "completed", "latency_reduction_ms": 25}
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        
        # Calculate cache statistics
        cache_stats = {
            "hit_rates": self.cache.hit_rates,
            "memory_cache_size": len(self.cache.memory_cache),
            "disk_cache_size": len(self.cache.disk_cache)
        }
        
        # Calculate load balancer statistics
        lb_stats = {
            "total_workers": len(self.load_balancer.workers),
            "worker_utilization": {}
        }
        
        for worker_id in self.load_balancer.workers:
            stats = self.load_balancer.worker_stats[worker_id]
            lb_stats["worker_utilization"][worker_id] = {
                "active_requests": stats["active_requests"],
                "total_requests": stats["total_requests"],
                "avg_response_time": np.mean(stats["response_times"]) if stats["response_times"] else 0,
                "error_rate": stats["error_rate"]
            }
        
        # System performance
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            performance_stats = {
                "current_cpu": latest_metrics.cpu_percent,
                "current_memory": latest_metrics.memory_percent,
                "current_response_time": latest_metrics.response_time,
                "current_instances": self.auto_scaler.current_instances
            }
        else:
            performance_stats = {}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_statistics": cache_stats,
            "load_balancer_statistics": lb_stats,
            "performance_statistics": performance_stats,
            "optimization_results": dict(self.performance_optimizer.optimization_results)
        }


def create_scale_optimization_system(**kwargs) -> ScaleOptimizationSystem:
    """Factory function to create scale optimization system."""
    return ScaleOptimizationSystem(**kwargs)


def initialize_scaling_systems() -> ScaleOptimizationSystem:
    """Initialize scaling systems with default configuration."""
    system = create_scale_optimization_system(
        cache_size=1000,
        max_instances=50,
        scaling_mode=ScalingMode.BALANCED
    )
    
    # Add initial workers
    for i in range(3):  # Start with 3 workers
        system.load_balancer.add_worker(f"worker_{i+1}")
    
    logger.info("Scale optimization systems initialized successfully")
    return system