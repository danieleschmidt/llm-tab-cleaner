"""
HyperScale Performance Engine - Generation 4 SDLC Implementation
Ultra-high performance optimization with adaptive scaling and intelligent resource management
"""

import logging
import asyncio
import time
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import multiprocessing as mp
import numpy as np
import psutil
from functools import lru_cache, wraps
import redis
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    throughput: float          # requests/sec
    latency_p50: float        # milliseconds
    latency_p95: float        # milliseconds  
    latency_p99: float        # milliseconds
    cpu_utilization: float    # percentage
    memory_usage: float       # MB
    cache_hit_rate: float     # percentage
    error_rate: float         # percentage
    concurrent_connections: int
    queue_depth: int
    resource_efficiency: float

@dataclass
class ScalingDecision:
    """Auto-scaling decision record."""
    timestamp: float
    decision_type: str        # SCALE_UP, SCALE_DOWN, MAINTAIN
    trigger_metric: str
    current_value: float
    threshold: float
    target_instances: int
    confidence: float
    estimated_impact: Dict[str, float]

@dataclass
class CacheStrategy:
    """Intelligent caching strategy configuration."""
    strategy_name: str
    hit_rate_target: float
    eviction_policy: str      # LRU, LFU, ADAPTIVE, PREDICTIVE
    ttl_seconds: int
    max_memory_mb: int
    prefetch_enabled: bool
    compression_enabled: bool

class AdaptiveCacheManager:
    """Intelligent multi-level caching system."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = None  # Redis cache
        self.cache_stats = defaultdict(int)
        self.access_patterns = defaultdict(deque)
        self.prefetch_queue = asyncio.Queue()
        self.cache_lock = threading.RLock()
        
        # Initialize Redis connection if available
        if redis_url:
            try:
                import redis as redis_lib
                self.l2_cache = redis_lib.from_url(redis_url)
                self.l2_cache.ping()
                logger.info("Redis L2 cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.l2_cache = None
        
        # Cache configuration
        self.l1_max_size = 10000
        self.l2_max_size = 100000
        self.ttl_default = 3600  # 1 hour
        self.compression_threshold = 1024  # bytes
        
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from multi-level cache."""
        
        cache_key = f"{namespace}:{key}"
        
        # Try L1 cache first
        with self.cache_lock:
            if cache_key in self.l1_cache:
                item = self.l1_cache[cache_key]
                if time.time() < item['expires']:
                    self.cache_stats['l1_hits'] += 1
                    self._record_access_pattern(cache_key)
                    return self._decompress_if_needed(item['data'])
                else:
                    # Expired, remove from L1
                    del self.l1_cache[cache_key]
        
        # Try L2 cache (Redis)
        if self.l2_cache:
            try:
                cached_data = self.l2_cache.get(cache_key)
                if cached_data:
                    self.cache_stats['l2_hits'] += 1
                    self._record_access_pattern(cache_key)
                    
                    # Promote to L1 cache
                    data = json.loads(cached_data)
                    await self._set_l1_cache(cache_key, data, self.ttl_default)
                    
                    return data
            except Exception as e:
                logger.error(f"L2 cache error: {e}")
        
        # Cache miss
        self.cache_stats['misses'] += 1
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        namespace: str = "default"
    ):
        """Set value in multi-level cache."""
        
        cache_key = f"{namespace}:{key}"
        ttl = ttl or self.ttl_default
        
        # Set in L1 cache
        await self._set_l1_cache(cache_key, value, ttl)
        
        # Set in L2 cache (Redis)
        if self.l2_cache:
            try:
                serialized_data = json.dumps(value)
                self.l2_cache.setex(cache_key, ttl, serialized_data)
                self.cache_stats['l2_sets'] += 1
            except Exception as e:
                logger.error(f"L2 cache set error: {e}")
        
        self._record_access_pattern(cache_key)
    
    async def _set_l1_cache(self, key: str, value: Any, ttl: int):
        """Set value in L1 cache with compression and eviction."""
        
        with self.cache_lock:
            # Evict if at capacity
            if len(self.l1_cache) >= self.l1_max_size:
                await self._evict_l1_cache()
            
            # Compress if needed
            compressed_data = self._compress_if_needed(value)
            
            self.l1_cache[key] = {
                'data': compressed_data,
                'expires': time.time() + ttl,
                'access_count': 1,
                'last_access': time.time()
            }
            
            self.cache_stats['l1_sets'] += 1
    
    async def _evict_l1_cache(self):
        """Intelligent cache eviction using adaptive strategy."""
        
        # Get eviction candidates
        current_time = time.time()
        candidates = []
        
        for key, item in self.l1_cache.items():
            # Skip if recently accessed
            if current_time - item['last_access'] < 60:
                continue
            
            # Calculate eviction score (lower = more likely to evict)
            age_factor = current_time - item['last_access']
            access_factor = 1.0 / max(1, item['access_count'])
            size_factor = len(str(item['data'])) / 1024  # KB
            
            eviction_score = age_factor * access_factor * (1 + size_factor)
            candidates.append((eviction_score, key))
        
        # Evict worst 10% or at least 1 item
        num_to_evict = max(1, len(candidates) // 10)
        candidates.sort(reverse=True)  # Highest scores first
        
        for _, key in candidates[:num_to_evict]:
            del self.l1_cache[key]
            self.cache_stats['l1_evictions'] += 1
    
    def _compress_if_needed(self, data: Any) -> Any:
        """Compress data if it exceeds threshold."""
        
        serialized = json.dumps(data) if not isinstance(data, str) else data
        
        if len(serialized.encode()) > self.compression_threshold:
            try:
                import gzip
                compressed = gzip.compress(serialized.encode())
                return {'compressed': True, 'data': compressed}
            except Exception as e:
                logger.error(f"Compression failed: {e}")
                return data
        
        return data
    
    def _decompress_if_needed(self, data: Any) -> Any:
        """Decompress data if compressed."""
        
        if isinstance(data, dict) and data.get('compressed'):
            try:
                import gzip
                decompressed = gzip.decompress(data['data']).decode()
                return json.loads(decompressed)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                return data
        
        return data
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for predictive caching."""
        
        pattern_key = key.split(':')[0] if ':' in key else key
        self.access_patterns[pattern_key].append(time.time())
        
        # Keep only recent accesses (last hour)
        cutoff_time = time.time() - 3600
        while (self.access_patterns[pattern_key] and 
               self.access_patterns[pattern_key][0] < cutoff_time):
            self.access_patterns[pattern_key].popleft()
    
    async def prefetch_predictions(self):
        """Predictive prefetching based on access patterns."""
        
        current_time = time.time()
        
        for pattern_key, accesses in self.access_patterns.items():
            if len(accesses) < 5:  # Need minimum pattern data
                continue
            
            # Calculate access frequency
            recent_accesses = [a for a in accesses if current_time - a < 1800]  # 30 min
            if len(recent_accesses) < 3:
                continue
            
            # Predict next access time
            intervals = [recent_accesses[i] - recent_accesses[i-1] 
                        for i in range(1, len(recent_accesses))]
            avg_interval = sum(intervals) / len(intervals)
            
            # If pattern suggests access soon, trigger prefetch
            if avg_interval < 300:  # Less than 5 minutes
                predicted_next = recent_accesses[-1] + avg_interval
                if predicted_next - current_time < 60:  # Predict access in 1 minute
                    await self._trigger_prefetch(pattern_key)
    
    async def _trigger_prefetch(self, pattern_key: str):
        """Trigger prefetch for predicted access pattern."""
        
        logger.debug(f"Triggering prefetch for pattern: {pattern_key}")
        # In production, this would prefetch related data
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        total_requests = (self.cache_stats['l1_hits'] + 
                         self.cache_stats['l2_hits'] + 
                         self.cache_stats['misses'])
        
        if total_requests == 0:
            hit_rate = 0.0
        else:
            hit_rate = ((self.cache_stats['l1_hits'] + self.cache_stats['l2_hits']) / 
                       total_requests * 100)
        
        return {
            'l1_size': len(self.l1_cache),
            'l1_hits': self.cache_stats['l1_hits'],
            'l1_sets': self.cache_stats['l1_sets'],
            'l1_evictions': self.cache_stats['l1_evictions'],
            'l2_hits': self.cache_stats['l2_hits'],
            'l2_sets': self.cache_stats['l2_sets'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'access_patterns': len(self.access_patterns)
        }

class IntelligentLoadBalancer:
    """AI-powered load balancing with predictive scaling."""
    
    def __init__(self, initial_workers: int = 4):
        self.workers = []
        self.worker_stats = {}
        self.request_queue = asyncio.Queue()
        self.target_workers = initial_workers
        self.min_workers = 1
        self.max_workers = multiprocessing.cpu_count() * 2
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        
        # Load balancing algorithms
        self.lb_algorithms = {
            'round_robin': self._round_robin_balance,
            'least_connections': self._least_connections_balance,
            'weighted_response': self._weighted_response_balance,
            'predictive': self._predictive_balance
        }
        
        self.current_algorithm = 'predictive'
        self.worker_index = 0
        
    async def initialize_workers(self):
        """Initialize worker pool with intelligent sizing."""
        
        # Calculate optimal initial worker count
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Heuristic for optimal worker count
        optimal_workers = min(
            self.max_workers,
            max(
                self.min_workers,
                int(cpu_count * 1.5),  # CPU-based scaling
                int(memory_gb / 2)     # Memory-based scaling (assuming 2GB per worker)
            )
        )
        
        self.target_workers = optimal_workers
        
        # Start initial workers
        for i in range(self.target_workers):
            await self._start_worker(f"worker_{i}")
        
        logger.info(f"Initialized {self.target_workers} workers")
    
    async def _start_worker(self, worker_id: str):
        """Start a new worker process/thread."""
        
        # In production, this would start actual worker processes
        # For demo, we simulate worker creation
        
        worker = {
            'id': worker_id,
            'status': 'active',
            'created_at': time.time(),
            'requests_handled': 0,
            'avg_response_time': 0.0,
            'current_connections': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        }
        
        self.workers.append(worker)
        self.worker_stats[worker_id] = {
            'response_times': deque(maxlen=100),
            'error_count': 0,
            'success_count': 0
        }
        
        logger.debug(f"Worker started: {worker_id}")
    
    async def _stop_worker(self, worker_id: str):
        """Gracefully stop a worker."""
        
        # Find and remove worker
        self.workers = [w for w in self.workers if w['id'] != worker_id]
        if worker_id in self.worker_stats:
            del self.worker_stats[worker_id]
        
        logger.debug(f"Worker stopped: {worker_id}")
    
    async def distribute_request(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Distribute request using intelligent load balancing."""
        
        if not self.workers:
            raise RuntimeError("No workers available")
        
        # Select worker using current algorithm
        selected_worker = await self.lb_algorithms[self.current_algorithm](request_data)
        
        # Process request (simulated)
        start_time = time.time()
        result = await self._process_request_on_worker(selected_worker, request_data)
        processing_time = time.time() - start_time
        
        # Update worker stats
        self._update_worker_stats(selected_worker['id'], processing_time, result['success'])
        
        # Record performance metrics
        self.performance_history.append(PerformanceMetrics(
            timestamp=time.time(),
            throughput=self._calculate_current_throughput(),
            latency_p50=self._calculate_percentile(50),
            latency_p95=self._calculate_percentile(95),
            latency_p99=self._calculate_percentile(99),
            cpu_utilization=self._get_average_cpu_usage(),
            memory_usage=self._get_total_memory_usage(),
            cache_hit_rate=85.0,  # Would be actual cache hit rate
            error_rate=self._calculate_error_rate(),
            concurrent_connections=sum(w['current_connections'] for w in self.workers),
            queue_depth=self.request_queue.qsize(),
            resource_efficiency=self._calculate_resource_efficiency()
        ))
        
        # Auto-scaling decision
        await self._evaluate_scaling()
        
        return result
    
    async def _round_robin_balance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Round-robin load balancing."""
        
        worker = self.workers[self.worker_index % len(self.workers)]
        self.worker_index += 1
        return worker
    
    async def _least_connections_balance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Least connections load balancing."""
        
        return min(self.workers, key=lambda w: w['current_connections'])
    
    async def _weighted_response_balance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted response time load balancing."""
        
        # Calculate weights based on inverse response time
        weights = []
        for worker in self.workers:
            avg_response = worker['avg_response_time'] or 0.1
            weight = 1.0 / avg_response
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return self.workers[0]
        
        # Weighted random selection
        import random
        random_val = random.random() * total_weight
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_val <= cumulative_weight:
                return self.workers[i]
        
        return self.workers[-1]
    
    async def _predictive_balance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predictive load balancing using machine learning insights."""
        
        # Analyze request characteristics
        request_complexity = self._analyze_request_complexity(request_data)
        
        # Score workers based on multiple factors
        worker_scores = []
        for worker in self.workers:
            # Base score from current performance
            base_score = 1.0
            
            # Penalize high current load
            connection_penalty = worker['current_connections'] * 0.1
            
            # Penalize slow response times
            response_penalty = worker['avg_response_time'] * 0.5
            
            # Penalize high resource usage
            resource_penalty = (worker['cpu_usage'] + worker['memory_usage']) * 0.01
            
            # Bonus for successful handling
            success_rate = (self.worker_stats[worker['id']]['success_count'] / 
                          max(1, self.worker_stats[worker['id']]['success_count'] + 
                              self.worker_stats[worker['id']]['error_count']))
            success_bonus = success_rate * 0.3
            
            # Calculate final score
            final_score = (base_score - connection_penalty - response_penalty - 
                          resource_penalty + success_bonus)
            
            worker_scores.append((final_score, worker))
        
        # Select worker with highest score
        best_worker = max(worker_scores, key=lambda x: x[0])[1]
        return best_worker
    
    def _analyze_request_complexity(self, request_data: Dict[str, Any]) -> float:
        """Analyze request complexity for predictive balancing."""
        
        complexity_score = 0.0
        
        # Data size factor
        data_size = len(str(request_data))
        complexity_score += min(1.0, data_size / 10000)  # Normalize to 0-1
        
        # Request type factor
        request_type = request_data.get('type', 'simple')
        type_complexity = {
            'simple': 0.1,
            'medium': 0.5,
            'complex': 0.9,
            'batch': 1.0
        }
        complexity_score += type_complexity.get(request_type, 0.5)
        
        # Processing requirements
        if request_data.get('requires_ml', False):
            complexity_score += 0.5
        if request_data.get('requires_io', False):
            complexity_score += 0.3
        
        return min(1.0, complexity_score)
    
    async def _process_request_on_worker(
        self, 
        worker: Dict[str, Any], 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request on selected worker (simulated)."""
        
        worker['current_connections'] += 1
        
        try:
            # Simulate processing time based on request complexity
            complexity = self._analyze_request_complexity(request_data)
            base_time = 0.1 + complexity * 0.5  # 0.1 to 0.6 seconds
            
            # Add worker-specific variation
            worker_efficiency = 1.0 - (worker['avg_response_time'] * 0.1)
            processing_time = base_time / max(0.1, worker_efficiency)
            
            await asyncio.sleep(processing_time)
            
            # Simulate success/failure
            success_rate = 0.95 - complexity * 0.05  # Higher complexity = more failures
            success = np.random.random() < success_rate
            
            worker['requests_handled'] += 1
            
            return {
                'success': success,
                'processing_time': processing_time,
                'worker_id': worker['id'],
                'result_data': f"Processed by {worker['id']}" if success else None,
                'error': None if success else "Processing failed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'processing_time': 0.0,
                'worker_id': worker['id'],
                'result_data': None,
                'error': str(e)
            }
        finally:
            worker['current_connections'] -= 1
    
    def _update_worker_stats(self, worker_id: str, processing_time: float, success: bool):
        """Update worker performance statistics."""
        
        # Update worker object
        worker = next((w for w in self.workers if w['id'] == worker_id), None)
        if worker:
            # Update average response time (exponential moving average)
            alpha = 0.1
            if worker['avg_response_time'] == 0:
                worker['avg_response_time'] = processing_time
            else:
                worker['avg_response_time'] = (alpha * processing_time + 
                                             (1 - alpha) * worker['avg_response_time'])
            
            # Simulate CPU and memory usage updates
            worker['cpu_usage'] = min(100, worker['cpu_usage'] * 0.9 + 
                                    processing_time * 10)
            worker['memory_usage'] = min(100, worker['memory_usage'] * 0.95 + 
                                       processing_time * 5)
        
        # Update detailed stats
        if worker_id in self.worker_stats:
            stats = self.worker_stats[worker_id]
            stats['response_times'].append(processing_time)
            if success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
    
    async def _evaluate_scaling(self):
        """Evaluate if auto-scaling is needed."""
        
        if len(self.performance_history) < 10:
            return  # Need more data
        
        recent_metrics = list(self.performance_history)[-10:]
        
        # Calculate average metrics
        avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_latency_p95 = sum(m.latency_p95 for m in recent_metrics) / len(recent_metrics)
        avg_queue_depth = sum(m.queue_depth for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # Scaling decision logic
        scaling_decision = None
        
        # Scale up conditions
        if (avg_cpu > 80 or avg_latency_p95 > 2000 or avg_queue_depth > 50 or avg_error_rate > 5):
            if len(self.workers) < self.max_workers:
                scaling_decision = ScalingDecision(
                    timestamp=time.time(),
                    decision_type="SCALE_UP",
                    trigger_metric="performance_threshold",
                    current_value=max(avg_cpu, avg_latency_p95/10, avg_queue_depth, avg_error_rate*20),
                    threshold=80.0,
                    target_instances=len(self.workers) + 1,
                    confidence=0.8,
                    estimated_impact={
                        'cpu_reduction': 15.0,
                        'latency_reduction': 200.0,
                        'throughput_increase': 25.0
                    }
                )
        
        # Scale down conditions
        elif (avg_cpu < 30 and avg_latency_p95 < 500 and avg_queue_depth < 10 and avg_error_rate < 1):
            if len(self.workers) > self.min_workers:
                scaling_decision = ScalingDecision(
                    timestamp=time.time(),
                    decision_type="SCALE_DOWN",
                    trigger_metric="underutilization",
                    current_value=avg_cpu,
                    threshold=30.0,
                    target_instances=len(self.workers) - 1,
                    confidence=0.7,
                    estimated_impact={
                        'cost_reduction': 20.0,
                        'efficiency_increase': 10.0
                    }
                )
        
        # Execute scaling decision
        if scaling_decision:
            await self._execute_scaling_decision(scaling_decision)
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute auto-scaling decision."""
        
        logger.info(f"Executing scaling decision: {decision.decision_type} to {decision.target_instances} instances")
        
        current_workers = len(self.workers)
        
        if decision.decision_type == "SCALE_UP":
            # Add workers
            workers_to_add = decision.target_instances - current_workers
            for i in range(workers_to_add):
                worker_id = f"worker_{current_workers + i}_{int(time.time())}"
                await self._start_worker(worker_id)
        
        elif decision.decision_type == "SCALE_DOWN":
            # Remove workers (gracefully)
            workers_to_remove = current_workers - decision.target_instances
            
            # Select workers to remove (least active ones)
            workers_by_load = sorted(self.workers, 
                                   key=lambda w: w['current_connections'] + w['avg_response_time'])
            
            for worker in workers_by_load[:workers_to_remove]:
                await self._stop_worker(worker['id'])
        
        # Record scaling decision
        self.scaling_history.append(decision)
        
        logger.info(f"Scaling completed: {len(self.workers)} workers active")
    
    # Helper methods for metrics calculation
    def _calculate_current_throughput(self) -> float:
        """Calculate current throughput in requests/sec."""
        
        if len(self.performance_history) < 2:
            return 0.0
        
        recent_metrics = list(self.performance_history)[-10:]
        total_requests = sum(w['requests_handled'] for w in self.workers)
        time_window = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        
        return total_requests / max(1, time_window)
    
    def _calculate_percentile(self, percentile: int) -> float:
        """Calculate latency percentile from recent response times."""
        
        all_times = []
        for worker_id, stats in self.worker_stats.items():
            all_times.extend(stats['response_times'])
        
        if not all_times:
            return 0.0
        
        return np.percentile(all_times, percentile) * 1000  # Convert to milliseconds
    
    def _get_average_cpu_usage(self) -> float:
        """Get average CPU usage across workers."""
        
        if not self.workers:
            return 0.0
        
        return sum(w['cpu_usage'] for w in self.workers) / len(self.workers)
    
    def _get_total_memory_usage(self) -> float:
        """Get total memory usage across workers."""
        
        return sum(w['memory_usage'] for w in self.workers)
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate percentage."""
        
        total_success = sum(stats['success_count'] for stats in self.worker_stats.values())
        total_errors = sum(stats['error_count'] for stats in self.worker_stats.values())
        total_requests = total_success + total_errors
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100
    
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource efficiency score."""
        
        if not self.workers:
            return 0.0
        
        avg_cpu = self._get_average_cpu_usage()
        throughput = self._calculate_current_throughput()
        error_rate = self._calculate_error_rate()
        
        # Efficiency score: high throughput, low CPU, low errors
        efficiency = (throughput * 10) / max(1, avg_cpu + error_rate * 10)
        return min(100, efficiency)

class HyperScalePerformanceEngine:
    """Main hyperscale performance optimization engine."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.cache_manager = AdaptiveCacheManager(redis_url)
        self.load_balancer = IntelligentLoadBalancer()
        self.performance_optimizer = PerformanceOptimizer()
        self.is_initialized = False
        
        # Performance monitoring
        self.performance_monitor_task = None
        self.optimization_task = None
        
    async def initialize(self):
        """Initialize the hyperscale performance engine."""
        
        if self.is_initialized:
            return
        
        logger.info("Initializing HyperScale Performance Engine")
        
        # Initialize components
        await self.load_balancer.initialize_workers()
        
        # Start background tasks
        self.performance_monitor_task = asyncio.create_task(self._performance_monitoring_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.is_initialized = True
        logger.info("HyperScale Performance Engine initialized successfully")
    
    async def shutdown(self):
        """Gracefully shutdown the performance engine."""
        
        logger.info("Shutting down HyperScale Performance Engine")
        
        # Cancel background tasks
        if self.performance_monitor_task:
            self.performance_monitor_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        # Wait for tasks to complete
        try:
            if self.performance_monitor_task:
                await self.performance_monitor_task
            if self.optimization_task:
                await self.optimization_task
        except asyncio.CancelledError:
            pass
        
        self.is_initialized = False
        logger.info("HyperScale Performance Engine shutdown complete")
    
    async def process_request(
        self, 
        request_data: Dict[str, Any],
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process request with full performance optimization."""
        
        start_time = time.time()
        
        # Try cache first
        if cache_key:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return {
                    'success': True,
                    'result': cached_result,
                    'cache_hit': True,
                    'processing_time': time.time() - start_time
                }
        
        # Process through load balancer
        result = await self.load_balancer.distribute_request(request_data)
        
        # Cache result if successful
        if result['success'] and cache_key:
            await self.cache_manager.set(cache_key, result['result_data'])
        
        # Add performance metadata
        result['cache_hit'] = False
        result['total_processing_time'] = time.time() - start_time
        
        return result
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop."""
        
        while self.is_initialized:
            try:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                
                # Analyze performance trends
                await self._analyze_performance_trends(metrics)
                
                # Predictive caching
                await self.cache_manager.prefetch_predictions()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _optimization_loop(self):
        """Continuous optimization loop."""
        
        while self.is_initialized:
            try:
                # Optimize cache strategies
                await self._optimize_cache_strategies()
                
                # Optimize load balancing algorithm
                await self._optimize_load_balancing()
                
                # System-level optimizations
                await self._system_optimizations()
                
                # Wait for next optimization cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Worker performance metrics
        worker_metrics = {
            'active_workers': len(self.load_balancer.workers),
            'total_requests': sum(w['requests_handled'] for w in self.load_balancer.workers),
            'average_response_time': sum(w['avg_response_time'] for w in self.load_balancer.workers) / max(1, len(self.load_balancer.workers)),
            'total_connections': sum(w['current_connections'] for w in self.load_balancer.workers),
            'error_rate': self.load_balancer._calculate_error_rate()
        }
        
        # System metrics
        system_metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        return {
            'timestamp': time.time(),
            'cache': cache_stats,
            'workers': worker_metrics,
            'system': system_metrics
        }
    
    async def _analyze_performance_trends(self, metrics: Dict[str, Any]):
        """Analyze performance trends for optimization opportunities."""
        
        # Cache performance analysis
        if metrics['cache']['hit_rate'] < 80:
            logger.info(f"Cache hit rate below target: {metrics['cache']['hit_rate']:.1f}%")
            # Trigger cache optimization
        
        # Worker performance analysis
        if metrics['workers']['error_rate'] > 2:
            logger.warning(f"High error rate detected: {metrics['workers']['error_rate']:.2f}%")
            # Trigger error rate investigation
        
        # System resource analysis
        if metrics['system']['cpu_usage'] > 80:
            logger.warning(f"High CPU usage: {metrics['system']['cpu_usage']:.1f}%")
            # Trigger CPU optimization
        
        if metrics['system']['memory_usage'] > 85:
            logger.warning(f"High memory usage: {metrics['system']['memory_usage']:.1f}%")
            # Trigger memory optimization
    
    async def _optimize_cache_strategies(self):
        """Optimize caching strategies based on performance data."""
        
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Adjust cache sizes based on hit rates
        if cache_stats['hit_rate'] < 70:
            # Increase L1 cache size
            self.cache_manager.l1_max_size = min(20000, self.cache_manager.l1_max_size * 1.2)
            logger.info(f"Increased L1 cache size to {self.cache_manager.l1_max_size}")
        
        elif cache_stats['hit_rate'] > 95:
            # Decrease L1 cache size to save memory
            self.cache_manager.l1_max_size = max(5000, self.cache_manager.l1_max_size * 0.9)
            logger.info(f"Decreased L1 cache size to {self.cache_manager.l1_max_size}")
    
    async def _optimize_load_balancing(self):
        """Optimize load balancing algorithm based on performance."""
        
        # Test different algorithms and choose best performing
        algorithms = ['round_robin', 'least_connections', 'weighted_response', 'predictive']
        
        current_perf = self.load_balancer._calculate_resource_efficiency()
        
        # Occasionally test different algorithms
        if len(self.load_balancer.scaling_history) % 10 == 0:
            # Switch to different algorithm for testing
            current_idx = algorithms.index(self.load_balancer.current_algorithm)
            next_algorithm = algorithms[(current_idx + 1) % len(algorithms)]
            
            logger.info(f"Testing load balancing algorithm: {next_algorithm}")
            self.load_balancer.current_algorithm = next_algorithm
    
    async def _system_optimizations(self):
        """Perform system-level optimizations."""
        
        # Memory optimization
        try:
            import gc
            gc.collect()
        except Exception as e:
            logger.debug(f"Garbage collection failed: {e}")
        
        # Process priority optimization (if possible)
        try:
            import os
            current_priority = os.getpriority(os.PRIO_PROCESS, 0)
            if current_priority > -5:  # Not already high priority
                os.setpriority(os.PRIO_PROCESS, 0, -5)
                logger.debug("Increased process priority")
        except Exception as e:
            logger.debug(f"Priority adjustment failed: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        metrics = await self._collect_performance_metrics()
        
        # Calculate performance scores
        cache_score = min(100, metrics['cache']['hit_rate'])
        throughput_score = min(100, self.load_balancer._calculate_current_throughput() / 10)  # Normalize
        latency_score = max(0, 100 - self.load_balancer._calculate_percentile(95) / 10)
        error_score = max(0, 100 - metrics['workers']['error_rate'] * 20)
        resource_score = self.load_balancer._calculate_resource_efficiency()
        
        overall_score = (cache_score + throughput_score + latency_score + error_score + resource_score) / 5
        
        return {
            'timestamp': time.time(),
            'overall_performance_score': overall_score,
            'component_scores': {
                'cache_performance': cache_score,
                'throughput_performance': throughput_score,
                'latency_performance': latency_score,
                'error_performance': error_score,
                'resource_efficiency': resource_score
            },
            'detailed_metrics': metrics,
            'scaling_history': list(self.load_balancer.scaling_history),
            'recommendations': self._generate_performance_recommendations(metrics)
        }
    
    def _generate_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        
        recommendations = []
        
        if metrics['cache']['hit_rate'] < 80:
            recommendations.append("Consider increasing cache size or improving cache key strategies")
        
        if metrics['workers']['error_rate'] > 2:
            recommendations.append("Investigate and reduce error rate in worker processes")
        
        if metrics['system']['cpu_usage'] > 80:
            recommendations.append("Consider scaling up CPU resources or optimizing algorithms")
        
        if metrics['system']['memory_usage'] > 85:
            recommendations.append("Consider increasing memory or optimizing memory usage")
        
        if metrics['workers']['average_response_time'] > 1.0:
            recommendations.append("Optimize processing algorithms to reduce response time")
        
        return recommendations

class PerformanceOptimizer:
    """Additional performance optimization utilities."""
    
    @staticmethod
    def memoize_with_ttl(ttl_seconds: int = 3600):
        """Decorator for memoizing function results with TTL."""
        
        def decorator(func):
            cache = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = str(args) + str(sorted(kwargs.items()))
                key_hash = hashlib.md5(key.encode()).hexdigest()
                
                # Check cache
                if key_hash in cache:
                    result, timestamp = cache[key_hash]
                    if time.time() - timestamp < ttl_seconds:
                        return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache[key_hash] = (result, time.time())
                
                # Clean old entries periodically
                if len(cache) > 1000:
                    current_time = time.time()
                    expired_keys = [k for k, (_, ts) in cache.items() 
                                  if current_time - ts > ttl_seconds]
                    for k in expired_keys:
                        del cache[k]
                
                return result
            
            return wrapper
        return decorator

# Global performance engine instance
_global_performance_engine = None
_engine_lock = threading.Lock()

def get_global_performance_engine() -> HyperScalePerformanceEngine:
    """Get or create global performance engine."""
    global _global_performance_engine
    
    if _global_performance_engine is None:
        with _engine_lock:
            if _global_performance_engine is None:
                _global_performance_engine = HyperScalePerformanceEngine()
    
    return _global_performance_engine

async def initialize_hyperscale_performance(
    redis_url: Optional[str] = None,
    auto_start: bool = True
) -> HyperScalePerformanceEngine:
    """Initialize hyperscale performance engine."""
    
    global _global_performance_engine
    with _engine_lock:
        _global_performance_engine = HyperScalePerformanceEngine(redis_url)
    
    if auto_start:
        await _global_performance_engine.initialize()
    
    logger.info("HyperScale Performance Engine ready for maximum throughput")
    
    return _global_performance_engine