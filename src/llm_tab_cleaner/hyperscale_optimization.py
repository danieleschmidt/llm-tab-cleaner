"""Hyperscale optimization engine with intelligent caching and distributed processing."""

import asyncio
import logging
import time
import hashlib
import pickle
import threading
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, AsyncGenerator
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import queue
import weakref
import json
import gzip
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies for different data types."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ProcessingTask:
    """Processing task with optimization metadata."""
    task_id: str
    data: Any
    processor_func: Callable
    priority: int = 5
    estimated_duration: float = 1.0
    memory_requirement: int = 1024 * 1024  # 1MB default
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class OptimizationMetrics:
    """Performance optimization metrics."""
    cache_hit_rate: float
    avg_processing_time: float
    memory_utilization: float
    cpu_utilization: float
    throughput_per_second: float
    queue_depth: int
    error_rate: float
    cache_size_mb: float
    active_workers: int


class IntelligentCache:
    """Intelligent multi-level caching system with adaptive strategies."""
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_persistence: bool = True,
        persistence_path: str = "./cache",
        redis_config: Optional[Dict] = None
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path)
        
        # Multi-level storage
        self.l1_cache: Dict[str, CacheEntry] = {}  # In-memory hot cache
        self.l2_cache: Dict[str, str] = {}  # Compressed cache paths
        self.redis_client = None
        
        # Metadata
        self.current_memory_usage = 0
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.performance_metrics = {}
        
        self._lock = threading.RLock()
        
        # Initialize Redis if available
        if HAS_REDIS and redis_config:
            try:
                self.redis_client = redis.Redis(**redis_config)
                self.redis_client.ping()
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
        
        # Initialize persistence
        if enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Background optimization
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache with intelligent promotion."""
        with self._lock:
            # L1 cache (hot memory)
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not self._is_expired(entry):
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self._update_access_pattern(key)
                    return entry.value
                else:
                    # Expired entry
                    self._remove_from_l1(key)
            
            # L2 cache (compressed disk)
            if key in self.l2_cache:
                value = self._load_from_l2(key)
                if value is not None:
                    # Promote to L1 if frequently accessed
                    if self._should_promote_to_l1(key):
                        self._store_in_l1(key, value, ttl_seconds=self.default_ttl)
                    return value
            
            # Distributed cache (Redis)
            if self.redis_client:
                try:
                    redis_value = self.redis_client.get(key)
                    if redis_value:
                        value = pickle.loads(redis_value)
                        # Store in local cache for future access
                        self._store_in_l1(key, value, ttl_seconds=self.default_ttl)
                        return value
                except Exception as e:
                    logger.warning(f"Redis get error for key {key}: {e}")
        
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache with intelligent placement."""
        ttl = ttl_seconds or self.default_ttl
        tags = tags or []
        
        with self._lock:
            # Calculate value size
            try:
                value_size = len(pickle.dumps(value))
            except Exception:
                value_size = 1024  # Default size estimate
            
            # Decide cache level based on size and access pattern
            access_frequency = self._get_access_frequency(key)
            
            if value_size < 1024 * 1024 and access_frequency > 0.1:  # Small, frequently accessed
                success = self._store_in_l1(key, value, ttl, tags, value_size)
                if not success:
                    # L1 full, try L2
                    self._store_in_l2(key, value, ttl, tags)
            else:  # Large or infrequently accessed
                self._store_in_l2(key, value, ttl, tags)
            
            # Store in Redis for distributed access
            if self.redis_client:
                try:
                    self.redis_client.setex(key, ttl, pickle.dumps(value))
                except Exception as e:
                    logger.warning(f"Redis set error for key {key}: {e}")
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across all levels."""
        with self._lock:
            removed = False
            
            # Remove from L1
            if key in self.l1_cache:
                self._remove_from_l1(key)
                removed = True
            
            # Remove from L2
            if key in self.l2_cache:
                cache_file = self.persistence_path / self.l2_cache[key]
                if cache_file.exists():
                    cache_file.unlink()
                del self.l2_cache[key]
                removed = True
            
            # Remove from Redis
            if self.redis_client:
                try:
                    self.redis_client.delete(key)
                    removed = True
                except Exception as e:
                    logger.warning(f"Redis delete error for key {key}: {e}")
            
            return removed
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with matching tags."""
        removed_count = 0
        
        with self._lock:
            # L1 cache
            keys_to_remove = []
            for key, entry in self.l1_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_from_l1(key)
                removed_count += 1
            
            # L2 cache - would need metadata storage for full implementation
            # For now, just clear all L2 if any tag matches
            # (In production, implement proper tag indexing)
        
        return removed_count
    
    def _store_in_l1(
        self,
        key: str,
        value: Any,
        ttl_seconds: int,
        tags: List[str] = None,
        size_bytes: int = None
    ) -> bool:
        """Store value in L1 (hot memory) cache."""
        if size_bytes is None:
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024
        
        # Check memory limit
        if self.current_memory_usage + size_bytes > self.max_memory_bytes:
            # Evict entries to make space
            if not self._make_space(size_bytes):
                return False  # Can't make enough space
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds,
            tags=tags or []
        )
        
        # Remove existing entry if present
        if key in self.l1_cache:
            self._remove_from_l1(key)
        
        # Store new entry
        self.l1_cache[key] = entry
        self.current_memory_usage += size_bytes
        
        return True
    
    def _store_in_l2(self, key: str, value: Any, ttl_seconds: int, tags: List[str] = None):
        """Store value in L2 (compressed disk) cache."""
        if not self.enable_persistence:
            return False
        
        try:
            # Create compressed cache file
            cache_filename = f"{hashlib.md5(key.encode()).hexdigest()}.gz"
            cache_path = self.persistence_path / cache_filename
            
            # Serialize and compress
            data = {
                'value': value,
                'created_at': datetime.now().timestamp(),
                'ttl_seconds': ttl_seconds,
                'tags': tags or []
            }
            
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update L2 index
            self.l2_cache[key] = cache_filename
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache store error for key {key}: {e}")
            return False
    
    def _load_from_l2(self, key: str) -> Any:
        """Load value from L2 cache."""
        if not self.enable_persistence or key not in self.l2_cache:
            return None
        
        try:
            cache_filename = self.l2_cache[key]
            cache_path = self.persistence_path / cache_filename
            
            if not cache_path.exists():
                # File missing, remove from index
                del self.l2_cache[key]
                return None
            
            # Load and decompress
            with gzip.open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check expiration
            created_at = datetime.fromtimestamp(data['created_at'])
            ttl_seconds = data['ttl_seconds']
            
            if ttl_seconds and (datetime.now() - created_at).total_seconds() > ttl_seconds:
                # Expired, remove file
                cache_path.unlink()
                del self.l2_cache[key]
                return None
            
            return data['value']
            
        except Exception as e:
            logger.error(f"L2 cache load error for key {key}: {e}")
            # Remove corrupted entry
            if key in self.l2_cache:
                del self.l2_cache[key]
            return None
    
    def _remove_from_l1(self, key: str):
        """Remove entry from L1 cache."""
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            self.current_memory_usage -= entry.size_bytes
            del self.l1_cache[key]
    
    def _make_space(self, required_bytes: int) -> bool:
        """Make space in L1 cache using eviction strategy."""
        if self.strategy == CacheStrategy.LRU:
            return self._evict_lru(required_bytes)
        elif self.strategy == CacheStrategy.LFU:
            return self._evict_lfu(required_bytes)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            return self._evict_adaptive(required_bytes)
        else:
            return self._evict_lru(required_bytes)  # Default to LRU
    
    def _evict_lru(self, required_bytes: int) -> bool:
        """Evict least recently used entries."""
        # Sort by last accessed time
        sorted_entries = sorted(
            self.l1_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_bytes = 0
        for key, entry in sorted_entries:
            if freed_bytes >= required_bytes:
                break
            
            # Move to L2 before evicting from L1
            if self.enable_persistence:
                self._store_in_l2(key, entry.value, entry.ttl_seconds, entry.tags)
            
            freed_bytes += entry.size_bytes
            self._remove_from_l1(key)
        
        return freed_bytes >= required_bytes
    
    def _evict_lfu(self, required_bytes: int) -> bool:
        """Evict least frequently used entries."""
        # Sort by access count
        sorted_entries = sorted(
            self.l1_cache.items(),
            key=lambda x: x[1].access_count
        )
        
        freed_bytes = 0
        for key, entry in sorted_entries:
            if freed_bytes >= required_bytes:
                break
            
            if self.enable_persistence:
                self._store_in_l2(key, entry.value, entry.ttl_seconds, entry.tags)
            
            freed_bytes += entry.size_bytes
            self._remove_from_l1(key)
        
        return freed_bytes >= required_bytes
    
    def _evict_adaptive(self, required_bytes: int) -> bool:
        """Adaptive eviction based on access patterns and age."""
        # Calculate composite score for each entry
        current_time = datetime.now()
        scored_entries = []
        
        for key, entry in self.l1_cache.items():
            age_hours = (current_time - entry.created_at).total_seconds() / 3600
            time_since_access = (current_time - entry.last_accessed).total_seconds() / 3600
            
            # Composite score (lower = more likely to evict)
            score = (
                entry.access_count * 0.4 +  # Frequency weight
                (1 / (time_since_access + 1)) * 0.4 +  # Recency weight
                (1 / (age_hours + 1)) * 0.2  # Age weight
            )
            
            scored_entries.append((key, entry, score))
        
        # Sort by score (lowest first)
        scored_entries.sort(key=lambda x: x[2])
        
        freed_bytes = 0
        for key, entry, score in scored_entries:
            if freed_bytes >= required_bytes:
                break
            
            if self.enable_persistence:
                self._store_in_l2(key, entry.value, entry.ttl_seconds, entry.tags)
            
            freed_bytes += entry.size_bytes
            self._remove_from_l1(key)
        
        return freed_bytes >= required_bytes
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl_seconds is None:
            return False
        
        age_seconds = (datetime.now() - entry.created_at).total_seconds()
        return age_seconds > entry.ttl_seconds
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if L2 entry should be promoted to L1."""
        access_frequency = self._get_access_frequency(key)
        return access_frequency > 0.05  # Promote if accessed > 5% of the time
    
    def _get_access_frequency(self, key: str) -> float:
        """Get access frequency for a key."""
        if key not in self.access_patterns:
            return 0.0
        
        recent_accesses = [
            t for t in self.access_patterns[key]
            if (datetime.now() - t).total_seconds() < 3600  # Last hour
        ]
        
        return len(recent_accesses) / 3600  # Accesses per second
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for a key."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(datetime.now())
        
        # Keep only recent patterns
        if len(self.access_patterns[key]) > 1000:
            self.access_patterns[key] = self.access_patterns[key][-1000:]
    
    def _optimization_loop(self):
        """Background optimization loop."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                with self._lock:
                    # Clean up expired entries
                    expired_keys = [
                        key for key, entry in self.l1_cache.items()
                        if self._is_expired(entry)
                    ]
                    
                    for key in expired_keys:
                        self._remove_from_l1(key)
                    
                    # Update performance metrics
                    self.performance_metrics.update({
                        'l1_size': len(self.l1_cache),
                        'l2_size': len(self.l2_cache),
                        'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                        'expired_cleaned': len(expired_keys)
                    })
                    
                    if expired_keys:
                        logger.debug(f"Cache optimization: cleaned {len(expired_keys)} expired entries")
                
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_accesses = sum(len(patterns) for patterns in self.access_patterns.values())
            
            return {
                'l1_entries': len(self.l1_cache),
                'l2_entries': len(self.l2_cache),
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self.current_memory_usage / self.max_memory_bytes,
                'total_keys': len(set(list(self.l1_cache.keys()) + list(self.l2_cache.keys()))),
                'total_accesses': total_accesses,
                'strategy': self.strategy.value,
                'redis_enabled': self.redis_client is not None,
                'persistence_enabled': self.enable_persistence,
                'performance_metrics': self.performance_metrics.copy()
            }


class HyperscaleProcessor:
    """High-performance distributed processing engine."""
    
    def __init__(
        self,
        max_workers: int = None,
        mode: ProcessingMode = ProcessingMode.HYBRID,
        cache_config: Optional[Dict] = None,
        enable_ray: bool = True
    ):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4) * 2)
        self.mode = mode
        self.enable_ray = enable_ray and HAS_RAY
        
        # Initialize cache
        self.cache = IntelligentCache(**(cache_config or {}))
        
        # Processing infrastructure
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers // 2,
            thread_name_prefix="hyperscale-thread"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(self.max_workers // 4, os.cpu_count() or 4)
        )
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, Any] = {}
        self.task_metadata: Dict[str, ProcessingTask] = {}
        
        # Performance tracking
        self.processing_stats = {
            'tasks_completed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        # Initialize Ray if available
        if self.enable_ray:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                logger.info("Ray distributed processing initialized")
            except Exception as e:
                logger.warning(f"Ray initialization failed: {e}")
                self.enable_ray = False
        
        self._shutdown_event = threading.Event()
        self._worker_threads = []
        
        # Start worker threads
        for i in range(min(4, self.max_workers // 8)):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"hyperscale-worker-{i}",
                daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)
    
    def submit_task(
        self,
        func: Callable,
        *args,
        priority: int = 5,
        cache_key: Optional[str] = None,
        cache_ttl: int = 3600,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a processing task."""
        # Generate task ID
        task_id = f"task_{int(time.time() * 1000)}_{hash((func.__name__, args, tuple(kwargs.items())))}"
        
        # Check cache first
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.processing_stats['cache_hits'] += 1
                self.completed_tasks[task_id] = {
                    'result': cached_result,
                    'cached': True,
                    'completed_at': datetime.now()
                }
                return task_id
            else:
                self.processing_stats['cache_misses'] += 1
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            data={'func': func, 'args': args, 'kwargs': kwargs},
            processor_func=func,
            priority=priority
        )
        
        self.task_metadata[task_id] = task
        
        # Queue task
        self.task_queue.put((priority, time.time(), task_id, task))
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result (blocking)."""
        start_time = time.time()
        
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(0.01)  # Small delay
        
        result_data = self.completed_tasks[task_id]
        
        if 'error' in result_data:
            raise result_data['error']
        
        return result_data['result']
    
    def submit_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        priority: int = 5,
        cache_prefix: Optional[str] = None
    ) -> List[str]:
        """Submit a batch of tasks."""
        task_ids = []
        
        for i, (func, args, kwargs) in enumerate(tasks):
            cache_key = f"{cache_prefix}_{i}" if cache_prefix else None
            task_id = self.submit_task(
                func, *args, priority=priority, cache_key=cache_key, **kwargs
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def get_batch_results(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Get results for a batch of tasks."""
        return [self.get_result(task_id, timeout) for task_id in task_ids]
    
    async def submit_async(
        self,
        func: Callable,
        *args,
        priority: int = 5,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Submit task asynchronously."""
        task_id = self.submit_task(
            func, *args, priority=priority, cache_key=cache_key, **kwargs
        )
        
        # Async polling for result
        while task_id not in self.completed_tasks:
            await asyncio.sleep(0.01)
        
        result_data = self.completed_tasks[task_id]
        
        if 'error' in result_data:
            raise result_data['error']
        
        return result_data['result']
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue (blocking with timeout)
                try:
                    priority, queued_at, task_id, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                start_time = time.time()
                
                try:
                    # Determine processing method based on mode and task characteristics
                    if self.mode == ProcessingMode.DISTRIBUTED and self.enable_ray:
                        result = self._process_with_ray(task)
                    elif self.mode == ProcessingMode.MULTIPROCESS:
                        result = self._process_with_multiprocessing(task)
                    elif self.mode == ProcessingMode.HYBRID:
                        result = self._process_hybrid(task)
                    else:
                        result = self._process_direct(task)
                    
                    processing_time = time.time() - start_time
                    
                    # Store result
                    self.completed_tasks[task_id] = {
                        'result': result,
                        'processing_time': processing_time,
                        'completed_at': datetime.now(),
                        'cached': False
                    }
                    
                    # Update stats
                    self.processing_stats['tasks_completed'] += 1
                    self.processing_stats['total_processing_time'] += processing_time
                    
                    # Cache result if applicable
                    cache_key = task.data.get('cache_key')
                    if cache_key and result is not None:
                        self.cache.set(cache_key, result)
                    
                    logger.debug(f"Task {task_id} completed in {processing_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    self.completed_tasks[task_id] = {
                        'error': e,
                        'completed_at': datetime.now()
                    }
                    self.processing_stats['errors'] += 1
                
                finally:
                    self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _process_direct(self, task: ProcessingTask) -> Any:
        """Process task directly in current thread."""
        func = task.data['func']
        args = task.data['args']
        kwargs = task.data['kwargs']
        
        return func(*args, **kwargs)
    
    def _process_with_multiprocessing(self, task: ProcessingTask) -> Any:
        """Process task using multiprocessing."""
        func = task.data['func']
        args = task.data['args']
        kwargs = task.data['kwargs']
        
        future = self.process_pool.submit(func, *args, **kwargs)
        return future.result()
    
    def _process_with_ray(self, task: ProcessingTask) -> Any:
        """Process task using Ray distributed computing."""
        if not self.enable_ray:
            return self._process_direct(task)
        
        try:
            @ray.remote
            def ray_task_wrapper(func, args, kwargs):
                return func(*args, **kwargs)
            
            func = task.data['func']
            args = task.data['args']
            kwargs = task.data['kwargs']
            
            future = ray_task_wrapper.remote(func, args, kwargs)
            return ray.get(future)
            
        except Exception as e:
            logger.warning(f"Ray processing failed, falling back to direct: {e}")
            return self._process_direct(task)
    
    def _process_hybrid(self, task: ProcessingTask) -> Any:
        """Intelligently choose processing method based on task characteristics."""
        # Simple heuristics for processing method selection
        estimated_duration = task.estimated_duration
        memory_requirement = task.memory_requirement
        
        if estimated_duration > 5.0 and self.enable_ray:
            # Long-running tasks -> Ray
            return self._process_with_ray(task)
        elif memory_requirement > 100 * 1024 * 1024:  # > 100MB
            # Memory-intensive tasks -> multiprocessing
            return self._process_with_multiprocessing(task)
        else:
            # Quick tasks -> direct processing
            return self._process_direct(task)
    
    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get comprehensive optimization metrics."""
        cache_stats = self.cache.get_stats()
        
        total_cache_operations = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
        cache_hit_rate = (
            self.processing_stats['cache_hits'] / total_cache_operations
            if total_cache_operations > 0 else 0.0
        )
        
        avg_processing_time = (
            self.processing_stats['total_processing_time'] / self.processing_stats['tasks_completed']
            if self.processing_stats['tasks_completed'] > 0 else 0.0
        )
        
        return OptimizationMetrics(
            cache_hit_rate=cache_hit_rate,
            avg_processing_time=avg_processing_time,
            memory_utilization=cache_stats['memory_utilization'],
            cpu_utilization=0.0,  # Would need psutil integration
            throughput_per_second=0.0,  # Would need time window calculation
            queue_depth=self.task_queue.qsize(),
            error_rate=self.processing_stats['errors'] / max(1, self.processing_stats['tasks_completed']),
            cache_size_mb=cache_stats['memory_usage_mb'],
            active_workers=len([t for t in self._worker_threads if t.is_alive()])
        )
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the processing engine."""
        logger.info("Shutting down Hyperscale Processor...")
        
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=timeout / len(self._worker_threads))
        
        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Shutdown Ray if initialized
        if self.enable_ray and ray.is_initialized():
            ray.shutdown()
        
        logger.info("Hyperscale Processor shutdown complete")


# Optimization decorators
def cached_result(cache_key_func: Optional[Callable] = None, ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(kwargs.items())))}"
            
            # Try to get from cache
            if hasattr(wrapper, '_cache'):
                result = wrapper._cache.get(cache_key)
                if result is not None:
                    return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            if hasattr(wrapper, '_cache'):
                wrapper._cache.set(cache_key, result, ttl_seconds=ttl)
            
            return result
        
        # Attach cache to function
        wrapper._cache = IntelligentCache(max_memory_mb=512)
        return wrapper
    return decorator


def parallel_batch_processing(batch_size: int = 100, max_workers: int = None):
    """Decorator for parallel batch processing of iterable inputs."""
    def decorator(func):
        @wraps(func)
        def wrapper(data_iterable, *args, **kwargs):
            processor = HyperscaleProcessor(max_workers=max_workers)
            
            try:
                # Split data into batches
                batches = []
                current_batch = []
                
                for item in data_iterable:
                    current_batch.append(item)
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
                
                if current_batch:
                    batches.append(current_batch)
                
                # Process batches in parallel
                batch_tasks = []
                for batch in batches:
                    task_id = processor.submit_task(func, batch, *args, **kwargs)
                    batch_tasks.append(task_id)
                
                # Collect results
                results = []
                for task_id in batch_tasks:
                    batch_result = processor.get_result(task_id)
                    if isinstance(batch_result, (list, tuple)):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                
                return results
                
            finally:
                processor.shutdown()
        
        return wrapper
    return decorator


# Global hyperscale processor instance
_global_processor: Optional[HyperscaleProcessor] = None


def get_global_hyperscale_processor(**config) -> HyperscaleProcessor:
    """Get or create global hyperscale processor instance."""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = HyperscaleProcessor(**config)
    
    return _global_processor


def initialize_hyperscale_optimization(
    max_workers: int = None,
    cache_memory_mb: int = 1024,
    enable_distributed: bool = True,
    redis_config: Optional[Dict] = None
) -> HyperscaleProcessor:
    """Initialize the hyperscale optimization system."""
    cache_config = {
        'max_memory_mb': cache_memory_mb,
        'strategy': CacheStrategy.ADAPTIVE,
        'enable_persistence': True,
        'redis_config': redis_config
    }
    
    mode = ProcessingMode.DISTRIBUTED if enable_distributed else ProcessingMode.HYBRID
    
    processor = HyperscaleProcessor(
        max_workers=max_workers,
        mode=mode,
        cache_config=cache_config,
        enable_ray=enable_distributed
    )
    
    logger.info(f"Hyperscale optimization initialized with {processor.max_workers} workers")
    return processor