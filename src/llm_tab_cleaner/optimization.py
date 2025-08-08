"""Performance optimization and scaling utilities."""

import asyncio
import hashlib
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from threading import Lock
import threading

import pandas as pd
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    
    # Caching
    enable_caching: bool = True
    cache_type: str = "memory"  # memory, redis, memcache
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000
    
    # Concurrency
    max_workers: int = 4
    chunk_size: int = 1000
    enable_parallel_processing: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    low_memory_mode: bool = False
    batch_processing_threshold: int = 10000
    
    # Performance monitoring
    enable_profiling: bool = False
    performance_sampling_rate: float = 0.1
    
    # Auto-scaling
    enable_auto_scaling: bool = False
    scale_up_threshold: float = 0.8  # CPU/memory usage
    scale_down_threshold: float = 0.3
    min_workers: int = 1
    max_workers_limit: int = 16


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear entire cache."""
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize memory cache."""
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                self.delete(key)
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Remove oldest item if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            if key in self.cache:
                self.cache.move_to_end(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False
    
    def clear(self) -> bool:
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
            return True
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


class RedisCache(CacheBackend):
    """Redis-backed cache implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: Optional[str] = None, ttl: int = 3600):
        """Initialize Redis cache."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        self.ttl = ttl
        self.client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password,
            decode_responses=False
        )
        
        # Test connection
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Redis: {e}")
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            serialized_value = pickle.dumps(value)
            return self.client.setex(key, ttl or self.ttl, serialized_value)
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear entire cache."""
        try:
            return self.client.flushdb()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "ttl": self.ttl
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}


class CacheManager:
    """Manages caching operations with multiple backend support."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize cache manager."""
        self.config = config
        self.enabled = config.enable_caching
        
        if not self.enabled:
            self.backend = None
            return
        
        # Initialize cache backend
        if config.cache_type == "memory":
            self.backend = MemoryCache(
                max_size=config.max_cache_size,
                ttl=config.cache_ttl
            )
        elif config.cache_type == "redis":
            self.backend = RedisCache(ttl=config.cache_ttl)
        elif config.cache_type == "memcache":
            if not MEMCACHE_AVAILABLE:
                logger.warning("Memcache not available, falling back to memory cache")
                self.backend = MemoryCache(
                    max_size=config.max_cache_size,
                    ttl=config.cache_ttl
                )
        else:
            logger.warning(f"Unknown cache type: {config.cache_type}, using memory cache")
            self.backend = MemoryCache(
                max_size=config.max_cache_size,
                ttl=config.cache_ttl
            )
        
        logger.info(f"Initialized cache manager with {config.cache_type} backend")
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if not self.enabled or not self.backend:
            return None
        
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.enabled or not self.backend:
            return False
        
        return self.backend.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.enabled or not self.backend:
            return False
        
        return self.backend.delete(key)
    
    def clear(self) -> bool:
        """Clear entire cache."""
        if not self.enabled or not self.backend:
            return False
        
        return self.backend.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.backend:
            return {"enabled": False}
        
        stats = self.backend.stats()
        stats["enabled"] = True
        stats["backend_type"] = self.config.cache_type
        return stats
    
    def make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        # Create deterministic key from arguments
        key_parts = []
        
        # Add positional args
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Use shape and column info for DataFrames
                key_parts.append(f"df_{arg.shape}_{list(arg.columns)}")
            else:
                key_parts.append(str(arg))
        
        # Add keyword args
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def cached(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cache_manager.enabled:
                return func(*args, **kwargs)
            
            # Create cache key
            cache_key = f"{func.__name__}_{cache_manager.make_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator


class ParallelProcessor:
    """Handles parallel processing of data cleaning operations."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize parallel processor."""
        self.config = config
        self.max_workers = config.max_workers
        self.chunk_size = config.chunk_size
        self.enabled = config.enable_parallel_processing
    
    def process_chunks_threaded(self, 
                               data_chunks: List[pd.DataFrame], 
                               process_func: Callable,
                               **kwargs) -> List[Any]:
        """Process data chunks using thread pool."""
        if not self.enabled or len(data_chunks) == 1:
            return [process_func(chunk, **kwargs) for chunk in data_chunks]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_func, chunk, **kwargs): i
                for i, chunk in enumerate(data_chunks)
            }
            
            # Collect results in order
            results = [None] * len(data_chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    results[chunk_index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    results[chunk_index] = None
        
        return results
    
    def process_chunks_async(self, 
                            data_chunks: List[pd.DataFrame],
                            process_func: Callable,
                            **kwargs) -> List[Any]:
        """Process data chunks using async processing."""
        if not self.enabled:
            return [process_func(chunk, **kwargs) for chunk in data_chunks]
        
        async def process_chunk_async(chunk: pd.DataFrame) -> Any:
            """Process single chunk asynchronously."""
            return await asyncio.to_thread(process_func, chunk, **kwargs)
        
        async def process_all_async() -> List[Any]:
            """Process all chunks concurrently."""
            tasks = [process_chunk_async(chunk) for chunk in data_chunks]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run async processing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(process_all_async())
    
    def split_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into chunks for parallel processing."""
        if len(df) <= self.chunk_size:
            return [df]
        
        chunks = []
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            chunks.append(chunk)
        
        logger.info(f"Split DataFrame of {len(df)} rows into {len(chunks)} chunks")
        return chunks
    
    def combine_results(self, 
                       results: List[Tuple[pd.DataFrame, Any]], 
                       combine_reports: bool = True) -> Tuple[pd.DataFrame, Any]:
        """Combine results from parallel processing."""
        if not results:
            return pd.DataFrame(), None
        
        # Filter out None results (errors)
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            logger.error("No valid results from parallel processing")
            return pd.DataFrame(), None
        
        # Combine DataFrames
        dfs = [result[0] for result in valid_results]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Combine reports if requested
        if combine_reports and len(valid_results) > 0:
            reports = [result[1] for result in valid_results]
            combined_report = self._combine_cleaning_reports(reports)
            return combined_df, combined_report
        
        return combined_df, valid_results[0][1] if valid_results else None
    
    def _combine_cleaning_reports(self, reports: List[Any]) -> Any:
        """Combine multiple cleaning reports into one."""
        if not reports:
            return None
        
        # This is a simplified combination - in practice, you'd want to
        # properly merge all the report fields
        from .core import CleaningReport, Fix
        
        total_fixes = sum(report.total_fixes for report in reports if report)
        avg_quality_score = sum(report.quality_score for report in reports if report) / len(reports)
        
        all_fixes = []
        for report in reports:
            if report and hasattr(report, 'fixes'):
                all_fixes.extend(report.fixes)
        
        total_processing_time = sum(report.processing_time for report in reports if report)
        
        return CleaningReport(
            total_fixes=total_fixes,
            quality_score=avg_quality_score,
            fixes=all_fixes,
            processing_time=total_processing_time,
            profile_summary={},
            audit_trail=[]
        )


class MemoryOptimizer:
    """Optimizes memory usage for large dataset processing."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize memory optimizer."""
        self.config = config
        self.enabled = config.enable_memory_optimization
        self.low_memory_mode = config.low_memory_mode
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        if not self.enabled:
            return df
        
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']):
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']):
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize object columns to category where beneficial
        for col in optimized_df.select_dtypes(include=['object']):
            if not optimized_df[col].isnull().all():
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])
                
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    try:
                        optimized_df[col] = optimized_df[col].astype('category')
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to category: {e}")
        
        # Log memory savings
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        savings = ((original_memory - optimized_memory) / original_memory) * 100
        
        logger.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
                   f"({savings:.1f}% reduction)")
        
        return optimized_df
    
    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get detailed memory usage information."""
        memory_usage = df.memory_usage(deep=True)
        total_memory_mb = memory_usage.sum() / 1024 / 1024
        
        return {
            "total_mb": total_memory_mb,
            "per_column_mb": {
                col: memory_usage[col] / 1024 / 1024 
                for col in df.columns
            },
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape
        }


class AutoScaler:
    """Auto-scaling functionality for dynamic resource management."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize auto scaler."""
        self.config = config
        self.enabled = config.enable_auto_scaling
        self.current_workers = config.max_workers
        self.min_workers = config.min_workers
        self.max_workers = config.max_workers_limit
        self.scale_up_threshold = config.scale_up_threshold
        self.scale_down_threshold = config.scale_down_threshold
        self.last_scale_time = 0
        self.scale_cooldown = 60  # seconds
    
    def should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Check if system should scale up."""
        if not self.enabled:
            return False
        
        if self.current_workers >= self.max_workers:
            return False
        
        # Check if enough time has passed since last scaling
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check metrics
        cpu_usage = metrics.get('cpu_usage', 0.0)
        memory_usage = metrics.get('memory_usage', 0.0)
        queue_depth = metrics.get('queue_depth', 0.0)
        
        return (cpu_usage > self.scale_up_threshold or 
                memory_usage > self.scale_up_threshold or
                queue_depth > 10)
    
    def should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Check if system should scale down."""
        if not self.enabled:
            return False
        
        if self.current_workers <= self.min_workers:
            return False
        
        # Check if enough time has passed since last scaling
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check metrics
        cpu_usage = metrics.get('cpu_usage', 0.0)
        memory_usage = metrics.get('memory_usage', 0.0)
        queue_depth = metrics.get('queue_depth', 0.0)
        
        return (cpu_usage < self.scale_down_threshold and 
                memory_usage < self.scale_down_threshold and
                queue_depth < 2)
    
    def scale_up(self) -> int:
        """Scale up workers."""
        new_worker_count = min(self.current_workers + 1, self.max_workers)
        logger.info(f"Scaling up: {self.current_workers} -> {new_worker_count} workers")
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()
        return new_worker_count
    
    def scale_down(self) -> int:
        """Scale down workers."""
        new_worker_count = max(self.current_workers - 1, self.min_workers)
        logger.info(f"Scaling down: {self.current_workers} -> {new_worker_count} workers")
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()
        return new_worker_count


class OptimizationEngine:
    """Main optimization engine that coordinates all optimization features."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize optimization engine."""
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.cache_manager = CacheManager(self.config)
        self.parallel_processor = ParallelProcessor(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.auto_scaler = AutoScaler(self.config)
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.memory_usage_history = []
        
        logger.info("Initialized optimization engine")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for processing."""
        if self.config.enable_memory_optimization:
            return self.memory_optimizer.optimize_dataframe_dtypes(df)
        return df
    
    def should_use_parallel_processing(self, df: pd.DataFrame) -> bool:
        """Determine if parallel processing should be used."""
        if not self.config.enable_parallel_processing:
            return False
        
        # Use parallel processing for large datasets
        return len(df) > self.config.batch_processing_threshold
    
    def get_optimization_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get optimization recommendations for a DataFrame."""
        memory_info = self.memory_optimizer.get_memory_usage(df)
        
        recommendations = {
            "memory_optimization": {
                "current_memory_mb": memory_info["total_mb"],
                "recommend_dtype_optimization": memory_info["total_mb"] > 100,
                "recommend_categorical_conversion": len(df.select_dtypes(include=['object']).columns) > 0
            },
            "parallel_processing": {
                "recommend_parallel": len(df) > self.config.batch_processing_threshold,
                "optimal_chunk_size": min(self.config.chunk_size, len(df) // self.config.max_workers),
                "estimated_chunks": max(1, len(df) // self.config.chunk_size)
            },
            "caching": {
                "cache_enabled": self.config.enable_caching,
                "cache_stats": self.cache_manager.stats()
            }
        }
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics."""
        cache_stats = self.cache_manager.stats()
        
        return {
            "cache": cache_stats,
            "parallel_processing": {
                "enabled": self.config.enable_parallel_processing,
                "max_workers": self.config.max_workers,
                "chunk_size": self.config.chunk_size
            },
            "memory_optimization": {
                "enabled": self.config.enable_memory_optimization,
                "low_memory_mode": self.config.low_memory_mode
            },
            "auto_scaling": {
                "enabled": self.config.enable_auto_scaling,
                "current_workers": self.auto_scaler.current_workers
            }
        }


# Global optimization engine instance
_global_optimizer: Optional[OptimizationEngine] = None


def get_global_optimizer() -> OptimizationEngine:
    """Get or create global optimization engine."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = OptimizationEngine()
    return _global_optimizer


def optimize_for_performance(func: Callable) -> Callable:
    """Decorator to apply performance optimizations to functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = get_global_optimizer()
        
        # Record start time
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Record performance metrics
            duration = time.time() - start_time
            optimizer.operation_times[func.__name__].append(duration)
            
            return result
            
        except Exception as e:
            # Log error and re-raise
            logger.error(f"Error in optimized function {func.__name__}: {e}")
            raise
    
    return wrapper