"""Advanced caching and performance optimization."""

import hashlib
import json
import logging
import pickle
import time
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zlib

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0


class LRUCache:
    """Thread-safe LRU cache with size limit."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.metrics = CacheMetrics()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        
        with self._lock:
            if key not in self.cache:
                self.metrics.misses += 1
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                self._remove_key(key)
                self.metrics.misses += 1
                return None
            
            # Move to end (most recently used)
            value = self.cache[key]
            self.cache.move_to_end(key)
            self.access_counts[key] += 1
            self.metrics.hits += 1
            
            # Update metrics
            access_time = time.time() - start_time
            self._update_access_time(access_time)
            
            return value
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        with self._lock:
            # Remove if exists
            if key in self.cache:
                self._remove_key(key)
            
            # Check size limit
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_counts[key] = 1
            self.metrics.total_size += self._get_size(value)
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_counts.clear()
            self.metrics = CacheMetrics()
    
    def _remove_key(self, key: str):
        """Remove key and update metrics."""
        if key in self.cache:
            value = self.cache[key]
            del self.cache[key]
            del self.timestamps[key]
            del self.access_counts[key]
            self.metrics.total_size -= self._get_size(value)
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.cache:
            lru_key = next(iter(self.cache))
            self._remove_key(lru_key)
            self.metrics.evictions += 1
    
    def _get_size(self, value: Any) -> int:
        """Estimate size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return 64  # Default estimate
    
    def _update_access_time(self, access_time: float):
        """Update average access time."""
        total_accesses = self.metrics.hits + self.metrics.misses
        if total_accesses > 0:
            self.metrics.average_access_time = (
                (self.metrics.average_access_time * (total_accesses - 1) + access_time) /
                total_accesses
            )
        
        self.metrics.hit_rate = (
            self.metrics.hits / total_accesses if total_accesses > 0 else 0
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "metrics": {
                    "hits": self.metrics.hits,
                    "misses": self.metrics.misses,
                    "evictions": self.metrics.evictions,
                    "hit_rate": self.metrics.hit_rate,
                    "total_size_bytes": self.metrics.total_size,
                    "average_access_time": self.metrics.average_access_time
                },
                "top_accessed": dict(
                    sorted(self.access_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                )
            }


class CompressedCache:
    """Cache with compression for large values."""
    
    def __init__(self, max_size: int = 500, compression_threshold: int = 1024):
        """Initialize compressed cache.
        
        Args:
            max_size: Maximum number of entries
            compression_threshold: Compress values larger than this (bytes)
        """
        self.base_cache = LRUCache(max_size=max_size)
        self.compression_threshold = compression_threshold
        self.compression_stats = {
            "compressed_entries": 0,
            "total_uncompressed_size": 0,
            "total_compressed_size": 0,
            "compression_ratio": 0.0
        }
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value, decompressing if needed."""
        with self._lock:
            cached_data = self.base_cache.get(key)
            if cached_data is None:
                return None
            
            if isinstance(cached_data, dict) and cached_data.get("compressed"):
                # Decompress
                try:
                    compressed_data = cached_data["data"]
                    decompressed = zlib.decompress(compressed_data)
                    return pickle.loads(decompressed)
                except Exception as e:
                    logger.warning(f"Decompression failed for key {key}: {e}")
                    self.base_cache.remove(key)
                    return None
            else:
                return cached_data
    
    def put(self, key: str, value: Any) -> bool:
        """Put value, compressing if large enough."""
        with self._lock:
            try:
                # Serialize value
                serialized = pickle.dumps(value)
                original_size = len(serialized)
                
                # Check if compression is beneficial
                if original_size > self.compression_threshold:
                    compressed = zlib.compress(serialized)
                    compressed_size = len(compressed)
                    
                    # Only use compression if it saves space
                    if compressed_size < original_size * 0.8:
                        cache_value = {
                            "compressed": True,
                            "data": compressed,
                            "original_size": original_size
                        }
                        
                        # Update compression stats
                        self.compression_stats["compressed_entries"] += 1
                        self.compression_stats["total_uncompressed_size"] += original_size
                        self.compression_stats["total_compressed_size"] += compressed_size
                        self._update_compression_ratio()
                        
                        return self.base_cache.put(key, cache_value)
                
                # Store uncompressed
                return self.base_cache.put(key, value)
                
            except Exception as e:
                logger.warning(f"Failed to cache value for key {key}: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        return self.base_cache.remove(key)
    
    def clear(self):
        """Clear cache and reset stats."""
        with self._lock:
            self.base_cache.clear()
            self.compression_stats = {
                "compressed_entries": 0,
                "total_uncompressed_size": 0,
                "total_compressed_size": 0,
                "compression_ratio": 0.0
            }
    
    def _update_compression_ratio(self):
        """Update compression ratio statistic."""
        if self.compression_stats["total_uncompressed_size"] > 0:
            self.compression_stats["compression_ratio"] = (
                self.compression_stats["total_compressed_size"] /
                self.compression_stats["total_uncompressed_size"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including compression stats."""
        with self._lock:
            base_stats = self.base_cache.get_stats()
            base_stats["compression"] = self.compression_stats.copy()
            base_stats["compression_threshold"] = self.compression_threshold
            return base_stats


class MultiLevelCache:
    """Multi-level cache with different storage tiers."""
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 500,
        l3_size: int = 2000,
        enable_disk_cache: bool = True,
        cache_dir: str = "./cache"
    ):
        """Initialize multi-level cache.
        
        Args:
            l1_size: L1 cache size (in-memory, fast)
            l2_size: L2 cache size (in-memory, compressed)
            l3_size: L3 cache size (disk-based)
            enable_disk_cache: Enable disk-based L3 cache
            cache_dir: Directory for disk cache
        """
        # L1: Fast in-memory cache
        self.l1_cache = LRUCache(max_size=l1_size, ttl=1800)  # 30 min
        
        # L2: Compressed in-memory cache
        self.l2_cache = CompressedCache(max_size=l2_size)
        
        # L3: Disk-based cache
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = Path(cache_dir)
        if enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.l3_index = LRUCache(max_size=l3_size, ttl=7200)  # 2 hours
        
        self.access_stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "total_misses": 0
        }
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        with self._lock:
            # Try L1 cache first
            value = self.l1_cache.get(key)
            if value is not None:
                self.access_stats["l1_hits"] += 1
                return value
            
            # Try L2 cache
            value = self.l2_cache.get(key)
            if value is not None:
                self.access_stats["l2_hits"] += 1
                # Promote to L1
                self.l1_cache.put(key, value)
                return value
            
            # Try L3 cache (disk)
            if self.enable_disk_cache:
                value = self._get_from_disk(key)
                if value is not None:
                    self.access_stats["l3_hits"] += 1
                    # Promote to L2 and L1
                    self.l2_cache.put(key, value)
                    self.l1_cache.put(key, value)
                    return value
            
            self.access_stats["total_misses"] += 1
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in multi-level cache."""
        with self._lock:
            try:
                # Put in all levels
                self.l1_cache.put(key, value)
                self.l2_cache.put(key, value)
                
                if self.enable_disk_cache:
                    self._put_to_disk(key, value)
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to put value in multi-level cache: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """Remove key from all cache levels."""
        with self._lock:
            removed = False
            
            if self.l1_cache.remove(key):
                removed = True
            
            if self.l2_cache.remove(key):
                removed = True
            
            if self.enable_disk_cache:
                if self._remove_from_disk(key):
                    removed = True
            
            return removed
    
    def clear(self):
        """Clear all cache levels."""
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            
            if self.enable_disk_cache:
                self.l3_index.clear()
                self._clear_disk_cache()
            
            self.access_stats = {
                "l1_hits": 0,
                "l2_hits": 0,
                "l3_hits": 0,
                "total_misses": 0
            }
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            # Check if key exists in index
            file_info = self.l3_index.get(key)
            if file_info is None:
                return None
            
            file_path = self.cache_dir / f"{key}.cache"
            if not file_path.exists():
                self.l3_index.remove(key)
                return None
            
            # Load and decompress
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            decompressed = zlib.decompress(compressed_data)
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def _put_to_disk(self, key: str, value: Any) -> bool:
        """Put value to disk cache."""
        try:
            # Serialize and compress
            serialized = pickle.dumps(value)
            compressed = zlib.compress(serialized)
            
            # Write to disk
            file_path = self.cache_dir / f"{key}.cache"
            with open(file_path, 'wb') as f:
                f.write(compressed)
            
            # Update index
            file_info = {
                "path": str(file_path),
                "size": len(compressed),
                "timestamp": time.time()
            }
            self.l3_index.put(key, file_info)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
            return False
    
    def _remove_from_disk(self, key: str) -> bool:
        """Remove key from disk cache."""
        try:
            file_path = self.cache_dir / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
            
            return self.l3_index.remove(key)
            
        except Exception as e:
            logger.warning(f"Failed to remove from disk cache: {e}")
            return False
    
    def _clear_disk_cache(self):
        """Clear disk cache directory."""
        try:
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_accesses = sum(self.access_stats.values())
            
            stats = {
                "access_stats": self.access_stats.copy(),
                "total_accesses": total_accesses,
                "cache_levels": {
                    "l1": self.l1_cache.get_stats(),
                    "l2": self.l2_cache.get_stats(),
                }
            }
            
            if self.enable_disk_cache:
                stats["cache_levels"]["l3"] = {
                    "size": len(self.l3_index.cache),
                    "max_size": self.l3_index.max_size,
                    "disk_usage_mb": self._get_disk_usage() / (1024 * 1024)
                }
            
            # Calculate hit rates
            if total_accesses > 0:
                stats["hit_rates"] = {
                    "l1_rate": self.access_stats["l1_hits"] / total_accesses,
                    "l2_rate": self.access_stats["l2_hits"] / total_accesses,
                    "l3_rate": self.access_stats["l3_hits"] / total_accesses,
                    "miss_rate": self.access_stats["total_misses"] / total_accesses,
                    "overall_hit_rate": (total_accesses - self.access_stats["total_misses"]) / total_accesses
                }
            
            return stats
    
    def _get_disk_usage(self) -> int:
        """Get total disk usage of cache."""
        try:
            total_size = 0
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    total_size += cache_file.stat().st_size
            return total_size
        except:
            return 0


class CacheKeyGenerator:
    """Generate consistent cache keys for complex objects."""
    
    @staticmethod
    def generate_key(data: Any, params: Dict[str, Any] = None) -> str:
        """Generate cache key for data and parameters.
        
        Args:
            data: Data to generate key for
            params: Additional parameters
            
        Returns:
            Cache key string
        """
        try:
            # Handle pandas DataFrame
            if isinstance(data, pd.DataFrame):
                data_hash = CacheKeyGenerator._hash_dataframe(data)
            else:
                # Generic serialization
                data_str = str(data)
                data_hash = hashlib.md5(data_str.encode()).hexdigest()
            
            # Include parameters if provided
            if params:
                params_str = json.dumps(params, sort_keys=True, default=str)
                params_hash = hashlib.md5(params_str.encode()).hexdigest()
                return f"{data_hash}_{params_hash}"
            
            return data_hash
            
        except Exception as e:
            # Fallback to timestamp-based key
            logger.warning(f"Failed to generate consistent key: {e}")
            return f"fallback_{int(time.time() * 1000)}"
    
    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """Generate hash for pandas DataFrame."""
        try:
            # Create a deterministic representation
            info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "index_hash": hashlib.md5(str(df.index.tolist()).encode()).hexdigest(),
                "values_hash": hashlib.md5(df.values.tobytes()).hexdigest() if df.size < 10000 else "large_df"
            }
            
            info_str = json.dumps(info, sort_keys=True)
            return hashlib.md5(info_str.encode()).hexdigest()
            
        except Exception:
            # Fallback to shape and column info
            basic_info = f"{df.shape}_{list(df.columns)}"
            return hashlib.md5(basic_info.encode()).hexdigest()


# Global cache instances
_global_cache = None
_global_key_generator = None


def get_global_cache(
    cache_type: str = "multi_level",
    **kwargs
) -> Union[LRUCache, CompressedCache, MultiLevelCache]:
    """Get or create global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        if cache_type == "lru":
            _global_cache = LRUCache(**kwargs)
        elif cache_type == "compressed":
            _global_cache = CompressedCache(**kwargs)
        elif cache_type == "multi_level":
            _global_cache = MultiLevelCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    return _global_cache


def get_global_key_generator() -> CacheKeyGenerator:
    """Get global cache key generator."""
    global _global_key_generator
    
    if _global_key_generator is None:
        _global_key_generator = CacheKeyGenerator()
    
    return _global_key_generator