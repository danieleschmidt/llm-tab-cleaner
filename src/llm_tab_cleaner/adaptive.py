"""Adaptive cleaning features with self-learning capabilities."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import threading

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import Fix, CleaningReport

logger = logging.getLogger(__name__)


@dataclass
class CleaningPattern:
    """Represents a learned cleaning pattern."""
    input_pattern: str
    output_pattern: str
    confidence: float
    frequency: int
    last_used: float
    metadata: Dict[str, Any]


@dataclass
class CacheEntry:
    """Cache entry for cleaning results."""
    input_hash: str
    output_value: Any
    confidence: float
    timestamp: float
    access_count: int
    column_type: str


class AdaptiveCache:
    """Intelligent caching system that learns from usage patterns."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live for cached entries in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns = defaultdict(list)
        self._lock = threading.RLock()
        
    def _make_key(self, value: Any, column: str, context: Dict[str, Any]) -> str:
        """Create cache key from inputs."""
        key_data = {
            "value": str(value),
            "column": column,
            "context_hash": hashlib.md5(
                json.dumps(context, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
    
    def get(self, value: Any, column: str, context: Dict[str, Any]) -> Optional[Tuple[Any, float]]:
        """Get cached cleaning result."""
        key = self._make_key(value, column, context)
        
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if entry is expired
            if time.time() - entry.timestamp > self.ttl:
                del self.cache[key]
                return None
            
            # Update access pattern
            entry.access_count += 1
            self.access_patterns[key].append(time.time())
            
            # Keep only recent access times
            cutoff = time.time() - self.ttl
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff
            ]
            
            logger.debug(f"Cache hit for column {column}, confidence: {entry.confidence:.3f}")
            return entry.output_value, entry.confidence
    
    def put(self, value: Any, column: str, context: Dict[str, Any], 
            output_value: Any, confidence: float):
        """Store cleaning result in cache."""
        key = self._make_key(value, column, context)
        
        with self._lock:
            # Evict old entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            entry = CacheEntry(
                input_hash=key,
                output_value=output_value,
                confidence=confidence,
                timestamp=time.time(),
                access_count=1,
                column_type=context.get("data_type", "unknown")
            )
            
            self.cache[key] = entry
            self.access_patterns[key] = [time.time()]
            
            logger.debug(f"Cache stored for column {column}")
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.cache:
            return
        
        # Calculate access frequency scores
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Recent access frequency
            recent_accesses = len(self.access_patterns.get(key, []))
            
            # Age penalty
            age_penalty = (current_time - entry.timestamp) / self.ttl
            
            # Confidence bonus
            confidence_bonus = entry.confidence
            
            scores[key] = recent_accesses * confidence_bonus / (1 + age_penalty)
        
        # Remove lowest scoring entries
        to_remove = sorted(scores.keys(), key=lambda k: scores[k])[:max(1, len(self.cache) // 10)]
        
        for key in to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
        
        logger.debug(f"Cache evicted {len(to_remove)} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            avg_confidence = np.mean([entry.confidence for entry in self.cache.values()]) if self.cache else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "average_confidence": avg_confidence,
                "column_type_distribution": Counter(
                    entry.column_type for entry in self.cache.values()
                )
            }


class PatternLearner:
    """Learns cleaning patterns from successful fixes."""
    
    def __init__(self, max_patterns: int = 1000):
        """Initialize pattern learner.
        
        Args:
            max_patterns: Maximum number of patterns to store
        """
        self.max_patterns = max_patterns
        self.patterns: List[CleaningPattern] = []
        self.pattern_index = {}  # For fast lookup
        self._lock = threading.Lock()
        
    def learn_from_fix(self, fix: Fix, context: Dict[str, Any]):
        """Learn a new pattern from a successful fix."""
        if fix.confidence < 0.8:  # Only learn from high-confidence fixes
            return
        
        pattern = CleaningPattern(
            input_pattern=str(fix.original),
            output_pattern=str(fix.cleaned),
            confidence=fix.confidence,
            frequency=1,
            last_used=time.time(),
            metadata={
                "column": fix.column,
                "rule": fix.rule_applied,
                "data_type": context.get("data_type", "unknown")
            }
        )
        
        pattern_key = f"{fix.original}->{fix.cleaned}"
        
        with self._lock:
            if pattern_key in self.pattern_index:
                # Update existing pattern
                existing = self.patterns[self.pattern_index[pattern_key]]
                existing.frequency += 1
                existing.confidence = max(existing.confidence, fix.confidence)
                existing.last_used = time.time()
            else:
                # Add new pattern
                if len(self.patterns) >= self.max_patterns:
                    self._evict_old_patterns()
                
                self.patterns.append(pattern)
                self.pattern_index[pattern_key] = len(self.patterns) - 1
            
            logger.debug(f"Learned pattern: {fix.original} -> {fix.cleaned}")
    
    def suggest_fix(self, value: Any, column: str, context: Dict[str, Any]) -> Optional[Tuple[Any, float]]:
        """Suggest a fix based on learned patterns."""
        value_str = str(value)
        
        with self._lock:
            # Look for exact match first
            exact_key = f"{value_str}->*"
            matching_patterns = [
                p for p in self.patterns 
                if p.input_pattern == value_str
                and p.metadata.get("data_type") == context.get("data_type", "unknown")
            ]
            
            if matching_patterns:
                # Return most frequent, recent, and confident pattern
                best = max(matching_patterns, key=lambda p: (
                    p.frequency * p.confidence * (1 / (time.time() - p.last_used + 1))
                ))
                best.last_used = time.time()
                logger.debug(f"Pattern match for {value}: {best.output_pattern}")
                return best.output_pattern, best.confidence
            
            # Look for fuzzy matches using clustering
            return self._fuzzy_pattern_match(value_str, context)
    
    def _fuzzy_pattern_match(self, value: str, context: Dict[str, Any]) -> Optional[Tuple[Any, float]]:
        """Find fuzzy pattern matches using text similarity."""
        try:
            data_type = context.get("data_type", "unknown")
            relevant_patterns = [
                p for p in self.patterns 
                if p.metadata.get("data_type") == data_type
                and p.frequency > 1  # Only patterns seen multiple times
            ]
            
            if len(relevant_patterns) < 2:
                return None
            
            # Use TF-IDF to find similar input patterns
            input_patterns = [p.input_pattern for p in relevant_patterns]
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            
            try:
                pattern_vectors = vectorizer.fit_transform(input_patterns + [value])
                target_vector = pattern_vectors[-1]
                similarity_scores = (pattern_vectors[:-1] * target_vector.T).toarray().flatten()
                
                # Find most similar pattern
                best_idx = np.argmax(similarity_scores)
                if similarity_scores[best_idx] > 0.7:  # Threshold for similarity
                    best_pattern = relevant_patterns[best_idx]
                    confidence = best_pattern.confidence * similarity_scores[best_idx]
                    
                    logger.debug(f"Fuzzy match for {value}: {best_pattern.output_pattern} (sim: {similarity_scores[best_idx]:.3f})")
                    return best_pattern.output_pattern, confidence
                    
            except ValueError:
                # Fallback for edge cases
                pass
                
        except Exception as e:
            logger.warning(f"Error in fuzzy pattern matching: {e}")
        
        return None
    
    def _evict_old_patterns(self):
        """Remove old, infrequent patterns."""
        # Sort by score (frequency * confidence / age)
        current_time = time.time()
        scored_patterns = []
        
        for i, pattern in enumerate(self.patterns):
            age_penalty = (current_time - pattern.last_used) / 3600  # hours
            score = pattern.frequency * pattern.confidence / (1 + age_penalty)
            scored_patterns.append((score, i, pattern))
        
        # Keep top patterns
        scored_patterns.sort(reverse=True, key=lambda x: x[0])
        keep_count = int(self.max_patterns * 0.8)  # Keep 80%
        
        self.patterns = [x[2] for x in scored_patterns[:keep_count]]
        
        # Rebuild index
        self.pattern_index = {}
        for i, pattern in enumerate(self.patterns):
            key = f"{pattern.input_pattern}->{pattern.output_pattern}"
            self.pattern_index[key] = i
        
        logger.debug(f"Evicted {len(scored_patterns) - keep_count} old patterns")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pattern learning statistics."""
        with self._lock:
            if not self.patterns:
                return {"pattern_count": 0}
            
            return {
                "pattern_count": len(self.patterns),
                "max_patterns": self.max_patterns,
                "average_confidence": np.mean([p.confidence for p in self.patterns]),
                "average_frequency": np.mean([p.frequency for p in self.patterns]),
                "data_type_distribution": Counter(
                    p.metadata.get("data_type", "unknown") for p in self.patterns
                )
            }


class AutoScalingProcessor:
    """Automatically scales processing based on workload."""
    
    def __init__(self, initial_batch_size: int = 100, max_batch_size: int = 5000):
        """Initialize auto-scaling processor.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
        # Performance tracking
        self.performance_history = []
        self.load_factor = 1.0
        self._lock = threading.Lock()
        
    def process_batch(self, items: List[Any], processor_func, **kwargs) -> List[Any]:
        """Process items in optimally-sized batches."""
        if not items:
            return []
        
        start_time = time.time()
        results = []
        
        # Calculate optimal batch size based on recent performance
        batch_size = self._calculate_optimal_batch_size(len(items))
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_start = time.time()
            
            try:
                batch_results = processor_func(batch, **kwargs)
                results.extend(batch_results)
                
                # Record performance
                batch_time = time.time() - batch_start
                throughput = len(batch) / batch_time if batch_time > 0 else 0
                
                self._record_performance(len(batch), throughput, success=True)
                
            except Exception as e:
                logger.warning(f"Batch processing failed for batch size {len(batch)}: {e}")
                
                # Fallback to smaller batches
                if len(batch) > 1:
                    smaller_batch_size = max(1, len(batch) // 2)
                    for j in range(0, len(batch), smaller_batch_size):
                        small_batch = batch[j:j + smaller_batch_size]
                        try:
                            small_results = processor_func(small_batch, **kwargs)
                            results.extend(small_results)
                        except Exception:
                            # Process individually as last resort
                            for item in small_batch:
                                try:
                                    item_result = processor_func([item], **kwargs)
                                    results.extend(item_result)
                                except Exception:
                                    logger.error(f"Failed to process item: {item}")
                
                self._record_performance(len(batch), 0, success=False)
        
        total_time = time.time() - start_time
        logger.info(f"Processed {len(items)} items in {total_time:.2f}s using batch size {batch_size}")
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on performance history."""
        with self._lock:
            if not self.performance_history:
                return min(self.current_batch_size, total_items)
            
            # Analyze recent performance
            recent_perf = self.performance_history[-10:]  # Last 10 batches
            
            if len(recent_perf) >= 3:
                # Calculate average throughput for different batch sizes
                size_performance = defaultdict(list)
                for perf in recent_perf:
                    if perf["success"]:
                        size_performance[perf["batch_size"]].append(perf["throughput"])
                
                if size_performance:
                    # Find batch size with best average throughput
                    best_size = max(
                        size_performance.keys(),
                        key=lambda size: np.mean(size_performance[size])
                    )
                    
                    # Adjust current batch size towards optimal
                    if best_size > self.current_batch_size:
                        self.current_batch_size = min(
                            int(self.current_batch_size * 1.2),
                            self.max_batch_size
                        )
                    elif best_size < self.current_batch_size:
                        self.current_batch_size = max(
                            int(self.current_batch_size * 0.8),
                            self.initial_batch_size
                        )
            
            return min(self.current_batch_size, total_items)
    
    def _record_performance(self, batch_size: int, throughput: float, success: bool):
        """Record batch processing performance."""
        with self._lock:
            perf_record = {
                "timestamp": time.time(),
                "batch_size": batch_size,
                "throughput": throughput,
                "success": success
            }
            
            self.performance_history.append(perf_record)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self._lock:
            if not self.performance_history:
                return {
                    "current_batch_size": self.current_batch_size,
                    "performance_samples": 0,
                    "success_rate": 1.0,
                    "average_throughput": 0.0
                }
            
            recent_perf = [p for p in self.performance_history if p["success"]]
            
            return {
                "current_batch_size": self.current_batch_size,
                "max_batch_size": self.max_batch_size,
                "performance_samples": len(self.performance_history),
                "success_rate": len(recent_perf) / len(self.performance_history) if self.performance_history else 1.0,
                "average_throughput": np.mean([p["throughput"] for p in recent_perf]) if recent_perf else 0,
                "optimal_batch_sizes": Counter([p["batch_size"] for p in recent_perf])
            }


def save_adaptive_state(cache: AdaptiveCache, learner: PatternLearner, 
                       processor: AutoScalingProcessor, filepath: str):
    """Save adaptive components state to file."""
    try:
        state = {
            "timestamp": time.time(),
            "cache_stats": cache.get_stats(),
            "pattern_stats": learner.get_stats(),
            "processor_stats": processor.get_stats(),
            "patterns": [asdict(p) for p in learner.patterns],
            "version": "1.0"
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Adaptive state saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save adaptive state: {e}")


def load_adaptive_state(filepath: str) -> Optional[Dict[str, Any]]:
    """Load adaptive components state from file."""
    try:
        if not Path(filepath).exists():
            return None
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        logger.info(f"Adaptive state loaded from {filepath}")
        return state
        
    except Exception as e:
        logger.error(f"Failed to load adaptive state: {e}")
        return None