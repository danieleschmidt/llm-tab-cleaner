"""Auto-scaling and distributed processing enhancements."""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading
import queue
import psutil
import math
import json
from pathlib import Path

import pandas as pd
import numpy as np

from .core import TableCleaner, CleaningReport, Fix


logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_threads: int
    active_processes: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Decision about how to scale processing."""
    target_workers: int
    target_batch_size: int
    use_distributed: bool
    use_async: bool
    reasoning: str
    confidence: float


class ResourceMonitor:
    """Monitors system resources for scaling decisions."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._metrics_history: List[ResourceMetrics] = []
        self._max_history = 60  # Keep 60 samples max
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            
            # Process counts
            active_threads = threading.active_count()
            active_processes = len(psutil.pids())
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_sent_mb=net_sent_mb,
                network_io_recv_mb=net_recv_mb,
                active_threads=active_threads,
                active_processes=active_processes
            )
            
        except Exception as e:
            logger.warning(f"Error getting resource metrics: {e}")
            return ResourceMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_average_metrics(self, window_seconds: int = 30) -> Optional[ResourceMetrics]:
        """Get average metrics over a time window."""
        with self._lock:
            if not self._metrics_history:
                return None
            
            cutoff_time = time.time() - window_seconds
            recent_metrics = [m for m in self._metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return None
            
            # Calculate averages
            return ResourceMetrics(
                cpu_percent=sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                memory_percent=sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                disk_io_read_mb=sum(m.disk_io_read_mb for m in recent_metrics) / len(recent_metrics),
                disk_io_write_mb=sum(m.disk_io_write_mb for m in recent_metrics) / len(recent_metrics),
                network_io_sent_mb=sum(m.network_io_sent_mb for m in recent_metrics) / len(recent_metrics),
                network_io_recv_mb=sum(m.network_io_recv_mb for m in recent_metrics) / len(recent_metrics),
                active_threads=int(sum(m.active_threads for m in recent_metrics) / len(recent_metrics)),
                active_processes=int(sum(m.active_processes for m in recent_metrics) / len(recent_metrics))
            )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                
                with self._lock:
                    self._metrics_history.append(metrics)
                    
                    # Trim history if too long
                    if len(self._metrics_history) > self._max_history:
                        self._metrics_history.pop(0)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)


class IntelligentScaler:
    """Intelligent scaling engine for optimal resource utilization."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = None,
        min_batch_size: int = 100,
        max_batch_size: int = 10000,
        target_cpu_utilization: float = 0.7,
        target_memory_utilization: float = 0.8
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count()
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_cpu_utilization = target_cpu_utilization
        self.target_memory_utilization = target_memory_utilization
        
        self.resource_monitor = ResourceMonitor()
        self.scaling_history: List[Dict[str, Any]] = []
        
    def make_scaling_decision(
        self, 
        data_size: int,
        current_workers: int = 1,
        current_batch_size: int = 1000,
        performance_history: Optional[List[Dict[str, float]]] = None
    ) -> ScalingDecision:
        """Make intelligent scaling decision based on current conditions."""
        
        # Get current resource metrics
        current_metrics = self.resource_monitor.get_current_metrics()
        avg_metrics = self.resource_monitor.get_average_metrics(30)
        
        if avg_metrics is None:
            avg_metrics = current_metrics
        
        # Analyze workload characteristics
        workload_intensity = self._estimate_workload_intensity(data_size, performance_history)
        
        # Calculate optimal worker count
        target_workers = self._calculate_optimal_workers(
            data_size, avg_metrics, workload_intensity
        )
        
        # Calculate optimal batch size
        target_batch_size = self._calculate_optimal_batch_size(
            data_size, target_workers, avg_metrics
        )
        
        # Decide on processing strategy
        use_distributed = data_size > 100000 or target_workers > mp.cpu_count()
        use_async = workload_intensity < 0.5  # For I/O bound tasks
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            current_metrics, avg_metrics, data_size, target_workers, target_batch_size, use_distributed, use_async
        )
        
        # Calculate confidence based on data quality and resource stability
        confidence = self._calculate_confidence(avg_metrics, performance_history)
        
        decision = ScalingDecision(
            target_workers=target_workers,
            target_batch_size=target_batch_size,
            use_distributed=use_distributed,
            use_async=use_async,
            reasoning=reasoning,
            confidence=confidence
        )
        
        # Log the decision
        self.scaling_history.append({
            "timestamp": time.time(),
            "data_size": data_size,
            "current_workers": current_workers,
            "current_batch_size": current_batch_size,
            "decision": decision.__dict__,
            "metrics": avg_metrics.__dict__
        })
        
        return decision
    
    def _estimate_workload_intensity(
        self, 
        data_size: int, 
        performance_history: Optional[List[Dict[str, float]]]
    ) -> float:
        """Estimate computational intensity of the workload (0.0 = I/O bound, 1.0 = CPU bound)."""
        
        if not performance_history:
            # Default estimate based on data size
            return min(1.0, data_size / 100000)
        
        # Analyze CPU vs I/O wait patterns from history
        cpu_times = [p.get("cpu_time", 0) for p in performance_history]
        total_times = [p.get("total_time", 1) for p in performance_history]
        
        if not cpu_times or not total_times:
            return 0.5
        
        avg_cpu_ratio = sum(cpu_times) / sum(total_times) if sum(total_times) > 0 else 0.5
        return min(1.0, max(0.0, avg_cpu_ratio))
    
    def _calculate_optimal_workers(
        self, 
        data_size: int, 
        metrics: ResourceMetrics, 
        workload_intensity: float
    ) -> int:
        """Calculate optimal number of workers."""
        
        # Start with available CPU cores
        available_cores = mp.cpu_count()
        
        # Adjust based on current CPU utilization
        if metrics.cpu_percent > 80:
            # System is already loaded, be conservative
            target_workers = max(1, available_cores // 2)
        elif metrics.cpu_percent < 20:
            # System has capacity, can use more workers
            target_workers = available_cores
        else:
            # Scale proportionally to available capacity
            capacity_factor = (100 - metrics.cpu_percent) / 100
            target_workers = max(1, int(available_cores * capacity_factor))
        
        # Adjust based on memory constraints
        if metrics.memory_percent > 85:
            target_workers = max(1, target_workers // 2)
        
        # Adjust based on workload intensity
        if workload_intensity > 0.8:  # CPU-intensive
            target_workers = min(target_workers, available_cores)
        else:  # I/O intensive, can benefit from more workers
            target_workers = min(target_workers * 2, available_cores * 2)
        
        # Adjust based on data size
        if data_size < 1000:
            target_workers = 1  # Small datasets don't benefit from parallelization
        elif data_size > 1000000:
            target_workers = max(target_workers, available_cores)
        
        return max(self.min_workers, min(self.max_workers, target_workers))
    
    def _calculate_optimal_batch_size(
        self, 
        data_size: int, 
        target_workers: int, 
        metrics: ResourceMetrics
    ) -> int:
        """Calculate optimal batch size for processing."""
        
        # Base batch size on data size and worker count
        base_batch_size = max(self.min_batch_size, data_size // (target_workers * 4))
        
        # Adjust based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate memory per batch (rough heuristic: 1GB per 100k rows)
        memory_per_batch = (base_batch_size / 100000) * 1.0
        max_batches_by_memory = max(1, int(available_memory_gb / memory_per_batch))
        
        if max_batches_by_memory < target_workers:
            # Reduce batch size to fit memory constraints
            base_batch_size = max(self.min_batch_size, 
                                 int(base_batch_size * max_batches_by_memory / target_workers))
        
        # Adjust based on I/O patterns
        if metrics.disk_io_read_mb + metrics.disk_io_write_mb > 100:  # High I/O
            # Smaller batches for better I/O patterns
            base_batch_size = max(self.min_batch_size, base_batch_size // 2)
        
        return max(self.min_batch_size, min(self.max_batch_size, base_batch_size))
    
    def _generate_reasoning(
        self, 
        current_metrics: ResourceMetrics, 
        avg_metrics: ResourceMetrics, 
        data_size: int, 
        target_workers: int, 
        target_batch_size: int, 
        use_distributed: bool, 
        use_async: bool
    ) -> str:
        """Generate human-readable reasoning for the scaling decision."""
        
        factors = []
        
        if avg_metrics.cpu_percent > 80:
            factors.append(f"high CPU usage ({avg_metrics.cpu_percent:.1f}%)")
        elif avg_metrics.cpu_percent < 20:
            factors.append(f"low CPU usage ({avg_metrics.cpu_percent:.1f}%)")
        
        if avg_metrics.memory_percent > 85:
            factors.append(f"high memory usage ({avg_metrics.memory_percent:.1f}%)")
        
        if data_size > 1000000:
            factors.append(f"large dataset ({data_size:,} rows)")
        elif data_size < 1000:
            factors.append(f"small dataset ({data_size} rows)")
        
        if use_distributed:
            factors.append("requires distributed processing")
        
        if use_async:
            factors.append("I/O-bound workload detected")
        
        reasoning = f"Scaling to {target_workers} workers with batch size {target_batch_size:,}"
        if factors:
            reasoning += f" due to: {', '.join(factors)}"
        
        return reasoning
    
    def _calculate_confidence(
        self, 
        avg_metrics: ResourceMetrics, 
        performance_history: Optional[List[Dict[str, float]]]
    ) -> float:
        """Calculate confidence in the scaling decision."""
        
        confidence = 0.8  # Base confidence
        
        # Reduce confidence if metrics are unstable
        current_metrics = self.resource_monitor.get_current_metrics()
        cpu_variance = abs(current_metrics.cpu_percent - avg_metrics.cpu_percent)
        memory_variance = abs(current_metrics.memory_percent - avg_metrics.memory_percent)
        
        if cpu_variance > 20 or memory_variance > 20:
            confidence -= 0.2
        
        # Increase confidence if we have good performance history
        if performance_history and len(performance_history) >= 3:
            # Check for consistent performance
            times = [p.get("total_time", 1) for p in performance_history[-3:]]
            if times and max(times) / min(times) < 2.0:  # Less than 2x variance
                confidence += 0.1
        
        # Reduce confidence for extreme resource conditions
        if avg_metrics.cpu_percent > 95 or avg_metrics.memory_percent > 95:
            confidence -= 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.resource_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.resource_monitor.stop_monitoring()


class AdaptiveTableCleaner:
    """Auto-scaling table cleaner with intelligent resource management."""
    
    def __init__(
        self,
        base_cleaner_factory: Callable[[], TableCleaner],
        scaler: Optional[IntelligentScaler] = None,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        self.base_cleaner_factory = base_cleaner_factory
        self.scaler = scaler or IntelligentScaler()
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self._cache: Dict[str, Tuple[pd.DataFrame, CleaningReport]] = {}
        self._cache_access_order: List[str] = []
        
        # Start monitoring
        self.scaler.start_monitoring()
        
        logger.info("AdaptiveTableCleaner initialized with auto-scaling")
    
    def clean(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sample_rate: float = 1.0,
        force_workers: Optional[int] = None,
        force_batch_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean DataFrame with automatic scaling optimization."""
        
        start_time = time.time()
        data_size = len(df)
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(df, columns, sample_rate)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for dataset (size: {data_size})")
                return cached_result
        
        # Make scaling decision
        scaling_decision = self.scaler.make_scaling_decision(
            data_size=data_size,
            performance_history=self.performance_history
        )
        
        # Override with forced parameters if provided
        target_workers = force_workers or scaling_decision.target_workers
        target_batch_size = force_batch_size or scaling_decision.target_batch_size
        
        logger.info(f"Scaling decision: {scaling_decision.reasoning} (confidence: {scaling_decision.confidence:.2f})")
        
        # Execute cleaning with optimal strategy
        if scaling_decision.use_distributed and data_size > target_batch_size:
            cleaned_df, report = self._clean_distributed(df, columns, sample_rate, target_workers, target_batch_size)
        elif scaling_decision.use_async:
            cleaned_df, report = asyncio.run(self._clean_async(df, columns, sample_rate, target_workers))
        else:
            cleaned_df, report = self._clean_single_process(df, columns, sample_rate)
        
        # Record performance metrics
        processing_time = time.time() - start_time
        self._record_performance(data_size, processing_time, target_workers, target_batch_size)
        
        # Update cache
        if self.enable_caching:
            self._add_to_cache(cache_key, (cleaned_df, report))
        
        logger.info(f"Cleaning completed in {processing_time:.2f}s using {target_workers} workers")
        
        return cleaned_df, report
    
    def _clean_distributed(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]], 
        sample_rate: float,
        num_workers: int,
        batch_size: int
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean using distributed processing."""
        
        logger.info(f"Starting distributed cleaning with {num_workers} workers, batch size {batch_size}")
        
        # Split DataFrame into batches
        batches = self._split_dataframe(df, batch_size)
        
        # Process batches in parallel
        cleaned_batches = []
        all_fixes = []
        total_processing_time = 0.0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch, batch, columns, sample_rate): i
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    cleaned_batch, batch_report = future.result()
                    cleaned_batches.append((batch_idx, cleaned_batch))
                    all_fixes.extend(batch_report.fixes)
                    total_processing_time += batch_report.processing_time
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Use original batch if processing failed
                    cleaned_batches.append((batch_idx, batches[batch_idx]))
        
        # Reassemble DataFrame
        cleaned_batches.sort(key=lambda x: x[0])  # Sort by batch index
        cleaned_df = pd.concat([batch for _, batch in cleaned_batches], ignore_index=True)
        
        # Create combined report
        quality_score = self._calculate_quality_score(df, cleaned_df, all_fixes)
        
        report = CleaningReport(
            total_fixes=len([f for f in all_fixes if f.confidence >= 0.85]),
            quality_score=quality_score,
            fixes=all_fixes,
            processing_time=total_processing_time / num_workers  # Approximate parallel time
        )
        
        return cleaned_df, report
    
    async def _clean_async(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]], 
        sample_rate: float,
        num_workers: int
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean using async processing for I/O-bound tasks."""
        
        logger.info(f"Starting async cleaning with {num_workers} workers")
        
        # For async processing, we use thread-based parallelism
        loop = asyncio.get_event_loop()
        
        # Split work among threads
        batch_size = max(100, len(df) // num_workers)
        batches = self._split_dataframe(df, batch_size)
        
        # Process batches concurrently
        tasks = [
            loop.run_in_executor(
                None, 
                self._process_batch, 
                batch, 
                columns, 
                sample_rate
            )
            for batch in batches
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        cleaned_batches = []
        all_fixes = []
        total_processing_time = 0.0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing async batch {i}: {result}")
                cleaned_batches.append(batches[i])
            else:
                cleaned_batch, batch_report = result
                cleaned_batches.append(cleaned_batch)
                all_fixes.extend(batch_report.fixes)
                total_processing_time += batch_report.processing_time
        
        # Reassemble DataFrame
        cleaned_df = pd.concat(cleaned_batches, ignore_index=True)
        
        # Create combined report
        quality_score = self._calculate_quality_score(df, cleaned_df, all_fixes)
        
        report = CleaningReport(
            total_fixes=len([f for f in all_fixes if f.confidence >= 0.85]),
            quality_score=quality_score,
            fixes=all_fixes,
            processing_time=total_processing_time / num_workers  # Approximate parallel time
        )
        
        return cleaned_df, report
    
    def _clean_single_process(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]], 
        sample_rate: float
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean using single process (fallback method)."""
        
        logger.info("Starting single-process cleaning")
        cleaner = self.base_cleaner_factory()
        return cleaner.clean(df, columns, sample_rate)
    
    def _process_batch(
        self, 
        batch: pd.DataFrame, 
        columns: Optional[List[str]], 
        sample_rate: float
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Process a single batch of data."""
        
        cleaner = self.base_cleaner_factory()
        return cleaner.clean(batch, columns, sample_rate)
    
    def _split_dataframe(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Split DataFrame into batches."""
        
        if batch_size >= len(df):
            return [df]
        
        batches = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].copy()
            batches.append(batch)
        
        return batches
    
    def _calculate_quality_score(
        self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame, 
        fixes: List[Fix]
    ) -> float:
        """Calculate overall quality improvement score."""
        
        if len(fixes) == 0:
            return 1.0
        
        successful_fixes = len([f for f in fixes if f.confidence >= 0.85])
        total_cells = original_df.shape[0] * original_df.shape[1]
        
        improvement_ratio = successful_fixes / total_cells if total_cells > 0 else 0
        avg_confidence = sum(f.confidence for f in fixes if f.confidence >= 0.85) / max(successful_fixes, 1)
        
        return min(1.0, 0.8 + improvement_ratio * 0.1 + (avg_confidence - 0.85) * 0.1)
    
    def _record_performance(
        self, 
        data_size: int, 
        processing_time: float, 
        workers: int, 
        batch_size: int
    ):
        """Record performance metrics for future scaling decisions."""
        
        metrics = {
            "timestamp": time.time(),
            "data_size": data_size,
            "processing_time": processing_time,
            "workers": workers,
            "batch_size": batch_size,
            "throughput": data_size / processing_time if processing_time > 0 else 0
        }
        
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def _generate_cache_key(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]], 
        sample_rate: float
    ) -> str:
        """Generate cache key for DataFrame cleaning request."""
        
        # Create a hash based on DataFrame content and parameters
        import hashlib
        
        # Sample some rows for hashing (to avoid hashing entire large DataFrames)
        sample_size = min(1000, len(df))
        sample_df = df.head(sample_size)
        
        content_hash = hashlib.md5(
            pd.util.hash_pandas_object(sample_df).values.tobytes()
        ).hexdigest()[:16]
        
        columns_str = str(sorted(columns)) if columns else "all"
        
        cache_key = f"{content_hash}_{len(df)}_{columns_str}_{sample_rate}"
        
        return cache_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, CleaningReport]]:
        """Get result from cache."""
        
        if cache_key in self._cache:
            # Update access order
            self._cache_access_order.remove(cache_key)
            self._cache_access_order.append(cache_key)
            return self._cache[cache_key]
        
        return None
    
    def _add_to_cache(self, cache_key: str, result: Tuple[pd.DataFrame, CleaningReport]):
        """Add result to cache."""
        
        # Remove oldest items if cache is full
        while len(self._cache) >= self.cache_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
        self._cache_access_order.append(cache_key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_history = self.performance_history[-20:]  # Last 20 operations
        
        throughputs = [h["throughput"] for h in recent_history]
        processing_times = [h["processing_time"] for h in recent_history]
        data_sizes = [h["data_size"] for h in recent_history]
        
        return {
            "total_operations": len(self.performance_history),
            "recent_operations": len(recent_history),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "avg_data_size": sum(data_sizes) / len(data_sizes) if data_sizes else 0,
            "cache_hit_ratio": len(self._cache) / max(1, len(self.performance_history)),
            "scaling_decisions": len(self.scaler.scaling_history)
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.scaler.stop_monitoring()


# Factory function
def create_adaptive_cleaner(
    llm_provider: str = "local",
    confidence_threshold: float = 0.85,
    **cleaner_kwargs
) -> AdaptiveTableCleaner:
    """Create an adaptive table cleaner with auto-scaling."""
    
    def cleaner_factory():
        return TableCleaner(
            llm_provider=llm_provider,
            confidence_threshold=confidence_threshold,
            **cleaner_kwargs
        )
    
    return AdaptiveTableCleaner(cleaner_factory)