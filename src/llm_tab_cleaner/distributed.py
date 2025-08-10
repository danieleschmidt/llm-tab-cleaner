"""Distributed processing and auto-scaling capabilities."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union
import multiprocessing
import threading
from pathlib import Path

import pandas as pd
import numpy as np

from .core import TableCleaner, CleaningReport
from .adaptive import AdaptiveCache, PatternLearner

logger = logging.getLogger(__name__)


@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed system."""
    node_id: str
    capacity: int
    current_load: int
    last_heartbeat: float
    status: str  # "active", "busy", "offline"
    performance_metrics: Dict[str, float]


@dataclass
class ProcessingTask:
    """Task for distributed processing."""
    task_id: str
    data_chunk: pd.DataFrame
    processing_params: Dict[str, Any]
    priority: int = 1
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class LoadBalancer:
    """Intelligent load balancer for distributed processing."""
    
    def __init__(self, strategy: str = "adaptive"):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy ("round_robin", "least_loaded", "adaptive")
        """
        self.strategy = strategy
        self.nodes: Dict[str, ProcessingNode] = {}
        self.current_index = 0
        self._lock = threading.RLock()
        
        # Performance tracking
        self.task_history = []
        self.performance_weights = {
            "cpu_usage": 0.3,
            "memory_usage": 0.3,
            "throughput": 0.4
        }
    
    def register_node(self, node: ProcessingNode):
        """Register a processing node."""
        with self._lock:
            self.nodes[node.node_id] = node
            logger.info(f"Registered processing node: {node.node_id} (capacity: {node.capacity})")
    
    def unregister_node(self, node_id: str):
        """Unregister a processing node."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered processing node: {node_id}")
    
    def select_node(self, task: ProcessingTask) -> Optional[ProcessingNode]:
        """Select the best node for a task."""
        with self._lock:
            active_nodes = [node for node in self.nodes.values() if node.status == "active"]
            
            if not active_nodes:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(active_nodes)
            elif self.strategy == "least_loaded":
                return self._least_loaded_selection(active_nodes)
            elif self.strategy == "adaptive":
                return self._adaptive_selection(active_nodes, task)
            else:
                return active_nodes[0]  # Default to first available
    
    def _round_robin_selection(self, nodes: List[ProcessingNode]) -> ProcessingNode:
        """Round-robin node selection."""
        selected = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return selected
    
    def _least_loaded_selection(self, nodes: List[ProcessingNode]) -> ProcessingNode:
        """Select node with least current load."""
        return min(nodes, key=lambda n: n.current_load / n.capacity)
    
    def _adaptive_selection(self, nodes: List[ProcessingNode], task: ProcessingTask) -> ProcessingNode:
        """Adaptive node selection based on performance metrics."""
        if not self.task_history:
            return self._least_loaded_selection(nodes)
        
        # Calculate composite scores
        node_scores = {}
        for node in nodes:
            score = 0.0
            
            # Load factor (lower is better)
            load_factor = node.current_load / node.capacity
            score += (1 - load_factor) * self.performance_weights["cpu_usage"]
            
            # Throughput factor
            throughput = node.performance_metrics.get("throughput", 1.0)
            score += min(throughput / 1000, 1.0) * self.performance_weights["throughput"]
            
            # Memory efficiency
            memory_efficiency = 1 - node.performance_metrics.get("memory_usage", 0.5)
            score += memory_efficiency * self.performance_weights["memory_usage"]
            
            # Task size consideration
            data_size_mb = len(task.data_chunk) * len(task.data_chunk.columns) * 8 / (1024 * 1024)
            if data_size_mb > 100:  # Large task
                # Prefer nodes with higher capacity
                capacity_bonus = node.capacity / max(n.capacity for n in nodes)
                score += capacity_bonus * 0.2
            
            node_scores[node.node_id] = score
        
        # Select node with highest score
        best_node_id = max(node_scores.keys(), key=lambda k: node_scores[k])
        return self.nodes[best_node_id]
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update performance metrics for a node."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].performance_metrics.update(metrics)
                self.nodes[node_id].last_heartbeat = time.time()
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        with self._lock:
            total_capacity = sum(node.capacity for node in self.nodes.values())
            total_load = sum(node.current_load for node in self.nodes.values())
            active_nodes = len([n for n in self.nodes.values() if n.status == "active"])
            
            return {
                "total_nodes": len(self.nodes),
                "active_nodes": active_nodes,
                "total_capacity": total_capacity,
                "current_load": total_load,
                "utilization": (total_load / total_capacity) if total_capacity > 0 else 0,
                "nodes": {
                    node_id: {
                        "capacity": node.capacity,
                        "load": node.current_load,
                        "utilization": node.current_load / node.capacity,
                        "status": node.status,
                        "last_heartbeat": node.last_heartbeat
                    }
                    for node_id, node in self.nodes.items()
                }
            }


class DistributedCleaner:
    """Distributed data cleaning with auto-scaling."""
    
    def __init__(
        self,
        base_cleaner_config: Dict[str, Any],
        max_workers: int = None,
        chunk_size: int = 10000,
        enable_process_pool: bool = True,
        load_balancer_strategy: str = "adaptive"
    ):
        """Initialize distributed cleaner.
        
        Args:
            base_cleaner_config: Configuration for base TableCleaner instances
            max_workers: Maximum number of worker processes/threads
            chunk_size: Size of data chunks for parallel processing
            enable_process_pool: Use process pool (True) vs thread pool (False)
            load_balancer_strategy: Load balancing strategy
        """
        self.base_cleaner_config = base_cleaner_config
        self.chunk_size = chunk_size
        self.enable_process_pool = enable_process_pool
        
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers = min(cpu_count * 2, 16) if enable_process_pool else min(cpu_count * 4, 32)
        
        self.max_workers = max_workers
        
        # Initialize components
        self.load_balancer = LoadBalancer(strategy=load_balancer_strategy)
        self.task_queue: List[ProcessingTask] = []
        self.completed_tasks: Dict[str, Any] = {}
        
        # Performance tracking
        self.processing_stats = {
            "tasks_completed": 0,
            "total_processing_time": 0.0,
            "total_records_processed": 0,
            "average_throughput": 0.0,
            "worker_utilization": 0.0
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"Initialized DistributedCleaner with {max_workers} workers, "
                   f"chunk size: {chunk_size}, process pool: {enable_process_pool}")
    
    def clean_distributed(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sample_rate: float = 1.0,
        priority: int = 1
    ) -> CleaningReport:
        """Clean DataFrame using distributed processing.
        
        Args:
            df: Input DataFrame
            columns: Columns to clean
            sample_rate: Sampling rate
            priority: Task priority
            
        Returns:
            Combined cleaning report
        """
        start_time = time.time()
        logger.info(f"Starting distributed cleaning for DataFrame with {len(df)} rows")
        
        try:
            # Create data chunks
            chunks = self._create_data_chunks(df, columns, sample_rate)
            logger.info(f"Created {len(chunks)} data chunks for processing")
            
            # Create processing tasks
            tasks = []
            for i, chunk in enumerate(chunks):
                task = ProcessingTask(
                    task_id=f"task_{int(time.time())}_{i}",
                    data_chunk=chunk,
                    processing_params={
                        "columns": columns,
                        "sample_rate": sample_rate,
                        "config": self.base_cleaner_config
                    },
                    priority=priority
                )
                tasks.append(task)
            
            # Process tasks
            if self.enable_process_pool:
                results = self._process_with_process_pool(tasks)
            else:
                results = self._process_with_thread_pool(tasks)
            
            # Combine results
            combined_report = self._combine_results(results, df, start_time)
            
            # Update stats
            self._update_processing_stats(len(df), time.time() - start_time, len(tasks))
            
            logger.info(f"Distributed cleaning completed: {combined_report.total_fixes} fixes in {combined_report.processing_time:.2f}s")
            return combined_report
            
        except Exception as e:
            logger.error(f"Distributed cleaning failed: {e}")
            raise
    
    def _create_data_chunks(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]], 
        sample_rate: float
    ) -> List[pd.DataFrame]:
        """Create data chunks for parallel processing."""
        # Apply sampling if requested
        if sample_rate < 1.0:
            sample_size = int(len(df) * sample_rate)
            df_sample = df.sample(n=sample_size)
        else:
            df_sample = df
        
        # Filter columns if specified
        if columns:
            available_columns = [col for col in columns if col in df_sample.columns]
            if not available_columns:
                raise ValueError("No valid columns found for cleaning")
            df_filtered = df_sample[available_columns].copy()
        else:
            df_filtered = df_sample.copy()
        
        # Create chunks
        chunks = []
        for i in range(0, len(df_filtered), self.chunk_size):
            chunk = df_filtered.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
        
        return chunks
    
    def _process_with_process_pool(self, tasks: List[ProcessingTask]) -> List[CleaningReport]:
        """Process tasks using process pool."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(_process_chunk_worker, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed task {task.task_id}")
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    # Create empty result
                    empty_result = CleaningReport(
                        total_fixes=0,
                        quality_score=0.0,
                        fixes=[],
                        processing_time=0.0,
                        audit_trail=[{"error": str(e), "task_id": task.task_id}]
                    )
                    results.append(empty_result)
        
        return results
    
    def _process_with_thread_pool(self, tasks: List[ProcessingTask]) -> List[CleaningReport]:
        """Process tasks using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(self._process_single_task, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed task {task.task_id}")
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    # Create empty result
                    empty_result = CleaningReport(
                        total_fixes=0,
                        quality_score=0.0,
                        fixes=[],
                        processing_time=0.0,
                        audit_trail=[{"error": str(e), "task_id": task.task_id}]
                    )
                    results.append(empty_result)
        
        return results
    
    def _process_single_task(self, task: ProcessingTask) -> CleaningReport:
        """Process a single task."""
        # Create cleaner instance
        cleaner = TableCleaner(**task.processing_params["config"])
        
        # Clean the data chunk
        _, report = cleaner.clean(
            task.data_chunk,
            columns=task.processing_params.get("columns"),
            sample_rate=1.0  # Already sampled at chunk level
        )
        
        return report
    
    def _combine_results(
        self, 
        results: List[CleaningReport], 
        original_df: pd.DataFrame, 
        start_time: float
    ) -> CleaningReport:
        """Combine results from distributed processing."""
        # Aggregate statistics
        total_fixes = sum(report.total_fixes for report in results)
        total_processing_time = time.time() - start_time
        
        # Combine fixes
        all_fixes = []
        for report in results:
            all_fixes.extend(report.fixes)
        
        # Combine audit trails
        all_audit_trails = []
        for report in results:
            if report.audit_trail:
                all_audit_trails.extend(report.audit_trail)
        
        # Calculate combined quality score
        if results:
            quality_scores = [r.quality_score for r in results if r.quality_score > 0]
            combined_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        else:
            combined_quality_score = 0.0
        
        # Create combined profile summary
        combined_profile_summary = {
            "distributed_processing": True,
            "chunks_processed": len(results),
            "parallel_workers": self.max_workers,
            "original_shape": original_df.shape,
            "processing_mode": "process_pool" if self.enable_process_pool else "thread_pool"
        }
        
        return CleaningReport(
            total_fixes=total_fixes,
            quality_score=combined_quality_score,
            fixes=all_fixes,
            processing_time=total_processing_time,
            profile_summary=combined_profile_summary,
            audit_trail=all_audit_trails
        )
    
    def _update_processing_stats(self, records_processed: int, processing_time: float, tasks_count: int):
        """Update processing statistics."""
        with self._lock:
            self.processing_stats["tasks_completed"] += tasks_count
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["total_records_processed"] += records_processed
            
            # Calculate averages
            if self.processing_stats["total_processing_time"] > 0:
                self.processing_stats["average_throughput"] = (
                    self.processing_stats["total_records_processed"] / 
                    self.processing_stats["total_processing_time"]
                )
            
            if tasks_count > 0:
                self.processing_stats["worker_utilization"] = min(tasks_count / self.max_workers, 1.0)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                **self.processing_stats,
                "max_workers": self.max_workers,
                "chunk_size": self.chunk_size,
                "processing_mode": "process_pool" if self.enable_process_pool else "thread_pool",
                "load_balancer_strategy": self.load_balancer.strategy,
                "cluster_status": self.load_balancer.get_cluster_status()
            }


def _process_chunk_worker(task: ProcessingTask) -> CleaningReport:
    """Worker function for process pool (must be at module level for pickling)."""
    try:
        # Create cleaner instance
        cleaner = TableCleaner(**task.processing_params["config"])
        
        # Clean the data chunk
        _, report = cleaner.clean(
            task.data_chunk,
            columns=task.processing_params.get("columns"),
            sample_rate=1.0  # Already sampled at chunk level
        )
        
        return report
        
    except Exception as e:
        # Return error report
        return CleaningReport(
            total_fixes=0,
            quality_score=0.0,
            fixes=[],
            processing_time=0.0,
            audit_trail=[{"error": str(e), "task_id": task.task_id}]
        )


class AutoScaler:
    """Automatic scaling based on workload and system resources."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None, target_utilization: float = 0.7):
        """Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_utilization: Target CPU utilization (0.0-1.0)
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or min(multiprocessing.cpu_count() * 4, 32)
        self.target_utilization = target_utilization
        
        self.current_workers = min_workers
        self.scaling_history = []
        self._lock = threading.Lock()
        
        logger.info(f"Initialized AutoScaler: {min_workers}-{self.max_workers} workers, target utilization: {target_utilization:.1%}")
    
    def recommend_scaling(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Recommend scaling action based on current metrics.
        
        Args:
            current_metrics: Current system metrics
            
        Returns:
            Scaling recommendation
        """
        with self._lock:
            cpu_utilization = current_metrics.get("cpu_utilization", 0.5)
            memory_utilization = current_metrics.get("memory_utilization", 0.5)
            queue_length = current_metrics.get("queue_length", 0)
            
            # Calculate scaling decision
            scale_decision = "maintain"
            recommended_workers = self.current_workers
            reason = "No scaling needed"
            
            # Scale up conditions
            if (cpu_utilization > self.target_utilization + 0.1 or 
                queue_length > self.current_workers * 2):
                if self.current_workers < self.max_workers:
                    scale_decision = "scale_up"
                    recommended_workers = min(self.current_workers + 1, self.max_workers)
                    reason = f"High utilization (CPU: {cpu_utilization:.1%}, Queue: {queue_length})"
            
            # Scale down conditions
            elif (cpu_utilization < self.target_utilization - 0.2 and 
                  queue_length == 0 and 
                  self.current_workers > self.min_workers):
                scale_decision = "scale_down"
                recommended_workers = max(self.current_workers - 1, self.min_workers)
                reason = f"Low utilization (CPU: {cpu_utilization:.1%})"
            
            recommendation = {
                "action": scale_decision,
                "current_workers": self.current_workers,
                "recommended_workers": recommended_workers,
                "reason": reason,
                "metrics": current_metrics,
                "timestamp": time.time()
            }
            
            # Record scaling decision
            self.scaling_history.append(recommendation)
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-50:]
            
            return recommendation
    
    def apply_scaling(self, recommendation: Dict[str, Any]) -> bool:
        """Apply scaling recommendation.
        
        Args:
            recommendation: Scaling recommendation from recommend_scaling
            
        Returns:
            True if scaling was applied
        """
        if recommendation["action"] == "maintain":
            return False
        
        with self._lock:
            old_workers = self.current_workers
            self.current_workers = recommendation["recommended_workers"]
            
            logger.info(f"Scaling {recommendation['action']}: {old_workers} -> {self.current_workers} workers ({recommendation['reason']})")
            return True
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        with self._lock:
            if not self.scaling_history:
                return {
                    "current_workers": self.current_workers,
                    "scaling_events": 0
                }
            
            scale_ups = len([h for h in self.scaling_history if h["action"] == "scale_up"])
            scale_downs = len([h for h in self.scaling_history if h["action"] == "scale_down"])
            
            return {
                "current_workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "target_utilization": self.target_utilization,
                "scaling_events": len(self.scaling_history),
                "scale_ups": scale_ups,
                "scale_downs": scale_downs,
                "recent_decisions": self.scaling_history[-10:] if self.scaling_history else []
            }