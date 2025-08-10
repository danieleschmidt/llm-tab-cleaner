"""Real-time streaming data cleaning capabilities."""

import asyncio
import json
import logging
import time
from asyncio import Queue
from dataclasses import dataclass, asdict
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd

from .core import TableCleaner, Fix, CleaningReport
from .adaptive import AdaptiveCache, PatternLearner, AutoScalingProcessor

logger = logging.getLogger(__name__)


@dataclass
class StreamRecord:
    """Individual record in a data stream."""
    id: str
    data: Dict[str, Any]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass  
class StreamBatch:
    """Batch of stream records."""
    records: List[StreamRecord]
    batch_id: str
    created_at: float
    size: int


class StreamingCleaner:
    """Real-time streaming data cleaner with adaptive capabilities."""
    
    def __init__(
        self,
        base_cleaner: TableCleaner,
        batch_size: int = 1000,
        batch_timeout: float = 5.0,
        max_queue_size: int = 10000,
        enable_adaptive: bool = True,
        checkpoint_interval: int = 100,
        **kwargs
    ):
        """Initialize streaming cleaner.
        
        Args:
            base_cleaner: Base TableCleaner instance
            batch_size: Maximum records per batch
            batch_timeout: Maximum time to wait before processing partial batch
            max_queue_size: Maximum records in processing queue
            enable_adaptive: Enable adaptive learning features
            checkpoint_interval: Batches between state checkpoints
        """
        self.base_cleaner = base_cleaner
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        self.enable_adaptive = enable_adaptive
        self.checkpoint_interval = checkpoint_interval
        
        # Processing queues
        self.input_queue: Queue[StreamRecord] = Queue(maxsize=max_queue_size)
        self.output_queue: Queue[StreamRecord] = Queue()
        
        # Adaptive components
        if enable_adaptive:
            self.cache = AdaptiveCache(max_size=20000, ttl=7200)  # 2 hour TTL
            self.pattern_learner = PatternLearner(max_patterns=5000)
            self.processor = AutoScalingProcessor(
                initial_batch_size=batch_size,
                max_batch_size=batch_size * 10
            )
        else:
            self.cache = None
            self.pattern_learner = None
            self.processor = None
        
        # Processing state
        self.is_running = False
        self.processed_batches = 0
        self.total_records_processed = 0
        self.error_count = 0
        
        # Statistics tracking
        self.stats = {
            "records_processed": 0,
            "batches_processed": 0,
            "errors": 0,
            "cache_hits": 0,
            "pattern_matches": 0,
            "average_latency": 0.0,
            "throughput": 0.0
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.stats_lock = threading.Lock()
        
    async def start(self):
        """Start the streaming cleaner."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting streaming cleaner")
        
        # Start processing tasks
        processing_task = asyncio.create_task(self._batch_processor())
        
        try:
            await processing_task
        except asyncio.CancelledError:
            logger.info("Streaming cleaner cancelled")
        finally:
            self.is_running = False
    
    async def stop(self):
        """Stop the streaming cleaner."""
        logger.info("Stopping streaming cleaner")
        self.is_running = False
        
        # Process remaining records
        await self._drain_queues()
        
        # Cleanup
        self.executor.shutdown(wait=True)
    
    async def add_record(self, record: StreamRecord) -> bool:
        """Add a record to the processing queue.
        
        Args:
            record: Stream record to process
            
        Returns:
            True if record was added, False if queue is full
        """
        try:
            await asyncio.wait_for(
                self.input_queue.put(record),
                timeout=0.1
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Input queue full, dropping record")
            return False
    
    async def get_cleaned_record(self, timeout: float = None) -> Optional[StreamRecord]:
        """Get a cleaned record from output queue.
        
        Args:
            timeout: Maximum time to wait for record
            
        Returns:
            Cleaned record or None if timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=timeout
                )
            else:
                return await self.output_queue.get()
        except asyncio.TimeoutError:
            return None
    
    async def clean_batch_stream(
        self, 
        record_stream: AsyncGenerator[StreamRecord, None]
    ) -> AsyncGenerator[StreamRecord, None]:
        """Clean a stream of records.
        
        Args:
            record_stream: Async generator of records
            
        Yields:
            Cleaned records
        """
        # Start background processing
        processing_task = asyncio.create_task(self.start())
        
        try:
            # Feed records into processor
            async def feed_records():
                async for record in record_stream:
                    await self.add_record(record)
            
            feed_task = asyncio.create_task(feed_records())
            
            # Yield cleaned records
            while True:
                cleaned_record = await self.get_cleaned_record(timeout=1.0)
                if cleaned_record is None:
                    if feed_task.done():
                        break
                    continue
                yield cleaned_record
                
        finally:
            processing_task.cancel()
            await self.stop()
    
    async def _batch_processor(self):
        """Main batch processing loop."""
        logger.info("Started batch processor")
        
        while self.is_running:
            try:
                # Collect batch
                batch = await self._collect_batch()
                if not batch.records:
                    continue
                
                # Process batch
                start_time = time.time()
                cleaned_batch = await self._process_batch(batch)
                processing_time = time.time() - start_time
                
                # Update statistics
                self._update_stats(batch, processing_time)
                
                # Send cleaned records to output
                for record in cleaned_batch.records:
                    await self.output_queue.put(record)
                
                # Checkpoint if needed
                if self.processed_batches % self.checkpoint_interval == 0:
                    await self._checkpoint_state()
                
                logger.debug(f"Processed batch {batch.batch_id} with {len(batch.records)} records in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                self.error_count += 1
    
    async def _collect_batch(self) -> StreamBatch:
        """Collect records into a batch."""
        records = []
        start_time = time.time()
        
        # Collect records until batch size or timeout
        while len(records) < self.batch_size:
            remaining_time = self.batch_timeout - (time.time() - start_time)
            if remaining_time <= 0:
                break
            
            try:
                record = await asyncio.wait_for(
                    self.input_queue.get(),
                    timeout=min(remaining_time, 0.1)
                )
                records.append(record)
            except asyncio.TimeoutError:
                if records:  # Have some records, process them
                    break
                continue
        
        batch_id = f"batch_{self.processed_batches:06d}_{int(time.time() * 1000)}"
        return StreamBatch(
            records=records,
            batch_id=batch_id,
            created_at=time.time(),
            size=len(records)
        )
    
    async def _process_batch(self, batch: StreamBatch) -> StreamBatch:
        """Process a batch of records."""
        if not batch.records:
            return batch
        
        # Convert to DataFrame for processing
        data_rows = [record.data for record in batch.records]
        df = pd.DataFrame(data_rows)
        
        # Apply cleaning with caching and pattern learning
        cleaned_records = []
        
        for i, (record, row) in enumerate(zip(batch.records, df.itertuples(index=False))):
            cleaned_data = {}
            
            for column, value in record.data.items():
                if pd.isna(value):
                    cleaned_data[column] = value
                    continue
                
                # Try adaptive cache first
                cached_result = None
                if self.cache:
                    cached_result = self.cache.get(
                        value, column, {"data_type": str(type(value).__name__)}
                    )
                
                if cached_result:
                    cleaned_value, confidence = cached_result
                    with self.stats_lock:
                        self.stats["cache_hits"] += 1
                else:
                    # Try pattern learning
                    pattern_result = None
                    if self.pattern_learner:
                        pattern_result = self.pattern_learner.suggest_fix(
                            value, column, {"data_type": str(type(value).__name__)}
                        )
                    
                    if pattern_result and pattern_result[1] > 0.8:
                        cleaned_value, confidence = pattern_result
                        with self.stats_lock:
                            self.stats["pattern_matches"] += 1
                    else:
                        # Fall back to LLM cleaning
                        try:
                            cleaned_value, confidence = await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                self._clean_single_value,
                                value, column, {"data_type": str(type(value).__name__)}
                            )
                        except Exception as e:
                            logger.warning(f"Failed to clean value {value} in column {column}: {e}")
                            cleaned_value, confidence = value, 0.0
                    
                    # Cache the result
                    if self.cache and confidence > 0.7:
                        self.cache.put(
                            value, column, {"data_type": str(type(value).__name__)},
                            cleaned_value, confidence
                        )
                    
                    # Learn pattern if high confidence
                    if self.pattern_learner and confidence > 0.8 and cleaned_value != value:
                        fix = Fix(
                            column=column,
                            row_index=i,
                            original=value,
                            cleaned=cleaned_value,
                            confidence=confidence,
                            reasoning="streaming_cleaning"
                        )
                        self.pattern_learner.learn_from_fix(fix, {"data_type": str(type(value).__name__)})
                
                cleaned_data[column] = cleaned_value
            
            # Create cleaned record
            cleaned_record = StreamRecord(
                id=record.id,
                data=cleaned_data,
                timestamp=record.timestamp,
                metadata={
                    **(record.metadata or {}),
                    "cleaned_at": time.time(),
                    "batch_id": batch.batch_id
                }
            )
            cleaned_records.append(cleaned_record)
        
        return StreamBatch(
            records=cleaned_records,
            batch_id=batch.batch_id,
            created_at=batch.created_at,
            size=len(cleaned_records)
        )
    
    def _clean_single_value(self, value: Any, column: str, context: Dict[str, Any]) -> tuple[Any, float]:
        """Clean a single value using base cleaner."""
        try:
            # Create minimal DataFrame for cleaning
            df = pd.DataFrame({column: [value]})
            cleaned_df, report = self.base_cleaner.clean(df, columns=[column])
            
            cleaned_value = cleaned_df.iloc[0][column]
            
            # Find confidence from fixes
            confidence = 0.8  # Default confidence
            if report.fixes:
                matching_fixes = [f for f in report.fixes if f.column == column]
                if matching_fixes:
                    confidence = matching_fixes[0].confidence
            
            return cleaned_value, confidence
            
        except Exception as e:
            logger.error(f"Error cleaning single value: {e}")
            return value, 0.0
    
    def _update_stats(self, batch: StreamBatch, processing_time: float):
        """Update processing statistics."""
        with self.stats_lock:
            self.stats["records_processed"] += len(batch.records)
            self.stats["batches_processed"] += 1
            
            # Update averages
            total_records = self.stats["records_processed"]
            if total_records > 0:
                self.stats["throughput"] = total_records / (time.time() - getattr(self, '_start_time', time.time()))
            
            # Running average of latency
            current_latency = processing_time / len(batch.records) if batch.records else 0
            self.stats["average_latency"] = (
                self.stats["average_latency"] * 0.9 + current_latency * 0.1
            )
        
        self.processed_batches += 1
        self.total_records_processed += len(batch.records)
    
    async def _checkpoint_state(self):
        """Save adaptive state checkpoint."""
        if not self.enable_adaptive:
            return
        
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "processed_batches": self.processed_batches,
                "total_records": self.total_records_processed,
                "stats": self.stats.copy(),
                "cache_stats": self.cache.get_stats() if self.cache else {},
                "pattern_stats": self.pattern_learner.get_stats() if self.pattern_learner else {},
                "processor_stats": self.processor.get_stats() if self.processor else {}
            }
            
            # In a real implementation, you'd save this to persistent storage
            logger.debug(f"Checkpoint: {checkpoint_data['stats']}")
            
        except Exception as e:
            logger.error(f"Failed to checkpoint state: {e}")
    
    async def _drain_queues(self):
        """Process any remaining records in queues."""
        logger.info("Draining processing queues")
        
        # Process remaining input records
        remaining_records = []
        while not self.input_queue.empty():
            try:
                record = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                remaining_records.append(record)
            except asyncio.TimeoutError:
                break
        
        if remaining_records:
            final_batch = StreamBatch(
                records=remaining_records,
                batch_id=f"final_batch_{int(time.time() * 1000)}",
                created_at=time.time(),
                size=len(remaining_records)
            )
            
            cleaned_batch = await self._process_batch(final_batch)
            for record in cleaned_batch.records:
                await self.output_queue.put(record)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.stats_lock:
            base_stats = self.stats.copy()
        
        base_stats.update({
            "is_running": self.is_running,
            "processed_batches": self.processed_batches,
            "total_records_processed": self.total_records_processed,
            "error_count": self.error_count,
            "queue_sizes": {
                "input": self.input_queue.qsize(),
                "output": self.output_queue.qsize()
            }
        })
        
        if self.enable_adaptive:
            base_stats["adaptive"] = {
                "cache": self.cache.get_stats() if self.cache else {},
                "patterns": self.pattern_learner.get_stats() if self.pattern_learner else {},
                "processor": self.processor.get_stats() if self.processor else {}
            }
        
        return base_stats


class StreamingAPI:
    """HTTP API wrapper for streaming cleaning."""
    
    def __init__(self, cleaner: StreamingCleaner):
        """Initialize streaming API.
        
        Args:
            cleaner: StreamingCleaner instance
        """
        self.cleaner = cleaner
    
    async def health_check(self) -> Dict[str, Any]:
        """Get service health status."""
        stats = self.cleaner.get_stats()
        
        return {
            "status": "healthy" if self.cleaner.is_running else "stopped",
            "uptime": time.time() - getattr(self.cleaner, '_start_time', time.time()),
            "stats": stats,
            "version": "1.0"
        }
    
    async def process_records(self, records_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records via API."""
        # Convert to StreamRecord format
        stream_records = []
        for i, data in enumerate(records_data):
            record = StreamRecord(
                id=f"api_record_{i}_{int(time.time() * 1000)}",
                data=data,
                timestamp=time.time()
            )
            stream_records.append(record)
        
        # Process records
        cleaned_records = []
        for record in stream_records:
            await self.cleaner.add_record(record)
        
        # Collect results
        for _ in range(len(stream_records)):
            cleaned_record = await self.cleaner.get_cleaned_record(timeout=30.0)
            if cleaned_record:
                cleaned_records.append(cleaned_record.data)
        
        return cleaned_records