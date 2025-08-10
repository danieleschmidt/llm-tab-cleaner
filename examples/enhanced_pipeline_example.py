#!/usr/bin/env python3
"""Enhanced pipeline example with adaptive features."""

import asyncio
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

from llm_tab_cleaner import TableCleaner
from llm_tab_cleaner.adaptive import AdaptiveCache, PatternLearner, AutoScalingProcessor
from llm_tab_cleaner.streaming import StreamingCleaner, StreamRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_messy_data(size: int = 1000) -> pd.DataFrame:
    """Create sample messy data for testing."""
    np.random.seed(42)
    
    # Generate base data
    data = {
        'customer_id': range(1, size + 1),
        'name': [f"Customer {i}" if np.random.random() > 0.1 else "N/A" for i in range(1, size + 1)],
        'email': [
            f"user{i}@example.com" if np.random.random() > 0.15 
            else np.random.choice(["missing", "N/A", "invalid-email", f"user{i}@"])
            for i in range(1, size + 1)
        ],
        'phone': [
            f"555-{i:04d}" if np.random.random() > 0.2
            else np.random.choice(["N/A", "555-INVALID", f"555{i}", ""])
            for i in range(1, size + 1)
        ],
        'age': [
            np.random.randint(18, 80) if np.random.random() > 0.1
            else np.random.choice([-1, 999, "unknown", "N/A"])
            for _ in range(size)
        ],
        'state': [
            np.random.choice(["CA", "NY", "TX", "FL"]) if np.random.random() > 0.1
            else np.random.choice(["California", "New York", "texas", "FLA", "N/A"])
            for _ in range(size)
        ],
        'salary': [
            np.random.randint(30000, 120000) if np.random.random() > 0.15
            else np.random.choice(["N/A", "$invalid", "0", "-1000"])
            for _ in range(size)
        ]
    }
    
    return pd.DataFrame(data)


async def enhanced_cleaning_example():
    """Demonstrate enhanced cleaning with adaptive features."""
    logger.info("=== Enhanced Pipeline Example ===")
    
    # Create sample data
    logger.info("Creating sample messy data...")
    df = create_sample_messy_data(size=5000)
    logger.info(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Initialize enhanced cleaner with adaptive features
    logger.info("Initializing TableCleaner with adaptive features...")
    cleaner = TableCleaner(
        llm_provider="local",  # Use local provider for demo
        confidence_threshold=0.80,
        enable_profiling=True,
        enable_monitoring=True,
        max_concurrent_operations=6,
        circuit_breaker_config={
            "failure_threshold": 3,
            "timeout": 30
        }
    )
    
    # Initialize adaptive components
    cache = AdaptiveCache(max_size=5000, ttl=1800)  # 30 min TTL
    pattern_learner = PatternLearner(max_patterns=2000)
    processor = AutoScalingProcessor(initial_batch_size=200, max_batch_size=1000)
    
    # Simulate cleaning with pattern learning
    logger.info("Starting enhanced cleaning process...")
    start_time = time.time()
    
    # Process in batches to simulate streaming
    batch_size = 1000
    total_fixes = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size].copy()
        logger.info(f"Processing batch {i//batch_size + 1}: rows {i} to {min(i + batch_size, len(df))}")
        
        # Clean batch
        cleaned_batch, report = cleaner.clean(batch_df, sample_rate=0.3)  # Sample 30% for speed
        
        # Learn patterns from successful fixes
        for fix in report.fixes:
            if fix.confidence > 0.8:
                pattern_learner.learn_from_fix(fix, {"data_type": "string"})
                
                # Cache high-confidence results
                cache.put(
                    fix.original, fix.column, {"data_type": "string"},
                    fix.cleaned, fix.confidence
                )
        
        total_fixes += report.total_fixes
        logger.info(f"Batch {i//batch_size + 1}: {report.total_fixes} fixes, quality score: {report.quality_score:.2%}")
    
    processing_time = time.time() - start_time
    
    # Display results
    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    logger.info(f"Total fixes applied: {total_fixes}")
    logger.info(f"Processing rate: {len(df) / processing_time:.1f} records/second")
    
    # Show adaptive component stats
    logger.info(f"\n=== Adaptive Components Stats ===")
    cache_stats = cache.get_stats()
    logger.info(f"Cache: {cache_stats['cache_size']} entries, {cache_stats['total_accesses']} accesses")
    logger.info(f"Cache average confidence: {cache_stats['average_confidence']:.3f}")
    
    pattern_stats = pattern_learner.get_stats()
    logger.info(f"Patterns: {pattern_stats['pattern_count']} learned")
    logger.info(f"Pattern average confidence: {pattern_stats.get('average_confidence', 0):.3f}")
    
    processor_stats = processor.get_stats()
    logger.info(f"Auto-scaling: current batch size {processor_stats['current_batch_size']}")
    logger.info(f"Success rate: {processor_stats['success_rate']:.2%}")


async def streaming_example():
    """Demonstrate real-time streaming cleaning."""
    logger.info("\n=== Streaming Example ===")
    
    # Initialize base cleaner
    base_cleaner = TableCleaner(
        llm_provider="local",
        confidence_threshold=0.75,
        enable_profiling=False  # Disable for streaming performance
    )
    
    # Initialize streaming cleaner
    streaming_cleaner = StreamingCleaner(
        base_cleaner=base_cleaner,
        batch_size=100,
        batch_timeout=2.0,
        max_queue_size=1000,
        enable_adaptive=True,
        checkpoint_interval=10
    )
    
    # Generate stream of records
    async def generate_record_stream():
        """Generate a stream of messy records."""
        messy_data = create_sample_messy_data(size=500)
        
        for i, row in messy_data.iterrows():
            record = StreamRecord(
                id=f"stream_record_{i}",
                data=row.to_dict(),
                timestamp=time.time()
            )
            yield record
            await asyncio.sleep(0.01)  # Simulate real-time arrival
    
    # Process streaming data
    logger.info("Starting streaming cleaning...")
    start_time = time.time()
    processed_count = 0
    
    async for cleaned_record in streaming_cleaner.clean_batch_stream(generate_record_stream()):
        processed_count += 1
        if processed_count % 50 == 0:
            logger.info(f"Processed {processed_count} streaming records...")
    
    processing_time = time.time() - start_time
    
    # Show streaming stats
    stats = streaming_cleaner.get_stats()
    logger.info(f"\n=== Streaming Results ===")
    logger.info(f"Total records processed: {stats['records_processed']}")
    logger.info(f"Total batches processed: {stats['batches_processed']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Throughput: {stats['throughput']:.1f} records/second")
    logger.info(f"Average latency: {stats['average_latency']:.3f} seconds/record")
    logger.info(f"Cache hits: {stats['cache_hits']}, Pattern matches: {stats['pattern_matches']}")


def data_quality_analysis():
    """Analyze data quality improvements."""
    logger.info("\n=== Data Quality Analysis ===")
    
    # Create test data with known quality issues
    original_data = create_sample_messy_data(size=1000)
    
    # Calculate baseline quality metrics
    def calculate_quality_metrics(df):
        metrics = {}
        
        for column in df.columns:
            series = df[column]
            
            # Missing/null values
            null_count = series.isnull().sum() + (series.astype(str).str.lower().isin(['n/a', 'na', 'null', 'missing', ''])).sum()
            
            # Completeness
            metrics[f"{column}_completeness"] = (len(series) - null_count) / len(series)
            
            # Consistency (for categorical columns)
            if column in ['state', 'name']:
                unique_ratio = series.nunique() / len(series)
                metrics[f"{column}_consistency"] = 1 - unique_ratio  # Higher is more consistent
        
        return metrics
    
    original_metrics = calculate_quality_metrics(original_data)
    
    # Clean the data
    cleaner = TableCleaner(llm_provider="local", confidence_threshold=0.8)
    cleaned_data, report = cleaner.clean(original_data, sample_rate=0.2)  # Sample for speed
    
    cleaned_metrics = calculate_quality_metrics(cleaned_data)
    
    # Show improvements
    logger.info("Data Quality Improvements:")
    for metric, original_value in original_metrics.items():
        cleaned_value = cleaned_metrics.get(metric, original_value)
        improvement = cleaned_value - original_value
        logger.info(f"{metric}: {original_value:.3f} → {cleaned_value:.3f} ({improvement:+.3f})")
    
    # Show fix distribution
    fix_distribution = {}
    for fix in report.fixes:
        fix_distribution[fix.column] = fix_distribution.get(fix.column, 0) + 1
    
    logger.info(f"\nFixes by column:")
    for column, count in sorted(fix_distribution.items()):
        logger.info(f"  {column}: {count} fixes")


async def main():
    """Run all examples."""
    try:
        # Run enhanced cleaning example
        await enhanced_cleaning_example()
        
        # Run streaming example
        await streaming_example()
        
        # Run data quality analysis
        data_quality_analysis()
        
        logger.info("\n=== All Examples Complete ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())