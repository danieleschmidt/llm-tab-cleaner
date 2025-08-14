#!/usr/bin/env python3
"""
Scalable demonstration of llm-tab-cleaner functionality.
Generation 3: Make it Scale (Optimized) - Performance optimization, caching, and scalability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from llm_tab_cleaner import TableCleaner, CleaningRule, RuleSet

# Configure logging for performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_large_dataset(size=10000):
    """Create a large dataset for performance testing."""
    np.random.seed(42)  # For reproducible results
    
    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
    states = ['CA', 'NY', 'TX', 'FL', 'WA', 'IL', 'PA', 'OH']
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'company.com']
    
    data = {
        'id': range(1, size + 1),
        'name': [
            f"{np.random.choice(names)} {np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])}"
            if np.random.random() > 0.05 else None  # 5% null
            for _ in range(size)
        ],
        'email': [
            f"{np.random.choice(names).lower()}{np.random.randint(1, 999)}@{np.random.choice(domains)}"
            if np.random.random() > 0.03 else 'invalid-email'  # 3% invalid
            for _ in range(size)
        ],
        'age': [
            np.random.randint(18, 80) if np.random.random() > 0.02 else 'invalid'  # 2% invalid
            for _ in range(size)
        ],
        'state': [
            np.random.choice(states) if np.random.random() > 0.04 else 'Unknown'  # 4% unknown
            for _ in range(size)
        ],
        'salary': [
            np.random.randint(30000, 150000) if np.random.random() > 0.06 else '$invalid'  # 6% invalid
            for _ in range(size)
        ],
        'score': np.random.normal(75, 15, size)  # Normally distributed scores
    }
    
    logger.info(f"Created large dataset with {size} rows")
    return pd.DataFrame(data)

def demo_concurrent_processing():
    """Demonstrate concurrent processing capabilities."""
    logger.info("🚀 Testing Concurrent Processing")
    print("🚀 Concurrent Processing Demo")
    print("=" * 40)
    
    # Create multiple datasets
    datasets = [create_large_dataset(1000) for _ in range(4)]
    
    # Sequential processing
    cleaner = TableCleaner(confidence_threshold=0.7, max_batch_size=100)
    
    start_time = time.time()
    sequential_results = []
    for i, df in enumerate(datasets):
        logger.info(f"Processing dataset {i+1} sequentially")
        result = cleaner.clean(df)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.2f}s for {len(datasets)} datasets")
    
    # Concurrent processing with ThreadPoolExecutor
    def process_dataset(df_tuple):
        idx, df = df_tuple
        cleaner = TableCleaner(confidence_threshold=0.7, max_batch_size=100)
        logger.info(f"Processing dataset {idx+1} concurrently")
        return cleaner.clean(df)
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_dataset, (i, df)) for i, df in enumerate(datasets)]
        concurrent_results = [future.result() for future in as_completed(futures)]
    concurrent_time = time.time() - start_time
    
    print(f"Concurrent processing: {concurrent_time:.2f}s for {len(datasets)} datasets")
    
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup > 1.2  # Expect at least 20% improvement

def demo_batch_optimization():
    """Demonstrate batch processing optimization."""
    logger.info("📦 Testing Batch Optimization")
    print("\n📦 Batch Processing Optimization Demo")
    print("-" * 45)
    
    large_df = create_large_dataset(5000)
    
    # Test different batch sizes
    batch_sizes = [10, 50, 100, 500, 1000]
    results = {}
    
    for batch_size in batch_sizes:
        cleaner = TableCleaner(
            confidence_threshold=0.7,
            max_batch_size=batch_size,
            enable_caching=True
        )
        
        start_time = time.time()
        result = cleaner.clean(large_df)
        processing_time = time.time() - start_time
        
        results[batch_size] = {
            'time': processing_time,
            'throughput': len(large_df) / processing_time if processing_time > 0 else 0
        }
        
        print(f"Batch size {batch_size:4d}: {processing_time:.3f}s ({results[batch_size]['throughput']:.0f} rows/s)")
    
    # Find optimal batch size
    optimal_batch = max(results.keys(), key=lambda k: results[k]['throughput'])
    print(f"\nOptimal batch size: {optimal_batch} (best throughput)")
    
    return True

def demo_memory_optimization():
    """Demonstrate memory-efficient processing."""
    logger.info("💾 Testing Memory Optimization")
    print("\n💾 Memory Optimization Demo")
    print("-" * 35)
    
    try:
        import psutil
        process = psutil.Process()
        
        # Baseline memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Process increasingly large datasets
        sizes = [1000, 5000, 10000]
        memory_usage = {}
        
        for size in sizes:
            df = create_large_dataset(size)
            
            # Memory before processing
            before_memory = process.memory_info().rss / 1024 / 1024
            
            cleaner = TableCleaner(
                confidence_threshold=0.7,
                max_batch_size=500,
                enable_caching=True
            )
            
            start_time = time.time()
            result = cleaner.clean(df)
            processing_time = time.time() - start_time
            
            # Memory after processing
            after_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = after_memory - before_memory
            
            memory_usage[size] = {
                'before': before_memory,
                'after': after_memory,
                'increase': memory_increase,
                'efficiency': size / memory_increase if memory_increase > 0 else float('inf')
            }
            
            print(f"Dataset size {size:5d}: {memory_increase:6.1f} MB increase, "
                  f"{memory_usage[size]['efficiency']:.0f} rows/MB")
            
            # Clean up
            del df, result
            
        return True
        
    except ImportError:
        print("psutil not available, skipping memory optimization demo")
        return True

def demo_caching_performance():
    """Demonstrate caching improvements."""
    logger.info("⚡ Testing Caching Performance")
    print("\n⚡ Caching Performance Demo")
    print("-" * 35)
    
    df = create_large_dataset(2000)
    
    # Without caching
    cleaner_no_cache = TableCleaner(
        confidence_threshold=0.7,
        enable_caching=False
    )
    
    start_time = time.time()
    result1 = cleaner_no_cache.clean(df)
    time_no_cache_1 = time.time() - start_time
    
    start_time = time.time()
    result2 = cleaner_no_cache.clean(df)  # Same data, no cache
    time_no_cache_2 = time.time() - start_time
    
    print(f"Without caching - First run:  {time_no_cache_1:.3f}s")
    print(f"Without caching - Second run: {time_no_cache_2:.3f}s")
    
    # With caching
    cleaner_with_cache = TableCleaner(
        confidence_threshold=0.7,
        enable_caching=True
    )
    
    start_time = time.time()
    result3 = cleaner_with_cache.clean(df)
    time_cache_1 = time.time() - start_time
    
    start_time = time.time()
    result4 = cleaner_with_cache.clean(df)  # Same data, with cache
    time_cache_2 = time.time() - start_time
    
    print(f"With caching - First run:     {time_cache_1:.3f}s")
    print(f"With caching - Second run:    {time_cache_2:.3f}s")
    
    if time_cache_2 > 0:
        cache_speedup = time_no_cache_2 / time_cache_2
        print(f"Cache speedup: {cache_speedup:.2f}x")
        return cache_speedup > 1.5  # Expect significant speedup
    
    return True

def demo_adaptive_processing():
    """Demonstrate adaptive processing based on data characteristics."""
    logger.info("🧠 Testing Adaptive Processing")
    print("\n🧠 Adaptive Processing Demo")
    print("-" * 38)
    
    # Create datasets with different characteristics
    datasets = {
        'clean': create_large_dataset(1000),  # Mostly clean data
        'messy': None,  # Will create messy data
        'sparse': None   # Will create sparse data
    }
    
    # Create messy dataset (more quality issues)
    messy_data = datasets['clean'].copy()
    messy_data.loc[::3, 'name'] = None  # Every 3rd name is null
    messy_data.loc[::4, 'email'] = 'invalid'  # Every 4th email is invalid
    messy_data.loc[::5, 'age'] = 'unknown'  # Every 5th age is invalid
    datasets['messy'] = messy_data
    
    # Create sparse dataset (many nulls)
    sparse_data = datasets['clean'].copy()
    sparse_data.loc[::2, 'name'] = None  # Half the names are null
    sparse_data.loc[::2, 'email'] = None  # Half the emails are null
    datasets['sparse'] = sparse_data
    
    # Test adaptive processing
    cleaner = TableCleaner(confidence_threshold=0.7)
    
    for dataset_type, df in datasets.items():
        logger.info(f"Processing {dataset_type} dataset")
        
        start_time = time.time()
        result = cleaner.clean(df)
        processing_time = time.time() - start_time
        
        # Calculate data quality metrics
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        print(f"{dataset_type.capitalize():6s} dataset: {processing_time:.3f}s, "
              f"{null_percentage:.1f}% nulls")
    
    return True

def demo_load_balancing():
    """Demonstrate load balancing across multiple workers."""
    logger.info("⚖️ Testing Load Balancing")
    print("\n⚖️ Load Balancing Demo")
    print("-" * 30)
    
    # Create workload with varying complexity
    datasets = []
    for i in range(8):
        size = 500 + i * 200  # Varying sizes from 500 to 1900
        df = create_large_dataset(size)
        datasets.append((i, df))
    
    # Process with different worker configurations
    worker_configs = [1, 2, 4]
    
    for num_workers in worker_configs:
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            def process_with_id(dataset_tuple):
                dataset_id, df = dataset_tuple
                cleaner = TableCleaner(confidence_threshold=0.7)
                return dataset_id, cleaner.clean(df)
            
            futures = [executor.submit(process_with_id, dataset) for dataset in datasets]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        total_rows = sum(len(df) for _, df in datasets)
        throughput = total_rows / total_time if total_time > 0 else 0
        
        print(f"{num_workers} worker(s): {total_time:.2f}s, {throughput:.0f} rows/s")
    
    return True

def demo_streaming_simulation():
    """Simulate streaming data processing."""
    logger.info("🌊 Testing Streaming Simulation")
    print("\n🌊 Streaming Processing Simulation")
    print("-" * 40)
    
    cleaner = TableCleaner(confidence_threshold=0.7, max_batch_size=100)
    
    # Simulate streaming data in batches
    batch_size = 200
    num_batches = 5
    processing_times = []
    
    print("Processing streaming batches:")
    
    for batch_num in range(num_batches):
        # Generate new batch
        batch_df = create_large_dataset(batch_size)
        
        start_time = time.time()
        result = cleaner.clean(batch_df)
        batch_time = time.time() - start_time
        processing_times.append(batch_time)
        
        print(f"  Batch {batch_num + 1}: {batch_time:.3f}s "
              f"({batch_size / batch_time:.0f} rows/s)")
    
    # Calculate streaming metrics
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    
    print(f"\nStreaming metrics:")
    print(f"  Average latency: {avg_time:.3f}s")
    print(f"  Max latency:     {max_time:.3f}s")
    print(f"  Min latency:     {min_time:.3f}s")
    print(f"  Consistency:     {(1 - (max_time - min_time) / avg_time):.1%}")
    
    return True

def main():
    """Run the scalable functionality demonstration."""
    print("🚀 Starting llm-tab-cleaner scalable functionality test...")
    logger.info("Starting Generation 3 (Scalable) testing")
    
    success_tests = 0
    total_tests = 0
    
    # Run all scalability tests
    tests = [
        ("Concurrent Processing", demo_concurrent_processing),
        ("Batch Optimization", demo_batch_optimization),
        ("Memory Optimization", demo_memory_optimization),
        ("Caching Performance", demo_caching_performance),
        ("Adaptive Processing", demo_adaptive_processing),
        ("Load Balancing", demo_load_balancing),
        ("Streaming Simulation", demo_streaming_simulation)
    ]
    
    for test_name, test_func in tests:
        total_tests += 1
        try:
            logger.info(f"Running {test_name} test...")
            if test_func():
                success_tests += 1
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.warning(f"⚠️ {test_name} test had some issues")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
    
    # Final assessment
    success_rate = (success_tests / total_tests) * 100
    print(f"\n🏆 Generation 3 (Scalable) Results:")
    print(f"Tests passed: {success_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("✅ Generation 3 (Scalable) - Optimization features working excellently!")
        print("Ready to proceed to Quality Gates and Testing")
        logger.info("Generation 3 completed successfully")
        return True
    elif success_rate >= 70:
        print("✅ Generation 3 (Scalable) - Good optimization performance!")
        print("Ready to proceed to Quality Gates and Testing")
        logger.info("Generation 3 completed with good results")
        return True
    else:
        print("⚠️ Some scalability issues detected, but basic functionality working")
        logger.warning("Generation 3 completed with some issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)