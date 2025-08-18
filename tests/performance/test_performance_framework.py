"""Performance testing framework and benchmarks for LLM Tab Cleaner."""

import time
from typing import Dict, List, Tuple
import pytest
import pandas as pd
import numpy as np

from llm_tab_cleaner import TableCleaner
from tests.utils import (
    TestTimer, PerformanceAssertion, DataGenerator,
    parametrize_dataframes, skip_if_no_api_key
)


class PerformanceBenchmark:
    """Performance benchmark suite for LLM Tab Cleaner."""
    
    def __init__(self):
        self.results = {}
        self.baselines = {
            # Expected performance baselines (records per second)
            "small_dataset": 1000,   # 1K RPS for datasets < 1K records
            "medium_dataset": 800,   # 800 RPS for datasets 1K-10K records  
            "large_dataset": 500,    # 500 RPS for datasets > 10K records
        }
    
    def benchmark_cleaning_performance(self, df: pd.DataFrame, cleaner: TableCleaner) -> Dict:
        """Benchmark the cleaning performance for a given DataFrame."""
        with TestTimer("data_cleaning") as timer:
            cleaned_df, report = cleaner.clean(df)
        
        records_processed = len(df)
        throughput = records_processed / timer.elapsed_time
        
        return {
            "records_processed": records_processed,
            "elapsed_time": timer.elapsed_time,
            "throughput_rps": throughput,
            "memory_usage_mb": self._get_memory_usage(),
            "fixes_applied": report.total_fixes if hasattr(report, 'total_fixes') else 0,
            "quality_score": report.quality_score if hasattr(report, 'quality_score') else 0.0
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


@pytest.fixture
def performance_benchmark():
    """Provide performance benchmark instance."""
    return PerformanceBenchmark()


@pytest.fixture
def mock_cleaner():
    """Provide a mocked cleaner for performance testing."""
    from unittest.mock import MagicMock
    
    cleaner = MagicMock()
    cleaner.clean.return_value = (
        pd.DataFrame(),  # Empty cleaned DataFrame
        MagicMock(total_fixes=10, quality_score=0.95)
    )
    return cleaner


class TestCleaningPerformance:
    """Test suite for cleaning performance benchmarks."""
    
    @parametrize_dataframes()
    def test_cleaning_throughput(self, df_size: int, performance_benchmark, mock_cleaner):
        """Test cleaning throughput for different dataset sizes."""
        # Generate test data
        df = DataGenerator.create_dataframe_with_issues(df_size)
        
        # Benchmark performance
        results = performance_benchmark.benchmark_cleaning_performance(df, mock_cleaner)
        
        # Determine expected baseline
        if df_size < 1000:
            baseline = performance_benchmark.baselines["small_dataset"]
        elif df_size < 10000:
            baseline = performance_benchmark.baselines["medium_dataset"]
        else:
            baseline = performance_benchmark.baselines["large_dataset"]
        
        # Assert performance meets baseline
        PerformanceAssertion.assert_throughput_above(
            results["records_processed"],
            results["elapsed_time"],
            baseline * 0.8  # Allow 20% variance
        )
        
        print(f"Processed {results['records_processed']} records in "
              f"{results['elapsed_time']:.3f}s ({results['throughput_rps']:.1f} RPS)")
    
    @pytest.mark.slow
    def test_memory_usage_scaling(self, performance_benchmark, mock_cleaner):
        """Test memory usage doesn't grow excessively with dataset size."""
        sizes = [1000, 5000, 10000, 20000]
        memory_usage = []
        
        for size in sizes:
            df = DataGenerator.create_dataframe_with_issues(size)
            results = performance_benchmark.benchmark_cleaning_performance(df, mock_cleaner)
            memory_usage.append(results["memory_usage_mb"])
        
        # Assert memory growth is roughly linear (not exponential)
        # Memory usage shouldn't grow more than 2x when dataset size grows 20x
        if memory_usage[-1] > 0 and memory_usage[0] > 0:
            memory_growth_ratio = memory_usage[-1] / memory_usage[0]
            dataset_growth_ratio = sizes[-1] / sizes[0]
            
            assert memory_growth_ratio < dataset_growth_ratio * 0.1, (
                f"Memory usage grew {memory_growth_ratio:.2f}x while dataset grew "
                f"{dataset_growth_ratio:.2f}x - possible memory leak"
            )
    
    @pytest.mark.benchmark
    def test_concurrent_cleaning_performance(self, performance_benchmark):
        """Test performance with concurrent cleaning operations."""
        import concurrent.futures
        import threading
        from unittest.mock import MagicMock
        
        # Create multiple datasets
        datasets = [
            DataGenerator.create_dataframe_with_issues(500) 
            for _ in range(4)
        ]
        
        # Mock cleaner that simulates work
        def mock_clean(df):
            time.sleep(0.1)  # Simulate processing time
            return df, MagicMock(total_fixes=5, quality_score=0.9)
        
        cleaner = MagicMock()
        cleaner.clean.side_effect = mock_clean
        
        # Test sequential processing
        start_time = time.time()
        for df in datasets:
            cleaner.clean(df)
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cleaner.clean, df) for df in datasets]
            concurrent.futures.wait(futures)
        concurrent_time = time.time() - start_time
        
        # Concurrent should be faster (but not necessarily 4x due to overhead)
        speedup = sequential_time / concurrent_time
        assert speedup > 1.5, f"Concurrent processing only {speedup:.2f}x faster"
        
        print(f"Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s, "
              f"Speedup: {speedup:.2f}x")
    
    @pytest.mark.slow
    @skip_if_no_api_key("openai")
    def test_real_llm_performance(self, performance_benchmark):
        """Test performance with actual LLM API calls (requires API key)."""
        # Create small dataset for real LLM testing
        df = DataGenerator.create_dataframe_with_issues(10)
        
        cleaner = TableCleaner(
            llm_provider="openai",
            confidence_threshold=0.8,
            max_batch_size=5
        )
        
        results = performance_benchmark.benchmark_cleaning_performance(df, cleaner)
        
        # Real LLM calls will be slower, adjust expectations
        min_throughput = 1.0  # At least 1 record per second
        PerformanceAssertion.assert_throughput_above(
            results["records_processed"],
            results["elapsed_time"],
            min_throughput
        )
        
        # Assert reasonable response time per record
        avg_time_per_record = results["elapsed_time"] / results["records_processed"]
        assert avg_time_per_record < 10.0, (
            f"Average time per record ({avg_time_per_record:.2f}s) is too high"
        )
    
    def test_batch_size_optimization(self, performance_benchmark):
        """Test different batch sizes for optimal performance."""
        from unittest.mock import MagicMock
        
        df = DataGenerator.create_dataframe_with_issues(1000)
        batch_sizes = [1, 10, 50, 100, 200]
        results = {}
        
        for batch_size in batch_sizes:
            # Mock cleaner with batch size consideration
            cleaner = MagicMock()
            
            def mock_clean_with_batching(data):
                # Simulate batch processing overhead
                num_batches = len(data) / batch_size
                processing_time = 0.001 * len(data) + 0.01 * num_batches
                time.sleep(processing_time)
                return data, MagicMock(total_fixes=10, quality_score=0.9)
            
            cleaner.clean.side_effect = mock_clean_with_batching
            
            performance_result = performance_benchmark.benchmark_cleaning_performance(df, cleaner)
            results[batch_size] = performance_result["throughput_rps"]
        
        # Find optimal batch size (highest throughput)
        optimal_batch_size = max(results.keys(), key=lambda k: results[k])
        print(f"Batch size performance: {results}")
        print(f"Optimal batch size: {optimal_batch_size}")
        
        # Assert that there's a clear optimal batch size (not just the extremes)
        assert 1 < optimal_batch_size < max(batch_sizes), (
            "Optimal batch size should be between extremes"
        )


class TestScalabilityBenchmarks:
    """Test suite for scalability benchmarks."""
    
    @pytest.mark.slow
    def test_linear_scaling(self, performance_benchmark, mock_cleaner):
        """Test that performance scales linearly with dataset size."""
        sizes = [100, 200, 400, 800]
        throughputs = []
        
        for size in sizes:
            df = DataGenerator.create_dataframe_with_issues(size)
            results = performance_benchmark.benchmark_cleaning_performance(df, mock_cleaner)
            throughputs.append(results["throughput_rps"])
        
        # Calculate coefficient of variation for throughput
        # Should be low if scaling is linear
        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        cv = std_throughput / mean_throughput if mean_throughput > 0 else 0
        
        # Coefficient of variation should be less than 20% for good linear scaling
        assert cv < 0.2, (
            f"Throughput varies too much across dataset sizes (CV: {cv:.3f}). "
            f"Throughputs: {throughputs}"
        )
    
    @pytest.mark.slow 
    def test_memory_efficiency(self, performance_benchmark, mock_cleaner):
        """Test memory efficiency with large datasets."""
        # Test progressively larger datasets
        sizes = [1000, 5000, 10000]
        peak_memory = []
        
        for size in sizes:
            import gc
            gc.collect()  # Clean up before test
            
            df = DataGenerator.create_dataframe_with_issues(size)
            
            initial_memory = performance_benchmark._get_memory_usage()
            results = performance_benchmark.benchmark_cleaning_performance(df, mock_cleaner)
            peak_memory.append(results["memory_usage_mb"] - initial_memory)
            
            del df  # Cleanup
            gc.collect()
        
        # Memory usage per record should stay roughly constant
        memory_per_record = [mem / size for mem, size in zip(peak_memory, sizes)]
        
        # Variation in memory per record should be small
        if len(memory_per_record) > 1:
            max_per_record = max(memory_per_record)
            min_per_record = min(memory_per_record)
            if min_per_record > 0:
                variation = (max_per_record - min_per_record) / min_per_record
                assert variation < 0.5, (
                    f"Memory usage per record varies too much: {memory_per_record}"
                )


@pytest.mark.benchmark
def test_performance_regression_detection(performance_benchmark, mock_cleaner):
    """Test framework for detecting performance regressions."""
    df = DataGenerator.create_dataframe_with_issues(1000)
    
    # Simulate current performance
    current_results = performance_benchmark.benchmark_cleaning_performance(df, mock_cleaner)
    
    # Simulate baseline performance (20% faster)
    baseline_time = current_results["elapsed_time"] * 0.8
    
    # This should pass (no significant regression)
    PerformanceAssertion.assert_performance_regression(
        current_results["elapsed_time"],
        baseline_time,
        max_regression_percent=25.0  # Allow 25% regression
    )
    
    # This should fail (significant regression)
    with pytest.raises(AssertionError, match="Performance regression detected"):
        PerformanceAssertion.assert_performance_regression(
            current_results["elapsed_time"],
            baseline_time,
            max_regression_percent=10.0  # Only allow 10% regression
        )


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "--benchmark-only"])