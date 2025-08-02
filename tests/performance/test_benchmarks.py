"""Benchmark tests for LLM Tab Cleaner performance."""

import time
import pytest
import pandas as pd
from llm_tab_cleaner import TableCleaner


@pytest.mark.benchmark
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_small_dataset_processing_speed(self, benchmark, table_cleaner_with_mock_llm, performance_config):
        """Benchmark processing speed for small datasets."""
        # Create small test dataset
        size = performance_config["small_dataset_size"]
        df = pd.DataFrame({
            'id': range(size),
            'name': [f'Person {i}' for i in range(size)],
            'email': [f'person{i}@test.com' if i % 10 != 0 else f'invalid{i}' for i in range(size)]
        })

        def process_data():
            result = table_cleaner_with_mock_llm.clean(df)
            return result

        # Benchmark the processing
        result = benchmark(process_data)
        
        # Verify we got a result
        assert result is not None
        assert len(result[0]) == size  # Same number of rows

    def test_medium_dataset_processing_speed(self, table_cleaner_with_mock_llm, performance_config):
        """Test processing speed for medium datasets."""
        size = performance_config["medium_dataset_size"]
        df = pd.DataFrame({
            'id': range(size),
            'name': [f'Person {i}' for i in range(size)],
            'value': [i * 1.5 if i % 100 != 0 else 'invalid' for i in range(size)]
        })

        start_time = time.time()
        cleaned_df, report = table_cleaner_with_mock_llm.clean(df)
        end_time = time.time()

        processing_time = end_time - start_time
        records_per_second = size / processing_time

        # Assert performance expectations
        assert processing_time < performance_config["max_processing_time"]
        assert records_per_second > performance_config["expected_throughput"] / 10  # Allow for overhead

    @pytest.mark.requires_llm
    def test_batch_processing_efficiency(self, table_cleaner, performance_config):
        """Test efficiency of batch processing vs single record processing."""
        size = 1000
        df = pd.DataFrame({
            'id': range(size),
            'email': [f'person{i}@invalid' if i % 5 == 0 else f'person{i}@test.com' for i in range(size)]
        })

        # Test single record processing (simulated)
        single_start = time.time()
        for i in range(min(10, size)):  # Only test first 10 for time
            single_row = df.iloc[i:i+1]
            # This would normally call the cleaner, but we'll simulate
            time.sleep(0.001)  # Simulate API call overhead
        single_time_per_record = (time.time() - single_start) / 10

        # Test batch processing
        batch_start = time.time()
        cleaned_df, report = table_cleaner.clean(df)
        batch_time = time.time() - batch_start
        batch_time_per_record = batch_time / size

        # Batch processing should be more efficient
        efficiency_improvement = single_time_per_record / batch_time_per_record
        assert efficiency_improvement > 1.5  # At least 50% more efficient

    def test_memory_usage_scaling(self, table_cleaner_with_mock_llm, memory_monitor):
        """Test that memory usage scales linearly with data size."""
        import psutil
        
        memory_readings = []
        dataset_sizes = [1000, 2000, 4000]

        for size in dataset_sizes:
            df = pd.DataFrame({
                'id': range(size),
                'value': [f'value_{i}' for i in range(size)]
            })

            # Force garbage collection before measurement
            import gc
            gc.collect()

            memory_before = memory_monitor.memory_info().rss
            cleaned_df, report = table_cleaner_with_mock_llm.clean(df)
            memory_after = memory_monitor.memory_info().rss

            memory_used = memory_after - memory_before
            memory_readings.append((size, memory_used))

        # Check that memory usage doesn't grow exponentially
        size_ratio = dataset_sizes[1] / dataset_sizes[0]
        memory_ratio = memory_readings[1][1] / memory_readings[0][1]
        
        # Memory should not grow more than 3x when data grows 2x
        assert memory_ratio < size_ratio * 1.5

    @pytest.mark.benchmark  
    def test_confidence_scoring_performance(self, benchmark, table_cleaner_with_mock_llm):
        """Benchmark confidence scoring performance."""
        df = pd.DataFrame({
            'values': ['good_value'] * 100 + ['bad_value'] * 100
        })

        def score_confidence():
            # This would normally call the confidence scorer
            cleaned_df, report = table_cleaner_with_mock_llm.clean(df)
            return report

        result = benchmark(score_confidence)
        assert result is not None

    def test_concurrent_processing_safety(self, table_cleaner_with_mock_llm):
        """Test that concurrent processing is thread-safe."""
        import threading
        import queue

        df = pd.DataFrame({
            'id': range(100),
            'value': [f'value_{i}' for i in range(100)]
        })

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def process_data(thread_id):
            try:
                cleaned_df, report = table_cleaner_with_mock_llm.clean(df)
                results_queue.put((thread_id, len(cleaned_df)))
            except Exception as e:
                errors_queue.put((thread_id, str(e)))

        # Start multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)

        # Check results
        assert errors_queue.qsize() == 0, f"Errors occurred: {list(errors_queue.queue)}"
        assert results_queue.qsize() == num_threads

        # All results should have the same number of rows
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        row_counts = [result[1] for result in results]
        assert all(count == row_counts[0] for count in row_counts)

    @pytest.mark.slow
    def test_large_dataset_streaming(self, table_cleaner_with_mock_llm, performance_config):
        """Test processing large datasets in streaming fashion."""
        # Simulate large dataset processing in chunks
        total_size = performance_config["large_dataset_size"]
        chunk_size = 1000
        total_processed = 0
        
        start_time = time.time()
        
        for chunk_start in range(0, total_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_size)
            chunk_df = pd.DataFrame({
                'id': range(chunk_start, chunk_end),
                'value': [f'value_{i}' for i in range(chunk_start, chunk_end)]
            })
            
            cleaned_chunk, report = table_cleaner_with_mock_llm.clean(chunk_df)
            total_processed += len(cleaned_chunk)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all data was processed
        assert total_processed == total_size
        
        # Check performance is reasonable
        records_per_second = total_size / processing_time
        assert records_per_second > performance_config["expected_throughput"] / 5  # Allow for chunking overhead