"""Tests for auto-scaling functionality."""

import time
import pytest
import pandas as pd
from unittest.mock import Mock, patch

from llm_tab_cleaner.auto_scaling import (
    ResourceMonitor,
    ResourceMetrics,
    IntelligentScaler,
    AdaptiveTableCleaner,
    ScalingDecision,
    create_adaptive_cleaner
)
from llm_tab_cleaner.core import TableCleaner, CleaningReport, Fix


class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    def test_init(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(sample_interval=0.5)
        
        assert monitor.sample_interval == 0.5
        assert len(monitor._metrics_history) == 0
        assert not monitor._monitoring
    
    def test_get_current_metrics(self):
        """Test getting current resource metrics."""
        monitor = ResourceMonitor()
        
        metrics = monitor.get_current_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.active_threads > 0
        assert metrics.timestamp > 0
    
    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        monitor = ResourceMonitor(sample_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring
        
        # Let it collect some data
        time.sleep(0.3)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor._monitoring
        
        # Should have collected some metrics
        assert len(monitor._metrics_history) > 0
    
    def test_get_average_metrics(self):
        """Test getting average metrics."""
        monitor = ResourceMonitor()
        
        # Manually add some test metrics
        test_metrics = [
            ResourceMetrics(50.0, 60.0, 0.0, 0.0, 0.0, 0.0, 10, 100, time.time()),
            ResourceMetrics(70.0, 80.0, 0.0, 0.0, 0.0, 0.0, 12, 105, time.time())
        ]
        
        monitor._metrics_history = test_metrics
        
        avg = monitor.get_average_metrics(window_seconds=60)
        
        assert avg is not None
        assert avg.cpu_percent == 60.0  # Average of 50 and 70
        assert avg.memory_percent == 70.0  # Average of 60 and 80


class TestIntelligentScaler:
    """Test intelligent scaling decisions."""
    
    def test_init(self):
        """Test intelligent scaler initialization."""
        scaler = IntelligentScaler(
            min_workers=2,
            max_workers=8,
            min_batch_size=500,
            max_batch_size=5000
        )
        
        assert scaler.min_workers == 2
        assert scaler.max_workers == 8
        assert scaler.min_batch_size == 500
        assert scaler.max_batch_size == 5000
        assert isinstance(scaler.resource_monitor, ResourceMonitor)
    
    def test_make_scaling_decision_small_data(self):
        """Test scaling decision for small datasets."""
        scaler = IntelligentScaler()
        
        decision = scaler.make_scaling_decision(
            data_size=500,
            current_workers=1,
            current_batch_size=100
        )
        
        assert isinstance(decision, ScalingDecision)
        assert decision.target_workers >= scaler.min_workers
        assert decision.target_workers <= scaler.max_workers
        assert decision.target_batch_size >= scaler.min_batch_size
        assert not decision.use_distributed  # Small data shouldn't need distribution
        assert isinstance(decision.reasoning, str)
        assert 0.0 <= decision.confidence <= 1.0
    
    def test_make_scaling_decision_large_data(self):
        """Test scaling decision for large datasets."""
        scaler = IntelligentScaler()
        
        decision = scaler.make_scaling_decision(
            data_size=500000,  # Large dataset
            current_workers=1,
            current_batch_size=1000
        )
        
        assert decision.target_workers > 1  # Should scale up for large data
        assert decision.use_distributed  # Large data should use distribution
        assert "large dataset" in decision.reasoning.lower()
    
    def test_estimate_workload_intensity(self):
        """Test workload intensity estimation."""
        scaler = IntelligentScaler()
        
        # Test with no history (default estimation)
        intensity = scaler._estimate_workload_intensity(10000, None)
        assert 0.0 <= intensity <= 1.0
        
        # Test with performance history
        history = [
            {"cpu_time": 8.0, "total_time": 10.0},  # 80% CPU utilization
            {"cpu_time": 7.0, "total_time": 10.0},  # 70% CPU utilization
        ]
        
        intensity = scaler._estimate_workload_intensity(10000, history)
        assert 0.7 <= intensity <= 0.8  # Should be around 75%
    
    def test_calculate_optimal_workers(self):
        """Test optimal worker calculation."""
        scaler = IntelligentScaler(min_workers=1, max_workers=8)
        
        # Test with low CPU usage (should allow more workers)
        low_cpu_metrics = ResourceMetrics(10.0, 30.0, 0.0, 0.0, 0.0, 0.0, 5, 50)
        workers = scaler._calculate_optimal_workers(50000, low_cpu_metrics, 0.5)
        assert workers >= 2
        
        # Test with high CPU usage (should limit workers)
        high_cpu_metrics = ResourceMetrics(90.0, 30.0, 0.0, 0.0, 0.0, 0.0, 5, 50)
        workers = scaler._calculate_optimal_workers(50000, high_cpu_metrics, 0.5)
        assert workers <= 4  # Should be conservative


class TestAdaptiveTableCleaner:
    """Test adaptive table cleaner functionality."""
    
    def test_init(self):
        """Test adaptive cleaner initialization."""
        def mock_factory():
            return Mock(spec=TableCleaner)
        
        cleaner = AdaptiveTableCleaner(mock_factory)
        
        assert cleaner.base_cleaner_factory == mock_factory
        assert isinstance(cleaner.scaler, IntelligentScaler)
        assert cleaner.enable_caching
        assert len(cleaner.performance_history) == 0
    
    @patch('llm_tab_cleaner.auto_scaling.IntelligentScaler')
    def test_clean_single_process(self, mock_scaler_class):
        """Test single process cleaning."""
        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.make_scaling_decision.return_value = ScalingDecision(
            target_workers=1,
            target_batch_size=1000,
            use_distributed=False,
            use_async=False,
            reasoning="Single process",
            confidence=0.9
        )
        mock_scaler_class.return_value = mock_scaler
        
        # Mock base cleaner
        mock_base_cleaner = Mock()
        mock_report = CleaningReport(
            total_fixes=5,
            quality_score=0.95,
            fixes=[],
            processing_time=1.0
        )
        mock_base_cleaner.clean.return_value = (
            pd.DataFrame({'col': ['clean1', 'clean2']}),
            mock_report
        )
        
        def mock_factory():
            return mock_base_cleaner
        
        cleaner = AdaptiveTableCleaner(mock_factory, mock_scaler)
        
        # Test data
        df = pd.DataFrame({'col': ['dirty1', 'dirty2']})
        
        # Clean
        cleaned_df, report = cleaner.clean(df)
        
        # Verify
        assert len(cleaned_df) == 2
        assert report.total_fixes == 5
        assert len(cleaner.performance_history) == 1
        mock_base_cleaner.clean.assert_called_once()
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        def mock_factory():
            mock_cleaner = Mock()
            mock_cleaner.clean.return_value = (
                pd.DataFrame({'col': ['clean']}),
                CleaningReport(1, 0.9, [], 0.5)
            )
            return mock_cleaner
        
        cleaner = AdaptiveTableCleaner(mock_factory, enable_caching=True, cache_size=2)
        
        df = pd.DataFrame({'col': ['dirty']})
        
        # First call - should go through processing
        result1 = cleaner.clean(df)
        assert len(cleaner._cache) == 1
        
        # Second call with same data - should hit cache
        result2 = cleaner.clean(df)
        assert len(cleaner._cache) == 1
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1[0], result2[0])
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        def mock_factory():
            return Mock()
        
        cleaner = AdaptiveTableCleaner(mock_factory)
        
        df1 = pd.DataFrame({'col': ['a', 'b']})
        df2 = pd.DataFrame({'col': ['a', 'b']})
        df3 = pd.DataFrame({'col': ['c', 'd']})
        
        key1 = cleaner._generate_cache_key(df1, None, 1.0)
        key2 = cleaner._generate_cache_key(df2, None, 1.0)
        key3 = cleaner._generate_cache_key(df3, None, 1.0)
        
        assert key1 == key2  # Same data should have same key
        assert key1 != key3  # Different data should have different keys
        assert isinstance(key1, str)
    
    def test_split_dataframe(self):
        """Test DataFrame splitting functionality."""
        def mock_factory():
            return Mock()
        
        cleaner = AdaptiveTableCleaner(mock_factory)
        
        df = pd.DataFrame({'col': list(range(10))})
        
        # Split into batches of 3
        batches = cleaner._split_dataframe(df, 3)
        
        assert len(batches) == 4  # 10 rows / 3 per batch = 4 batches
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3  
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1  # Remainder
        
        # Verify all data is preserved
        combined = pd.concat(batches, ignore_index=True)
        pd.testing.assert_frame_equal(df, combined)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        def mock_factory():
            return Mock()
        
        cleaner = AdaptiveTableCleaner(mock_factory)
        
        # Add some mock performance data
        cleaner.performance_history = [
            {"throughput": 100, "processing_time": 1.0, "data_size": 100},
            {"throughput": 200, "processing_time": 0.5, "data_size": 100}
        ]
        cleaner._cache = {"key1": (Mock(), Mock())}
        
        stats = cleaner.get_performance_stats()
        
        assert stats["total_operations"] == 2
        assert stats["avg_throughput"] == 150.0
        assert stats["avg_processing_time"] == 0.75
        assert stats["avg_data_size"] == 100
        assert "cache_hit_ratio" in stats


class TestCreateAdaptiveCleaner:
    """Test adaptive cleaner factory function."""
    
    def test_create_adaptive_cleaner(self):
        """Test creating adaptive cleaner with factory function."""
        cleaner = create_adaptive_cleaner(
            llm_provider="local",
            confidence_threshold=0.9
        )
        
        assert isinstance(cleaner, AdaptiveTableCleaner)
        
        # Test that the factory creates cleaners with correct parameters
        base_cleaner = cleaner.base_cleaner_factory()
        assert isinstance(base_cleaner, TableCleaner)
        assert base_cleaner.confidence_threshold == 0.9