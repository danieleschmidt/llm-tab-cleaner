"""Tests for incremental cleaning module."""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from llm_tab_cleaner.incremental import IncrementalCleaner


class TestIncrementalCleaner:
    """Test suite for IncrementalCleaner class."""
    
    def test_init_with_default_state_path(self):
        """Test IncrementalCleaner initialization with default state path."""
        cleaner = IncrementalCleaner()
        assert cleaner is not None
        assert hasattr(cleaner, 'process_increment')
        assert hasattr(cleaner, 'reprocess_low_confidence')
    
    def test_init_with_custom_state_path(self):
        """Test IncrementalCleaner initialization with custom state path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "custom_state.db"
            cleaner = IncrementalCleaner(
                state_path=str(state_path),
                llm_provider="anthropic"
            )
            assert cleaner is not None
    
    def test_process_increment_empty_dataframe(self):
        """Test process_increment with empty DataFrame."""
        cleaner = IncrementalCleaner()
        empty_df = pd.DataFrame()
        
        result = cleaner.process_increment(empty_df)
        
        # Should handle empty DataFrame gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_process_increment_simple_dataframe(self):
        """Test process_increment with simple DataFrame."""
        cleaner = IncrementalCleaner()
        sample_df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'age': [25, 30],
            'email': ['john@example.com', 'jane@example.com']
        })
        
        result = cleaner.process_increment(sample_df)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
    
    def test_process_increment_with_update_statistics(self):
        """Test process_increment with update_statistics=True."""
        cleaner = IncrementalCleaner()
        sample_df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'score': [85, 92]
        })
        
        result = cleaner.process_increment(
            sample_df, 
            update_statistics=True
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_reprocess_low_confidence_default_threshold(self):
        """Test reprocess_low_confidence with default threshold."""
        cleaner = IncrementalCleaner()
        
        # Should execute without errors even if no previous data exists
        result = cleaner.reprocess_low_confidence()
        
        # Implementation may return None or empty result when no data to reprocess
        assert result is None or isinstance(result, pd.DataFrame)
    
    def test_reprocess_low_confidence_custom_threshold(self):
        """Test reprocess_low_confidence with custom threshold."""
        cleaner = IncrementalCleaner()
        
        result = cleaner.reprocess_low_confidence(
            confidence_threshold=0.8,
            new_model="claude-3"
        )
        
        assert result is None or isinstance(result, pd.DataFrame)
    
    def test_reprocess_low_confidence_invalid_threshold(self):
        """Test reprocess_low_confidence with invalid threshold."""
        cleaner = IncrementalCleaner()
        
        # Test with threshold > 1.0
        with pytest.raises(ValueError):
            cleaner.reprocess_low_confidence(confidence_threshold=1.5)
        
        # Test with negative threshold
        with pytest.raises(ValueError):
            cleaner.reprocess_low_confidence(confidence_threshold=-0.1)
    
    def test_state_persistence(self):
        """Test that state is persisted between operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "test_state.db"
            
            # First cleaner instance
            cleaner1 = IncrementalCleaner(state_path=str(state_path))
            sample_df = pd.DataFrame({
                'data': ['test1', 'test2']
            })
            
            cleaner1.process_increment(sample_df)
            
            # Second cleaner instance (should load previous state)
            cleaner2 = IncrementalCleaner(state_path=str(state_path))
            
            # Should be able to create new cleaner with same state path
            assert cleaner2 is not None
    
    def test_different_llm_providers(self):
        """Test initialization with different LLM providers."""
        providers = ["anthropic", "openai", "local"]
        
        for provider in providers:
            cleaner = IncrementalCleaner(llm_provider=provider)
            assert cleaner is not None
    
    def test_process_increment_with_different_dtypes(self):
        """Test process_increment with various data types."""
        cleaner = IncrementalCleaner()
        
        mixed_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=3)
        })
        
        result = cleaner.process_increment(mixed_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_df)
        
    def test_cleanup_on_del(self):
        """Test cleanup when cleaner is deleted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "cleanup_test.db"
            
            cleaner = IncrementalCleaner(state_path=str(state_path))
            sample_df = pd.DataFrame({'test': [1, 2, 3]})
            cleaner.process_increment(sample_df)
            
            # Delete cleaner
            del cleaner
            
            # This test mainly ensures no exceptions are raised during cleanup