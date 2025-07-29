"""Tests for core table cleaning functionality."""

import pandas as pd
import pytest
from llm_tab_cleaner import TableCleaner


class TestTableCleaner:
    """Test cases for TableCleaner class."""
    
    def test_init_default_params(self):
        """Test TableCleaner initialization with default parameters."""
        cleaner = TableCleaner()
        assert cleaner.llm_provider == "anthropic"
        assert cleaner.confidence_threshold == 0.85
        assert cleaner.rules is None
        
    def test_init_custom_params(self):
        """Test TableCleaner initialization with custom parameters."""
        cleaner = TableCleaner(
            llm_provider="openai",
            confidence_threshold=0.9
        )
        assert cleaner.llm_provider == "openai"
        assert cleaner.confidence_threshold == 0.9
        
    def test_clean_empty_dataframe(self):
        """Test cleaning an empty DataFrame."""
        cleaner = TableCleaner()
        df = pd.DataFrame()
        
        cleaned_df, report = cleaner.clean(df)
        
        assert cleaned_df.empty
        assert report.total_fixes == 0
        assert report.quality_score == 0.95
        
    def test_clean_simple_dataframe(self):
        """Test cleaning a simple DataFrame."""
        cleaner = TableCleaner()
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })
        
        cleaned_df, report = cleaner.clean(df)
        
        assert len(cleaned_df) == 2
        assert list(cleaned_df.columns) == ['name', 'age']
        assert isinstance(report.quality_score, float)