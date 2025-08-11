"""Tests for research algorithms and benchmarking."""

import asyncio
import pytest
import pandas as pd
import numpy as np

from llm_tab_cleaner.research import (
    EnsembleLLMCleaner, 
    AdaptiveLLMCleaner, 
    ResearchBenchmarker,
    create_synthetic_benchmark
)


class TestEnsembleLLMCleaner:
    """Test ensemble LLM cleaning algorithms."""
    
    def test_init_majority_voting(self):
        """Test ensemble cleaner initialization with majority voting."""
        cleaner = EnsembleLLMCleaner(
            providers=["local"], 
            voting_strategy="majority"
        )
        
        assert cleaner.name == "ensemble_llm_majority"
        assert cleaner.voting_strategy == "majority"
        assert len(cleaner.providers) == 1
    
    def test_init_confidence_weighted(self):
        """Test ensemble cleaner initialization with confidence weighting."""
        cleaner = EnsembleLLMCleaner(
            providers=["local"], 
            voting_strategy="confidence_weighted"
        )
        
        assert cleaner.name == "ensemble_llm_confidence_weighted"
        assert cleaner.voting_strategy == "confidence_weighted"
    
    @pytest.mark.asyncio
    async def test_clean_async_simple(self):
        """Test async cleaning with simple data."""
        cleaner = EnsembleLLMCleaner(providers=["local"])
        
        df = pd.DataFrame({
            'email': ['test@example.com', 'invalid@'],
            'name': ['Alice', 'Bob']
        })
        
        cleaned_df, report = await cleaner.clean_async(df)
        
        assert len(cleaned_df) == 2
        assert isinstance(report.total_fixes, int)
        assert isinstance(report.quality_score, float)
        assert 0.0 <= report.quality_score <= 1.0


class TestAdaptiveLLMCleaner:
    """Test adaptive LLM cleaning algorithm."""
    
    def test_init(self):
        """Test adaptive cleaner initialization."""
        cleaner = AdaptiveLLMCleaner(
            base_provider="local",
            learning_rate=0.2,
            memory_size=500
        )
        
        assert cleaner.name == "adaptive_llm_local"
        assert cleaner.learning_rate == 0.2
        assert cleaner.memory_size == 500
        assert len(cleaner.correction_memory) == 0
    
    @pytest.mark.asyncio
    async def test_clean_async_with_ground_truth(self):
        """Test adaptive cleaning with ground truth for learning."""
        cleaner = AdaptiveLLMCleaner(base_provider="local")
        
        dirty_df = pd.DataFrame({
            'email': ['test@example.com', 'invalid@domain'],
            'name': ['Alice', 'Bob']
        })
        
        ground_truth = pd.DataFrame({
            'email': ['test@example.com', 'invalid@domain.com'],
            'name': ['Alice', 'Bob']
        })
        
        cleaned_df, report = await cleaner.clean_async(dirty_df, ground_truth)
        
        assert len(cleaned_df) == 2
        assert isinstance(report.total_fixes, int)
        assert len(cleaner.correction_memory) >= 0  # Should have learned something
    
    def test_values_similar(self):
        """Test similarity detection for learning transfer."""
        cleaner = AdaptiveLLMCleaner()
        
        # Exact match
        assert cleaner._values_similar("test@example.com", "test@example.com")
        
        # Email similarity
        assert cleaner._values_similar("user@domain.com", "admin@company.org")
        
        # Number similarity
        assert cleaner._values_similar("123", "456")
        
        # Phone similarity
        assert cleaner._values_similar("123-456-7890", "987-654-3210")
        
        # Different types
        assert not cleaner._values_similar("email@test.com", "123-456-7890")


class TestResearchBenchmarker:
    """Test research benchmarking functionality."""
    
    def test_init(self):
        """Test benchmarker initialization."""
        benchmarker = ResearchBenchmarker(output_dir="./test_results")
        
        assert benchmarker.output_dir.name == "test_results"
        assert benchmarker.output_dir.exists()
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        benchmarker = ResearchBenchmarker()
        
        # Simple test case
        dirty_df = pd.DataFrame({'col': ['A', 'B', 'wrong']})
        cleaned_df = pd.DataFrame({'col': ['A', 'B', 'correct']})
        ground_truth = pd.DataFrame({'col': ['A', 'B', 'correct']})
        
        from llm_tab_cleaner.core import Fix
        fixes = [Fix(
            column='col', 
            row_index=2, 
            original='wrong', 
            cleaned='correct', 
            confidence=0.9
        )]
        
        metrics = benchmarker._calculate_metrics(dirty_df, cleaned_df, ground_truth, fixes)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    
    @pytest.mark.asyncio
    async def test_run_comparative_study_small(self):
        """Test running a small comparative study."""
        # Create minimal algorithms for testing
        algorithms = [
            EnsembleLLMCleaner(providers=["local"], voting_strategy="majority")
        ]
        
        # Create minimal benchmark
        benchmark = create_synthetic_benchmark()
        
        benchmarker = ResearchBenchmarker(output_dir="./test_study_results")
        
        # Run with just 1 run to keep test fast
        results = await benchmarker.run_comparative_study(
            algorithms=algorithms,
            benchmark_suite=benchmark,
            runs_per_algorithm=1
        )
        
        assert len(results) == 1
        assert "ensemble_llm_majority" in results
        assert len(results["ensemble_llm_majority"]) > 0
        
        # Check result structure
        result = results["ensemble_llm_majority"][0]
        assert hasattr(result, 'algorithm_name')
        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'processing_time')


class TestSyntheticBenchmark:
    """Test synthetic benchmark creation."""
    
    def test_create_synthetic_benchmark(self):
        """Test synthetic benchmark creation."""
        benchmark = create_synthetic_benchmark()
        
        assert benchmark.name == "synthetic_benchmark"
        assert len(benchmark.datasets) == 2  # Email and phone datasets
        assert "description" in benchmark.metadata
        
        # Check first dataset (emails)
        dirty_df, clean_df = benchmark.datasets[0]
        assert "email" in dirty_df.columns
        assert "email" in clean_df.columns
        assert len(dirty_df) == len(clean_df)
        
        # Check second dataset (phones)  
        dirty_df2, clean_df2 = benchmark.datasets[1]
        assert "phone" in dirty_df2.columns
        assert "phone" in clean_df2.columns
        assert len(dirty_df2) == len(clean_df2)