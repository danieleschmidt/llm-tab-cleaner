"""Testing utilities and helpers for LLM Tab Cleaner."""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestTimer:
    """Context manager for timing test operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = 0
        self.end_time = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time
        
    def assert_time_under(self, max_seconds: float):
        """Assert that the operation took less than max_seconds."""
        if self.elapsed_time > max_seconds:
            pytest.fail(f"{self.name} took {self.elapsed_time:.2f}s, expected under {max_seconds}s")


class DataFrameAssertion:
    """Helper class for DataFrame assertions in tests."""
    
    @staticmethod
    def assert_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
        """Assert that two DataFrames are equal."""
        try:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
        except AssertionError as e:
            pytest.fail(f"DataFrames are not equal: {e}")
    
    @staticmethod
    def assert_columns_equal(df1: pd.DataFrame, df2: pd.DataFrame):
        """Assert that two DataFrames have the same columns."""
        if list(df1.columns) != list(df2.columns):
            pytest.fail(f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}")
    
    @staticmethod
    def assert_no_nulls(df: pd.DataFrame, columns: Optional[List[str]] = None):
        """Assert that specified columns have no null values."""
        columns = columns or df.columns.tolist()
        for col in columns:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                pytest.fail(f"Column '{col}' has {null_count} null values")
    
    @staticmethod
    def assert_data_types(df: pd.DataFrame, expected_types: Dict[str, str]):
        """Assert that DataFrame columns have expected data types."""
        for col, expected_type in expected_types.items():
            actual_type = str(df[col].dtype)
            if not actual_type.startswith(expected_type):
                pytest.fail(f"Column '{col}' has type '{actual_type}', expected '{expected_type}'")
    
    @staticmethod
    def assert_unique_values(df: pd.DataFrame, column: str):
        """Assert that a column has all unique values."""
        if df[column].duplicated().any():
            duplicates = df[df[column].duplicated()]
            pytest.fail(f"Column '{column}' has duplicate values: {duplicates[column].tolist()}")


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None):
        self.responses = responses or [
            {
                "fixed_value": "corrected_value",
                "confidence": 0.9,
                "reasoning": "Test correction"
            }
        ]
        self.call_count = 0
        self.call_history = []
    
    def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Mock completion method."""
        self.call_count += 1
        self.call_history.append({"prompt": prompt, "kwargs": kwargs})
        
        # Return the next response, or cycle back to the beginning
        response_index = (self.call_count - 1) % len(self.responses)
        return self.responses[response_index]
    
    def batch_complete(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Mock batch completion method."""
        return [self.complete(prompt, **kwargs) for prompt in prompts]
    
    def reset(self):
        """Reset call count and history."""
        self.call_count = 0
        self.call_history.clear()


def requires_env_var(var_name: str):
    """Skip test if environment variable is not set."""
    import os
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if var_name not in os.environ:
                pytest.skip(f"Test requires environment variable: {var_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Retry test on failure (useful for flaky tests)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
                    
            raise last_exception
        return wrapper
    return decorator


@contextmanager
def temporary_environment_variables(**env_vars):
    """Temporarily set environment variables for a test."""
    import os
    original_env = {}
    
    # Store original values
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


class PerformanceAssertion:
    """Helper for performance-related assertions."""
    
    @staticmethod
    def assert_performance_regression(
        current_time: float,
        baseline_time: float,
        max_regression_percent: float = 20.0
    ):
        """Assert that performance hasn't regressed beyond acceptable threshold."""
        regression = ((current_time - baseline_time) / baseline_time) * 100
        
        if regression > max_regression_percent:
            pytest.fail(
                f"Performance regression detected: {regression:.1f}% slower "
                f"(current: {current_time:.3f}s, baseline: {baseline_time:.3f}s)"
            )
    
    @staticmethod
    def assert_memory_usage_under(max_memory_mb: float):
        """Assert that memory usage is under the specified threshold."""
        import psutil
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > max_memory_mb:
            pytest.fail(f"Memory usage {memory_mb:.1f}MB exceeds limit {max_memory_mb}MB")
    
    @staticmethod
    def assert_throughput_above(records_processed: int, elapsed_time: float, min_rps: float):
        """Assert that throughput is above the minimum threshold."""
        actual_rps = records_processed / elapsed_time
        
        if actual_rps < min_rps:
            pytest.fail(f"Throughput {actual_rps:.1f} RPS below minimum {min_rps} RPS")


class DataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_messy_names(count: int) -> List[str]:
        """Generate names with various formatting issues."""
        base_names = [
            "john smith", "JANE DOE", "Bob Johnson", "alice BROWN",
            "charlie wilson", "EVE DAVIS", "frank Miller", "grace GARCIA"
        ]
        
        names = []
        for i in range(count):
            base = base_names[i % len(base_names)]
            # Add various formatting issues
            if i % 3 == 0:
                names.append(base.lower())
            elif i % 3 == 1:
                names.append(base.upper())
            else:
                names.append(base.title())
                
        return names
    
    @staticmethod
    def create_messy_emails(count: int) -> List[str]:
        """Generate email addresses with various issues."""
        issues = [
            "user@domain.com",  # Valid
            "user@domain",      # Missing TLD
            "user.domain.com",  # Missing @
            "@domain.com",      # Missing user
            "user@.com",        # Missing domain
            "user email@domain.com",  # Space in user
            "user@domain.",     # Trailing dot
            "",                 # Empty
        ]
        
        emails = []
        for i in range(count):
            emails.append(issues[i % len(issues)])
            
        return emails
    
    @staticmethod
    def create_dataframe_with_issues(rows: int) -> pd.DataFrame:
        """Create a DataFrame with various data quality issues."""
        return pd.DataFrame({
            'id': list(range(rows)) + [None] * (rows // 10),  # Some nulls
            'name': DataGenerator.create_messy_names(rows + rows // 10),
            'email': DataGenerator.create_messy_emails(rows + rows // 10),
            'age': [25 + i % 50 if i % 20 != 0 else 'invalid' for i in range(rows + rows // 10)],
            'salary': [50000 + i * 1000 if i % 15 != 0 else '$invalid' for i in range(rows + rows // 10)]
        })


def parametrize_dataframes():
    """Decorator to parametrize tests with different DataFrame sizes."""
    sizes = [
        pytest.param(10, id="small"),
        pytest.param(100, id="medium"),
        pytest.param(1000, id="large", marks=pytest.mark.slow),
    ]
    
    def decorator(func):
        return pytest.mark.parametrize("df_size", sizes)(func)
    
    return decorator


def skip_if_no_api_key(provider: str):
    """Skip test if API key for provider is not available."""
    import os
    
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure": "AZURE_OPENAI_KEY"
    }
    
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.getenv(env_var):
                pytest.skip(f"Test requires {env_var} environment variable")
            return func(*args, **kwargs)
        return wrapper
    return decorator