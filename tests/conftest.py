"""Shared test fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from llm_tab_cleaner import TableCleaner, CleaningRule, RuleSet


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM API access"
    )
    config.addinivalue_line(
        "markers", "requires_spark: mark test as requiring Spark"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Add slow marker for tests with specific names
        if "slow" in item.name or "large" in item.name:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# ENVIRONMENT AND SETUP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "sqlite:///:memory:",
        "CACHE_TTL_SECONDS": "1",
        "CONFIDENCE_THRESHOLD": "0.5",
        "MAX_BATCH_SIZE": "100",
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response for testing."""
    return {
        "fixed_value": "corrected_value",
        "confidence": 0.9,
        "reasoning": "Fixed based on pattern matching"
    }


@pytest.fixture
def mock_llm_provider(mock_llm_response):
    """Mock LLM provider for testing."""
    with patch('llm_tab_cleaner.llm_providers.get_provider') as mock:
        instance = MagicMock()
        instance.complete.return_value = mock_llm_response
        instance.batch_complete.return_value = [mock_llm_response] * 5
        mock.return_value = instance
        yield instance


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN', None, ''],
        'email': ['alice@test.com', 'BOB@TEST.COM', 'charlie.test.com', 'invalid-email', None],
        'age': [25, 'thirty', 35, -5, 150],
        'state': ['California', 'NY', 'tx', 'Unknown', None],
        'salary': [50000, '$60,000', '70k', 'confidential', None]
    })


@pytest.fixture
def large_dataframe():
    """Create a large DataFrame for performance testing."""
    size = 10000
    return pd.DataFrame({
        'id': range(size),
        'name': [f'Person {i}' for i in range(size)],
        'email': [f'person{i}@test.com' if i % 10 != 0 else f'invalid{i}' for i in range(size)],
        'age': [25 + (i % 50) if i % 100 != 0 else 'invalid' for i in range(size)],
        'category': [f'Cat{i % 5}' if i % 20 != 0 else None for i in range(size)]
    })


@pytest.fixture
def messy_dataframe():
    """Create a DataFrame with various data quality issues."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 3, 5, None, 7, 8, 9, 10],  # Duplicates, nulls
        'first_name': ['John', 'jane', 'BOB', '', None, 'Alice', 'Charlie', 'dave', 'Eve', 'Frank'],
        'last_name': ['Smith', 'DOE', 'johnson', 'Brown', 'Wilson', '', None, 'Miller', 'Davis', 'garcia'],
        'email': [
            'john@example.com', 'jane.doe@test', 'bob@invalid', 
            'brown@example.com', 'wilson@test.com', 'alice@example.com',
            'charlie.test.com', 'dave@', '@miller.com', 'frank@example.com'
        ],
        'phone': [
            '(555) 123-4567', '555.987.6543', '5551234567', 
            '555-123-4567', 'invalid', None, '(555) 111-2222',
            '555 333 4444', '1-555-555-5555', '555-999-0000'
        ],
        'birth_date': [
            '1990-01-15', '1985/12/25', 'Dec 3, 1988', 
            '1992-13-45', None, '1987-02-28', '1995/04/10',
            'invalid', '1980-06-15', '1993-09-20'
        ],
        'income': [
            50000, '$75,000', '60k', -1000, None, 
            '90,000', 'confidential', 55000, '$0', 120000
        ],
        'state': [
            'California', 'ca', 'TX', 'New York', 'FL',
            'Unknown', None, 'Massachusetts', 'nevada', 'OR'
        ]
    })


@pytest.fixture
def time_series_data():
    """Create time series data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': [100 + i + (10 if i % 10 == 0 else 0) for i in range(100)],  # With outliers
        'category': ['A' if i % 2 == 0 else 'B' for i in range(100)]
    })


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def table_cleaner():
    """Create a TableCleaner instance for testing."""
    return TableCleaner(
        confidence_threshold=0.5,
        max_batch_size=10,
        enable_caching=False  # Disable caching for tests
    )


@pytest.fixture
def table_cleaner_with_mock_llm(mock_llm_provider):
    """Create a TableCleaner with mocked LLM provider."""
    cleaner = TableCleaner(confidence_threshold=0.5)
    cleaner.llm_provider = mock_llm_provider
    return cleaner


@pytest.fixture
def sample_rules():
    """Create sample cleaning rules for testing."""
    return [
        CleaningRule(
            name="standardize_states",
            description="Convert state names to 2-letter codes",
            examples=[
                ("California", "CA"),
                ("New York", "NY"),
                ("Texas", "TX"),
                ("Florida", "FL")
            ]
        ),
        CleaningRule(
            name="fix_emails",
            description="Fix malformed email addresses",
            pattern=r"[\w\.-]+@[\w\.-]+\.\w+",
            transform="Add missing @ or .com"
        ),
        CleaningRule(
            name="standardize_names",
            description="Capitalize names properly",
            examples=[
                ("john smith", "John Smith"),
                ("JANE DOE", "Jane Doe"),
                ("bob johnson", "Bob Johnson")
            ]
        )
    ]


@pytest.fixture
def sample_ruleset(sample_rules):
    """Create a RuleSet with sample rules."""
    return RuleSet(sample_rules)


@pytest.fixture
def complex_ruleset():
    """Create a complex RuleSet for advanced testing."""
    rules = [
        CleaningRule(
            name="phone_normalization",
            description="Normalize phone numbers to (XXX) XXX-XXXX format",
            pattern=r"[\d\s\-\(\)]+",
            examples=[
                ("555-123-4567", "(555) 123-4567"),
                ("555.123.4567", "(555) 123-4567"),
                ("5551234567", "(555) 123-4567")
            ]
        ),
        CleaningRule(
            name="currency_normalization",
            description="Normalize currency values",
            examples=[
                ("$50,000", "50000"),
                ("60k", "60000"),
                ("$1.5M", "1500000")
            ]
        ),
        CleaningRule(
            name="date_standardization",
            description="Standardize date formats to YYYY-MM-DD",
            examples=[
                ("12/25/2023", "2023-12-25"),
                ("Dec 25, 2023", "2023-12-25"),
                ("25-Dec-23", "2023-12-25")
            ]
        )
    ]
    return RuleSet(rules)


# =============================================================================
# PERFORMANCE AND LOAD TESTING FIXTURES
# =============================================================================

@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "small_dataset_size": 1000,
        "medium_dataset_size": 10000,
        "large_dataset_size": 100000,
        "max_processing_time": 30.0,  # seconds
        "expected_throughput": 1000,  # records per second
    }


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Record initial memory
    initial_memory = process.memory_info().rss
    
    yield process
    
    # Force garbage collection and check for leaks
    gc.collect()
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Warn if memory increased significantly (>100MB)
    if memory_increase > 100 * 1024 * 1024:
        pytest.warns(UserWarning, f"Possible memory leak: {memory_increase / 1024 / 1024:.1f}MB increase")


# =============================================================================
# SPARK AND DISTRIBUTED TESTING FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for testing."""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("LLMTabCleanerTests") \
            .master("local[2]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("WARN")
        
        yield spark
        
        spark.stop()
    except ImportError:
        pytest.skip("PySpark not available")


@pytest.fixture
def spark_dataframe(spark_session, sample_dataframe):
    """Create a Spark DataFrame from pandas DataFrame."""
    return spark_session.createDataFrame(sample_dataframe)


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_database_url():
    """Provide test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture
def mock_api_responses():
    """Mock responses for various API calls."""
    return {
        "openai": {
            "choices": [
                {
                    "message": {
                        "content": '{"fixed_value": "corrected", "confidence": 0.9}'
                    }
                }
            ]
        },
        "anthropic": {
            "content": [
                {
                    "text": '{"fixed_value": "corrected", "confidence": 0.9}'
                }
            ]
        }
    }


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    
    # Clear any global state
    import gc
    gc.collect()


@pytest.fixture(scope="function")
def isolated_test():
    """Ensure test isolation by resetting global state."""
    # Store original state
    original_globals = {}
    
    yield
    
    # Restore original state if needed
    # This is a placeholder for any global state restoration