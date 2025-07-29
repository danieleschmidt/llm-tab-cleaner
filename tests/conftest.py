"""Shared test fixtures and configuration."""

import pandas as pd
import pytest
from llm_tab_cleaner import TableCleaner, CleaningRule, RuleSet


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN'],
        'email': ['alice@test.com', 'BOB@TEST.COM', 'charlie.test.com'],
        'age': [25, 'thirty', 35],
        'state': ['California', 'NY', 'tx']
    })


@pytest.fixture
def table_cleaner():
    """Create a TableCleaner instance for testing."""
    return TableCleaner()


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
                ("Texas", "TX")
            ]
        ),
        CleaningRule(
            name="fix_emails",
            description="Fix malformed email addresses",
            pattern=r"[\w\.-]+@[\w\.-]+\.\w+",
            transform="Add missing @ or .com"
        )
    ]


@pytest.fixture
def sample_ruleset(sample_rules):
    """Create a RuleSet with sample rules."""
    return RuleSet(sample_rules)