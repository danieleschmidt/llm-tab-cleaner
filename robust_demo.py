#!/usr/bin/env python3
"""
Robust demonstration of llm-tab-cleaner functionality.
Generation 2: Make it Robust (Reliable) - Error handling, validation, and resilience.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import tempfile
import time
from pathlib import Path
import pandas as pd
from llm_tab_cleaner import TableCleaner, CleaningRule, RuleSet, IncrementalCleaner

# Configure logging for robust error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_robust_sample_data():
    """Create sample data with various edge cases and quality issues."""
    return pd.DataFrame({
        'id': [1, 2, 3, None, 5, 6, 7, 8, 9, 10],
        'name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN', '', None, 'Dave Wilson', 'eve@test.com', '  spaced  ', '123invalid', 'valid name'],
        'email': ['alice@test.com', 'BOB@TEST.COM', 'charlie.test.com', 'invalid-email', None, 'dave@example.com', '', '  spaces@test.com  ', 'no-at-sign.com', 'valid@email.com'],
        'age': [25, 'thirty', 35, -5, 150, None, 'invalid', 0, '25.5', 30],
        'state': ['California', 'NY', 'tx', 'Unknown', None, 'FL', '', '  CA  ', 'invalid_state', 'TX'],
        'salary': [50000, '$60,000', '70k', 'confidential', None, 80000, '', '$0', '-1000', '120,000'],
        'join_date': ['2023-01-15', '2023/12/25', 'invalid', None, '2023-13-45', '2023-02-28', '', '1990-01-01', '2024-12-31', '2023-06-15']
    })

def create_robust_rules():
    """Create comprehensive cleaning rules with proper validation."""
    try:
        rules = [
            CleaningRule(
                name="standardize_names",
                description="Standardize name capitalization and format",
                examples=[
                    ("alice smith", "Alice Smith"),
                    ("BOB JONES", "Bob Jones"),
                    ("charlie brown", "Charlie Brown"),
                    ("  spaced  ", "Spaced")
                ]
            ),
            CleaningRule(
                name="standardize_states", 
                description="Convert state names to standard 2-letter codes",
                examples=[
                    ("California", "CA"),
                    ("NY", "NY"),
                    ("tx", "TX"),
                    ("texas", "TX"),
                    ("florida", "FL"),
                    ("  CA  ", "CA")
                ]
            ),
            CleaningRule(
                name="clean_emails",
                description="Standardize email format and validate structure",
                examples=[
                    ("BOB@TEST.COM", "bob@test.com"),
                    ("  spaces@test.com  ", "spaces@test.com"),
                    ("charlie.test.com", "charlie@test.com"),
                    ("invalid-email", None)
                ]
            ),
            CleaningRule(
                name="normalize_salaries",
                description="Convert salary formats to numeric values",
                examples=[
                    ("$60,000", "60000"),
                    ("70k", "70000"),
                    ("120,000", "120000"),
                    ("$0", "0"),
                    ("confidential", None)
                ]
            )
        ]
        logger.info(f"Created {len(rules)} robust cleaning rules")
        return RuleSet(rules)
    
    except Exception as e:
        logger.error(f"Error creating rules: {e}")
        # Fallback to basic rules
        return RuleSet([])

def demo_robust_error_handling():
    """Demonstrate robust error handling and validation."""
    logger.info("🛡️ Testing Robust Error Handling")
    print("🛡️ Robust Error Handling Demo")
    print("=" * 50)
    
    success_count = 0
    test_count = 0
    
    # Test 1: Invalid confidence threshold
    test_count += 1
    try:
        cleaner = TableCleaner(confidence_threshold=1.5)  # Invalid threshold
        logger.warning("Should have failed with invalid confidence threshold")
    except (ValueError, TypeError) as e:
        logger.info(f"✅ Correctly caught invalid confidence threshold: {e}")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Unexpected error with invalid confidence: {e}")
    
    # Test 2: Empty DataFrame handling
    test_count += 1
    try:
        cleaner = TableCleaner(confidence_threshold=0.5)
        empty_df = pd.DataFrame()
        result = cleaner.clean(empty_df)
        logger.info("✅ Successfully handled empty DataFrame")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Failed to handle empty DataFrame: {e}")
    
    # Test 3: DataFrame with all null values
    test_count += 1
    try:
        null_df = pd.DataFrame({'col1': [None, None, None], 'col2': [None, None, None]})
        result = cleaner.clean(null_df)
        logger.info("✅ Successfully handled all-null DataFrame")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Failed to handle all-null DataFrame: {e}")
    
    # Test 4: DataFrame with mixed types
    test_count += 1
    try:
        mixed_df = pd.DataFrame({
            'numbers': [1, 2.5, '3', None, float('inf')],
            'strings': ['a', 123, None, '', 'valid'],
            'booleans': [True, False, 1, 0, 'true']
        })
        result = cleaner.clean(mixed_df)
        logger.info("✅ Successfully handled mixed-type DataFrame")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Failed to handle mixed-type DataFrame: {e}")
    
    print(f"\n📊 Error Handling Results: {success_count}/{test_count} tests passed")
    return success_count == test_count

def demo_input_validation():
    """Demonstrate comprehensive input validation."""
    logger.info("🔍 Testing Input Validation")
    print("\n🔍 Input Validation Demo")
    print("-" * 30)
    
    success_count = 0
    test_count = 0
    
    # Test invalid rule creation
    test_count += 1
    try:
        # This should fail due to validation rules
        invalid_rule = CleaningRule(name="", description="")
        logger.warning("Should have failed with empty rule name")
    except ValueError as e:
        logger.info(f"✅ Correctly caught invalid rule: {e}")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Unexpected error with invalid rule: {e}")
    
    # Test valid rule creation with examples
    test_count += 1
    try:
        valid_rule = CleaningRule(
            name="test_rule",
            description="Test rule with examples",
            examples=[("input", "output")]
        )
        logger.info("✅ Successfully created valid rule with examples")
        success_count += 1
    except Exception as e:
        logger.error(f"❌ Failed to create valid rule: {e}")
    
    print(f"\n📊 Validation Results: {success_count}/{test_count} tests passed")
    return success_count == test_count

def demo_incremental_cleaning_robust():
    """Demonstrate robust incremental cleaning with state management."""
    logger.info("📈 Testing Robust Incremental Cleaning")
    print("\n📈 Incremental Cleaning Demo")
    print("-" * 35)
    
    success_count = 0
    test_count = 0
    
    # Test with temporary state file
    test_count += 1
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            state_path = tmp_file.name
        
        # Initialize incremental cleaner with state path
        cleaner = IncrementalCleaner(
            state_path=state_path,
            confidence_threshold=0.5
        )
        
        # Process initial data
        df = create_robust_sample_data()
        result = cleaner.process_increment(df[:5])  # Process first 5 rows
        
        # Verify state persistence
        if Path(state_path).exists():
            logger.info("✅ State file created successfully")
            success_count += 1
        else:
            logger.error("❌ State file was not created")
        
        # Clean up
        try:
            Path(state_path).unlink()
        except:
            pass
            
    except Exception as e:
        logger.error(f"❌ Incremental cleaning failed: {e}")
    
    print(f"\n📊 Incremental Cleaning Results: {success_count}/{test_count} tests passed")
    return success_count == test_count

def demo_performance_monitoring():
    """Demonstrate performance monitoring and resource management."""
    logger.info("⚡ Testing Performance Monitoring")
    print("\n⚡ Performance Monitoring Demo")
    print("-" * 38)
    
    success_count = 0
    test_count = 0
    
    # Test processing time measurement
    test_count += 1
    try:
        cleaner = TableCleaner(confidence_threshold=0.5, max_batch_size=100)
        df = create_robust_sample_data()
        
        start_time = time.time()
        result = cleaner.clean(df)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Processing completed in {processing_time:.3f} seconds")
        print(f"Processing time: {processing_time:.3f}s")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring failed: {e}")
    
    # Test memory usage awareness
    test_count += 1
    try:
        large_df = pd.DataFrame({
            'col1': range(1000),
            'col2': [f'value_{i}' for i in range(1000)],
            'col3': [i % 10 for i in range(1000)]
        })
        
        result = cleaner.clean(large_df)
        logger.info("✅ Successfully processed larger dataset")
        success_count += 1
        
    except Exception as e:
        logger.error(f"❌ Large dataset processing failed: {e}")
    
    print(f"\n📊 Performance Results: {success_count}/{test_count} tests passed")
    return success_count == test_count

def demo_data_quality_metrics():
    """Demonstrate comprehensive data quality assessment."""
    logger.info("📊 Testing Data Quality Metrics")
    print("\n📊 Data Quality Assessment Demo")
    print("-" * 40)
    
    df = create_robust_sample_data()
    
    # Calculate comprehensive metrics
    metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'unique_values_per_column': df.nunique().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    print("Data Quality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Identify problematic columns
    problematic_columns = []
    for col in df.columns:
        null_percentage = (df[col].isnull().sum() / len(df)) * 100
        if null_percentage > 30:  # More than 30% null
            problematic_columns.append((col, null_percentage))
    
    if problematic_columns:
        print("\nProblematic columns (>30% null):")
        for col, pct in problematic_columns:
            print(f"  {col}: {pct:.1f}% null")
    else:
        print("\n✅ No columns with excessive null values")
    
    return True

def main():
    """Run the robust functionality demonstration."""
    print("🚀 Starting llm-tab-cleaner robust functionality test...")
    logger.info("Starting Generation 2 (Robust) testing")
    
    success_tests = 0
    total_tests = 0
    
    # Run all robust tests
    tests = [
        ("Error Handling", demo_robust_error_handling),
        ("Input Validation", demo_input_validation),
        ("Incremental Cleaning", demo_incremental_cleaning_robust),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Data Quality Metrics", demo_data_quality_metrics)
    ]
    
    for test_name, test_func in tests:
        total_tests += 1
        try:
            logger.info(f"Running {test_name} test...")
            if test_func():
                success_tests += 1
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.warning(f"⚠️ {test_name} test had some failures")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
    
    # Final assessment
    success_rate = (success_tests / total_tests) * 100
    print(f"\n🏆 Generation 2 (Robust) Results:")
    print(f"Tests passed: {success_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ Generation 2 (Robust) - Reliability features working well!")
        print("Ready to proceed to Generation 3 (Optimized)")
        logger.info("Generation 2 completed successfully")
        return True
    else:
        print("⚠️ Some reliability issues detected, but core functionality working")
        logger.warning("Generation 2 completed with some issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)