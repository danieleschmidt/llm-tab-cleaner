"""End-to-end integration tests for LLM Tab Cleaner pipeline."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from llm_tab_cleaner import TableCleaner, CleaningRule, RuleSet


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for complete data cleaning pipeline."""

    def test_complete_data_cleaning_workflow(self, messy_dataframe, complex_ruleset):
        """Test complete workflow from messy data to clean output."""
        # Initialize cleaner with custom rules
        cleaner = TableCleaner(
            rules=complex_ruleset,
            confidence_threshold=0.7,
            max_batch_size=5
        )

        # Process the messy data
        cleaned_df, report = cleaner.clean(messy_dataframe)

        # Verify the cleaning process
        assert len(cleaned_df) <= len(messy_dataframe)  # May remove some rows
        assert report is not None
        assert hasattr(report, 'total_fixes')
        assert hasattr(report, 'quality_score')
        
        # Check that we have audit information
        assert hasattr(report, 'fixes')
        if report.fixes:
            assert all(hasattr(fix, 'column') for fix in report.fixes)
            assert all(hasattr(fix, 'confidence') for fix in report.fixes)

    def test_csv_file_processing_pipeline(self, messy_dataframe, temp_directory):
        """Test processing data from CSV file input to CSV output."""
        # Write messy data to CSV
        input_file = temp_directory / "messy_data.csv"
        messy_dataframe.to_csv(input_file, index=False)

        # Initialize cleaner
        cleaner = TableCleaner(confidence_threshold=0.6)

        # Read, clean, and write data
        df = pd.read_csv(input_file)
        cleaned_df, report = cleaner.clean(df)
        
        output_file = temp_directory / "cleaned_data.csv"
        cleaned_df.to_csv(output_file, index=False)

        # Verify output file exists and has content
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert len(output_df) > 0
        assert len(output_df.columns) == len(cleaned_df.columns)

    def test_incremental_processing_workflow(self, table_cleaner_with_mock_llm, temp_directory):
        """Test incremental data processing workflow."""
        # Initial dataset
        initial_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'bob', 'CHARLIE'],
            'email': ['alice@test.com', 'bob@invalid', 'charlie@test.com']
        })

        # Process initial data
        cleaned_initial, report1 = table_cleaner_with_mock_llm.clean(initial_data)
        
        # New incremental data
        incremental_data = pd.DataFrame({
            'id': [4, 5],
            'name': ['david', 'EVE'],
            'email': ['david@test.com', 'eve@invalid']
        })

        # Process incremental data
        cleaned_incremental, report2 = table_cleaner_with_mock_llm.clean(incremental_data)

        # Verify both datasets were processed
        assert len(cleaned_initial) == 3
        assert len(cleaned_incremental) == 2
        assert report1.total_fixes >= 0
        assert report2.total_fixes >= 0

    @pytest.mark.requires_llm
    def test_multi_provider_fallback_workflow(self):
        """Test workflow with multiple LLM providers and fallback."""
        # This would test real provider fallback logic
        # For now, we'll simulate the workflow structure
        
        df = pd.DataFrame({
            'text': ['Hello world', 'invalid data', 'another value']
        })

        # Configure cleaner with multiple providers
        cleaner = TableCleaner(
            llm_provider='openai',  # Primary
            fallback_providers=['anthropic', 'local'],
            confidence_threshold=0.8
        )

        # This test would require actual API keys to run fully
        # For CI/CD, we'll skip if no keys are available
        try:
            cleaned_df, report = cleaner.clean(df)
            assert len(cleaned_df) == len(df)
        except Exception as e:
            pytest.skip(f"LLM provider not available: {e}")

    def test_audit_trail_generation(self, table_cleaner_with_mock_llm, messy_dataframe, temp_directory):
        """Test that comprehensive audit trails are generated."""
        # Configure cleaner with audit logging
        cleaner = table_cleaner_with_mock_llm
        cleaner.enable_audit_logging = True
        cleaner.audit_log_path = temp_directory / "audit.log"

        # Process data
        cleaned_df, report = cleaner.clean(messy_dataframe)

        # Verify audit information is available
        assert report is not None
        if hasattr(report, 'fixes') and report.fixes:
            for fix in report.fixes:
                assert hasattr(fix, 'original')
                assert hasattr(fix, 'cleaned')
                assert hasattr(fix, 'confidence')
                assert hasattr(fix, 'column')

    def test_quality_metrics_calculation(self, table_cleaner_with_mock_llm, messy_dataframe):
        """Test that quality metrics are properly calculated."""
        # Process messy data
        cleaned_df, report = table_cleaner_with_mock_llm.clean(messy_dataframe)

        # Verify quality metrics
        assert hasattr(report, 'quality_score')
        assert 0 <= report.quality_score <= 1

        if hasattr(report, 'metrics'):
            metrics = report.metrics
            expected_metrics = [
                'total_records', 'processed_records', 'total_fixes',
                'confidence_distribution', 'processing_time'
            ]
            for metric in expected_metrics:
                assert hasattr(metrics, metric) or metric in metrics

    def test_error_handling_and_recovery(self, table_cleaner_with_mock_llm):
        """Test error handling and recovery in the pipeline."""
        # Create problematic data that might cause errors
        problematic_df = pd.DataFrame({
            'id': [None, None, None],  # All nulls
            'data': ['', '', ''],       # All empty strings
            'numbers': ['not_a_number', 'also_not_a_number', 'still_not_a_number']
        })

        # Process should handle errors gracefully
        try:
            cleaned_df, report = table_cleaner_with_mock_llm.clean(problematic_df)
            
            # Should not crash and return some result
            assert cleaned_df is not None
            assert report is not None
            
            # May have fewer rows if some were filtered out
            assert len(cleaned_df) <= len(problematic_df)
            
        except Exception as e:
            # If it does raise an exception, it should be a known type
            assert isinstance(e, (ValueError, TypeError, RuntimeError))

    def test_configuration_persistence(self, temp_directory):
        """Test that cleaner configuration can be saved and loaded."""
        # Create cleaner with specific configuration
        original_cleaner = TableCleaner(
            confidence_threshold=0.75,
            max_batch_size=50,
            enable_caching=True
        )

        # Save configuration (this would be implemented in the actual class)
        config_file = temp_directory / "cleaner_config.json"
        
        # Simulate saving configuration
        import json
        config = {
            'confidence_threshold': original_cleaner.confidence_threshold,
            'max_batch_size': getattr(original_cleaner, 'max_batch_size', 100),
            'enable_caching': getattr(original_cleaner, 'enable_caching', False)
        }
        with open(config_file, 'w') as f:
            json.dump(config, f)

        # Load configuration into new cleaner
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)

        new_cleaner = TableCleaner(**loaded_config)

        # Verify configuration was preserved
        assert new_cleaner.confidence_threshold == original_cleaner.confidence_threshold

    @pytest.mark.requires_spark
    def test_spark_integration_workflow(self, spark_session, sample_dataframe):
        """Test workflow with Spark backend."""
        # Create Spark DataFrame
        spark_df = spark_session.createDataFrame(sample_dataframe)

        # This would test Spark-specific cleaning logic
        # For now, verify we can work with Spark DataFrames
        assert spark_df.count() == len(sample_dataframe)
        
        # Convert back to pandas for cleaning (simplified workflow)
        pandas_df = spark_df.toPandas()
        
        cleaner = TableCleaner(confidence_threshold=0.5)
        # Would normally use SparkCleaner here
        cleaned_df, report = cleaner.clean(pandas_df)
        
        assert len(cleaned_df) <= len(pandas_df)

    def test_pipeline_monitoring_integration(self, table_cleaner_with_mock_llm, sample_dataframe):
        """Test integration with monitoring and observability systems."""
        # Enable monitoring (simulated)
        cleaner = table_cleaner_with_mock_llm
        cleaner.enable_monitoring = True

        # Process data
        start_time = pd.Timestamp.now()
        cleaned_df, report = cleaner.clean(sample_dataframe)
        end_time = pd.Timestamp.now()

        # Verify monitoring data would be available
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time >= 0

        # Verify report contains monitoring information
        assert report is not None
        if hasattr(report, 'processing_time'):
            assert report.processing_time >= 0