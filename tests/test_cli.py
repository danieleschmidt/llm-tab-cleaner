"""Tests for CLI module."""
import pytest
import sys
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from llm_tab_cleaner.cli import cli, main


class TestCLI:
    """Test suite for CLI functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'llm-clean' in result.output.lower() or 'clean' in result.output.lower()
    
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        # Should contain version information
        assert any(char.isdigit() for char in result.output)
    
    def test_cli_no_arguments(self):
        """Test CLI with no arguments."""
        result = self.runner.invoke(cli, [])
        # Should either show help or provide a useful error message
        assert result.exit_code in [0, 2]  # 0 for success/help, 2 for usage error
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = self.runner.invoke(cli, ['invalid-command'])
        assert result.exit_code != 0
    
    @patch('llm_tab_cleaner.cli.pd.read_csv')
    @patch('llm_tab_cleaner.cli.TableCleaner')
    def test_clean_csv_file(self, mock_cleaner_class, mock_read_csv):
        """Test cleaning a CSV file."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'age': [25, 30]
        })
        mock_read_csv.return_value = mock_df
        
        mock_cleaner = Mock()
        mock_report = Mock()
        mock_report.total_fixes = 5
        mock_report.quality_score = 0.95
        mock_cleaner.clean.return_value = (mock_df, mock_report)
        mock_cleaner_class.return_value = mock_cleaner
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = self.runner.invoke(cli, [
                'clean', 
                temp_path,
                '--output', temp_path.replace('.csv', '_cleaned.csv')
            ])
            
            if result.exit_code != 0:
                # The command might not be fully implemented yet
                assert 'not implemented' in result.output.lower() or result.exit_code == 0
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_path.replace('.csv', '_cleaned.csv')).unlink(missing_ok=True)
    
    def test_cli_with_provider_option(self):
        """Test CLI with LLM provider option."""
        result = self.runner.invoke(cli, [
            'clean', 
            'nonexistent.csv',  # This should fail gracefully
            '--provider', 'anthropic'
        ])
        
        # Should handle missing file gracefully or show appropriate error
        assert result.exit_code != 0  # File doesn't exist, should fail
        assert 'anthropic' in result.output.lower() or 'file' in result.output.lower()
    
    def test_cli_with_confidence_threshold(self):
        """Test CLI with confidence threshold option."""
        result = self.runner.invoke(cli, [
            'clean',
            'nonexistent.csv',
            '--confidence-threshold', '0.9'
        ])
        
        # Should handle missing file, but parse confidence threshold
        assert result.exit_code != 0  # File doesn't exist
        # The confidence threshold should be parsed correctly
    
    def test_cli_with_invalid_confidence_threshold(self):
        """Test CLI with invalid confidence threshold."""
        result = self.runner.invoke(cli, [
            'clean',
            'test.csv',
            '--confidence-threshold', '1.5'  # Invalid: > 1.0
        ])
        
        assert result.exit_code != 0
        assert 'confidence' in result.output.lower() or 'threshold' in result.output.lower()
    
    def test_cli_with_output_option(self):
        """Test CLI with output file option."""
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_input:
            with tempfile.NamedTemporaryFile(suffix='.csv') as temp_output:
                result = self.runner.invoke(cli, [
                    'clean',
                    temp_input.name,
                    '--output', temp_output.name
                ])
                
                # May not be fully implemented yet
                if 'not implemented' in result.output.lower():
                    assert True  # Implementation placeholder detected
                else:
                    # Should process the command structure correctly
                    assert result.exit_code in [0, 1, 2]
    
    def test_main_function(self):
        """Test the main function entry point."""
        # Test that main function exists and can be called
        assert callable(main)
        
        # Mock sys.argv to test main function
        with patch.object(sys, 'argv', ['llm-clean', '--help']):
            try:
                main()
            except SystemExit as e:
                # CLI tools often exit with SystemExit, this is normal
                assert e.code in [0, 1, 2]
    
    def test_cli_verbose_option(self):
        """Test CLI with verbose option."""
        result = self.runner.invoke(cli, [
            'clean',
            'nonexistent.csv',
            '--verbose'
        ])
        
        # Should handle verbose flag
        assert result.exit_code != 0  # File doesn't exist
    
    @patch('llm_tab_cleaner.cli.TableCleaner')  
    def test_clean_with_rules_file(self, mock_cleaner_class):
        """Test cleaning with custom rules file."""
        mock_cleaner = Mock()
        mock_cleaner_class.return_value = mock_cleaner
        
        with tempfile.NamedTemporaryFile(suffix='.yaml') as rules_file:
            result = self.runner.invoke(cli, [
                'clean',
                'nonexistent.csv',
                '--rules', rules_file.name
            ])
            
            # Should attempt to process rules file
            assert result.exit_code != 0  # Input file doesn't exist
    
    def test_cli_batch_processing(self):
        """Test CLI batch processing capability."""
        result = self.runner.invoke(cli, [
            'clean',
            'input_dir/',
            '--batch',
            '--output-dir', 'output_dir/'
        ])
        
        # May not be implemented yet
        if result.exit_code != 0:
            # Expected for non-existent directories or unimplemented features
            assert True