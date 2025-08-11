"""Tests for advanced security features."""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from llm_tab_cleaner.advanced_security import (
    SecurityPolicy,
    SensitivityLevel, 
    PrivacyTechnique,
    DataClassifier,
    TokenizationPreserver,
    DataMaskingPreserver,
    SecureTableCleaner,
    SecurityViolation,
    create_secure_cleaner
)
from llm_tab_cleaner.core import TableCleaner


class TestSecurityPolicy:
    """Test security policy configuration."""
    
    def test_default_policy(self):
        """Test default security policy."""
        policy = SecurityPolicy()
        
        assert policy.max_data_size == 100000
        assert policy.sensitivity_level == SensitivityLevel.INTERNAL
        assert PrivacyTechnique.DATA_MASKING in policy.privacy_techniques
        assert len(policy.blocked_patterns) > 0
    
    def test_custom_policy(self):
        """Test custom security policy."""
        policy = SecurityPolicy(
            max_data_size=50000,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            privacy_techniques=[PrivacyTechnique.TOKENIZATION]
        )
        
        assert policy.max_data_size == 50000
        assert policy.sensitivity_level == SensitivityLevel.RESTRICTED
        assert policy.privacy_techniques == [PrivacyTechnique.TOKENIZATION]


class TestDataClassifier:
    """Test automatic data classification."""
    
    def test_classify_public_data(self):
        """Test classification of public data."""
        classifier = DataClassifier()
        df = pd.DataFrame({
            'product_name': ['Widget A', 'Widget B'],
            'price': [10.99, 15.99]
        })
        
        classifications = classifier.classify_dataframe(df)
        
        assert classifications['product_name'] == SensitivityLevel.PUBLIC
        assert classifications['price'] == SensitivityLevel.PUBLIC
    
    def test_classify_sensitive_data(self):
        """Test classification of sensitive data."""
        classifier = DataClassifier()
        df = pd.DataFrame({
            'employee_email': ['user@company.com', 'admin@company.com'],
            'salary': [50000, 75000],
            'ssn': ['123-45-6789', '987-65-4321']
        })
        
        classifications = classifier.classify_dataframe(df)
        
        assert classifications['employee_email'] == SensitivityLevel.INTERNAL
        assert classifications['salary'] == SensitivityLevel.CONFIDENTIAL
        # SSN should be RESTRICTED (highest sensitivity)
        assert classifications['ssn'] == SensitivityLevel.RESTRICTED


class TestTokenizationPreserver:
    """Test tokenization privacy preservation."""
    
    def test_preserve_and_restore(self):
        """Test tokenization and restoration."""
        preserver = TokenizationPreserver()
        
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'id': [1, 2, 3]
        })
        
        # Preserve
        preserved_df, preservation_info = preserver.preserve(df, ['name'])
        
        assert 'mappings' in preservation_info
        assert 'name' in preservation_info['mappings']
        assert preservation_info['technique'] == 'tokenization'
        
        # Check that names are tokenized
        assert preserved_df['name'].iloc[0] != 'Alice'
        assert preserved_df['name'].iloc[1] != 'Bob'
        # Same original values should have same tokens
        assert preserved_df['name'].iloc[0] == preserved_df['name'].iloc[2]
        
        # ID column should be unchanged
        assert preserved_df['id'].equals(df['id'])
        
        # Restore
        restored_df = preserver.restore(preserved_df, preservation_info)
        
        assert restored_df['name'].equals(df['name'])
        assert restored_df['id'].equals(df['id'])


class TestDataMaskingPreserver:
    """Test data masking privacy preservation."""
    
    def test_mask_email(self):
        """Test email masking."""
        preserver = DataMaskingPreserver()
        
        masked = preserver._mask_email('test@example.com')
        assert masked == 't***@example.com'
        
        masked_short = preserver._mask_email('a@b.com')
        assert masked_short == '*@b.com'
    
    def test_mask_phone(self):
        """Test phone number masking."""
        preserver = DataMaskingPreserver()
        
        masked = preserver._mask_phone('123-456-7890')
        assert masked.startswith('123')
        assert masked.endswith('7890')
        assert '*' in masked
    
    def test_mask_name(self):
        """Test name masking."""
        preserver = DataMaskingPreserver()
        
        masked = preserver._mask_name('Alice')
        assert masked == 'A***e'  # A + 3 stars + e = 5 chars total
        
        masked_short = preserver._mask_name('Al')
        assert masked_short == '**'
    
    def test_preserve_dataframe(self):
        """Test masking entire dataframe."""
        preserver = DataMaskingPreserver()
        
        df = pd.DataFrame({
            'email': ['test@example.com', 'user@domain.org'],
            'name': ['Alice', 'Bob']
        })
        
        preserved_df, preservation_info = preserver.preserve(df, ['email', 'name'])
        
        assert preservation_info['technique'] == 'data_masking'
        assert 'mappings' in preservation_info
        
        # Check that values are masked
        assert preserved_df['email'].iloc[0] != 'test@example.com'
        assert preserved_df['name'].iloc[0] != 'Alice'
        assert '@' in preserved_df['email'].iloc[0]  # Format preserved


class TestSecureTableCleaner:
    """Test secure table cleaner functionality."""
    
    def test_init(self):
        """Test secure cleaner initialization."""
        base_cleaner = Mock()
        policy = SecurityPolicy(max_data_size=1000)
        
        secure_cleaner = SecureTableCleaner(base_cleaner, policy)
        
        assert secure_cleaner.base_cleaner == base_cleaner
        assert secure_cleaner.security_policy == policy
        assert len(secure_cleaner.audit_log) == 0
    
    def test_validate_input_size_violation(self):
        """Test validation with size violation."""
        base_cleaner = Mock()
        policy = SecurityPolicy(max_data_size=5)
        secure_cleaner = SecureTableCleaner(base_cleaner, policy)
        
        large_df = pd.DataFrame({'col': range(10)})
        
        with pytest.raises(SecurityViolation, match="Dataset too large"):
            secure_cleaner._validate_input(large_df)
    
    def test_validate_input_allowed_columns(self):
        """Test validation with allowed columns."""
        base_cleaner = Mock()
        policy = SecurityPolicy(allowed_columns={'col1', 'col2'})
        secure_cleaner = SecureTableCleaner(base_cleaner, policy)
        
        # Valid dataframe
        valid_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        secure_cleaner._validate_input(valid_df)  # Should not raise
        
        # Invalid dataframe
        invalid_df = pd.DataFrame({'col1': [1, 2], 'col3': [3, 4]})
        with pytest.raises(SecurityViolation, match="Unauthorized columns"):
            secure_cleaner._validate_input(invalid_df)
    
    @patch('llm_tab_cleaner.advanced_security.DataClassifier')
    def test_secure_clean_basic(self, mock_classifier_class):
        """Test basic secure cleaning functionality."""
        # Mock the base cleaner
        base_cleaner = Mock()
        mock_report = Mock()
        mock_report.total_fixes = 5
        base_cleaner.clean.return_value = (pd.DataFrame({'col': ['clean1', 'clean2']}), mock_report)
        
        # Mock the classifier
        mock_classifier = Mock()
        mock_classifier.classify_dataframe.return_value = {'col': SensitivityLevel.PUBLIC}
        mock_classifier_class.return_value = mock_classifier
        
        # Create secure cleaner
        policy = SecurityPolicy(privacy_techniques=[])  # No privacy techniques for simplicity
        secure_cleaner = SecureTableCleaner(base_cleaner, policy)
        
        # Test data
        df = pd.DataFrame({'col': ['data1', 'data2']})
        
        # Perform secure cleaning
        cleaned_df, security_report = secure_cleaner.secure_clean(df, user_id="test_user")
        
        # Verify results
        assert len(cleaned_df) == 2
        assert 'request_id' in security_report
        assert security_report['user_id'] == 'test_user'
        assert 'classifications' in security_report
        assert len(secure_cleaner.audit_log) == 1
        
        # Verify base cleaner was called
        base_cleaner.clean.assert_called_once()


class TestCreateSecureCleaner:
    """Test secure cleaner factory function."""
    
    def test_create_default_secure_cleaner(self):
        """Test creating secure cleaner with defaults."""
        base_cleaner = Mock()
        
        secure_cleaner = create_secure_cleaner(base_cleaner)
        
        assert isinstance(secure_cleaner, SecureTableCleaner)
        assert secure_cleaner.base_cleaner == base_cleaner
        assert secure_cleaner.security_policy.sensitivity_level == SensitivityLevel.INTERNAL
        assert PrivacyTechnique.DATA_MASKING in secure_cleaner.security_policy.privacy_techniques
    
    def test_create_custom_secure_cleaner(self):
        """Test creating secure cleaner with custom settings."""
        base_cleaner = Mock()
        
        secure_cleaner = create_secure_cleaner(
            base_cleaner,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            privacy_techniques=[PrivacyTechnique.TOKENIZATION]
        )
        
        assert secure_cleaner.security_policy.sensitivity_level == SensitivityLevel.RESTRICTED
        assert secure_cleaner.security_policy.privacy_techniques == [PrivacyTechnique.TOKENIZATION]
        assert secure_cleaner.security_policy.encryption_required == True  # Should be True for RESTRICTED