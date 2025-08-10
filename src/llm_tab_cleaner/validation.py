"""Input validation and sanitization."""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        """Initialize input validator."""
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(OR|AND)\b\s+\w+\s*=\s*\w+)",
            r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]
        
        self.file_path_patterns = [
            r"\.\.\/",  # Directory traversal
            r"\/etc\/",  # System files
            r"\/proc\/",
            r"\/dev\/",
            r"\/sys\/",
        ]
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a pandas DataFrame."""
        errors = []
        warnings = []
        
        try:
            # Check if DataFrame is not empty
            if df.empty:
                errors.append("DataFrame is empty")
                return ValidationResult(False, errors, warnings)
            
            # Check for reasonable size
            if len(df) > 10_000_000:  # 10M rows
                warnings.append(f"Very large DataFrame ({len(df)} rows) may cause performance issues")
            
            if len(df.columns) > 1000:
                warnings.append(f"DataFrame has many columns ({len(df.columns)}) which may impact performance")
            
            # Validate column names
            invalid_columns = []
            for col in df.columns:
                if not isinstance(col, str):
                    invalid_columns.append(f"Column name '{col}' is not a string")
                elif len(col) == 0:
                    invalid_columns.append("Empty column name found")
                elif len(col) > 255:
                    invalid_columns.append(f"Column name too long: '{col[:50]}...'")
            
            if invalid_columns:
                errors.extend(invalid_columns)
            
            # Check for suspicious content in string columns
            for col in df.select_dtypes(include=[object]).columns:
                suspicious_values = []
                
                for idx, value in df[col].dropna().head(1000).items():  # Check first 1000 values
                    if isinstance(value, str):
                        if self._contains_malicious_content(value):
                            suspicious_values.append((idx, value[:100]))
                
                if suspicious_values:
                    warnings.append(f"Column '{col}' contains potentially malicious content in {len(suspicious_values)} values")
            
            # Check data types
            for col, dtype in df.dtypes.items():
                if dtype == object:
                    # Check if object column should be numeric
                    sample_values = df[col].dropna().head(100)
                    if len(sample_values) > 0:
                        numeric_count = 0
                        for val in sample_values:
                            if isinstance(val, str):
                                try:
                                    float(val.replace(',', '').replace('$', '').replace('%', ''))
                                    numeric_count += 1
                                except ValueError:
                                    pass
                        
                        if numeric_count / len(sample_values) > 0.8:
                            warnings.append(f"Column '{col}' appears to contain mostly numeric data but is stored as object")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings)
            
        except Exception as e:
            errors.append(f"DataFrame validation failed: {e}")
            return ValidationResult(False, errors, warnings)
    
    def validate_column_name(self, column_name: str) -> ValidationResult:
        """Validate a column name."""
        errors = []
        warnings = []
        
        if not isinstance(column_name, str):
            errors.append(f"Column name must be string, got {type(column_name)}")
            return ValidationResult(False, errors, warnings)
        
        # Length check
        if len(column_name) == 0:
            errors.append("Column name cannot be empty")
        elif len(column_name) > 255:
            errors.append(f"Column name too long ({len(column_name)} chars, max 255)")
        
        # Character validation
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name):
            warnings.append(f"Column name '{column_name}' contains non-standard characters")
        
        # Reserved words (basic SQL)
        reserved_words = {
            'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
            'table', 'index', 'view', 'where', 'order', 'group', 'having'
        }
        
        if column_name.lower() in reserved_words:
            warnings.append(f"Column name '{column_name}' is a reserved SQL keyword")
        
        # Malicious content
        if self._contains_malicious_content(column_name):
            errors.append(f"Column name contains potentially malicious content")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, column_name)
    
    def validate_value(self, value: Any, column_type: str = "unknown") -> ValidationResult:
        """Validate and sanitize a single value."""
        errors = []
        warnings = []
        sanitized_value = value
        
        try:
            # Handle None/NaN
            if pd.isna(value):
                return ValidationResult(True, errors, warnings, value)
            
            # String validation
            if isinstance(value, str):
                # Length check
                if len(value) > 100000:  # 100KB
                    warnings.append(f"Very long string value ({len(value)} chars)")
                
                # Malicious content check
                if self._contains_malicious_content(value):
                    errors.append("Value contains potentially malicious content")
                
                # Sanitize common issues
                sanitized_value = self._sanitize_string(value)
                if sanitized_value != value:
                    warnings.append("String value was sanitized")
            
            # Numeric validation
            elif isinstance(value, (int, float)):
                # Range checks
                if isinstance(value, float):
                    if np.isinf(value):
                        errors.append("Value is infinite")
                    elif np.isnan(value):
                        sanitized_value = None
                        warnings.append("NaN converted to None")
                
                # Reasonable range check based on column type
                if column_type == "age" and (value < 0 or value > 150):
                    warnings.append(f"Age value {value} is outside reasonable range")
                elif column_type == "salary" and value < 0:
                    warnings.append(f"Negative salary value: {value}")
            
            # List/array validation
            elif isinstance(value, (list, tuple)):
                if len(value) > 10000:
                    warnings.append(f"Very large list/array ({len(value)} elements)")
                
                # Validate each element
                for i, element in enumerate(value[:100]):  # Check first 100 elements
                    element_result = self.validate_value(element, column_type)
                    if not element_result.is_valid:
                        errors.extend([f"Element {i}: {error}" for error in element_result.errors])
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings, sanitized_value)
            
        except Exception as e:
            errors.append(f"Value validation failed: {e}")
            return ValidationResult(False, errors, warnings, value)
    
    def validate_cleaning_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate cleaning configuration."""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['llm_provider']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate specific fields
        if 'confidence_threshold' in config:
            threshold = config['confidence_threshold']
            if not isinstance(threshold, (int, float)):
                errors.append("confidence_threshold must be numeric")
            elif not 0.0 <= threshold <= 1.0:
                errors.append("confidence_threshold must be between 0.0 and 1.0")
        
        if 'llm_provider' in config:
            provider = config['llm_provider']
            if not isinstance(provider, str):
                errors.append("llm_provider must be a string")
            elif provider not in ['anthropic', 'openai', 'local']:
                warnings.append(f"Unknown llm_provider: {provider}")
        
        if 'max_fixes_per_column' in config:
            max_fixes = config['max_fixes_per_column']
            if not isinstance(max_fixes, int) or max_fixes < 0:
                errors.append("max_fixes_per_column must be non-negative integer")
        
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append("batch_size must be positive integer")
            elif batch_size > 10000:
                warnings.append(f"Large batch_size ({batch_size}) may cause memory issues")
        
        # Security checks
        for key, value in config.items():
            if isinstance(value, str) and self._contains_malicious_content(value):
                errors.append(f"Config field '{key}' contains potentially malicious content")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, config)
    
    def _contains_malicious_content(self, text: str) -> bool:
        """Check if text contains potentially malicious content."""
        if not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        
        # Check SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check file path traversal
        for pattern in self.file_path_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize a string value."""
        if not isinstance(text, str):
            return text
        
        # Remove null bytes
        sanitized = text.replace('\x00', '')
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Remove or escape potentially dangerous characters
        dangerous_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#x27;'
        }
        
        for char, replacement in dangerous_chars.items():
            if char in sanitized:
                sanitized = sanitized.replace(char, replacement)
        
        return sanitized
    
    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate a file path for security."""
        errors = []
        warnings = []
        
        if not isinstance(file_path, str):
            errors.append("File path must be string")
            return ValidationResult(False, errors, warnings)
        
        # Path traversal check
        if '..' in file_path:
            errors.append("Path traversal detected (..) in file path")
        
        # System directory access
        dangerous_paths = ['/etc/', '/proc/', '/dev/', '/sys/', '/root/']
        for dangerous_path in dangerous_paths:
            if file_path.startswith(dangerous_path):
                errors.append(f"Access to system directory not allowed: {dangerous_path}")
        
        # File extension validation
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.sh', '.ps1']
        for ext in dangerous_extensions:
            if file_path.lower().endswith(ext):
                warnings.append(f"Potentially executable file extension: {ext}")
        
        # Length check
        if len(file_path) > 4096:
            errors.append("File path too long")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, file_path)


class DataSanitizer:
    """Sanitizes data for safe processing."""
    
    def __init__(self):
        """Initialize data sanitizer."""
        self.validator = InputValidator()
    
    def sanitize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Sanitize a DataFrame and return warnings."""
        warnings = []
        sanitized_df = df.copy()
        
        try:
            # Sanitize column names
            column_mapping = {}
            for col in df.columns:
                result = self.validator.validate_column_name(str(col))
                if not result.is_valid:
                    # Create safe column name
                    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
                    safe_name = re.sub(r'^[0-9]', 'col_', safe_name)
                    column_mapping[col] = safe_name
                    warnings.append(f"Renamed column '{col}' to '{safe_name}'")
            
            if column_mapping:
                sanitized_df = sanitized_df.rename(columns=column_mapping)
            
            # Sanitize string values
            for col in sanitized_df.select_dtypes(include=[object]).columns:
                sanitized_values = []
                column_warnings = 0
                
                for value in sanitized_df[col]:
                    if pd.isna(value):
                        sanitized_values.append(value)
                    elif isinstance(value, str):
                        result = self.validator.validate_value(value)
                        sanitized_values.append(result.sanitized_value)
                        if result.warnings:
                            column_warnings += 1
                    else:
                        sanitized_values.append(value)
                
                sanitized_df[col] = sanitized_values
                
                if column_warnings > 0:
                    warnings.append(f"Sanitized {column_warnings} values in column '{col}'")
            
            # Handle infinite values
            numeric_cols = sanitized_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                inf_mask = np.isinf(sanitized_df[col])
                if inf_mask.any():
                    sanitized_df.loc[inf_mask, col] = np.nan
                    warnings.append(f"Converted {inf_mask.sum()} infinite values to NaN in column '{col}'")
            
            return sanitized_df, warnings
            
        except Exception as e:
            logger.error(f"DataFrame sanitization failed: {e}")
            return df, [f"Sanitization failed: {e}"]
    
    def sanitize_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Sanitize configuration dictionary."""
        warnings = []
        sanitized_config = config.copy()
        
        try:
            # Sanitize string values
            for key, value in config.items():
                if isinstance(value, str):
                    result = self.validator.validate_value(value)
                    if result.sanitized_value != value:
                        sanitized_config[key] = result.sanitized_value
                        warnings.append(f"Sanitized config value for '{key}'")
            
            # Ensure numeric bounds
            if 'confidence_threshold' in sanitized_config:
                threshold = sanitized_config['confidence_threshold']
                if isinstance(threshold, (int, float)):
                    sanitized_config['confidence_threshold'] = max(0.0, min(1.0, threshold))
            
            if 'max_fixes_per_column' in sanitized_config:
                max_fixes = sanitized_config['max_fixes_per_column']
                if isinstance(max_fixes, (int, float)):
                    sanitized_config['max_fixes_per_column'] = max(0, int(max_fixes))
            
            return sanitized_config, warnings
            
        except Exception as e:
            logger.error(f"Config sanitization failed: {e}")
            return config, [f"Config sanitization failed: {e}"]


# Global validator instance
_global_validator = None
_global_sanitizer = None


def get_global_validator() -> InputValidator:
    """Get global input validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator


def get_global_sanitizer() -> DataSanitizer:
    """Get global data sanitizer instance."""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = DataSanitizer()
    return _global_sanitizer