"""Security and validation utilities for data cleaning."""

import hashlib
import logging
import re
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Security configuration for data cleaning operations."""
    
    max_data_size: int = Field(default=100_000_000, description="Max data size in bytes")
    max_rows: int = Field(default=1_000_000, description="Max number of rows to process")
    max_columns: int = Field(default=1000, description="Max number of columns to process")
    allow_sensitive_columns: bool = Field(default=False, description="Allow processing of sensitive data")
    sensitive_patterns: List[str] = Field(
        default=["ssn", "social.*security", "credit.*card", "password", "secret", "api.*key", "token"],
        description="Regex patterns for sensitive column names"
    )
    max_processing_time: int = Field(default=3600, description="Max processing time in seconds")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    audit_log_path: Optional[str] = Field(default=None, description="Path for audit logs")
    
    @validator('max_data_size', 'max_rows', 'max_columns')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class DataValidator:
    """Validates data inputs for security and safety."""
    
    def __init__(self, config: SecurityConfig = None):
        """Initialize validator with security configuration."""
        self.config = config or SecurityConfig()
        self.sensitive_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.config.sensitive_patterns]
    
    def validate_dataframe(self, df: pd.DataFrame, operation_name: str = "unknown") -> Dict[str, Any]:
        """Validate a DataFrame for security compliance.
        
        Args:
            df: DataFrame to validate
            operation_name: Name of the operation for logging
            
        Returns:
            Validation results dictionary
            
        Raises:
            SecurityException: If security validation fails
        """
        logger.info(f"Validating DataFrame for operation: {operation_name}")
        
        validation_result = {
            "operation": operation_name,
            "timestamp": datetime.utcnow().isoformat(),
            "data_hash": self._calculate_data_hash(df),
            "passed": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check data size
            data_size = df.memory_usage(deep=True).sum()
            if data_size > self.config.max_data_size:
                raise SecurityException(f"Data size {data_size} exceeds maximum {self.config.max_data_size}")
            
            # Check row count
            if len(df) > self.config.max_rows:
                raise SecurityException(f"Row count {len(df)} exceeds maximum {self.config.max_rows}")
            
            # Check column count
            if len(df.columns) > self.config.max_columns:
                raise SecurityException(f"Column count {len(df.columns)} exceeds maximum {self.config.max_columns}")
            
            # Check for sensitive columns
            sensitive_columns = self._detect_sensitive_columns(df.columns)
            if sensitive_columns and not self.config.allow_sensitive_columns:
                raise SecurityException(f"Sensitive columns detected: {sensitive_columns}")
            elif sensitive_columns:
                validation_result["warnings"].append(f"Sensitive columns detected: {sensitive_columns}")
            
            # Check for potentially malicious content
            malicious_patterns = self._detect_malicious_patterns(df)
            if malicious_patterns:
                validation_result["warnings"].extend(malicious_patterns)
            
            # Validate column names
            invalid_columns = self._validate_column_names(df.columns)
            if invalid_columns:
                validation_result["warnings"].append(f"Invalid column names: {invalid_columns}")
            
            validation_result.update({
                "data_size_bytes": data_size,
                "row_count": len(df),
                "column_count": len(df.columns),
                "sensitive_columns": sensitive_columns,
                "data_types": df.dtypes.to_dict()
            })
            
        except SecurityException as e:
            validation_result["passed"] = False
            validation_result["errors"].append(str(e))
            logger.error(f"Security validation failed: {e}")
            raise
        except Exception as e:
            validation_result["passed"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Validation error: {e}")
            raise SecurityException(f"Validation failed: {str(e)}")
        
        logger.info(f"Validation completed. Passed: {validation_result['passed']}, Warnings: {len(validation_result['warnings'])}")
        return validation_result
    
    def sanitize_column_names(self, columns: List[str]) -> List[str]:
        """Sanitize column names to remove potentially harmful characters."""
        sanitized = []
        
        for col in columns:
            # Remove/replace harmful characters
            sanitized_col = re.sub(r'[^\w\-_.]', '_', str(col))
            
            # Ensure it doesn't start with number or special char
            if sanitized_col and not sanitized_col[0].isalpha():
                sanitized_col = 'col_' + sanitized_col
            
            # Limit length
            if len(sanitized_col) > 50:
                sanitized_col = sanitized_col[:47] + '...'
            
            sanitized.append(sanitized_col or f"col_{len(sanitized)}")
        
        return sanitized
    
    def _detect_sensitive_columns(self, columns: List[str]) -> List[str]:
        """Detect columns that may contain sensitive information."""
        sensitive_columns = []
        
        for col in columns:
            col_str = str(col).lower()
            for pattern in self.sensitive_patterns:
                if pattern.search(col_str):
                    sensitive_columns.append(col)
                    break
        
        return sensitive_columns
    
    def _detect_malicious_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect potentially malicious patterns in data."""
        warnings = []
        
        # Check for SQL injection patterns (sample first 100 rows)
        sample_df = df.head(100)
        sql_patterns = [
            r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(?i)(exec\s*\(|script\s*>|javascript:|on\w+\s*=)"
        ]
        
        for pattern in sql_patterns:
            regex = re.compile(pattern)
            for col in sample_df.select_dtypes(include=['object']).columns:
                if sample_df[col].astype(str).str.contains(regex, na=False).any():
                    warnings.append(f"Potential malicious pattern in column '{col}': {pattern}")
        
        return warnings
    
    def _validate_column_names(self, columns: List[str]) -> List[str]:
        """Validate column names for safety."""
        invalid = []
        
        for col in columns:
            col_str = str(col)
            
            # Check for executable patterns
            if any(pattern in col_str.lower() for pattern in ['exec', 'eval', 'system', '__']):
                invalid.append(col)
            
            # Check for extremely long names
            if len(col_str) > 100:
                invalid.append(col)
        
        return invalid
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash of the DataFrame for audit purposes."""
        # Use a sample for large datasets
        if len(df) > 1000:
            sample_df = df.sample(n=1000, random_state=42)
        else:
            sample_df = df
        
        # Create a string representation
        data_str = f"{sample_df.shape}|{list(sample_df.columns)}|{sample_df.dtypes.to_dict()}"
        
        # Add sample of actual data
        for col in sample_df.columns[:10]:  # First 10 columns
            sample_values = sample_df[col].dropna().head(5).astype(str).tolist()
            data_str += f"|{col}:{','.join(sample_values)}"
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class SecurityException(Exception):
    """Exception raised for security-related validation failures."""
    pass


class AuditLogger:
    """Handles audit logging for compliance and security monitoring."""
    
    def __init__(self, log_path: Optional[str] = None, enable_logging: bool = True):
        """Initialize audit logger.
        
        Args:
            log_path: Path to audit log file
            enable_logging: Whether to enable audit logging
        """
        self.enable_logging = enable_logging
        self.log_path = log_path
        self.session_id = secrets.token_hex(8)
        
        if self.enable_logging:
            # Setup audit logger
            self.audit_logger = logging.getLogger("llm_tab_cleaner.audit")
            self.audit_logger.setLevel(logging.INFO)
            
            # Create file handler if path provided
            if log_path:
                handler = logging.FileHandler(log_path)
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
                self.audit_logger.addHandler(handler)
    
    def log_operation_start(self, operation: str, metadata: Dict[str, Any] = None):
        """Log the start of an operation."""
        if not self.enable_logging:
            return
        
        log_data = {
            "event": "operation_start",
            "operation": operation,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.audit_logger.info(f"AUDIT: {log_data}")
    
    def log_operation_end(self, operation: str, success: bool, metadata: Dict[str, Any] = None):
        """Log the end of an operation."""
        if not self.enable_logging:
            return
        
        log_data = {
            "event": "operation_end",
            "operation": operation,
            "session_id": self.session_id,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.audit_logger.info(f"AUDIT: {log_data}")
    
    def log_security_event(self, event_type: str, details: str, severity: str = "INFO"):
        """Log a security-related event."""
        if not self.enable_logging:
            return
        
        log_data = {
            "event": "security_event",
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if severity == "ERROR":
            self.audit_logger.error(f"SECURITY: {log_data}")
        elif severity == "WARNING":
            self.audit_logger.warning(f"SECURITY: {log_data}")
        else:
            self.audit_logger.info(f"SECURITY: {log_data}")
    
    def log_data_access(self, data_hash: str, operation: str, user_info: Dict[str, Any] = None):
        """Log data access for compliance."""
        if not self.enable_logging:
            return
        
        log_data = {
            "event": "data_access",
            "data_hash": data_hash,
            "operation": operation,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_info": user_info or {}
        }
        
        self.audit_logger.info(f"DATA_ACCESS: {log_data}")


class RateLimiter:
    """Rate limiting for API calls and operations."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
    
    def check_rate_limit(self, identifier: str = "default") -> bool:
        """Check if request is within rate limit.
        
        Args:
            identifier: Identifier for the requester
            
        Returns:
            True if within rate limit, False otherwise
        """
        current_time = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        # Check if within limit
        if len(self.requests) >= self.max_requests:
            return False
        
        # Add current request
        self.requests.append(current_time)
        return True
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        current_time = time.time()
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        return max(0, self.max_requests - len(self.requests))


class SecurityManager:
    """Central security manager for coordinating all security measures."""
    
    def __init__(self, config: SecurityConfig = None):
        """Initialize security manager."""
        self.config = config or SecurityConfig()
        self.validator = DataValidator(self.config)
        self.audit_logger = AuditLogger(
            self.config.audit_log_path, 
            self.config.enable_audit_logging
        )
        self.rate_limiter = RateLimiter()
        self.active_operations: Dict[str, datetime] = {}
    
    def validate_and_prepare_data(self, df: pd.DataFrame, operation_name: str) -> Dict[str, Any]:
        """Validate data and prepare for processing."""
        # Check rate limit
        if not self.rate_limiter.check_rate_limit():
            raise SecurityException("Rate limit exceeded")
        
        # Validate data
        validation_result = self.validator.validate_dataframe(df, operation_name)
        
        # Log data access
        self.audit_logger.log_data_access(
            validation_result["data_hash"],
            operation_name
        )
        
        # Track operation
        operation_id = secrets.token_hex(8)
        self.active_operations[operation_id] = datetime.utcnow()
        
        # Log operation start
        self.audit_logger.log_operation_start(operation_name, {
            "operation_id": operation_id,
            "data_hash": validation_result["data_hash"],
            "row_count": len(df),
            "column_count": len(df.columns)
        })
        
        validation_result["operation_id"] = operation_id
        return validation_result
    
    def finalize_operation(self, operation_id: str, success: bool, metadata: Dict[str, Any] = None):
        """Finalize an operation and log results."""
        if operation_id in self.active_operations:
            start_time = self.active_operations.pop(operation_id)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            final_metadata = metadata or {}
            final_metadata["duration_seconds"] = duration
            
            self.audit_logger.log_operation_end(
                "data_cleaning",
                success,
                final_metadata
            )
        
        # Clean up old operations (older than 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.active_operations = {
            op_id: start_time for op_id, start_time in self.active_operations.items()
            if start_time > cutoff_time
        }
    
    def check_operation_timeout(self, operation_id: str) -> bool:
        """Check if operation has timed out."""
        if operation_id not in self.active_operations:
            return False
        
        start_time = self.active_operations[operation_id]
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        
        if elapsed > self.config.max_processing_time:
            self.audit_logger.log_security_event(
                "operation_timeout",
                f"Operation {operation_id} exceeded max processing time",
                "WARNING"
            )
            return True
        
        return False


def create_secure_cleaner(config: SecurityConfig = None, **cleaner_kwargs):
    """Factory function to create a TableCleaner with security measures."""
    from .core import TableCleaner
    
    security_manager = SecurityManager(config)
    
    # Create cleaner with security wrapper
    base_cleaner = TableCleaner(**cleaner_kwargs)
    
    # Monkey patch the clean method to add security
    original_clean = base_cleaner.clean
    
    def secure_clean(df, **kwargs):
        validation_result = security_manager.validate_and_prepare_data(df, "data_cleaning")
        operation_id = validation_result["operation_id"]
        
        try:
            # Check for timeout periodically during cleaning
            if security_manager.check_operation_timeout(operation_id):
                raise SecurityException("Operation timed out")
            
            # Perform cleaning
            cleaned_df, report = original_clean(df, **kwargs)
            
            # Finalize operation
            security_manager.finalize_operation(operation_id, True, {
                "fixes_applied": report.total_fixes,
                "quality_score": report.quality_score
            })
            
            return cleaned_df, report
            
        except Exception as e:
            security_manager.finalize_operation(operation_id, False, {
                "error": str(e)
            })
            raise
    
    base_cleaner.clean = secure_clean
    base_cleaner.security_manager = security_manager
    
    return base_cleaner