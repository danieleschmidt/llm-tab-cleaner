"""Advanced security features for LLM table cleaning with privacy preservation."""

import hashlib
import hmac
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import base64
from pathlib import Path

import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PrivacyTechnique(Enum):
    """Privacy preservation techniques."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    DATA_MASKING = "data_masking"
    TOKENIZATION = "tokenization"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_data_size: int = 100000  # Maximum rows for processing
    allowed_columns: Optional[Set[str]] = None
    blocked_patterns: List[str] = None
    encryption_required: bool = False
    audit_required: bool = True
    retention_days: int = 90
    sensitivity_level: SensitivityLevel = SensitivityLevel.INTERNAL
    privacy_techniques: List[PrivacyTechnique] = None
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email (for demo)
            ]
        if self.privacy_techniques is None:
            self.privacy_techniques = [PrivacyTechnique.DATA_MASKING]


class SecurityViolation(Exception):
    """Exception raised for security policy violations."""
    pass


class DataClassifier:
    """Automatic data classification and sensitivity detection."""
    
    def __init__(self):
        self.sensitivity_patterns = {
            SensitivityLevel.RESTRICTED: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
                r'\bpassword\b', r'\bssn\b', r'\btax_id\b'
            ],
            SensitivityLevel.CONFIDENTIAL: [
                r'\bsalary\b', r'\bincome\b', r'\bmedical\b',
                r'\bhealth\b', r'\bphone\b', r'\baddress\b'
            ],
            SensitivityLevel.INTERNAL: [
                r'\bemail\b', r'\bemployee_id\b', r'\bdepartment\b'
            ]
        }
    
    def classify_dataframe(self, df: pd.DataFrame) -> Dict[str, SensitivityLevel]:
        """Classify each column by sensitivity level."""
        classifications = {}
        
        # Sensitivity level ordering for comparison
        sensitivity_order = {
            SensitivityLevel.PUBLIC: 0,
            SensitivityLevel.INTERNAL: 1, 
            SensitivityLevel.CONFIDENTIAL: 2,
            SensitivityLevel.RESTRICTED: 3
        }
        
        for column in df.columns:
            column_lower = column.lower()
            sample_values = df[column].dropna().astype(str).head(100).str.lower()
            
            # Check column name patterns
            max_sensitivity = SensitivityLevel.PUBLIC
            
            for sensitivity, patterns in self.sensitivity_patterns.items():
                for pattern in patterns:
                    # Check column name
                    pattern_clean = pattern.replace(r'\b', '').replace('\\', '')
                    if pattern_clean in column_lower:
                        if sensitivity_order[sensitivity] > sensitivity_order[max_sensitivity]:
                            max_sensitivity = sensitivity
                    
                    # Check sample values (limited to avoid performance issues)
                    import re
                    if any(re.search(pattern.replace(r'\b', ''), str(val)) for val in sample_values.head(10)):
                        if sensitivity_order[sensitivity] > sensitivity_order[max_sensitivity]:
                            max_sensitivity = sensitivity
            
            classifications[column] = max_sensitivity
        
        return classifications


class PrivacyPreserver(ABC):
    """Abstract base for privacy preservation techniques."""
    
    @abstractmethod
    def preserve(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply privacy preservation to specified columns."""
        pass
    
    @abstractmethod
    def restore(self, df: pd.DataFrame, preservation_info: Dict[str, Any]) -> pd.DataFrame:
        """Restore data from privacy-preserved format."""
        pass


class DifferentialPrivacyPreserver(PrivacyPreserver):
    """Differential privacy implementation for numerical data."""
    
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
    
    def preserve(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Add Laplace noise for differential privacy."""
        preserved_df = df.copy()
        preservation_info = {"technique": "differential_privacy", "columns": columns}
        
        for column in columns:
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                # Add Laplace noise
                scale = self.sensitivity / self.epsilon
                noise = np.random.laplace(0, scale, size=len(df))
                preserved_df[column] = df[column] + noise
                
                preservation_info[f"{column}_noise_scale"] = scale
        
        return preserved_df, preservation_info
    
    def restore(self, df: pd.DataFrame, preservation_info: Dict[str, Any]) -> pd.DataFrame:
        """Cannot restore from differential privacy (by design)."""
        logger.warning("Cannot restore data from differential privacy - information is intentionally lost")
        return df


class TokenizationPreserver(PrivacyPreserver):
    """Tokenization-based privacy preservation."""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self.token_mapping = {}
    
    def preserve(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Tokenize sensitive values."""
        preserved_df = df.copy()
        preservation_info = {"technique": "tokenization", "columns": columns, "mappings": {}}
        
        for column in columns:
            if column in df.columns:
                column_mapping = {}
                unique_values = df[column].dropna().unique()
                
                for value in unique_values:
                    token = self._generate_token(str(value))
                    column_mapping[str(value)] = token
                
                # Apply tokenization
                preserved_df[column] = df[column].map(
                    lambda x: column_mapping.get(str(x), x) if pd.notna(x) else x
                )
                
                preservation_info["mappings"][column] = column_mapping
        
        return preserved_df, preservation_info
    
    def restore(self, df: pd.DataFrame, preservation_info: Dict[str, Any]) -> pd.DataFrame:
        """Restore from tokenized format."""
        restored_df = df.copy()
        
        for column, mapping in preservation_info["mappings"].items():
            if column in df.columns:
                # Create reverse mapping
                reverse_mapping = {token: value for value, token in mapping.items()}
                restored_df[column] = df[column].map(
                    lambda x: reverse_mapping.get(x, x) if pd.notna(x) else x
                )
        
        return restored_df
    
    def _generate_token(self, value: str) -> str:
        """Generate a secure token for a value."""
        # Use a hash for consistent tokenization
        hash_digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        return f"TOKEN_{hash_digest}"


class DataMaskingPreserver(PrivacyPreserver):
    """Data masking with format preservation."""
    
    def __init__(self):
        self.masking_rules = {
            'email': lambda x: self._mask_email(x),
            'phone': lambda x: self._mask_phone(x),
            'ssn': lambda x: self._mask_ssn(x),
            'name': lambda x: self._mask_name(x),
            'default': lambda x: self._mask_generic(x)
        }
    
    def preserve(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply data masking while preserving format."""
        preserved_df = df.copy()
        preservation_info = {"technique": "data_masking", "columns": columns, "mappings": {}}
        
        for column in columns:
            if column in df.columns:
                column_type = self._detect_column_type(df[column])
                masking_func = self.masking_rules.get(column_type, self.masking_rules['default'])
                
                column_mapping = {}
                for idx, value in df[column].items():
                    if pd.notna(value):
                        masked_value = masking_func(str(value))
                        column_mapping[str(value)] = masked_value
                        preserved_df.at[idx, column] = masked_value
                
                preservation_info["mappings"][column] = column_mapping
        
        return preserved_df, preservation_info
    
    def restore(self, df: pd.DataFrame, preservation_info: Dict[str, Any]) -> pd.DataFrame:
        """Cannot restore from masked data (by design)."""
        logger.warning("Cannot restore data from masking - original values are replaced")
        return df
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect the type of data in a column."""
        sample_values = series.dropna().astype(str).head(10)
        
        if any('@' in val for val in sample_values):
            return 'email'
        elif any(any(c.isdigit() for c in val) and len(val) >= 7 for val in sample_values):
            return 'phone'
        elif any(len(val.replace('-', '').replace(' ', '')) == 9 and val.replace('-', '').replace(' ', '').isdigit() for val in sample_values):
            return 'ssn'
        elif any(any(c.isalpha() for c in val) for val in sample_values):
            return 'name'
        else:
            return 'default'
    
    def _mask_email(self, email: str) -> str:
        """Mask email while preserving format."""
        if '@' in email:
            local, domain = email.split('@', 1)
            masked_local = local[0] + '*' * (len(local) - 1) if len(local) > 1 else '*'
            return f"{masked_local}@{domain}"
        return email
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number while preserving format."""
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) >= 7:
            # Keep first 3 and last 4 digits, mask middle
            masked_digits = digits[:3] + '*' * (len(digits) - 7) + digits[-4:]
            # Preserve original format
            result = phone
            digit_idx = 0
            for i, char in enumerate(phone):
                if char.isdigit() and digit_idx < len(masked_digits):
                    result = result[:i] + masked_digits[digit_idx] + result[i+1:]
                    digit_idx += 1
            return result
        return phone
    
    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN while preserving format."""
        digits = ''.join(c for c in ssn if c.isdigit())
        if len(digits) == 9:
            masked = 'XXX-XX-' + digits[-4:]
            return masked
        return ssn
    
    def _mask_name(self, name: str) -> str:
        """Mask name while preserving length."""
        if len(name) <= 2:
            return '*' * len(name)
        return name[0] + '*' * (len(name) - 2) + name[-1]
    
    def _mask_generic(self, value: str) -> str:
        """Generic masking for unknown data types."""
        if len(value) <= 3:
            return '*' * len(value)
        return value[:1] + '*' * (len(value) - 2) + value[-1:]


class SecureTableCleaner:
    """Security-enhanced table cleaner with privacy preservation."""
    
    def __init__(
        self, 
        base_cleaner,
        security_policy: Optional[SecurityPolicy] = None,
        enable_classification: bool = True
    ):
        self.base_cleaner = base_cleaner
        self.security_policy = security_policy or SecurityPolicy()
        self.enable_classification = enable_classification
        self.classifier = DataClassifier() if enable_classification else None
        self.audit_log = []
        
        # Initialize privacy preservers
        self.privacy_preservers = {
            PrivacyTechnique.DIFFERENTIAL_PRIVACY: DifferentialPrivacyPreserver(),
            PrivacyTechnique.TOKENIZATION: TokenizationPreserver(),
            PrivacyTechnique.DATA_MASKING: DataMaskingPreserver()
        }
    
    def secure_clean(
        self, 
        df: pd.DataFrame,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform secure cleaning with privacy preservation."""
        start_time = time.time()
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = secrets.token_hex(8)
        
        try:
            # Security validation
            self._validate_input(df)
            
            # Classify data sensitivity
            classifications = self.classifier.classify_dataframe(df) if self.classifier else {}
            
            # Apply privacy preservation for sensitive columns
            preserved_df, preservation_info = self._apply_privacy_preservation(df, classifications)
            
            # Perform cleaning on preserved data
            cleaned_df, cleaning_report = self.base_cleaner.clean(preserved_df)
            
            # Create security report
            security_report = {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": start_time,
                "processing_time": time.time() - start_time,
                "classifications": {k: v.value for k, v in classifications.items()},
                "preservation_info": preservation_info,
                "security_policy": {
                    "sensitivity_level": self.security_policy.sensitivity_level.value,
                    "privacy_techniques": [t.value for t in self.security_policy.privacy_techniques]
                },
                "data_stats": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "sensitive_columns": sum(1 for c in classifications.values() 
                                           if c in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.RESTRICTED])
                }
            }
            
            # Audit logging
            self._log_audit_event(security_report, cleaning_report)
            
            return cleaned_df, security_report
            
        except Exception as e:
            # Log security incident
            incident = {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": start_time,
                "error": str(e),
                "data_shape": df.shape if df is not None else None
            }
            self._log_security_incident(incident)
            raise
    
    def _validate_input(self, df: pd.DataFrame):
        """Validate input against security policy."""
        if len(df) > self.security_policy.max_data_size:
            raise SecurityViolation(f"Dataset too large: {len(df)} > {self.security_policy.max_data_size}")
        
        # Check for blocked patterns
        for pattern in self.security_policy.blocked_patterns:
            for column in df.columns:
                sample_values = df[column].dropna().astype(str).head(100)
                import re
                if any(re.search(pattern, str(val)) for val in sample_values):
                    logger.warning(f"Blocked pattern detected in column {column}: {pattern}")
        
        # Check allowed columns
        if self.security_policy.allowed_columns:
            unauthorized_columns = set(df.columns) - self.security_policy.allowed_columns
            if unauthorized_columns:
                raise SecurityViolation(f"Unauthorized columns: {unauthorized_columns}")
    
    def _apply_privacy_preservation(
        self, 
        df: pd.DataFrame, 
        classifications: Dict[str, SensitivityLevel]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply privacy preservation techniques based on sensitivity."""
        preserved_df = df.copy()
        preservation_info = {"techniques_applied": []}
        
        # Group columns by sensitivity level
        sensitive_columns = {
            SensitivityLevel.RESTRICTED: [],
            SensitivityLevel.CONFIDENTIAL: []
        }
        
        for column, sensitivity in classifications.items():
            if sensitivity in sensitive_columns:
                sensitive_columns[sensitivity].append(column)
        
        # Apply privacy techniques
        for technique in self.security_policy.privacy_techniques:
            if technique in self.privacy_preservers:
                preserver = self.privacy_preservers[technique]
                
                # Determine which columns to preserve
                columns_to_preserve = []
                if technique == PrivacyTechnique.DIFFERENTIAL_PRIVACY:
                    # Apply to all numerical sensitive columns
                    columns_to_preserve = [
                        col for col in sensitive_columns[SensitivityLevel.RESTRICTED] + 
                                          sensitive_columns[SensitivityLevel.CONFIDENTIAL]
                        if col in df.columns and df[col].dtype in ['int64', 'float64']
                    ]
                else:
                    # Apply to all sensitive columns
                    columns_to_preserve = [
                        col for col in sensitive_columns[SensitivityLevel.RESTRICTED]
                        if col in df.columns
                    ]
                
                if columns_to_preserve:
                    preserved_df, tech_info = preserver.preserve(preserved_df, columns_to_preserve)
                    preservation_info["techniques_applied"].append(tech_info)
        
        return preserved_df, preservation_info
    
    def _log_audit_event(self, security_report: Dict[str, Any], cleaning_report):
        """Log audit event for compliance."""
        audit_event = {
            "timestamp": time.time(),
            "event_type": "secure_cleaning",
            "request_id": security_report["request_id"],
            "user_id": security_report.get("user_id"),
            "data_classification": security_report["classifications"],
            "privacy_techniques": security_report["security_policy"]["privacy_techniques"],
            "fixes_applied": cleaning_report.total_fixes,
            "processing_time": security_report["processing_time"]
        }
        
        self.audit_log.append(audit_event)
        
        # In production, this would be sent to a secure audit system
        logger.info(f"Audit event logged: {audit_event['request_id']}")
    
    def _log_security_incident(self, incident: Dict[str, Any]):
        """Log security incident."""
        incident_record = {
            "timestamp": time.time(),
            "event_type": "security_incident",
            **incident
        }
        
        self.audit_log.append(incident_record)
        logger.error(f"Security incident logged: {incident_record}")
    
    def export_audit_log(self, filepath: str):
        """Export audit log for compliance reporting."""
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        
        logger.info(f"Audit log exported to {filepath}")


# Factory function for easy creation
def create_secure_cleaner(
    base_cleaner,
    sensitivity_level: SensitivityLevel = SensitivityLevel.INTERNAL,
    privacy_techniques: List[PrivacyTechnique] = None
) -> SecureTableCleaner:
    """Create a secure table cleaner with specified security settings."""
    if privacy_techniques is None:
        privacy_techniques = [PrivacyTechnique.DATA_MASKING]
    
    policy = SecurityPolicy(
        sensitivity_level=sensitivity_level,
        privacy_techniques=privacy_techniques,
        audit_required=True,
        encryption_required=sensitivity_level in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.RESTRICTED]
    )
    
    return SecureTableCleaner(base_cleaner, policy)