"""Compliance and regulatory support for data cleaning operations."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

import pandas as pd

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"  # PII
    SENSITIVE_PERSONAL = "sensitive_personal"  # Special category data


class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"  # GDPR
    US = "us"  # CCPA, HIPAA, etc.
    UK = "uk"  # UK GDPR
    CANADA = "canada"  # PIPEDA
    SINGAPORE = "singapore"  # PDPA
    AUSTRALIA = "australia"  # Privacy Act
    GLOBAL = "global"  # Global baseline


@dataclass
class ComplianceConfig:
    """Configuration for compliance requirements."""
    
    # Regional compliance
    regions: List[ComplianceRegion] = field(default_factory=lambda: [ComplianceRegion.GLOBAL])
    
    # Data retention
    max_retention_days: int = 2555  # 7 years default
    automatic_deletion: bool = True
    deletion_grace_period_days: int = 30
    
    # Consent management
    require_consent: bool = True
    consent_valid_days: int = 365
    explicit_consent_required: bool = False
    
    # Data subject rights
    enable_right_to_be_forgotten: bool = True
    enable_data_portability: bool = True
    enable_right_of_access: bool = True
    enable_right_to_rectification: bool = True
    
    # Audit and logging
    detailed_audit_logging: bool = True
    audit_retention_days: int = 2555  # 7 years
    log_data_access: bool = True
    log_processing_operations: bool = True
    
    # Data minimization
    enforce_data_minimization: bool = True
    purpose_limitation: bool = True
    
    # Cross-border transfers
    allow_cross_border_transfer: bool = False
    adequacy_decision_required: bool = True
    
    # Anonymization
    require_anonymization: bool = False
    pseudonymization_required: bool = False
    
    # Breach notification
    breach_notification_hours: int = 72  # GDPR requirement
    enable_breach_detection: bool = True


@dataclass
class ConsentRecord:
    """Record of data subject consent."""
    
    subject_id: str
    consent_type: str
    purpose: str
    granted: bool
    timestamp: datetime
    expiry_date: datetime
    withdrawal_date: Optional[datetime] = None
    legal_basis: str = "consent"
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRecord:
    """Record of data processing activity."""
    
    activity_id: str
    data_subject_id: str
    data_categories: List[str]
    processing_purpose: str
    legal_basis: str
    timestamp: datetime
    controller: str
    processor: str = "llm-tab-cleaner"
    recipients: List[str] = field(default_factory=list)
    retention_period: int = 2555  # days
    cross_border_transfer: bool = False
    transfer_safeguards: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "activity_id": self.activity_id,
            "data_subject_id": self.data_subject_id,
            "data_categories": self.data_categories,
            "processing_purpose": self.processing_purpose,
            "legal_basis": self.legal_basis,
            "timestamp": self.timestamp.isoformat(),
            "controller": self.controller,
            "processor": self.processor,
            "recipients": self.recipients,
            "retention_period": self.retention_period,
            "cross_border_transfer": self.cross_border_transfer,
            "transfer_safeguards": self.transfer_safeguards
        }


class ComplianceManager:
    """Manages compliance and regulatory requirements."""
    
    def __init__(self, config: ComplianceConfig):
        """Initialize compliance manager."""
        self.config = config
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.processing_records: List[ProcessingRecord] = []
        self.data_classifications: Dict[str, DataClassification] = {}
        
        # Setup audit logging
        if config.detailed_audit_logging:
            self._setup_audit_logging()
        
        logger.info("Initialized compliance manager")
    
    def _setup_audit_logging(self) -> None:
        """Setup detailed audit logging."""
        self.compliance_logger = logging.getLogger("llm_tab_cleaner.compliance")
        self.compliance_logger.setLevel(logging.INFO)
        
        # Add file handler for compliance logs
        log_dir = Path("compliance_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"compliance_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.compliance_logger.addHandler(handler)
    
    def classify_data(self, df: pd.DataFrame, column_classifications: Dict[str, DataClassification] = None) -> Dict[str, DataClassification]:
        """Classify data based on column content and metadata.
        
        Args:
            df: DataFrame to classify
            column_classifications: Manual column classifications
            
        Returns:
            Dictionary mapping column names to classifications
        """
        classifications = {}
        
        # Apply manual classifications first
        if column_classifications:
            classifications.update(column_classifications)
        
        # Auto-classify remaining columns
        for column in df.columns:
            if column in classifications:
                continue
                
            classification = self._auto_classify_column(column, df[column])
            classifications[column] = classification
        
        # Store classifications
        self.data_classifications.update(classifications)
        
        # Log classification
        if self.config.detailed_audit_logging:
            self.compliance_logger.info(f"Data classification completed: {classifications}")
        
        return classifications
    
    def _auto_classify_column(self, column_name: str, series: pd.Series) -> DataClassification:
        """Automatically classify a column based on name and content."""
        column_lower = column_name.lower()
        
        # Check for sensitive personal data patterns
        sensitive_patterns = [
            'ssn', 'social_security', 'passport', 'tax_id', 'national_id',
            'medical', 'health', 'diagnosis', 'prescription',
            'religion', 'political', 'sexual', 'race', 'ethnicity',
            'biometric', 'genetic'
        ]
        
        if any(pattern in column_lower for pattern in sensitive_patterns):
            return DataClassification.SENSITIVE_PERSONAL
        
        # Check for personal data patterns
        personal_patterns = [
            'name', 'email', 'phone', 'address', 'zip', 'postal',
            'birth', 'age', 'dob', 'gender', 'ip_address', 'user_id',
            'customer_id', 'account', 'login'
        ]
        
        if any(pattern in column_lower for pattern in personal_patterns):
            return DataClassification.PERSONAL
        
        # Check for confidential data patterns
        confidential_patterns = [
            'salary', 'wage', 'income', 'credit', 'financial',
            'password', 'secret', 'key', 'token'
        ]
        
        if any(pattern in column_lower for pattern in confidential_patterns):
            return DataClassification.CONFIDENTIAL
        
        # Default classification
        return DataClassification.INTERNAL
    
    def check_consent(self, data_subject_id: str, purpose: str) -> bool:
        """Check if we have valid consent for data processing.
        
        Args:
            data_subject_id: Unique identifier for data subject
            purpose: Processing purpose
            
        Returns:
            True if consent is valid, False otherwise
        """
        if not self.config.require_consent:
            return True
        
        consent_key = f"{data_subject_id}:{purpose}"
        
        if consent_key not in self.consent_records:
            logger.warning(f"No consent record found for {consent_key}")
            return False
        
        consent = self.consent_records[consent_key]
        
        if not consent.granted:
            return False
        
        if consent.withdrawal_date:
            return False
        
        if datetime.now() > consent.expiry_date:
            logger.warning(f"Consent expired for {consent_key}")
            return False
        
        return True
    
    def record_consent(self, consent: ConsentRecord) -> None:
        """Record data subject consent."""
        consent_key = f"{consent.subject_id}:{consent.purpose}"
        self.consent_records[consent_key] = consent
        
        # Log consent
        if self.config.detailed_audit_logging:
            self.compliance_logger.info(f"Consent recorded: {consent_key}, granted: {consent.granted}")
    
    def withdraw_consent(self, data_subject_id: str, purpose: str) -> bool:
        """Withdraw consent for data processing."""
        consent_key = f"{data_subject_id}:{purpose}"
        
        if consent_key not in self.consent_records:
            return False
        
        self.consent_records[consent_key].withdrawal_date = datetime.now()
        
        # Log withdrawal
        if self.config.detailed_audit_logging:
            self.compliance_logger.info(f"Consent withdrawn: {consent_key}")
        
        return True
    
    def record_processing_activity(self, record: ProcessingRecord) -> None:
        """Record data processing activity."""
        self.processing_records.append(record)
        
        # Log processing activity
        if self.config.log_processing_operations:
            self.compliance_logger.info(f"Processing activity: {json.dumps(record.to_dict())}")
    
    def validate_cross_border_transfer(self, target_region: str) -> bool:
        """Validate if cross-border data transfer is allowed."""
        if not self.config.allow_cross_border_transfer:
            return False
        
        # Check adequacy decision for EU transfers
        if ComplianceRegion.EU in self.config.regions:
            adequate_countries = [
                "andorra", "argentina", "canada", "faroe_islands", "guernsey",
                "israel", "isle_of_man", "japan", "jersey", "new_zealand",
                "switzerland", "united_kingdom", "uruguay"
            ]
            
            if self.config.adequacy_decision_required and target_region.lower() not in adequate_countries:
                logger.warning(f"Cross-border transfer to {target_region} requires additional safeguards")
                return False
        
        return True
    
    def check_retention_compliance(self, processing_timestamp: datetime, retention_days: int = None) -> bool:
        """Check if data retention complies with policy."""
        retention_period = retention_days or self.config.max_retention_days
        cutoff_date = processing_timestamp + timedelta(days=retention_period)
        
        return datetime.now() <= cutoff_date
    
    def get_expired_data(self) -> List[ProcessingRecord]:
        """Get processing records that have exceeded retention period."""
        expired = []
        
        for record in self.processing_records:
            if not self.check_retention_compliance(record.timestamp, record.retention_period):
                expired.append(record)
        
        return expired
    
    def anonymize_data(self, df: pd.DataFrame, classifications: Dict[str, DataClassification] = None) -> pd.DataFrame:
        """Anonymize personal data in DataFrame.
        
        Args:
            df: DataFrame to anonymize
            classifications: Optional column classifications
            
        Returns:
            Anonymized DataFrame
        """
        if not self.config.require_anonymization:
            return df
        
        classifications = classifications or self.data_classifications
        anonymized_df = df.copy()
        
        for column, classification in classifications.items():
            if column not in df.columns:
                continue
            
            if classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]:
                # Apply anonymization technique based on data type
                anonymized_df[column] = self._anonymize_column(df[column], classification)
        
        # Log anonymization
        if self.config.detailed_audit_logging:
            self.compliance_logger.info(f"Data anonymization applied to {len(classifications)} columns")
        
        return anonymized_df
    
    def _anonymize_column(self, series: pd.Series, classification: DataClassification) -> pd.Series:
        """Anonymize a specific column."""
        if classification == DataClassification.SENSITIVE_PERSONAL:
            # Full redaction for sensitive personal data
            return pd.Series(['[REDACTED]'] * len(series), index=series.index)
        
        # For personal data, apply pseudonymization
        anonymized = []
        for value in series:
            if pd.isna(value):
                anonymized.append(value)
            else:
                # Create consistent hash for pseudonymization
                hash_value = hashlib.sha256(str(value).encode()).hexdigest()[:8]
                anonymized.append(f"ANON_{hash_value}")
        
        return pd.Series(anonymized, index=series.index)
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "configuration": {
                "regions": [region.value for region in self.config.regions],
                "max_retention_days": self.config.max_retention_days,
                "require_consent": self.config.require_consent,
                "detailed_audit_logging": self.config.detailed_audit_logging
            },
            "statistics": {
                "total_consent_records": len(self.consent_records),
                "active_consents": len([c for c in self.consent_records.values() 
                                     if c.granted and not c.withdrawal_date]),
                "total_processing_activities": len(self.processing_records),
                "data_classifications": {
                    classification.value: len([c for c in self.data_classifications.values() 
                                             if c == classification])
                    for classification in DataClassification
                }
            },
            "compliance_checks": {
                "consent_compliance": self._check_consent_compliance(),
                "retention_compliance": self._check_retention_compliance(),
                "cross_border_compliance": self._check_cross_border_compliance()
            }
        }
        
        # Log report generation
        if self.config.detailed_audit_logging:
            self.compliance_logger.info(f"Compliance report generated: {report['report_id']}")
        
        return report
    
    def _check_consent_compliance(self) -> Dict[str, Any]:
        """Check consent compliance status."""
        if not self.config.require_consent:
            return {"status": "not_required", "compliant": True}
        
        total_records = len(self.consent_records)
        expired_records = len([c for c in self.consent_records.values() 
                              if datetime.now() > c.expiry_date])
        withdrawn_records = len([c for c in self.consent_records.values() 
                               if c.withdrawal_date])
        
        return {
            "status": "required",
            "total_records": total_records,
            "expired_records": expired_records,
            "withdrawn_records": withdrawn_records,
            "compliant": expired_records == 0
        }
    
    def _check_retention_compliance(self) -> Dict[str, Any]:
        """Check data retention compliance."""
        expired_records = self.get_expired_data()
        
        return {
            "total_processing_records": len(self.processing_records),
            "expired_records": len(expired_records),
            "compliant": len(expired_records) == 0,
            "automatic_deletion_enabled": self.config.automatic_deletion
        }
    
    def _check_cross_border_compliance(self) -> Dict[str, Any]:
        """Check cross-border transfer compliance."""
        cross_border_activities = [r for r in self.processing_records if r.cross_border_transfer]
        
        return {
            "cross_border_transfers_allowed": self.config.allow_cross_border_transfer,
            "total_cross_border_activities": len(cross_border_activities),
            "adequacy_decision_required": self.config.adequacy_decision_required,
            "compliant": all(len(r.transfer_safeguards) > 0 for r in cross_border_activities)
        }


def create_gdpr_config() -> ComplianceConfig:
    """Create GDPR-compliant configuration."""
    return ComplianceConfig(
        regions=[ComplianceRegion.EU],
        max_retention_days=2555,  # 7 years
        require_consent=True,
        explicit_consent_required=True,
        enable_right_to_be_forgotten=True,
        enable_data_portability=True,
        enable_right_of_access=True,
        enable_right_to_rectification=True,
        detailed_audit_logging=True,
        enforce_data_minimization=True,
        purpose_limitation=True,
        allow_cross_border_transfer=False,
        adequacy_decision_required=True,
        breach_notification_hours=72
    )


def create_ccpa_config() -> ComplianceConfig:
    """Create CCPA-compliant configuration."""
    return ComplianceConfig(
        regions=[ComplianceRegion.US],
        max_retention_days=2555,  # 7 years
        require_consent=False,  # CCPA uses opt-out model
        enable_right_to_be_forgotten=True,
        enable_data_portability=True,
        enable_right_of_access=True,
        detailed_audit_logging=True,
        enforce_data_minimization=True,
        allow_cross_border_transfer=True,
        breach_notification_hours=72
    )


def create_global_baseline_config() -> ComplianceConfig:
    """Create global baseline compliance configuration."""
    return ComplianceConfig(
        regions=[ComplianceRegion.GLOBAL],
        max_retention_days=1825,  # 5 years
        require_consent=True,
        enable_right_to_be_forgotten=True,
        enable_right_of_access=True,
        detailed_audit_logging=True,
        enforce_data_minimization=True,
        allow_cross_border_transfer=True,
        breach_notification_hours=72
    )