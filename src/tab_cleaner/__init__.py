"""tab-cleaner: LLM-ready data cleaning pipeline with audit trails and confidence scoring."""

from .cleaners import (
    BaseCleaningRule,
    DuplicateRemover,
    MissingValueImputer,
    OutlierDetector,
    TypeCoercer,
)
from .audit import AuditTrail
from .confidence import ConfidenceScorer
from .pipeline import CleaningPipeline

__version__ = "1.0.0"
__all__ = [
    "BaseCleaningRule",
    "MissingValueImputer",
    "TypeCoercer",
    "OutlierDetector",
    "DuplicateRemover",
    "AuditTrail",
    "ConfidenceScorer",
    "CleaningPipeline",
]
