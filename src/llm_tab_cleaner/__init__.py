"""LLM Tab Cleaner - Production data cleaning with language models."""

from .core import TableCleaner, CleaningReport, Fix
from .cleaning_rule import CleaningRule, RuleSet, create_default_rules, create_custom_rule
from .confidence import ConfidenceCalibrator, CalibrationMetrics, EnsembleCalibrator, create_ensemble_calibrator
from .incremental import IncrementalCleaner
from .profiler import DataProfiler, ColumnProfile, TableProfile
from .llm_providers import get_provider, AnthropicProvider, OpenAIProvider, LocalProvider

# Optional imports for advanced features
try:
    from .spark import SparkCleaner, StreamingCleaner, create_spark_cleaner
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False

try:
    from .benchmarks import PerformanceBenchmarker, BenchmarkResult, BenchmarkSuite, run_comprehensive_benchmark
    _BENCHMARKS_AVAILABLE = True
except ImportError:
    _BENCHMARKS_AVAILABLE = False

__version__ = "0.3.0"
__all__ = [
    # Core cleaning
    "TableCleaner",
    "CleaningReport", 
    "Fix",
    
    # Cleaning rules
    "CleaningRule", 
    "RuleSet",
    "create_default_rules",
    "create_custom_rule",
    
    # Confidence calibration
    "ConfidenceCalibrator",
    "CalibrationMetrics",
    "EnsembleCalibrator", 
    "create_ensemble_calibrator",
    
    # Data profiling
    "DataProfiler",
    "ColumnProfile",
    "TableProfile",
    
    # LLM providers
    "get_provider",
    "AnthropicProvider",
    "OpenAIProvider", 
    "LocalProvider",
    
    # Advanced features
    "IncrementalCleaner",
]

# Add Spark components if available
if _SPARK_AVAILABLE:
    __all__.extend([
        "SparkCleaner",
        "StreamingCleaner", 
        "create_spark_cleaner"
    ])

# Add benchmark components if available
if _BENCHMARKS_AVAILABLE:
    __all__.extend([
        "PerformanceBenchmarker",
        "BenchmarkResult",
        "BenchmarkSuite",
        "run_comprehensive_benchmark"
    ])

# Research modules
try:
    from .neural_confidence import NeuralCalibrator, CalibrationMetrics, NeuralCalibrationConfig
    from .federated_learning import FederatedDataQualityServer, FederatedDataQualityClient, FederatedConfig
    from .multimodal_cleaning import MultiModalProcessor, MultiModalSample, ModalityType
    from .adaptive_learning import AdaptiveLearningSystem, FeedbackSignal, AdaptationMetrics
    _RESEARCH_AVAILABLE = True
except ImportError:
    _RESEARCH_AVAILABLE = False

# Add research components if available
if _RESEARCH_AVAILABLE:
    __all__.extend([
        "NeuralCalibrator",
        "CalibrationMetrics", 
        "NeuralCalibrationConfig",
        "FederatedDataQualityServer",
        "FederatedDataQualityClient",
        "FederatedConfig",
        "MultiModalProcessor",
        "MultiModalSample",
        "ModalityType",
        "AdaptiveLearningSystem",
        "FeedbackSignal",
        "AdaptationMetrics"
    ])


def get_version_info():
    """Get detailed version and feature availability information."""
    return {
        "version": __version__,
        "features": {
            "core_cleaning": True,
            "llm_providers": True,
            "data_profiling": True,
            "confidence_calibration": True,
            "cleaning_rules": True,
            "spark_integration": _SPARK_AVAILABLE,
            "benchmarking": _BENCHMARKS_AVAILABLE,
            "incremental_cleaning": True,
            "research_modules": _RESEARCH_AVAILABLE,
            "neural_confidence": _RESEARCH_AVAILABLE,
            "federated_learning": _RESEARCH_AVAILABLE,
            "multimodal_cleaning": _RESEARCH_AVAILABLE,
            "adaptive_learning": _RESEARCH_AVAILABLE
        }
    }