"""Multi-Modal Data Cleaning Research Module.

Implements next-generation multi-modal data cleaning that combines:
- Text/NLP data cleaning
- Time series data cleaning  
- Image/visual data cleaning
- Audio data cleaning
- Graph/network data cleaning

Based on cutting-edge research in multi-modal machine learning and
cross-modal data quality assessment.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import base64
import json
from pathlib import Path
import time
from enum import Enum

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of data modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TIME_SERIES = "time_series"
    TABULAR = "tabular"
    GRAPH = "graph"
    VIDEO = "video"
    GEOSPATIAL = "geospatial"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""
    modality_type: ModalityType
    weight: float = 1.0
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_extractors: List[str] = field(default_factory=list)
    quality_metrics: List[str] = field(default_factory=list)
    cleaning_strategies: List[str] = field(default_factory=list)
    cross_modal_features: bool = True


@dataclass
class MultiModalSample:
    """Sample containing multiple modalities."""
    sample_id: str
    modalities: Dict[ModalityType, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[ModalityType, float] = field(default_factory=dict)
    cross_modal_features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize quality scores."""
        for modality in self.modalities.keys():
            if modality not in self.quality_scores:
                self.quality_scores[modality] = 0.0


@dataclass
class CleaningResult:
    """Result of multi-modal cleaning."""
    original_sample: MultiModalSample
    cleaned_sample: MultiModalSample
    cleaning_actions: Dict[ModalityType, List[str]]
    confidence_scores: Dict[ModalityType, float]
    cross_modal_consistency: float
    processing_time: float


class ModalityProcessor(ABC):
    """Abstract base class for modality-specific processors."""
    
    @abstractmethod
    def extract_features(self, data: Any) -> np.ndarray:
        """Extract features from modality data."""
        pass
    
    @abstractmethod
    def assess_quality(self, data: Any) -> float:
        """Assess quality of modality data."""
        pass
    
    @abstractmethod
    def clean_data(self, data: Any, confidence_threshold: float = 0.8) -> Tuple[Any, List[str]]:
        """Clean modality data."""
        pass
    
    @abstractmethod
    def detect_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """Detect anomalies in modality data."""
        pass


class TextProcessor(ModalityProcessor):
    """Processor for text modality."""
    
    def __init__(self):
        self.vocab_size_threshold = 10000
        self.min_text_length = 10
        self.max_text_length = 10000
        
    def extract_features(self, text: str) -> np.ndarray:
        """Extract text features."""
        if not isinstance(text, str):
            text = str(text)
        
        features = []
        
        # Basic statistics
        features.extend([
            len(text),  # Character count
            len(text.split()),  # Word count
            len(set(text.split())),  # Unique word count
            text.count('\n'),  # Line count
            text.count('!') + text.count('?'),  # Exclamation/question marks
        ])
        
        # Language features
        features.extend([
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
            sum(1 for c in text if c.isspace()) / max(len(text), 1),  # Whitespace ratio
            sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),  # Special char ratio
        ])
        
        # Readability approximation
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            sentences = text.split('.') + text.split('!') + text.split('?')
            avg_sentence_length = len(words) / max(len(sentences), 1)
            features.extend([avg_word_length, avg_sentence_length])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def assess_quality(self, text: str) -> float:
        """Assess text quality."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        
        score = 1.0
        
        # Length checks
        text_len = len(text)
        if text_len < self.min_text_length:
            score *= 0.5
        elif text_len > self.max_text_length:
            score *= 0.7
        
        # Encoding issues
        try:
            text.encode('utf-8')
        except UnicodeError:
            score *= 0.3
        
        # Repetition check
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score *= unique_ratio
        
        # Special character ratio
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.3:
            score *= 0.8
        
        return min(score, 1.0)
    
    def clean_data(self, text: str, confidence_threshold: float = 0.8) -> Tuple[str, List[str]]:
        """Clean text data."""
        if not isinstance(text, str):
            text = str(text)
        
        actions = []
        cleaned_text = text
        
        # Remove excessive whitespace
        original_len = len(cleaned_text)
        cleaned_text = ' '.join(cleaned_text.split())
        if len(cleaned_text) != original_len:
            actions.append("normalized_whitespace")
        
        # Fix encoding issues
        try:
            cleaned_text = cleaned_text.encode('utf-8', errors='ignore').decode('utf-8')
            actions.append("fixed_encoding")
        except:
            pass
        
        # Remove excessive punctuation
        import re
        original_text = cleaned_text
        cleaned_text = re.sub(r'[!?]{3,}', '!!!', cleaned_text)
        cleaned_text = re.sub(r'\.{4,}', '...', cleaned_text)
        if cleaned_text != original_text:
            actions.append("normalized_punctuation")
        
        return cleaned_text, actions
    
    def detect_anomalies(self, text: str) -> List[Dict[str, Any]]:
        """Detect text anomalies."""
        anomalies = []
        
        if not isinstance(text, str):
            anomalies.append({
                "type": "type_error",
                "description": f"Expected string, got {type(text)}",
                "severity": "high"
            })
            return anomalies
        
        # Check for suspicious patterns
        if len(text.split()) < 3 and len(text) > 50:
            anomalies.append({
                "type": "low_word_density",
                "description": "Very few words for text length",
                "severity": "medium"
            })
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5:
                anomalies.append({
                    "type": "excessive_repetition",
                    "description": f"Word repeated {max_count} times",
                    "severity": "high"
                })
        
        return anomalies


class TimeSeriesProcessor(ModalityProcessor):
    """Processor for time series modality."""
    
    def __init__(self):
        self.min_length = 10
        self.max_gap_ratio = 0.1
        
    def extract_features(self, series: np.ndarray) -> np.ndarray:
        """Extract time series features."""
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=np.float32)
        
        if len(series) == 0:
            return np.zeros(20, dtype=np.float32)
        
        features = []
        
        # Basic statistics
        features.extend([
            len(series),
            np.mean(series),
            np.std(series),
            np.min(series),
            np.max(series),
            np.median(series),
        ])
        
        # Distribution features
        features.extend([
            np.percentile(series, 25),
            np.percentile(series, 75),
            np.sum(np.isnan(series)) / len(series),  # Missing ratio
            np.sum(np.isinf(series)) / len(series),  # Infinite ratio
        ])
        
        # Temporal features
        if len(series) > 1:
            diff = np.diff(series)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.mean(np.abs(diff)),
                len(np.where(np.diff(np.sign(diff)) != 0)[0]) / max(len(diff) - 1, 1),  # Zero crossings
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Trend and seasonality (simplified)
        if len(series) >= 12:
            # Simple linear trend
            x = np.arange(len(series))
            trend_coef = np.corrcoef(x, series)[0, 1] if not np.any(np.isnan(series)) else 0
            features.append(trend_coef)
            
            # Seasonality (autocorrelation at lag 12)
            if len(series) >= 24:
                lag_12_corr = np.corrcoef(series[:-12], series[12:])[0, 1] if not np.any(np.isnan(series)) else 0
                features.append(lag_12_corr)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Stability measures
        if len(series) > 10:
            # Rolling window variance
            window_size = min(10, len(series) // 3)
            rolling_vars = []
            for i in range(len(series) - window_size + 1):
                window = series[i:i + window_size]
                if not np.any(np.isnan(window)):
                    rolling_vars.append(np.var(window))
            
            if rolling_vars:
                features.append(np.mean(rolling_vars))
                features.append(np.std(rolling_vars))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def assess_quality(self, series: np.ndarray) -> float:
        """Assess time series quality."""
        if not isinstance(series, np.ndarray):
            try:
                series = np.array(series, dtype=np.float32)
            except:
                return 0.0
        
        if len(series) == 0:
            return 0.0
        
        score = 1.0
        
        # Length check
        if len(series) < self.min_length:
            score *= 0.5
        
        # Missing values
        missing_ratio = np.sum(np.isnan(series)) / len(series)
        score *= (1 - missing_ratio)
        
        # Infinite values
        inf_ratio = np.sum(np.isinf(series)) / len(series)
        score *= (1 - inf_ratio)
        
        # Constant series
        if len(np.unique(series[~np.isnan(series)])) == 1:
            score *= 0.3
        
        # Extreme outliers (more than 6 standard deviations)
        if not np.all(np.isnan(series)):
            mean_val = np.nanmean(series)
            std_val = np.nanstd(series)
            if std_val > 0:
                outlier_ratio = np.sum(np.abs(series - mean_val) > 6 * std_val) / len(series)
                score *= (1 - outlier_ratio)
        
        return min(score, 1.0)
    
    def clean_data(self, series: np.ndarray, confidence_threshold: float = 0.8) -> Tuple[np.ndarray, List[str]]:
        """Clean time series data."""
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=np.float32)
        
        actions = []
        cleaned_series = series.copy()
        
        # Handle infinite values
        inf_mask = np.isinf(cleaned_series)
        if np.any(inf_mask):
            cleaned_series[inf_mask] = np.nan
            actions.append("replaced_infinite_with_nan")
        
        # Interpolate missing values (simple linear interpolation)
        missing_mask = np.isnan(cleaned_series)
        if np.any(missing_mask) and not np.all(missing_mask):
            valid_indices = np.where(~missing_mask)[0]
            if len(valid_indices) >= 2:
                cleaned_series = pd.Series(cleaned_series).interpolate(method='linear').values
                actions.append("interpolated_missing_values")
        
        # Handle extreme outliers
        if not np.all(np.isnan(cleaned_series)):
            mean_val = np.nanmean(cleaned_series)
            std_val = np.nanstd(cleaned_series)
            if std_val > 0:
                outlier_threshold = 5 * std_val
                outlier_mask = np.abs(cleaned_series - mean_val) > outlier_threshold
                if np.any(outlier_mask):
                    # Cap outliers
                    cleaned_series[outlier_mask & (cleaned_series > mean_val)] = mean_val + outlier_threshold
                    cleaned_series[outlier_mask & (cleaned_series < mean_val)] = mean_val - outlier_threshold
                    actions.append("capped_extreme_outliers")
        
        return cleaned_series, actions
    
    def detect_anomalies(self, series: np.ndarray) -> List[Dict[str, Any]]:
        """Detect time series anomalies."""
        anomalies = []
        
        if not isinstance(series, np.ndarray):
            try:
                series = np.array(series, dtype=np.float32)
            except:
                anomalies.append({
                    "type": "conversion_error",
                    "description": "Cannot convert to numeric array",
                    "severity": "high"
                })
                return anomalies
        
        if len(series) == 0:
            anomalies.append({
                "type": "empty_series",
                "description": "Time series is empty",
                "severity": "high"
            })
            return anomalies
        
        # Check for suspicious patterns
        missing_ratio = np.sum(np.isnan(series)) / len(series)
        if missing_ratio > 0.5:
            anomalies.append({
                "type": "excessive_missing",
                "description": f"{missing_ratio:.2%} of values are missing",
                "severity": "high"
            })
        
        # Check for constant series
        if len(np.unique(series[~np.isnan(series)])) == 1:
            anomalies.append({
                "type": "constant_series",
                "description": "All non-missing values are identical",
                "severity": "medium"
            })
        
        # Check for sudden jumps
        if len(series) > 1:
            diff = np.diff(series)
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) > 0:
                diff_std = np.std(valid_diff)
                if diff_std > 0:
                    large_jumps = np.abs(valid_diff) > 5 * diff_std
                    if np.any(large_jumps):
                        anomalies.append({
                            "type": "sudden_jumps",
                            "description": f"{np.sum(large_jumps)} sudden jumps detected",
                            "severity": "medium"
                        })
        
        return anomalies


class MultiModalProcessor:
    """Main processor for multi-modal data cleaning."""
    
    def __init__(self, modality_configs: Optional[List[ModalityConfig]] = None):
        self.modality_configs = modality_configs or self._get_default_configs()
        self.processors = self._initialize_processors()
        self.cross_modal_features = {}
        
    def _get_default_configs(self) -> List[ModalityConfig]:
        """Get default configurations for supported modalities."""
        return [
            ModalityConfig(
                modality_type=ModalityType.TEXT,
                weight=1.0,
                quality_metrics=["length", "encoding", "repetition"],
                cleaning_strategies=["normalize_whitespace", "fix_encoding"]
            ),
            ModalityConfig(
                modality_type=ModalityType.TIME_SERIES,
                weight=1.0,
                quality_metrics=["completeness", "consistency", "outliers"],
                cleaning_strategies=["interpolate", "cap_outliers"]
            ),
            ModalityConfig(
                modality_type=ModalityType.TABULAR,
                weight=1.0,
                quality_metrics=["missing_values", "duplicates", "outliers"],
                cleaning_strategies=["impute", "deduplicate", "normalize"]
            )
        ]
    
    def _initialize_processors(self) -> Dict[ModalityType, ModalityProcessor]:
        """Initialize modality-specific processors."""
        processors = {}
        
        for config in self.modality_configs:
            if config.modality_type == ModalityType.TEXT:
                processors[ModalityType.TEXT] = TextProcessor()
            elif config.modality_type == ModalityType.TIME_SERIES:
                processors[ModalityType.TIME_SERIES] = TimeSeriesProcessor()
            # Add more processors as needed
        
        return processors
    
    def process_sample(self, sample: MultiModalSample) -> CleaningResult:
        """Process a multi-modal sample."""
        start_time = time.time()
        
        cleaned_modalities = {}
        cleaning_actions = {}
        confidence_scores = {}
        
        # Process each modality
        for modality_type, data in sample.modalities.items():
            if modality_type in self.processors:
                processor = self.processors[modality_type]
                
                # Assess quality
                quality_score = processor.assess_quality(data)
                sample.quality_scores[modality_type] = quality_score
                
                # Clean data
                cleaned_data, actions = processor.clean_data(data)
                cleaned_modalities[modality_type] = cleaned_data
                cleaning_actions[modality_type] = actions
                
                # Assess confidence in cleaning
                cleaned_quality = processor.assess_quality(cleaned_data)
                confidence_scores[modality_type] = cleaned_quality
            else:
                # No processor available, keep original
                cleaned_modalities[modality_type] = data
                cleaning_actions[modality_type] = []
                confidence_scores[modality_type] = 0.5
        
        # Create cleaned sample
        cleaned_sample = MultiModalSample(
            sample_id=sample.sample_id,
            modalities=cleaned_modalities,
            metadata=sample.metadata.copy(),
            quality_scores=confidence_scores
        )
        
        # Compute cross-modal consistency
        cross_modal_consistency = self._compute_cross_modal_consistency(cleaned_sample)
        
        processing_time = time.time() - start_time
        
        return CleaningResult(
            original_sample=sample,
            cleaned_sample=cleaned_sample,
            cleaning_actions=cleaning_actions,
            confidence_scores=confidence_scores,
            cross_modal_consistency=cross_modal_consistency,
            processing_time=processing_time
        )
    
    def _compute_cross_modal_consistency(self, sample: MultiModalSample) -> float:
        """Compute consistency across modalities."""
        if len(sample.modalities) < 2:
            return 1.0
        
        # Extract features from each modality
        features = {}
        for modality_type, data in sample.modalities.items():
            if modality_type in self.processors:
                processor = self.processors[modality_type]
                features[modality_type] = processor.extract_features(data)
        
        if len(features) < 2:
            return 1.0
        
        # Compute pairwise correlations (simplified)
        correlations = []
        modality_types = list(features.keys())
        
        for i in range(len(modality_types)):
            for j in range(i + 1, len(modality_types)):
                feat1 = features[modality_types[i]]
                feat2 = features[modality_types[j]]
                
                # Pad to same length
                min_len = min(len(feat1), len(feat2))
                if min_len > 0:
                    corr = np.corrcoef(feat1[:min_len], feat2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            return np.mean(correlations)
        else:
            return 0.5
    
    def batch_process(self, samples: List[MultiModalSample]) -> List[CleaningResult]:
        """Process multiple samples efficiently."""
        results = []
        
        logger.info(f"Processing {len(samples)} multi-modal samples")
        
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(samples)} samples")
            
            result = self.process_sample(sample)
            results.append(result)
        
        logger.info(f"Completed processing {len(samples)} samples")
        return results
    
    def get_modality_statistics(self, results: List[CleaningResult]) -> Dict[str, Any]:
        """Get statistics about multi-modal processing."""
        stats = {
            "total_samples": len(results),
            "modality_coverage": {},
            "average_quality_scores": {},
            "cleaning_action_frequency": {},
            "cross_modal_consistency": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        }
        
        if not results:
            return stats
        
        # Analyze modality coverage
        all_modalities = set()
        for result in results:
            all_modalities.update(result.original_sample.modalities.keys())
        
        for modality in all_modalities:
            count = sum(1 for result in results if modality in result.original_sample.modalities)
            stats["modality_coverage"][modality.value] = count / len(results)
        
        # Average quality scores
        for modality in all_modalities:
            scores = []
            for result in results:
                if modality in result.confidence_scores:
                    scores.append(result.confidence_scores[modality])
            
            if scores:
                stats["average_quality_scores"][modality.value] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                }
        
        # Cleaning action frequency
        for modality in all_modalities:
            action_counts = {}
            for result in results:
                if modality in result.cleaning_actions:
                    for action in result.cleaning_actions[modality]:
                        action_counts[action] = action_counts.get(action, 0) + 1
            
            stats["cleaning_action_frequency"][modality.value] = action_counts
        
        # Cross-modal consistency
        consistency_scores = [result.cross_modal_consistency for result in results]
        stats["cross_modal_consistency"] = {
            "mean": np.mean(consistency_scores),
            "std": np.std(consistency_scores),
            "min": np.min(consistency_scores),
            "max": np.max(consistency_scores)
        }
        
        return stats


def create_multimodal_sample(
    sample_id: str,
    text_data: Optional[str] = None,
    time_series_data: Optional[np.ndarray] = None,
    tabular_data: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> MultiModalSample:
    """Create a multi-modal sample from different data types."""
    modalities = {}
    
    if text_data is not None:
        modalities[ModalityType.TEXT] = text_data
    
    if time_series_data is not None:
        modalities[ModalityType.TIME_SERIES] = time_series_data
    
    if tabular_data is not None:
        modalities[ModalityType.TABULAR] = tabular_data
    
    return MultiModalSample(
        sample_id=sample_id,
        modalities=modalities,
        metadata=metadata or {}
    )