"""ML-Driven Quality Gates - Enhanced Validation System.

This module implements advanced quality gates with machine learning-driven
validation, anomaly detection, and intelligent quality assessment.

Key Features:
- Neural network-based quality prediction
- Anomaly detection for data quality issues  
- Adaptive quality thresholds based on data characteristics
- Multi-dimensional quality scoring
- Automated quality improvement suggestions
- Statistical significance testing for quality improvements

Author: Terry (Terragon Labs)
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy" 
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    CONFORMITY = "conformity"


class QualityGateStatus(Enum):
    """Quality gate validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class AnomalyType(Enum):
    """Types of data quality anomalies."""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    DISTRIBUTION = "distribution"
    RELATIONSHIP = "relationship"
    SCHEMA = "schema"


@dataclass
class QualityMetric:
    """Individual quality metric measurement."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float
    methodology: str
    details: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result."""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    metrics: List[QualityMetric]
    gate_status: QualityGateStatus
    improvement_suggestions: List[str]
    risk_factors: List[str]
    statistical_significance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    minimum_overall_score: float = 0.7
    dimension_thresholds: Dict[QualityDimension, float] = field(default_factory=dict)
    enable_ml_prediction: bool = True
    enable_anomaly_detection: bool = True
    statistical_significance_level: float = 0.05
    adaptive_thresholds: bool = True


class NeuralQualityPredictor:
    """Neural network-based quality score prediction."""
    
    def __init__(self):
        """Initialize neural quality predictor."""
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for quality prediction."""
        features = []
        
        # Basic statistics
        features.extend([
            df.shape[0],  # Number of rows
            df.shape[1],  # Number of columns
            df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),  # Missing ratio
            df.duplicated().sum() / df.shape[0] if df.shape[0] > 0 else 0,  # Duplicate ratio
        ])
        
        # Data type diversity
        dtype_counts = df.dtypes.value_counts()
        features.extend([
            len(dtype_counts),  # Number of different data types
            dtype_counts.get('object', 0),  # Number of object columns
            dtype_counts.get('int64', 0) + dtype_counts.get('float64', 0),  # Numeric columns
        ])
        
        # Statistical properties
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            features.extend([
                numeric_df.std().mean(),  # Average standard deviation
                abs(numeric_df.skew()).mean(),  # Average skewness
                numeric_df.kurtosis().mean(),  # Average kurtosis
                np.corrcoef(numeric_df.T).mean() if len(numeric_cols) > 1 else 0,  # Average correlation
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # String/categorical properties
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            features.extend([
                df[object_cols].nunique().mean(),  # Average unique values
                df[object_cols].apply(lambda x: x.str.len().mean() if x.dtype == 'object' else 0).mean(),  # Avg string length
            ])
        else:
            features.extend([0, 0])
        
        # Pattern consistency
        pattern_scores = []
        for col in object_cols[:5]:  # Limit to first 5 for performance
            if df[col].dtype == 'object':
                unique_patterns = df[col].astype(str).apply(self._extract_pattern).nunique()
                pattern_scores.append(unique_patterns / max(1, df[col].nunique()))
        
        features.append(np.mean(pattern_scores) if pattern_scores else 0)
        
        return np.array(features)
    
    def _extract_pattern(self, value: str) -> str:
        """Extract pattern from string value."""
        if pd.isna(value) or value == '':
            return 'EMPTY'
        
        pattern = ''
        for char in str(value):
            if char.isalpha():
                pattern += 'A'
            elif char.isdigit():
                pattern += 'N'
            elif char.isspace():
                pattern += 'S'
            else:
                pattern += 'X'
        
        return pattern
    
    def train(self, training_data: List[Tuple[pd.DataFrame, float]]):
        """Train the neural quality predictor.
        
        Args:
            training_data: List of (dataframe, quality_score) tuples
        """
        if len(training_data) < 10:
            logger.warning("Insufficient training data for neural quality predictor")
            return
        
        try:
            # Extract features and targets
            X = []
            y = []
            
            for df, quality_score in training_data:
                features = self.extract_features(df)
                X.append(features)
                y.append(quality_score)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Evaluate
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=min(5, len(X)), scoring='r2')
            mse = mean_squared_error(y, self.model.predict(X_scaled))
            
            self.is_trained = True
            self.training_history.append({
                'timestamp': time.time(),
                'samples': len(X),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'mse': mse
            })
            
            logger.info(f"Trained neural quality predictor: R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
        except Exception as e:
            logger.error(f"Error training neural quality predictor: {e}")
    
    def predict_quality(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Predict quality score for a dataframe.
        
        Returns:
            Tuple of (predicted_score, confidence)
        """
        if not self.is_trained:
            # Fallback to simple heuristic
            return self._heuristic_quality_score(df), 0.5
        
        try:
            features = self.extract_features(df).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            predicted_score = self.model.predict(features_scaled)[0]
            
            # Estimate confidence based on training performance
            latest_training = self.training_history[-1] if self.training_history else {}
            confidence = min(0.95, max(0.3, latest_training.get('cv_r2_mean', 0.5)))
            
            return max(0.0, min(1.0, predicted_score)), confidence
            
        except Exception as e:
            logger.error(f"Error predicting quality: {e}")
            return self._heuristic_quality_score(df), 0.3
    
    def _heuristic_quality_score(self, df: pd.DataFrame) -> float:
        """Simple heuristic quality score as fallback."""
        if df.empty:
            return 0.0
        
        # Basic quality indicators
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_ratio = df.duplicated().sum() / df.shape[0] if df.shape[0] > 0 else 0
        
        # Simple scoring
        completeness_score = 1.0 - missing_ratio
        uniqueness_score = 1.0 - duplicate_ratio
        
        return (completeness_score + uniqueness_score) / 2


class AnomalyDetector:
    """Multi-dimensional anomaly detection for data quality."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.detectors = {
            AnomalyType.STATISTICAL: IsolationForest(contamination=0.1, random_state=42),
            AnomalyType.DISTRIBUTION: None,  # Will use statistical tests
            AnomalyType.PATTERN: None,  # Will use pattern analysis
        }
        self.baseline_stats = {}
        self.trained = False
        
    def fit_baseline(self, reference_data: List[pd.DataFrame]):
        """Fit baseline statistics from reference data."""
        if not reference_data:
            return
        
        try:
            # Combine reference data
            combined_features = []
            for df in reference_data:
                features = self._extract_anomaly_features(df)
                combined_features.append(features)
            
            if combined_features:
                feature_matrix = np.array(combined_features)
                
                # Fit statistical anomaly detector
                self.detectors[AnomalyType.STATISTICAL].fit(feature_matrix)
                
                # Compute baseline statistics
                self.baseline_stats = {
                    'feature_means': np.mean(feature_matrix, axis=0),
                    'feature_stds': np.std(feature_matrix, axis=0),
                    'feature_ranges': (np.min(feature_matrix, axis=0), np.max(feature_matrix, axis=0))
                }
                
                self.trained = True
                logger.info(f"Trained anomaly detector on {len(reference_data)} reference datasets")
                
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {e}")
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in data quality."""
        anomalies = []
        
        # Statistical anomalies
        statistical_anomalies = self._detect_statistical_anomalies(df)
        anomalies.extend(statistical_anomalies)
        
        # Distribution anomalies
        distribution_anomalies = self._detect_distribution_anomalies(df)
        anomalies.extend(distribution_anomalies)
        
        # Pattern anomalies
        pattern_anomalies = self._detect_pattern_anomalies(df)
        anomalies.extend(pattern_anomalies)
        
        # Schema anomalies
        schema_anomalies = self._detect_schema_anomalies(df)
        anomalies.extend(schema_anomalies)
        
        return anomalies
    
    def _extract_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = []
        
        # Basic shape features
        features.extend([df.shape[0], df.shape[1]])
        
        # Missing data patterns
        features.append(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        
        # Data type distribution
        dtype_counts = df.dtypes.value_counts()
        features.extend([
            dtype_counts.get('object', 0) / df.shape[1],
            dtype_counts.get('int64', 0) / df.shape[1],
            dtype_counts.get('float64', 0) / df.shape[1]
        ])
        
        # Statistical properties for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            features.extend([
                numeric_df.mean().mean(),
                numeric_df.std().mean(),
                abs(numeric_df.skew()).mean(),
                numeric_df.kurtosis().mean()
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect statistical anomalies."""
        anomalies = []
        
        if not self.trained:
            return anomalies
        
        try:
            features = self._extract_anomaly_features(df).reshape(1, -1)
            is_outlier = self.detectors[AnomalyType.STATISTICAL].predict(features)[0] == -1
            
            if is_outlier:
                anomaly_score = -self.detectors[AnomalyType.STATISTICAL].score_samples(features)[0]
                anomalies.append({
                    'type': AnomalyType.STATISTICAL.value,
                    'severity': 'high' if anomaly_score > 0.7 else 'medium',
                    'score': anomaly_score,
                    'description': 'Statistical properties deviate significantly from baseline',
                    'affected_dimensions': [QualityDimension.CONSISTENCY.value, QualityDimension.VALIDITY.value]
                })
                
        except Exception as e:
            logger.error(f"Error detecting statistical anomalies: {e}")
        
        return anomalies
    
    def _detect_distribution_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect distribution anomalies using statistical tests."""
        anomalies = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                data = df[col].dropna()
                if len(data) < 30:  # Need sufficient data for tests
                    continue
                
                # Test for normality
                _, p_value_normality = stats.shapiro(data.sample(min(5000, len(data))))
                
                # Test for outliers using Z-score
                z_scores = np.abs(stats.zscore(data))
                outlier_ratio = (z_scores > 3).mean()
                
                if p_value_normality < 0.001 and outlier_ratio > 0.05:
                    anomalies.append({
                        'type': AnomalyType.DISTRIBUTION.value,
                        'severity': 'medium',
                        'score': 1 - p_value_normality,
                        'description': f'Column {col} has non-normal distribution with {outlier_ratio:.1%} outliers',
                        'affected_dimensions': [QualityDimension.VALIDITY.value],
                        'column': col,
                        'outlier_ratio': outlier_ratio
                    })
                    
            except Exception as e:
                logger.error(f"Error testing distribution for column {col}: {e}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect pattern anomalies in string columns."""
        anomalies = []
        
        object_cols = df.select_dtypes(include=['object']).columns
        
        for col in object_cols[:10]:  # Limit to first 10 columns for performance
            try:
                if df[col].dtype != 'object':
                    continue
                
                # Analyze pattern consistency
                patterns = df[col].astype(str).apply(self._extract_pattern)
                pattern_counts = patterns.value_counts()
                
                # Check for pattern fragmentation
                total_values = len(df[col].dropna())
                unique_patterns = len(pattern_counts)
                
                if unique_patterns > total_values * 0.3:  # Too many different patterns
                    anomalies.append({
                        'type': AnomalyType.PATTERN.value,
                        'severity': 'medium',
                        'score': unique_patterns / total_values,
                        'description': f'Column {col} has inconsistent patterns ({unique_patterns} patterns for {total_values} values)',
                        'affected_dimensions': [QualityDimension.CONSISTENCY.value, QualityDimension.CONFORMITY.value],
                        'column': col,
                        'pattern_diversity': unique_patterns / total_values
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing patterns for column {col}: {e}")
        
        return anomalies
    
    def _detect_schema_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect schema-related anomalies."""
        anomalies = []
        
        # Check for columns with mixed types
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Check if column contains mixed numeric and string data
                    sample_values = df[col].dropna().astype(str).head(100)
                    numeric_count = sum(1 for val in sample_values if self._is_numeric_string(val))
                    string_count = len(sample_values) - numeric_count
                    
                    if numeric_count > 0 and string_count > 0 and min(numeric_count, string_count) / len(sample_values) > 0.1:
                        anomalies.append({
                            'type': AnomalyType.SCHEMA.value,
                            'severity': 'high',
                            'score': min(numeric_count, string_count) / len(sample_values),
                            'description': f'Column {col} contains mixed data types',
                            'affected_dimensions': [QualityDimension.VALIDITY.value, QualityDimension.CONFORMITY.value],
                            'column': col,
                            'mixed_type_ratio': min(numeric_count, string_count) / len(sample_values)
                        })
                        
            except Exception as e:
                logger.error(f"Error checking schema for column {col}: {e}")
        
        return anomalies
    
    def _extract_pattern(self, value: str) -> str:
        """Extract pattern from string value."""
        if pd.isna(value) or value == '':
            return 'EMPTY'
        
        pattern = ''
        for char in str(value):
            if char.isalpha():
                pattern += 'A'
            elif char.isdigit():
                pattern += 'N'
            elif char.isspace():
                pattern += 'S'
            else:
                pattern += 'X'
        
        return pattern
    
    def _is_numeric_string(self, value: str) -> bool:
        """Check if string represents a numeric value."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


class MLQualityGateValidator:
    """ML-driven quality gate validation system."""
    
    def __init__(self, config: QualityGateConfig = None):
        """Initialize ML quality gate validator.
        
        Args:
            config: Quality gate configuration
        """
        self.config = config or QualityGateConfig()
        self.neural_predictor = NeuralQualityPredictor()
        self.anomaly_detector = AnomalyDetector()
        
        # Historical data for learning
        self.quality_history = deque(maxlen=1000)
        self.validation_history = deque(maxlen=500)
        
        # Adaptive thresholds
        self.adaptive_thresholds = self.config.dimension_thresholds.copy()
        
        logger.info("Initialized ML Quality Gate Validator")
    
    def train_from_historical_data(self, historical_data: List[Tuple[pd.DataFrame, float]]):
        """Train models from historical quality data."""
        logger.info(f"Training quality models from {len(historical_data)} historical samples")
        
        # Train neural predictor
        self.neural_predictor.train(historical_data)
        
        # Train anomaly detector
        reference_dataframes = [df for df, score in historical_data if score > 0.8]
        self.anomaly_detector.fit_baseline(reference_dataframes)
        
        logger.info("Completed training of quality models")
    
    def validate_quality(
        self, 
        df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame] = None
    ) -> QualityAssessment:
        """Perform comprehensive quality validation.
        
        Args:
            df: DataFrame to validate
            reference_df: Optional reference DataFrame for comparison
            
        Returns:
            Comprehensive quality assessment
        """
        start_time = time.time()
        
        try:
            # Initialize assessment components
            metrics = []
            dimension_scores = {}
            improvement_suggestions = []
            risk_factors = []
            
            # Calculate individual dimension scores
            for dimension in QualityDimension:
                score, confidence, details = self._assess_dimension(df, dimension, reference_df)
                
                metric = QualityMetric(
                    dimension=dimension,
                    score=score,
                    confidence=confidence,
                    methodology=f"ml_hybrid_{dimension.value}",
                    details=details
                )
                
                metrics.append(metric)
                dimension_scores[dimension] = score
            
            # ML-based overall prediction
            if self.config.enable_ml_prediction:
                ml_score, ml_confidence = self.neural_predictor.predict_quality(df)
                
                ml_metric = QualityMetric(
                    dimension=QualityDimension.ACCURACY,  # Use accuracy as placeholder
                    score=ml_score,
                    confidence=ml_confidence,
                    methodology="neural_network_prediction",
                    details={'model_type': 'neural_network', 'confidence': ml_confidence}
                )
                metrics.append(ml_metric)
            
            # Anomaly detection
            anomalies = []
            if self.config.enable_anomaly_detection:
                anomalies = self.anomaly_detector.detect_anomalies(df)
                
                # Add anomalies to relevant metrics
                for anomaly in anomalies:
                    affected_dimensions = anomaly.get('affected_dimensions', [])
                    for dim_name in affected_dimensions:
                        try:
                            dimension = QualityDimension(dim_name)
                            for metric in metrics:
                                if metric.dimension == dimension:
                                    metric.anomalies.append(anomaly)
                        except ValueError:
                            continue
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_scores, anomalies)
            
            # Statistical significance testing if reference provided
            statistical_significance = None
            if reference_df is not None:
                statistical_significance = self._test_statistical_significance(df, reference_df)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                dimension_scores, anomalies, df
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(dimension_scores, anomalies)
            
            # Determine gate status
            gate_status = self._determine_gate_status(overall_score, dimension_scores, anomalies)
            
            # Update adaptive thresholds if enabled
            if self.config.adaptive_thresholds:
                self._update_adaptive_thresholds(dimension_scores)
            
            # Record validation
            self.validation_history.append({
                'timestamp': time.time(),
                'overall_score': overall_score,
                'gate_status': gate_status.value,
                'anomaly_count': len(anomalies),
                'processing_time': time.time() - start_time
            })
            
            return QualityAssessment(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                metrics=metrics,
                gate_status=gate_status,
                improvement_suggestions=improvement_suggestions,
                risk_factors=risk_factors,
                statistical_significance=statistical_significance,
                metadata={
                    'processing_time': time.time() - start_time,
                    'anomaly_count': len(anomalies),
                    'ml_prediction_used': self.config.enable_ml_prediction,
                    'adaptive_thresholds_used': self.config.adaptive_thresholds
                }
            )
            
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            
            # Return minimal assessment on error
            return QualityAssessment(
                overall_score=0.0,
                dimension_scores={},
                metrics=[],
                gate_status=QualityGateStatus.FAILED,
                improvement_suggestions=["Review data for quality issues"],
                risk_factors=[f"Validation error: {str(e)}"],
                metadata={'error': str(e)}
            )
    
    def _assess_dimension(
        self, 
        df: pd.DataFrame, 
        dimension: QualityDimension,
        reference_df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Assess a specific quality dimension.
        
        Returns:
            Tuple of (score, confidence, details)
        """
        if dimension == QualityDimension.COMPLETENESS:
            return self._assess_completeness(df)
        elif dimension == QualityDimension.ACCURACY:
            return self._assess_accuracy(df, reference_df)
        elif dimension == QualityDimension.CONSISTENCY:
            return self._assess_consistency(df)
        elif dimension == QualityDimension.VALIDITY:
            return self._assess_validity(df)
        elif dimension == QualityDimension.UNIQUENESS:
            return self._assess_uniqueness(df)
        elif dimension == QualityDimension.TIMELINESS:
            return self._assess_timeliness(df)
        elif dimension == QualityDimension.CONFORMITY:
            return self._assess_conformity(df)
        else:
            return 0.5, 0.3, {'method': 'unknown_dimension'}
    
    def _assess_completeness(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data completeness."""
        if df.empty:
            return 0.0, 1.0, {'reason': 'empty_dataframe'}
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1.0 - (missing_cells / total_cells)
        
        # Column-wise completeness
        column_completeness = 1.0 - (df.isnull().sum() / len(df))
        completeness_variance = column_completeness.var()
        
        # Confidence based on data size and variance
        confidence = min(0.95, 0.5 + (df.shape[0] / 1000) * 0.3 + (1 - completeness_variance) * 0.2)
        
        details = {
            'missing_cells': missing_cells,
            'total_cells': total_cells,
            'completeness_ratio': completeness_ratio,
            'column_completeness_variance': completeness_variance,
            'worst_columns': column_completeness.nsmallest(3).to_dict()
        }
        
        return completeness_ratio, confidence, details
    
    def _assess_accuracy(
        self, 
        df: pd.DataFrame, 
        reference_df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data accuracy."""
        if reference_df is not None:
            # Compare with reference if available
            return self._compare_with_reference(df, reference_df)
        
        # Heuristic accuracy assessment
        accuracy_score = 0.8  # Default assumption
        confidence = 0.6
        
        # Check for obvious accuracy issues
        accuracy_issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check for impossible values (e.g., negative ages, extreme outliers)
                    z_scores = np.abs(stats.zscore(col_data))
                    extreme_outliers = (z_scores > 4).mean()
                    
                    if extreme_outliers > 0.05:  # More than 5% extreme outliers
                        accuracy_score *= 0.9
                        accuracy_issues.append(f"Column {col}: {extreme_outliers:.1%} extreme outliers")
                        
            except Exception:
                continue
        
        details = {
            'method': 'heuristic_accuracy',
            'issues_found': accuracy_issues,
            'reference_available': reference_df is not None
        }
        
        return accuracy_score, confidence, details
    
    def _assess_consistency(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data consistency."""
        consistency_scores = []
        details = {'consistency_checks': []}
        
        # Check format consistency in string columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                if df[col].dtype == 'object':
                    patterns = df[col].astype(str).apply(self._extract_pattern)
                    pattern_consistency = 1.0 - (patterns.nunique() / max(1, len(patterns)))
                    consistency_scores.append(pattern_consistency)
                    
                    details['consistency_checks'].append({
                        'column': col,
                        'pattern_consistency': pattern_consistency,
                        'unique_patterns': patterns.nunique()
                    })
                    
            except Exception:
                continue
        
        # Check value consistency (e.g., categorical values)
        for col in df.columns:
            try:
                if df[col].nunique() < 20:  # Likely categorical
                    value_counts = df[col].value_counts()
                    # Check for similar values that might be inconsistent
                    # This is a simplified check
                    consistency_scores.append(0.8)  # Placeholder
                    
            except Exception:
                continue
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.5
        confidence = min(0.9, len(consistency_scores) / 10.0)
        
        return overall_consistency, confidence, details
    
    def _assess_validity(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data validity."""
        validity_scores = []
        details = {'validity_checks': []}
        
        # Check data type validity
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Check if numeric columns are stored as strings
                    sample_values = df[col].dropna().astype(str).head(100)
                    numeric_ratio = sum(1 for val in sample_values if self._is_numeric_string(val)) / len(sample_values)
                    
                    if numeric_ratio > 0.8:  # Mostly numeric but stored as string
                        validity_scores.append(0.6)  # Reduced validity
                        details['validity_checks'].append({
                            'column': col,
                            'issue': 'numeric_stored_as_string',
                            'numeric_ratio': numeric_ratio
                        })
                    else:
                        validity_scores.append(0.9)
                else:
                    validity_scores.append(0.95)  # Proper data type
                    
            except Exception:
                validity_scores.append(0.5)
        
        overall_validity = np.mean(validity_scores) if validity_scores else 0.5
        confidence = 0.8
        
        return overall_validity, confidence, details
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data uniqueness."""
        if df.empty:
            return 1.0, 1.0, {'reason': 'empty_dataframe'}
        
        duplicate_ratio = df.duplicated().sum() / len(df)
        uniqueness_score = 1.0 - duplicate_ratio
        
        # Column-wise uniqueness analysis
        column_uniqueness = {}
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            column_uniqueness[col] = unique_ratio
        
        confidence = 0.9
        
        details = {
            'duplicate_rows': df.duplicated().sum(),
            'total_rows': len(df),
            'duplicate_ratio': duplicate_ratio,
            'column_uniqueness': column_uniqueness
        }
        
        return uniqueness_score, confidence, details
    
    def _assess_timeliness(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data timeliness."""
        # This is a placeholder implementation
        # In real scenarios, this would check data freshness, update timestamps, etc.
        timeliness_score = 0.8  # Default assumption
        confidence = 0.5  # Low confidence without specific temporal information
        
        details = {
            'method': 'placeholder',
            'note': 'Timeliness assessment requires temporal metadata'
        }
        
        return timeliness_score, confidence, details
    
    def _assess_conformity(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Assess data conformity to expected formats/standards."""
        conformity_scores = []
        details = {'conformity_checks': []}
        
        # Check common format conformity
        for col in df.select_dtypes(include=['object']).columns:
            try:
                sample_values = df[col].dropna().astype(str).head(100)
                
                # Email format check
                if 'email' in col.lower() or 'mail' in col.lower():
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    email_conformity = sum(1 for val in sample_values if pd.Series([val]).str.match(email_pattern).iloc[0]) / len(sample_values)
                    conformity_scores.append(email_conformity)
                    details['conformity_checks'].append({
                        'column': col,
                        'format': 'email',
                        'conformity': email_conformity
                    })
                
                # Phone format check
                elif 'phone' in col.lower() or 'tel' in col.lower():
                    # Simple phone pattern
                    phone_pattern = r'^[\+]?[0-9\s\-\(\)]{10,}$'
                    phone_conformity = sum(1 for val in sample_values if pd.Series([val]).str.match(phone_pattern).iloc[0]) / len(sample_values)
                    conformity_scores.append(phone_conformity)
                    details['conformity_checks'].append({
                        'column': col,
                        'format': 'phone',
                        'conformity': phone_conformity
                    })
                else:
                    conformity_scores.append(0.8)  # Default for other columns
                    
            except Exception:
                conformity_scores.append(0.5)
        
        overall_conformity = np.mean(conformity_scores) if conformity_scores else 0.7
        confidence = 0.7
        
        return overall_conformity, confidence, details
    
    def _compare_with_reference(
        self, 
        df: pd.DataFrame, 
        reference_df: pd.DataFrame
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compare DataFrame with reference for accuracy assessment."""
        if df.shape != reference_df.shape:
            return 0.5, 0.8, {'reason': 'shape_mismatch', 'current_shape': df.shape, 'reference_shape': reference_df.shape}
        
        # Column-wise comparison
        column_similarities = {}
        overall_similarities = []
        
        for col in df.columns:
            if col in reference_df.columns:
                try:
                    if df[col].dtype == reference_df[col].dtype:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # Numeric comparison
                            correlation = df[col].corr(reference_df[col])
                            column_similarities[col] = correlation if not pd.isna(correlation) else 0.0
                        else:
                            # Categorical comparison
                            exact_matches = (df[col] == reference_df[col]).mean()
                            column_similarities[col] = exact_matches
                    else:
                        column_similarities[col] = 0.5  # Type mismatch
                        
                    overall_similarities.append(column_similarities[col])
                    
                except Exception:
                    column_similarities[col] = 0.0
        
        accuracy_score = np.mean(overall_similarities) if overall_similarities else 0.0
        confidence = 0.9
        
        details = {
            'method': 'reference_comparison',
            'column_similarities': column_similarities,
            'average_similarity': accuracy_score
        }
        
        return accuracy_score, confidence, details
    
    def _extract_pattern(self, value: str) -> str:
        """Extract pattern from string value."""
        if pd.isna(value) or value == '':
            return 'EMPTY'
        
        pattern = ''
        for char in str(value):
            if char.isalpha():
                pattern += 'A'
            elif char.isdigit():
                pattern += 'N'
            elif char.isspace():
                pattern += 'S'
            else:
                pattern += 'X'
        
        return pattern
    
    def _is_numeric_string(self, value: str) -> bool:
        """Check if string represents a numeric value."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _calculate_overall_score(
        self, 
        dimension_scores: Dict[QualityDimension, float], 
        anomalies: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall quality score."""
        if not dimension_scores:
            return 0.0
        
        # Weighted average of dimension scores
        weights = {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.VALIDITY: 0.15,
            QualityDimension.UNIQUENESS: 0.10,
            QualityDimension.TIMELINESS: 0.05,
            QualityDimension.CONFORMITY: 0.05
        }
        
        weighted_score = sum(
            dimension_scores.get(dim, 0.5) * weight
            for dim, weight in weights.items()
        )
        
        # Apply anomaly penalty
        anomaly_penalty = 0.0
        for anomaly in anomalies:
            if anomaly.get('severity') == 'high':
                anomaly_penalty += 0.1
            elif anomaly.get('severity') == 'medium':
                anomaly_penalty += 0.05
            else:
                anomaly_penalty += 0.02
        
        final_score = max(0.0, weighted_score - anomaly_penalty)
        
        return final_score
    
    def _test_statistical_significance(
        self, 
        df: pd.DataFrame, 
        reference_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Test statistical significance of differences."""
        significance_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_df.columns and pd.api.types.is_numeric_dtype(reference_df[col]):
                try:
                    current_data = df[col].dropna()
                    reference_data = reference_df[col].dropna()
                    
                    if len(current_data) > 10 and len(reference_data) > 10:
                        # T-test for means
                        t_stat, p_value = stats.ttest_ind(current_data, reference_data)
                        significance_results[f'{col}_mean_difference'] = p_value
                        
                        # KS test for distribution
                        ks_stat, ks_p_value = stats.ks_2samp(current_data, reference_data)
                        significance_results[f'{col}_distribution_difference'] = ks_p_value
                        
                except Exception as e:
                    logger.error(f"Error in significance test for {col}: {e}")
        
        return significance_results
    
    def _generate_improvement_suggestions(
        self, 
        dimension_scores: Dict[QualityDimension, float],
        anomalies: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        # Dimension-based suggestions
        for dimension, score in dimension_scores.items():
            if score < 0.7:
                if dimension == QualityDimension.COMPLETENESS:
                    suggestions.append("Address missing values through imputation or data collection")
                elif dimension == QualityDimension.ACCURACY:
                    suggestions.append("Review data sources and validation processes")
                elif dimension == QualityDimension.CONSISTENCY:
                    suggestions.append("Standardize formats and implement data validation rules")
                elif dimension == QualityDimension.VALIDITY:
                    suggestions.append("Implement data type validation and constraint checking")
                elif dimension == QualityDimension.UNIQUENESS:
                    suggestions.append("Remove duplicates and implement uniqueness constraints")
        
        # Anomaly-based suggestions
        for anomaly in anomalies:
            if anomaly.get('severity') in ['high', 'medium']:
                if anomaly['type'] == AnomalyType.STATISTICAL.value:
                    suggestions.append("Investigate statistical outliers and unusual patterns")
                elif anomaly['type'] == AnomalyType.DISTRIBUTION.value:
                    suggestions.append(f"Review distribution of {anomaly.get('column', 'columns')}")
                elif anomaly['type'] == AnomalyType.PATTERN.value:
                    suggestions.append(f"Standardize patterns in {anomaly.get('column', 'string columns')}")
                elif anomaly['type'] == AnomalyType.SCHEMA.value:
                    suggestions.append("Resolve data type inconsistencies")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _identify_risk_factors(
        self, 
        dimension_scores: Dict[QualityDimension, float],
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify quality risk factors."""
        risk_factors = []
        
        # Critical dimension scores
        critical_dimensions = [dim for dim, score in dimension_scores.items() if score < 0.5]
        if critical_dimensions:
            risk_factors.append(f"Critical quality issues in: {', '.join(d.value for d in critical_dimensions)}")
        
        # High severity anomalies
        high_severity_anomalies = [a for a in anomalies if a.get('severity') == 'high']
        if high_severity_anomalies:
            risk_factors.append(f"{len(high_severity_anomalies)} high-severity anomalies detected")
        
        # Overall assessment
        overall_score = self._calculate_overall_score(dimension_scores, anomalies)
        if overall_score < 0.6:
            risk_factors.append("Overall quality score below acceptable threshold")
        
        return risk_factors
    
    def _determine_gate_status(
        self, 
        overall_score: float,
        dimension_scores: Dict[QualityDimension, float],
        anomalies: List[Dict[str, Any]]
    ) -> QualityGateStatus:
        """Determine quality gate status."""
        # Check overall threshold
        if overall_score < self.config.minimum_overall_score:
            return QualityGateStatus.FAILED
        
        # Check dimension thresholds
        for dimension, threshold in self.adaptive_thresholds.items():
            if dimension_scores.get(dimension, 0.0) < threshold:
                return QualityGateStatus.FAILED
        
        # Check for critical anomalies
        critical_anomalies = [a for a in anomalies if a.get('severity') == 'high']
        if len(critical_anomalies) > 2:
            return QualityGateStatus.FAILED
        
        # Warning conditions
        warning_conditions = [
            overall_score < self.config.minimum_overall_score + 0.1,
            len(anomalies) > 5,
            any(score < 0.7 for score in dimension_scores.values())
        ]
        
        if any(warning_conditions):
            return QualityGateStatus.WARNING
        
        return QualityGateStatus.PASSED
    
    def _update_adaptive_thresholds(self, dimension_scores: Dict[QualityDimension, float]):
        """Update adaptive thresholds based on recent performance."""
        # Simple adaptive mechanism - in practice would be more sophisticated
        for dimension, score in dimension_scores.items():
            if dimension not in self.adaptive_thresholds:
                self.adaptive_thresholds[dimension] = 0.7
            
            # Gradually adjust thresholds based on recent performance
            current_threshold = self.adaptive_thresholds[dimension]
            adjustment = (score - current_threshold) * 0.01  # Small adjustment
            self.adaptive_thresholds[dimension] = max(0.5, min(0.9, current_threshold + adjustment))
    
    def get_validation_analytics(self) -> Dict[str, Any]:
        """Get validation analytics and performance metrics."""
        if not self.validation_history:
            return {}
        
        df = pd.DataFrame(list(self.validation_history))
        
        return {
            'total_validations': len(self.validation_history),
            'average_score': df['overall_score'].mean(),
            'score_trend': df['overall_score'].tail(10).tolist(),
            'gate_pass_rate': (df['gate_status'] == 'passed').mean(),
            'average_processing_time': df['processing_time'].mean(),
            'anomaly_detection_rate': (df['anomaly_count'] > 0).mean(),
            'recent_performance': {
                'last_24h_validations': len(df[df['timestamp'] > time.time() - 86400]),
                'last_24h_avg_score': df[df['timestamp'] > time.time() - 86400]['overall_score'].mean() if len(df) > 0 else 0
            },
            'model_status': {
                'neural_predictor_trained': self.neural_predictor.is_trained,
                'anomaly_detector_trained': self.anomaly_detector.trained,
                'adaptive_thresholds_active': self.config.adaptive_thresholds
            }
        }


# Global validator instance
_global_validator: Optional[MLQualityGateValidator] = None


def get_global_validator() -> MLQualityGateValidator:
    """Get global ML quality gate validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = MLQualityGateValidator()
    return _global_validator


def initialize_quality_gates(
    config: QualityGateConfig = None,
    historical_data: List[Tuple[pd.DataFrame, float]] = None
) -> MLQualityGateValidator:
    """Initialize ML-driven quality gates."""
    global _global_validator
    
    _global_validator = MLQualityGateValidator(config)
    
    if historical_data:
        _global_validator.train_from_historical_data(historical_data)
    
    logger.info("Initialized ML-driven quality gates")
    return _global_validator


if __name__ == "__main__":
    # Demo ML quality gates
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': range(1000),
        'name': [f'User_{i}' for i in range(1000)],
        'age': np.random.randint(18, 80, 1000),
        'email': [f'user{i}@example.com' for i in range(1000)],
        'score': np.random.normal(75, 15, 1000)
    })
    
    # Add some quality issues
    sample_data.loc[50:60, 'name'] = None  # Missing values
    sample_data.loc[100:110, :] = sample_data.loc[100:110, :]  # Duplicates
    sample_data.loc[200:205, 'email'] = 'invalid_email'  # Invalid emails
    
    # Initialize validator
    validator = initialize_quality_gates()
    
    # Validate quality
    assessment = validator.validate_quality(sample_data)
    
    print("Quality Assessment Results:")
    print(f"Overall Score: {assessment.overall_score:.3f}")
    print(f"Gate Status: {assessment.gate_status.value}")
    print(f"Dimension Scores:")
    for dim, score in assessment.dimension_scores.items():
        print(f"  {dim.value}: {score:.3f}")
    
    print(f"\nImprovement Suggestions:")
    for suggestion in assessment.improvement_suggestions:
        print(f"  - {suggestion}")
    
    print(f"\nRisk Factors:")
    for risk in assessment.risk_factors:
        print(f"  - {risk}")
    
    # Analytics
    analytics = validator.get_validation_analytics()
    print(f"\nValidation Analytics:")
    print(json.dumps(analytics, indent=2, default=str))