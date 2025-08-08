"""Confidence calibration for cleaning predictions."""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Metrics for confidence calibration assessment."""
    brier_score: float
    log_loss: float
    reliability: float
    sharpness: float
    calibration_bins: List[Tuple[float, float, int]]  # (bin_center, accuracy, count)


class ConfidenceCalibrator:
    """Advanced confidence calibration for cleaning predictions."""
    
    def __init__(
        self, 
        method: str = "isotonic",
        n_bins: int = 10,
        save_path: Optional[str] = None
    ):
        """Initialize confidence calibrator.
        
        Args:
            method: Calibration method ("isotonic", "sigmoid", "histogram")
            n_bins: Number of bins for calibration assessment
            save_path: Path to save/load calibration model
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Using basic confidence calibration.")
        
        self.method = method
        self.n_bins = n_bins
        self.save_path = save_path
        self.is_fitted = False
        
        # Calibration models
        self._calibrator = None
        self._feature_calibrators = {}  # Per-feature calibration
        
        # Calibration data for assessment
        self._calibration_history = []
        self._bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
    def fit(
        self, 
        predictions: List[Dict[str, Any]], 
        ground_truth: List[bool],
        features: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Fit calibrator on labeled data.
        
        Args:
            predictions: List of prediction dicts with 'confidence' and optionally other features
            ground_truth: True binary labels (True = correct prediction, False = incorrect)
            features: Optional additional features for calibration (data_type, column_name, etc.)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if len(predictions) < 10:
            logger.warning("Very few calibration samples, results may be unreliable")
        
        # Extract confidence scores
        confidences = np.array([pred.get('confidence', 0.5) for pred in predictions])
        labels = np.array(ground_truth, dtype=bool)
        
        # Fit main calibrator
        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds='clip')
            self._calibrator.fit(confidences, labels)
        elif self.method == "sigmoid":
            # Use logistic regression for Platt scaling
            self._calibrator = LogisticRegression()
            self._calibrator.fit(confidences.reshape(-1, 1), labels)
        elif self.method == "histogram":
            # Histogram-based calibration
            self._fit_histogram_calibrator(confidences, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Fit feature-specific calibrators if features provided
        if features:
            self._fit_feature_calibrators(confidences, labels, features)
        
        # Store calibration history
        self._calibration_history = list(zip(confidences, labels))
        
        self.is_fitted = True
        logger.info(f"Fitted {self.method} calibrator on {len(predictions)} samples")
        
        # Save if path provided
        if self.save_path:
            self.save(self.save_path)
    
    def calibrate(
        self, 
        confidence: float, 
        features: Optional[Dict[str, Any]] = None
    ) -> float:
        """Apply calibration to raw confidence score.
        
        Args:
            confidence: Raw confidence score from model
            features: Optional features for context-aware calibration
            
        Returns:
            Calibrated confidence score
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw confidence")
            return confidence
        
        # Clip confidence to valid range
        confidence = np.clip(confidence, 0.001, 0.999)
        
        # Apply main calibration
        if self.method == "isotonic":
            calibrated = float(self._calibrator.predict([confidence])[0])
        elif self.method == "sigmoid":
            calibrated = float(self._calibrator.predict_proba([[confidence]])[0, 1])
        elif self.method == "histogram":
            calibrated = self._apply_histogram_calibration(confidence)
        else:
            calibrated = confidence
        
        # Apply feature-specific calibration if available
        if features and self._feature_calibrators:
            calibrated = self._apply_feature_calibration(calibrated, features)
        
        return np.clip(calibrated, 0.0, 1.0)
    
    def assess_calibration(
        self, 
        predictions: List[float], 
        ground_truth: List[bool]
    ) -> CalibrationMetrics:
        """Assess calibration quality with various metrics.
        
        Args:
            predictions: Calibrated confidence scores
            ground_truth: True binary labels
            
        Returns:
            Calibration quality metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth, dtype=bool)
        
        # Calculate Brier score (lower is better)
        brier = brier_score_loss(ground_truth, predictions)
        
        # Calculate log loss (lower is better)
        # Clip predictions to avoid log(0)
        clipped_preds = np.clip(predictions, 1e-15, 1 - 1e-15)
        logloss = log_loss(ground_truth, clipped_preds)
        
        # Calculate reliability and sharpness
        reliability, sharpness, calibration_bins = self._calculate_reliability_sharpness(
            predictions, ground_truth
        )
        
        return CalibrationMetrics(
            brier_score=brier,
            log_loss=logloss,
            reliability=reliability,
            sharpness=sharpness,
            calibration_bins=calibration_bins
        )
    
    def get_calibration_curve(
        self, 
        predictions: List[float], 
        ground_truth: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibration curve for plotting.
        
        Args:
            predictions: Confidence scores
            ground_truth: True binary labels
            
        Returns:
            Tuple of (mean_predicted_probability, fraction_of_positives)
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth, dtype=bool)
        
        # Create bins
        bin_indices = np.digitize(predictions, self._bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        mean_predicted = []
        fraction_positive = []
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_predicted.append(predictions[mask].mean())
                fraction_positive.append(ground_truth[mask].mean())
            else:
                mean_predicted.append((self._bin_boundaries[i] + self._bin_boundaries[i+1]) / 2)
                fraction_positive.append(0.0)
        
        return np.array(mean_predicted), np.array(fraction_positive)
    
    def save(self, path: str) -> None:
        """Save calibrator to file."""
        save_data = {
            'method': self.method,
            'n_bins': self.n_bins,
            'is_fitted': self.is_fitted,
            'calibrator': self._calibrator,
            'feature_calibrators': self._feature_calibrators,
            'calibration_history': self._calibration_history,
            'bin_boundaries': self._bin_boundaries
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved calibrator to {path}")
    
    def load(self, path: str) -> None:
        """Load calibrator from file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.method = save_data['method']
        self.n_bins = save_data['n_bins']
        self.is_fitted = save_data['is_fitted']
        self._calibrator = save_data['calibrator']
        self._feature_calibrators = save_data['feature_calibrators']
        self._calibration_history = save_data['calibration_history']
        self._bin_boundaries = save_data['bin_boundaries']
        
        logger.info(f"Loaded calibrator from {path}")
    
    def _fit_histogram_calibrator(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit histogram-based calibrator."""
        bin_indices = np.digitize(confidences, self._bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        self._histogram_calibration = {}
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                accuracy = labels[mask].mean()
                self._histogram_calibration[i] = accuracy
            else:
                # Use bin center as default
                self._histogram_calibration[i] = (self._bin_boundaries[i] + self._bin_boundaries[i+1]) / 2
    
    def _apply_histogram_calibration(self, confidence: float) -> float:
        """Apply histogram-based calibration."""
        bin_idx = np.digitize([confidence], self._bin_boundaries)[0] - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
        return self._histogram_calibration.get(bin_idx, confidence)
    
    def _fit_feature_calibrators(
        self, 
        confidences: np.ndarray, 
        labels: np.ndarray, 
        features: List[Dict[str, Any]]
    ) -> None:
        """Fit calibrators for specific features."""
        # Group by data type
        data_type_groups = {}
        for i, feat in enumerate(features):
            data_type = feat.get('data_type', 'unknown')
            if data_type not in data_type_groups:
                data_type_groups[data_type] = {'confidences': [], 'labels': []}
            
            data_type_groups[data_type]['confidences'].append(confidences[i])
            data_type_groups[data_type]['labels'].append(labels[i])
        
        # Fit calibrator for each data type with enough samples
        for data_type, group_data in data_type_groups.items():
            if len(group_data['confidences']) >= 5:  # Minimum samples
                group_confidences = np.array(group_data['confidences'])
                group_labels = np.array(group_data['labels'])
                
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(group_confidences, group_labels)
                
                self._feature_calibrators[f'data_type_{data_type}'] = calibrator
    
    def _apply_feature_calibration(
        self, 
        confidence: float, 
        features: Dict[str, Any]
    ) -> float:
        """Apply feature-specific calibration."""
        data_type = features.get('data_type', 'unknown')
        calibrator_key = f'data_type_{data_type}'
        
        if calibrator_key in self._feature_calibrators:
            calibrator = self._feature_calibrators[calibrator_key]
            feature_calibrated = float(calibrator.predict([confidence])[0])
            
            # Blend with main calibration (weighted average)
            return 0.7 * confidence + 0.3 * feature_calibrated
        
        return confidence
    
    def _calculate_reliability_sharpness(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray
    ) -> Tuple[float, float, List[Tuple[float, float, int]]]:
        """Calculate reliability and sharpness metrics."""
        bin_indices = np.digitize(predictions, self._bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        reliability = 0.0
        sharpness = 0.0
        calibration_bins = []
        total_count = len(predictions)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            bin_count = mask.sum()
            
            if bin_count > 0:
                bin_predictions = predictions[mask]
                bin_labels = ground_truth[mask]
                
                bin_accuracy = bin_labels.mean()
                bin_confidence = bin_predictions.mean()
                
                # Reliability: weighted squared difference between confidence and accuracy
                reliability += (bin_count / total_count) * (bin_confidence - bin_accuracy) ** 2
                
                # Sharpness: variance of predictions
                sharpness += (bin_count / total_count) * bin_predictions.var()
                
                calibration_bins.append((bin_confidence, bin_accuracy, bin_count))
            else:
                bin_center = (self._bin_boundaries[i] + self._bin_boundaries[i+1]) / 2
                calibration_bins.append((bin_center, 0.0, 0))
        
        return reliability, sharpness, calibration_bins


def create_ensemble_calibrator(
    calibrators: List[ConfidenceCalibrator],
    weights: Optional[List[float]] = None
) -> "EnsembleCalibrator":
    """Create ensemble of multiple calibrators."""
    return EnsembleCalibrator(calibrators, weights)


class EnsembleCalibrator:
    """Ensemble of multiple confidence calibrators."""
    
    def __init__(
        self, 
        calibrators: List[ConfidenceCalibrator],
        weights: Optional[List[float]] = None
    ):
        """Initialize ensemble calibrator.
        
        Args:
            calibrators: List of fitted calibrators
            weights: Optional weights for ensemble (defaults to uniform)
        """
        self.calibrators = calibrators
        self.weights = weights or [1.0 / len(calibrators)] * len(calibrators)
        
        if len(self.weights) != len(calibrators):
            raise ValueError("Number of weights must match number of calibrators")
        
        if not all(cal.is_fitted for cal in calibrators):
            raise ValueError("All calibrators must be fitted")
    
    def calibrate(
        self, 
        confidence: float, 
        features: Optional[Dict[str, Any]] = None
    ) -> float:
        """Apply ensemble calibration."""
        calibrated_scores = []
        
        for calibrator in self.calibrators:
            calibrated = calibrator.calibrate(confidence, features)
            calibrated_scores.append(calibrated)
        
        # Weighted average
        ensemble_score = sum(
            w * score for w, score in zip(self.weights, calibrated_scores)
        )
        
        return np.clip(ensemble_score, 0.0, 1.0)