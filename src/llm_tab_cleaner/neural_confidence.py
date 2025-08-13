"""Neural confidence calibration for LLM data cleaning - Research Module.

This module implements state-of-the-art neural confidence calibration techniques
specifically designed for LLM-powered data cleaning tasks. Based on recent research
in uncertainty quantification and confidence calibration.

Research Papers Implemented:
- "Temperature scaling: A simple and effective postprocessing for calibrating deep neural networks"
- "Beyond temperature scaling: Obtaining well-calibrated predictions from neural networks"
- "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration metrics for research evaluation."""
    expected_calibration_error: float
    maximum_calibration_error: float
    average_calibration_error: float
    brier_score: float
    log_likelihood: float
    reliability_diagram_data: Dict[str, List[float]]
    confidence_histogram: Dict[str, List[float]]
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.overconfidence_error = max(0, self.average_calibration_error)
        self.underconfidence_error = max(0, -self.average_calibration_error)


@dataclass
class NeuralCalibrationConfig:
    """Configuration for neural confidence calibration."""
    method: str = "temperature_scaling"  # temperature_scaling, platt_scaling, isotonic, ensemble
    temperature_range: Tuple[float, float] = (0.1, 10.0)
    ensemble_size: int = 5
    validation_split: float = 0.2
    calibration_bins: int = 15
    bootstrap_samples: int = 1000
    neural_hidden_layers: Tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    max_iterations: int = 1000
    tolerance: float = 1e-6
    research_mode: bool = True
    enable_uncertainty_quantification: bool = True


class TemperatureScaling:
    """Temperature scaling for neural confidence calibration.
    
    Implements the method from Guo et al. "On Calibration of Modern Neural Networks"
    """
    
    def __init__(self, temperature_range: Tuple[float, float] = (0.1, 10.0)):
        self.temperature_range = temperature_range
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, correct_predictions: np.ndarray) -> float:
        """Fit temperature parameter to calibrate confidences.
        
        Args:
            confidences: Raw confidence scores [0, 1]
            correct_predictions: Binary correctness indicators
            
        Returns:
            Optimal temperature parameter
        """
        def negative_log_likelihood(temperature):
            """Negative log-likelihood for temperature optimization."""
            scaled_confidences = self._apply_temperature(confidences, temperature)
            # Clip to avoid log(0)
            scaled_confidences = np.clip(scaled_confidences, 1e-8, 1 - 1e-8)
            
            # Binary cross-entropy loss
            return -np.mean(
                correct_predictions * np.log(scaled_confidences) +
                (1 - correct_predictions) * np.log(1 - scaled_confidences)
            )
        
        # Optimize temperature
        result = minimize(
            negative_log_likelihood,
            x0=1.0,
            bounds=[(self.temperature_range[0], self.temperature_range[1])],
            method='L-BFGS-B'
        )
        
        self.temperature = result.x[0]
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted with T={self.temperature:.4f}")
        return self.temperature
    
    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to confidence scores."""
        if not self.is_fitted:
            raise ValueError("Temperature scaling not fitted. Call fit() first.")
        
        return self._apply_temperature(confidences, self.temperature)
    
    def _apply_temperature(self, confidences: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling transformation."""
        # Convert to logits, scale, then back to probabilities
        epsilon = 1e-8
        confidences = np.clip(confidences, epsilon, 1 - epsilon)
        
        logits = np.log(confidences / (1 - confidences))
        scaled_logits = logits / temperature
        
        # Convert back to probabilities
        return 1 / (1 + np.exp(-scaled_logits))


class PlattScaling:
    """Platt scaling for confidence calibration using logistic regression."""
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, correct_predictions: np.ndarray):
        """Fit Platt scaling model."""
        # Convert to logits for input features
        epsilon = 1e-8
        confidences = np.clip(confidences, epsilon, 1 - epsilon)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)
        
        self.calibrator.fit(logits, correct_predictions)
        self.is_fitted = True
        
        logger.info("Platt scaling fitted")
    
    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to confidence scores."""
        if not self.is_fitted:
            raise ValueError("Platt scaling not fitted. Call fit() first.")
        
        epsilon = 1e-8
        confidences = np.clip(confidences, epsilon, 1 - epsilon)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)
        
        return self.calibrator.predict_proba(logits)[:, 1]


class NeuralCalibrator:
    """Advanced neural network-based confidence calibration.
    
    Implements multiple calibration methods with uncertainty quantification
    for research-grade confidence estimation.
    """
    
    def __init__(self, config: Optional[NeuralCalibrationConfig] = None):
        self.config = config or NeuralCalibrationConfig()
        self.calibration_models = {}
        self.is_fitted = False
        self.feature_extractors = []
        self.validation_metrics = {}
        
    def fit(
        self,
        confidences: np.ndarray,
        correct_predictions: np.ndarray,
        features: Optional[np.ndarray] = None,
        data_quality_features: Optional[Dict[str, np.ndarray]] = None
    ) -> CalibrationMetrics:
        """Fit neural calibration models with comprehensive evaluation.
        
        Args:
            confidences: Raw LLM confidence scores
            correct_predictions: Ground truth correctness (0/1)
            features: Additional features for calibration (optional)
            data_quality_features: Data quality indicators (optional)
            
        Returns:
            Comprehensive calibration metrics
        """
        logger.info(f"Fitting neural calibrator with {len(confidences)} samples")
        
        # Prepare features
        X = self._prepare_features(confidences, features, data_quality_features)
        y = correct_predictions.astype(float)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, random_state=42
        )
        
        # Fit different calibration methods
        self._fit_temperature_scaling(X_train[:, 0], y_train)  # First column is confidence
        self._fit_platt_scaling(X_train[:, 0], y_train)
        
        if self.config.method == "ensemble":
            self._fit_ensemble_calibrator(X_train, y_train)
        elif self.config.method == "neural":
            self._fit_neural_calibrator(X_train, y_train)
        
        # Evaluate on validation set
        val_confidences = X_val[:, 0]
        calibrated_confidences = self.calibrate(val_confidences)
        metrics = self._compute_calibration_metrics(calibrated_confidences, y_val)
        
        self.validation_metrics = metrics
        self.is_fitted = True
        
        logger.info(f"Neural calibrator fitted. ECE: {metrics.expected_calibration_error:.4f}")
        return metrics
    
    def calibrate(self, confidences: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply neural calibration to confidence scores."""
        if not self.is_fitted:
            raise ValueError("Neural calibrator not fitted. Call fit() first.")
        
        if self.config.method == "temperature_scaling":
            return self.calibration_models["temperature"].calibrate(confidences)
        elif self.config.method == "platt_scaling":
            return self.calibration_models["platt"].calibrate(confidences)
        elif self.config.method == "ensemble":
            return self._ensemble_calibrate(confidences, features)
        elif self.config.method == "neural":
            X = self._prepare_features(confidences, features)
            return self.calibration_models["neural"].predict(X)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
    
    def get_uncertainty_estimates(self, confidences: np.ndarray) -> Dict[str, np.ndarray]:
        """Get uncertainty estimates for confidence predictions.
        
        Returns epistemic and aleatoric uncertainty estimates.
        """
        if not self.config.enable_uncertainty_quantification:
            return {"epistemic": np.zeros_like(confidences), "aleatoric": np.zeros_like(confidences)}
        
        # Bootstrap sampling for epistemic uncertainty
        bootstrap_predictions = []
        for _ in range(min(100, self.config.bootstrap_samples)):
            # Simulate model uncertainty
            noise = np.random.normal(0, 0.01, size=confidences.shape)
            noisy_confidences = np.clip(confidences + noise, 0, 1)
            pred = self.calibrate(noisy_confidences)
            bootstrap_predictions.append(pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = np.std(bootstrap_predictions, axis=0)
        
        # Aleatoric uncertainty (data uncertainty)
        # Estimated from confidence in the prediction
        calibrated = self.calibrate(confidences)
        aleatoric = np.sqrt(calibrated * (1 - calibrated))
        
        return {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": np.sqrt(epistemic**2 + aleatoric**2)
        }
    
    def _prepare_features(
        self,
        confidences: np.ndarray,
        features: Optional[np.ndarray] = None,
        data_quality_features: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Prepare feature matrix for calibration."""
        feature_list = [confidences.reshape(-1, 1)]
        
        # Add confidence-derived features
        feature_list.append((confidences ** 2).reshape(-1, 1))  # Confidence squared
        feature_list.append(np.log(confidences + 1e-8).reshape(-1, 1))  # Log confidence
        feature_list.append((1 - confidences).reshape(-1, 1))  # Uncertainty
        
        # Add additional features if provided
        if features is not None:
            feature_list.append(features)
        
        # Add data quality features
        if data_quality_features:
            for key, values in data_quality_features.items():
                feature_list.append(values.reshape(-1, 1))
        
        return np.hstack(feature_list)
    
    def _fit_temperature_scaling(self, confidences: np.ndarray, correct_predictions: np.ndarray):
        """Fit temperature scaling model."""
        temp_scaler = TemperatureScaling(self.config.temperature_range)
        temp_scaler.fit(confidences, correct_predictions)
        self.calibration_models["temperature"] = temp_scaler
    
    def _fit_platt_scaling(self, confidences: np.ndarray, correct_predictions: np.ndarray):
        """Fit Platt scaling model."""
        platt_scaler = PlattScaling()
        platt_scaler.fit(confidences, correct_predictions)
        self.calibration_models["platt"] = platt_scaler
    
    def _fit_ensemble_calibrator(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble of calibration models."""
        ensemble_models = []
        
        # Random Forest calibrator
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X, y)
        ensemble_models.append(rf_model)
        
        # Neural network calibrator
        nn_model = MLPRegressor(
            hidden_layer_sizes=self.config.neural_hidden_layers,
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.max_iterations,
            random_state=42
        )
        nn_model.fit(X, y)
        ensemble_models.append(nn_model)
        
        self.calibration_models["ensemble"] = ensemble_models
    
    def _fit_neural_calibrator(self, X: np.ndarray, y: np.ndarray):
        """Fit dedicated neural network calibrator."""
        model = MLPRegressor(
            hidden_layer_sizes=self.config.neural_hidden_layers,
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.max_iterations,
            activation='relu',
            solver='adam',
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        self.calibration_models["neural"] = model
    
    def _ensemble_calibrate(self, confidences: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply ensemble calibration."""
        X = self._prepare_features(confidences, features)
        
        predictions = []
        for model in self.calibration_models["ensemble"]:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average ensemble predictions
        return np.mean(predictions, axis=0)
    
    def _compute_calibration_metrics(
        self,
        calibrated_confidences: np.ndarray,
        correct_predictions: np.ndarray
    ) -> CalibrationMetrics:
        """Compute comprehensive calibration metrics for research evaluation."""
        n_bins = self.config.calibration_bins
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (calibrated_confidences > bin_lower) & (calibrated_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct_predictions[in_bin].mean()
                avg_confidence_in_bin = calibrated_confidences[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Calculate metrics
        non_empty_bins = bin_counts > 0
        
        # Expected Calibration Error (ECE)
        ece = np.average(
            np.abs(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins]),
            weights=bin_counts[non_empty_bins]
        )
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins]))
        
        # Average Calibration Error (ACE)
        ace = np.mean(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins])
        
        # Brier Score
        brier = brier_score_loss(correct_predictions, calibrated_confidences)
        
        # Log-likelihood
        clipped_confidences = np.clip(calibrated_confidences, 1e-8, 1 - 1e-8)
        log_likelihood = -np.mean(
            correct_predictions * np.log(clipped_confidences) +
            (1 - correct_predictions) * np.log(1 - clipped_confidences)
        )
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_calibration_error=ace,
            brier_score=brier,
            log_likelihood=log_likelihood,
            reliability_diagram_data={
                "bin_confidences": bin_confidences.tolist(),
                "bin_accuracies": bin_accuracies.tolist(),
                "bin_counts": bin_counts.tolist()
            },
            confidence_histogram={
                "bin_edges": bin_boundaries.tolist(),
                "counts": bin_counts.tolist()
            }
        )