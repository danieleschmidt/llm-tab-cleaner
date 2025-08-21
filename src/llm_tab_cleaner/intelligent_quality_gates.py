"""Intelligent ML-powered quality gates with adaptive thresholds."""

import logging
import time
import json
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from pathlib import Path
import statistics

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    HAS_ML = True
except ImportError:
    HAS_ML = False

logger = logging.getLogger(__name__)


class QualityGateResult(Enum):
    """Quality gate evaluation results."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class GateType(Enum):
    """Types of quality gates."""
    THRESHOLD = "threshold"
    TREND = "trend"
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"
    COMPOSITE = "composite"


@dataclass
class QualityMetric:
    """A quality metric measurement."""
    name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class QualityGateConfig:
    """Configuration for a quality gate."""
    name: str
    gate_type: GateType
    metric_name: str
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    trend_window: int = 10
    statistical_confidence: float = 0.95
    ml_model_path: Optional[str] = None
    adaptive: bool = True
    weight: float = 1.0
    enabled: bool = True
    custom_evaluator: Optional[Callable] = None


@dataclass
class GateEvaluationResult:
    """Result of a quality gate evaluation."""
    gate_name: str
    result: QualityGateResult
    score: float
    threshold_used: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on historical data."""
    
    def __init__(self, learning_rate: float = 0.1, min_samples: int = 20):
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.threshold_history: Dict[str, List[float]] = {}
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def update_threshold(self, gate_name: str, metric_value: float, quality_outcome: bool):
        """Update adaptive threshold based on outcome feedback."""
        with self._lock:
            if gate_name not in self.threshold_history:
                self.threshold_history[gate_name] = []
                self.adaptive_thresholds[gate_name] = {"min": 0.0, "max": 1.0}
            
            self.threshold_history[gate_name].append(metric_value)
            
            # Keep only recent history
            if len(self.threshold_history[gate_name]) > 1000:
                self.threshold_history[gate_name] = self.threshold_history[gate_name][-1000:]
            
            if len(self.threshold_history[gate_name]) >= self.min_samples:
                self._recalculate_threshold(gate_name, quality_outcome)
    
    def _recalculate_threshold(self, gate_name: str, quality_outcome: bool):
        """Recalculate adaptive threshold using statistical analysis."""
        values = self.threshold_history[gate_name]
        
        if len(values) < self.min_samples:
            return
        
        # Calculate statistical thresholds
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Percentile-based thresholds
        p25 = np.percentile(values, 25) if HAS_ML else min(values)
        p75 = np.percentile(values, 75) if HAS_ML else max(values)
        p95 = np.percentile(values, 95) if HAS_ML else max(values)
        p05 = np.percentile(values, 5) if HAS_ML else min(values)
        
        # Adaptive adjustment based on recent outcomes
        if quality_outcome:
            # Good outcome - slightly relax thresholds
            new_min = max(p05, mean_val - 2 * std_val) * (1 - self.learning_rate * 0.1)
            new_max = min(p95, mean_val + 2 * std_val) * (1 + self.learning_rate * 0.1)
        else:
            # Poor outcome - tighten thresholds
            new_min = max(p25, mean_val - std_val) * (1 + self.learning_rate * 0.1)
            new_max = min(p75, mean_val + std_val) * (1 - self.learning_rate * 0.1)
        
        # Update adaptive thresholds
        current = self.adaptive_thresholds[gate_name]
        current["min"] = current["min"] * (1 - self.learning_rate) + new_min * self.learning_rate
        current["max"] = current["max"] * (1 - self.learning_rate) + new_max * self.learning_rate
        
        logger.debug(f"Updated adaptive thresholds for {gate_name}: min={current['min']:.3f}, max={current['max']:.3f}")
    
    def get_threshold(self, gate_name: str, threshold_type: str = "max") -> Optional[float]:
        """Get current adaptive threshold."""
        with self._lock:
            if gate_name in self.adaptive_thresholds:
                return self.adaptive_thresholds[gate_name].get(threshold_type)
        return None


class MLQualityPredictor:
    """ML-based quality prediction and anomaly detection."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, List[Tuple]] = {}
        self.model_performance: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def train_model(self, gate_name: str, features: np.ndarray, targets: np.ndarray) -> bool:
        """Train ML model for quality prediction."""
        if not HAS_ML:
            logger.warning("ML dependencies not available")
            return False
        
        try:
            with self._lock:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features, targets, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model based on target type
                if len(np.unique(targets)) <= 2:
                    # Classification (pass/fail)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    performance = accuracy_score(y_test, y_pred)
                else:
                    # Regression (quality score)
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    performance = 1.0 - mean_squared_error(y_test, y_pred)  # Convert to score
                
                # Store model and scaler
                self.models[gate_name] = model
                self.scalers[gate_name] = scaler
                self.model_performance[gate_name] = performance
                
                logger.info(f"Trained ML model for {gate_name} with performance: {performance:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to train ML model for {gate_name}: {e}")
            return False
    
    def predict_quality(self, gate_name: str, features: np.ndarray) -> Optional[Tuple[float, float]]:
        """Predict quality using trained ML model."""
        if not HAS_ML or gate_name not in self.models:
            return None
        
        try:
            with self._lock:
                model = self.models[gate_name]
                scaler = self.scalers[gate_name]
                
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    # Classification - return probability of positive class
                    proba = model.predict_proba(features_scaled)[0]
                    prediction = proba[1] if len(proba) > 1 else proba[0]
                    confidence = max(proba)
                else:
                    # Regression - return predicted score
                    prediction = model.predict(features_scaled)[0]
                    confidence = min(1.0, abs(prediction))  # Simple confidence estimate
                
                return prediction, confidence
                
        except Exception as e:
            logger.error(f"Error predicting quality for {gate_name}: {e}")
            return None
    
    def add_training_sample(self, gate_name: str, features: np.ndarray, target: float):
        """Add training sample for online learning."""
        with self._lock:
            if gate_name not in self.training_data:
                self.training_data[gate_name] = []
            
            self.training_data[gate_name].append((features.copy(), target))
            
            # Retrain if enough new samples
            if len(self.training_data[gate_name]) % 100 == 0:
                logger.info(f"Retraining model for {gate_name} with {len(self.training_data[gate_name])} samples")
                self._retrain_model(gate_name)
    
    def _retrain_model(self, gate_name: str):
        """Retrain model with accumulated data."""
        if gate_name not in self.training_data:
            return
        
        samples = self.training_data[gate_name]
        if len(samples) < 20:  # Need minimum samples
            return
        
        features = np.array([sample[0] for sample in samples])
        targets = np.array([sample[1] for sample in samples])
        
        self.train_model(gate_name, features, targets)


class IntelligentQualityGateSystem:
    """Intelligent quality gate system with ML-powered adaptive thresholds."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.gates: Dict[str, QualityGateConfig] = {}
        self.metric_history: Dict[str, List[QualityMetric]] = {}
        self.evaluation_history: List[GateEvaluationResult] = []
        
        # Advanced components
        self.threshold_manager = AdaptiveThresholdManager()
        self.ml_predictor = MLQualityPredictor()
        
        # Performance tracking
        self.gate_performance: Dict[str, List[float]] = {}
        
        self._lock = threading.Lock()
        
        # Load configuration if provided
        if config_path:
            self.load_configuration(config_path)
        else:
            self._initialize_default_gates()
    
    def _initialize_default_gates(self):
        """Initialize default quality gates."""
        default_gates = [
            QualityGateConfig(
                name="data_quality_score",
                gate_type=GateType.THRESHOLD,
                metric_name="quality_score",
                threshold_min=0.85,
                threshold_max=1.0,
                adaptive=True,
                weight=2.0
            ),
            QualityGateConfig(
                name="error_rate_threshold",
                gate_type=GateType.THRESHOLD,
                metric_name="error_rate",
                threshold_min=0.0,
                threshold_max=5.0,
                adaptive=True,
                weight=1.5
            ),
            QualityGateConfig(
                name="processing_speed_trend",
                gate_type=GateType.TREND,
                metric_name="processing_throughput",
                trend_window=10,
                adaptive=True,
                weight=1.0
            ),
            QualityGateConfig(
                name="system_stability_composite",
                gate_type=GateType.COMPOSITE,
                metric_name="composite_stability",
                adaptive=True,
                weight=1.8
            ),
            QualityGateConfig(
                name="ml_quality_prediction",
                gate_type=GateType.ML_BASED,
                metric_name="ml_quality_score",
                adaptive=True,
                weight=1.2
            )
        ]
        
        for gate in default_gates:
            self.gates[gate.name] = gate
    
    def add_gate(self, gate_config: QualityGateConfig):
        """Add a new quality gate."""
        with self._lock:
            self.gates[gate_config.name] = gate_config
            logger.info(f"Added quality gate: {gate_config.name}")
    
    def remove_gate(self, gate_name: str) -> bool:
        """Remove a quality gate."""
        with self._lock:
            if gate_name in self.gates:
                del self.gates[gate_name]
                logger.info(f"Removed quality gate: {gate_name}")
                return True
        return False
    
    def record_metric(self, metric: QualityMetric):
        """Record a new quality metric."""
        with self._lock:
            if metric.name not in self.metric_history:
                self.metric_history[metric.name] = []
            
            self.metric_history[metric.name].append(metric)
            
            # Keep only recent history
            if len(self.metric_history[metric.name]) > 1000:
                self.metric_history[metric.name] = self.metric_history[metric.name][-1000:]
    
    def evaluate_gates(self, current_metrics: Dict[str, float]) -> Dict[str, GateEvaluationResult]:
        """Evaluate all quality gates against current metrics."""
        results = {}
        
        for gate_name, gate_config in self.gates.items():
            if not gate_config.enabled:
                continue
            
            try:
                result = self._evaluate_single_gate(gate_config, current_metrics)
                results[gate_name] = result
                
                # Update performance tracking
                self._update_gate_performance(gate_name, result.score)
                
                # Update adaptive thresholds
                if gate_config.adaptive and result.result in [QualityGateResult.PASS, QualityGateResult.FAIL]:
                    quality_outcome = result.result == QualityGateResult.PASS
                    if gate_config.metric_name in current_metrics:
                        self.threshold_manager.update_threshold(
                            gate_name, 
                            current_metrics[gate_config.metric_name],
                            quality_outcome
                        )
                
                # Add ML training sample if available
                if HAS_ML and gate_config.gate_type == GateType.ML_BASED:
                    features = self._extract_features_for_ml(current_metrics)
                    if features is not None:
                        target = 1.0 if result.result == QualityGateResult.PASS else 0.0
                        self.ml_predictor.add_training_sample(gate_name, features, target)
                
            except Exception as e:
                logger.error(f"Error evaluating gate {gate_name}: {e}")
                results[gate_name] = GateEvaluationResult(
                    gate_name=gate_name,
                    result=QualityGateResult.ERROR,
                    score=0.0,
                    message=f"Evaluation error: {str(e)}"
                )
        
        # Store evaluation history
        with self._lock:
            for result in results.values():
                self.evaluation_history.append(result)
            
            # Keep only recent history
            if len(self.evaluation_history) > 10000:
                self.evaluation_history = self.evaluation_history[-10000:]
        
        return results
    
    def _evaluate_single_gate(self, gate_config: QualityGateConfig, metrics: Dict[str, float]) -> GateEvaluationResult:
        """Evaluate a single quality gate."""
        if gate_config.gate_type == GateType.THRESHOLD:
            return self._evaluate_threshold_gate(gate_config, metrics)
        elif gate_config.gate_type == GateType.TREND:
            return self._evaluate_trend_gate(gate_config, metrics)
        elif gate_config.gate_type == GateType.STATISTICAL:
            return self._evaluate_statistical_gate(gate_config, metrics)
        elif gate_config.gate_type == GateType.ML_BASED:
            return self._evaluate_ml_gate(gate_config, metrics)
        elif gate_config.gate_type == GateType.COMPOSITE:
            return self._evaluate_composite_gate(gate_config, metrics)
        else:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.ERROR,
                score=0.0,
                message=f"Unknown gate type: {gate_config.gate_type}"
            )
    
    def _evaluate_threshold_gate(self, gate_config: QualityGateConfig, metrics: Dict[str, float]) -> GateEvaluationResult:
        """Evaluate threshold-based quality gate."""
        metric_value = metrics.get(gate_config.metric_name)
        
        if metric_value is None:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message=f"Metric {gate_config.metric_name} not available"
            )
        
        # Use adaptive thresholds if available
        threshold_min = gate_config.threshold_min
        threshold_max = gate_config.threshold_max
        
        if gate_config.adaptive:
            adaptive_min = self.threshold_manager.get_threshold(gate_config.name, "min")
            adaptive_max = self.threshold_manager.get_threshold(gate_config.name, "max")
            
            if adaptive_min is not None:
                threshold_min = adaptive_min
            if adaptive_max is not None:
                threshold_max = adaptive_max
        
        # Evaluate thresholds
        if threshold_min is not None and metric_value < threshold_min:
            score = max(0.0, metric_value / threshold_min)
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.FAIL,
                score=score,
                threshold_used=threshold_min,
                message=f"Value {metric_value:.3f} below minimum threshold {threshold_min:.3f}"
            )
        
        if threshold_max is not None and metric_value > threshold_max:
            score = max(0.0, 1.0 - (metric_value - threshold_max) / threshold_max)
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.FAIL,
                score=score,
                threshold_used=threshold_max,
                message=f"Value {metric_value:.3f} above maximum threshold {threshold_max:.3f}"
            )
        
        # Calculate score within acceptable range
        if threshold_min is not None and threshold_max is not None:
            range_size = threshold_max - threshold_min
            distance_from_center = abs(metric_value - (threshold_min + threshold_max) / 2)
            score = max(0.0, 1.0 - (distance_from_center / (range_size / 2)))
        else:
            score = 1.0
        
        result = QualityGateResult.PASS if score >= 0.8 else QualityGateResult.WARN
        
        return GateEvaluationResult(
            gate_name=gate_config.name,
            result=result,
            score=score,
            threshold_used=threshold_max or threshold_min,
            message=f"Value {metric_value:.3f} within acceptable range"
        )
    
    def _evaluate_trend_gate(self, gate_config: QualityGateConfig, metrics: Dict[str, float]) -> GateEvaluationResult:
        """Evaluate trend-based quality gate."""
        metric_name = gate_config.metric_name
        
        if metric_name not in self.metric_history:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="Insufficient historical data for trend analysis"
            )
        
        recent_metrics = self.metric_history[metric_name][-gate_config.trend_window:]
        
        if len(recent_metrics) < 3:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="Insufficient data points for trend analysis"
            )
        
        # Calculate trend
        values = [m.value for m in recent_metrics]
        if HAS_ML:
            # Linear regression to detect trend
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Normalize slope relative to value range
            value_range = max(values) - min(values) + 1e-6
            normalized_slope = slope / value_range
        else:
            # Simple trend calculation
            normalized_slope = (values[-1] - values[0]) / (max(values) - min(values) + 1e-6)
        
        # Evaluate trend
        if abs(normalized_slope) < 0.01:  # Stable trend
            score = 1.0
            result = QualityGateResult.PASS
            message = "Stable trend detected"
        elif normalized_slope > 0.05:  # Strong positive trend
            score = min(1.0, 0.8 + normalized_slope)
            result = QualityGateResult.PASS
            message = "Positive trend detected"
        elif normalized_slope < -0.05:  # Strong negative trend
            score = max(0.0, 0.8 + normalized_slope)
            result = QualityGateResult.WARN if score > 0.5 else QualityGateResult.FAIL
            message = "Negative trend detected"
        else:  # Mild trend
            score = 0.8
            result = QualityGateResult.PASS
            message = "Mild trend detected"
        
        return GateEvaluationResult(
            gate_name=gate_config.name,
            result=result,
            score=score,
            message=message,
            details={"trend_slope": normalized_slope, "data_points": len(values)}
        )
    
    def _evaluate_statistical_gate(self, gate_config: QualityGateConfig, metrics: Dict[str, float]) -> GateEvaluationResult:
        """Evaluate statistical quality gate using confidence intervals."""
        metric_name = gate_config.metric_name
        
        if metric_name not in self.metric_history:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="No historical data for statistical analysis"
            )
        
        recent_metrics = self.metric_history[metric_name][-100:]  # Use last 100 points
        
        if len(recent_metrics) < 10:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="Insufficient data for statistical analysis"
            )
        
        current_value = metrics.get(metric_name)
        if current_value is None:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="Current metric value not available"
            )
        
        # Calculate statistical bounds
        historical_values = [m.value for m in recent_metrics]
        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        
        # Calculate confidence interval
        confidence = gate_config.statistical_confidence
        z_score = 1.96 if confidence == 0.95 else 2.58  # For 95% or 99%
        
        lower_bound = mean_val - z_score * std_val
        upper_bound = mean_val + z_score * std_val
        
        # Evaluate current value against statistical bounds
        if lower_bound <= current_value <= upper_bound:
            # Within expected range
            distance_from_mean = abs(current_value - mean_val)
            normalized_distance = distance_from_mean / (std_val + 1e-6)
            score = max(0.0, 1.0 - normalized_distance / z_score)
            result = QualityGateResult.PASS
            message = f"Value within {confidence*100:.0f}% confidence interval"
        else:
            # Outside expected range
            if current_value < lower_bound:
                distance = lower_bound - current_value
                score = max(0.0, 1.0 - distance / (std_val + 1e-6))
                message = f"Value below {confidence*100:.0f}% confidence interval"
            else:
                distance = current_value - upper_bound
                score = max(0.0, 1.0 - distance / (std_val + 1e-6))
                message = f"Value above {confidence*100:.0f}% confidence interval"
            
            result = QualityGateResult.WARN if score > 0.5 else QualityGateResult.FAIL
        
        return GateEvaluationResult(
            gate_name=gate_config.name,
            result=result,
            score=score,
            message=message,
            details={
                "mean": mean_val,
                "std": std_val,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "current_value": current_value
            }
        )
    
    def _evaluate_ml_gate(self, gate_config: QualityGateConfig, metrics: Dict[str, float]) -> GateEvaluationResult:
        """Evaluate ML-based quality gate."""
        features = self._extract_features_for_ml(metrics)
        
        if features is None:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="Unable to extract features for ML prediction"
            )
        
        prediction_result = self.ml_predictor.predict_quality(gate_config.name, features)
        
        if prediction_result is None:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message="ML model not available or prediction failed"
            )
        
        prediction, confidence = prediction_result
        
        # Interpret prediction
        if prediction >= 0.8:
            result = QualityGateResult.PASS
            message = "ML model predicts high quality"
        elif prediction >= 0.6:
            result = QualityGateResult.WARN
            message = "ML model predicts moderate quality"
        else:
            result = QualityGateResult.FAIL
            message = "ML model predicts low quality"
        
        return GateEvaluationResult(
            gate_name=gate_config.name,
            result=result,
            score=prediction,
            message=message,
            details={
                "ml_prediction": prediction,
                "ml_confidence": confidence,
                "features_used": features.tolist()
            }
        )
    
    def _evaluate_composite_gate(self, gate_config: QualityGateConfig, metrics: Dict[str, float]) -> GateEvaluationResult:
        """Evaluate composite quality gate combining multiple metrics."""
        # Define composite metrics based on system stability
        required_metrics = ["cpu_usage", "memory_usage", "error_rate", "quality_score"]
        
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if missing_metrics:
            return GateEvaluationResult(
                gate_name=gate_config.name,
                result=QualityGateResult.SKIP,
                score=0.0,
                message=f"Missing required metrics: {missing_metrics}"
            )
        
        # Calculate composite score
        scores = []
        
        # CPU usage score (inverse - lower is better)
        cpu_score = max(0.0, 1.0 - metrics["cpu_usage"] / 100.0)
        scores.append(("cpu", cpu_score, 0.2))
        
        # Memory usage score (inverse - lower is better)  
        memory_score = max(0.0, 1.0 - metrics["memory_usage"] / 100.0)
        scores.append(("memory", memory_score, 0.2))
        
        # Error rate score (inverse - lower is better)
        error_score = max(0.0, 1.0 - metrics["error_rate"] / 10.0)  # Assume 10% is max
        scores.append(("error", error_score, 0.3))
        
        # Quality score (direct - higher is better)
        quality_score = metrics["quality_score"]
        scores.append(("quality", quality_score, 0.3))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in scores)
        composite_score = sum(score * weight for _, score, weight in scores) / total_weight
        
        # Determine result
        if composite_score >= 0.85:
            result = QualityGateResult.PASS
            message = "Composite system stability is excellent"
        elif composite_score >= 0.7:
            result = QualityGateResult.WARN
            message = "Composite system stability shows some concerns"
        else:
            result = QualityGateResult.FAIL
            message = "Composite system stability is poor"
        
        return GateEvaluationResult(
            gate_name=gate_config.name,
            result=result,
            score=composite_score,
            message=message,
            details={
                "component_scores": {name: score for name, score, _ in scores},
                "weights": {name: weight for name, _, weight in scores}
            }
        )
    
    def _extract_features_for_ml(self, metrics: Dict[str, float]) -> Optional[np.ndarray]:
        """Extract feature vector for ML processing."""
        if not HAS_ML:
            return None
        
        # Define standard feature set
        feature_names = [
            "cpu_usage", "memory_usage", "error_rate", "quality_score",
            "processing_throughput", "response_time_p95"
        ]
        
        features = []
        for name in feature_names:
            if name in metrics:
                features.append(metrics[name])
            else:
                features.append(0.0)  # Default value for missing metrics
        
        return np.array(features) if features else None
    
    def _update_gate_performance(self, gate_name: str, score: float):
        """Update performance tracking for a quality gate."""
        with self._lock:
            if gate_name not in self.gate_performance:
                self.gate_performance[gate_name] = []
            
            self.gate_performance[gate_name].append(score)
            
            # Keep only recent performance data
            if len(self.gate_performance[gate_name]) > 1000:
                self.gate_performance[gate_name] = self.gate_performance[gate_name][-1000:]
    
    def get_system_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        with self._lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_gates": len(self.gates),
                "enabled_gates": sum(1 for g in self.gates.values() if g.enabled),
                "recent_evaluations": len([r for r in self.evaluation_history 
                                         if (datetime.now() - r.timestamp).total_seconds() < 3600]),
                "gate_performance": {}
            }
            
            # Calculate performance metrics for each gate
            for gate_name, performance_data in self.gate_performance.items():
                if performance_data:
                    report["gate_performance"][gate_name] = {
                        "average_score": statistics.mean(performance_data),
                        "min_score": min(performance_data),
                        "max_score": max(performance_data),
                        "recent_trend": performance_data[-10:] if len(performance_data) >= 10 else performance_data,
                        "evaluation_count": len(performance_data)
                    }
            
            # Overall system quality
            if self.evaluation_history:
                recent_results = [r for r in self.evaluation_history 
                                if (datetime.now() - r.timestamp).total_seconds() < 3600]
                
                if recent_results:
                    pass_rate = len([r for r in recent_results if r.result == QualityGateResult.PASS]) / len(recent_results)
                    avg_score = statistics.mean([r.score for r in recent_results])
                    
                    report["overall_quality"] = {
                        "pass_rate": pass_rate,
                        "average_score": avg_score,
                        "total_evaluations": len(recent_results)
                    }
            
            return report
    
    def load_configuration(self, config_path: str):
        """Load quality gate configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for gate_data in config_data.get("gates", []):
                gate_config = QualityGateConfig(
                    name=gate_data["name"],
                    gate_type=GateType(gate_data["gate_type"]),
                    metric_name=gate_data["metric_name"],
                    threshold_min=gate_data.get("threshold_min"),
                    threshold_max=gate_data.get("threshold_max"),
                    trend_window=gate_data.get("trend_window", 10),
                    statistical_confidence=gate_data.get("statistical_confidence", 0.95),
                    adaptive=gate_data.get("adaptive", True),
                    weight=gate_data.get("weight", 1.0),
                    enabled=gate_data.get("enabled", True)
                )
                self.add_gate(gate_config)
            
            logger.info(f"Loaded {len(config_data.get('gates', []))} quality gates from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    def save_configuration(self, config_path: str):
        """Save current quality gate configuration to file."""
        try:
            config_data = {
                "gates": [
                    {
                        "name": gate.name,
                        "gate_type": gate.gate_type.value,
                        "metric_name": gate.metric_name,
                        "threshold_min": gate.threshold_min,
                        "threshold_max": gate.threshold_max,
                        "trend_window": gate.trend_window,
                        "statistical_confidence": gate.statistical_confidence,
                        "adaptive": gate.adaptive,
                        "weight": gate.weight,
                        "enabled": gate.enabled
                    }
                    for gate in self.gates.values()
                ]
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved quality gate configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")


def initialize_intelligent_quality_gates(
    config_path: Optional[str] = None,
    **kwargs
) -> IntelligentQualityGateSystem:
    """Initialize the intelligent quality gate system."""
    system = IntelligentQualityGateSystem(config_path=config_path)
    logger.info("Intelligent Quality Gate System initialized successfully")
    return system