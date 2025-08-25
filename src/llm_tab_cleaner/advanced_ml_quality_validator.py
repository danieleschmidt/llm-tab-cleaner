"""
Advanced ML Quality Validator - Neural Quality Gates with Ensemble Learning
Multi-modal quality assessment using deep learning and statistical validation
"""

import logging
import asyncio
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import threading

logger = logging.getLogger(__name__)

@dataclass
class QualityGateConfig:
    """Advanced quality gate configuration."""
    gate_name: str
    threshold: float
    weight: float
    critical: bool = False
    ml_model: str = "ensemble"
    feature_set: List[str] = None
    confidence_required: float = 0.85

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    coverage_percentage: float
    performance_score: float
    security_score: float
    maintainability_score: float
    reliability_score: float
    scalability_score: float
    overall_quality_score: float
    confidence_level: float

@dataclass
class ValidationResult:
    """Advanced validation result with ML predictions."""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    confidence: float
    ml_prediction: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    recommendations: List[str] = None

@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp: float
    overall_passed: bool
    quality_metrics: QualityMetrics
    gate_results: List[ValidationResult]
    ml_insights: Dict[str, Any]
    trends: Dict[str, List[float]]
    recommendations: List[str]
    risk_assessment: Dict[str, float]

class FeatureExtractor:
    """Extract features from code and system metrics."""
    
    def extract_code_features(self, code_path: Path) -> Dict[str, float]:
        """Extract features from codebase."""
        
        features = {}
        
        try:
            # Analyze Python files
            python_files = list(code_path.rglob("*.py"))
            features["total_files"] = len(python_files)
            features["lines_of_code"] = 0
            features["complexity_score"] = 0
            features["function_count"] = 0
            features["class_count"] = 0
            features["import_count"] = 0
            
            for file_path in python_files[:50]:  # Limit to avoid performance issues
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        features["lines_of_code"] += len(lines)
                        
                        # Simple complexity metrics
                        for line in lines:
                            line = line.strip()
                            if line.startswith("def "):
                                features["function_count"] += 1
                            elif line.startswith("class "):
                                features["class_count"] += 1
                            elif line.startswith("import ") or line.startswith("from "):
                                features["import_count"] += 1
                            elif "if " in line or "for " in line or "while " in line:
                                features["complexity_score"] += 1
                                
                except Exception as e:
                    logger.warning(f"Could not analyze file {file_path}: {e}")
                    continue
            
            # Calculate derived metrics
            if features["total_files"] > 0:
                features["avg_lines_per_file"] = features["lines_of_code"] / features["total_files"]
                features["avg_functions_per_file"] = features["function_count"] / features["total_files"]
                features["avg_classes_per_file"] = features["class_count"] / features["total_files"]
            else:
                features["avg_lines_per_file"] = 0
                features["avg_functions_per_file"] = 0
                features["avg_classes_per_file"] = 0
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            features = {
                "total_files": 0, "lines_of_code": 0, "complexity_score": 0,
                "function_count": 0, "class_count": 0, "import_count": 0,
                "avg_lines_per_file": 0, "avg_functions_per_file": 0, "avg_classes_per_file": 0
            }
        
        return features
    
    def extract_test_features(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from test results."""
        
        return {
            "test_count": test_results.get("total_tests", 0),
            "test_pass_rate": test_results.get("pass_rate", 0.0),
            "test_coverage": test_results.get("coverage_percentage", 0.0),
            "test_execution_time": test_results.get("execution_time", 0.0),
            "failed_tests": test_results.get("failed_tests", 0),
            "skipped_tests": test_results.get("skipped_tests", 0)
        }
    
    def extract_performance_features(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance-related features."""
        
        return {
            "response_time_p50": performance_data.get("response_time_p50", 0.0),
            "response_time_p95": performance_data.get("response_time_p95", 0.0),
            "response_time_p99": performance_data.get("response_time_p99", 0.0),
            "throughput": performance_data.get("throughput", 0.0),
            "error_rate": performance_data.get("error_rate", 0.0),
            "cpu_usage": performance_data.get("cpu_usage", 0.0),
            "memory_usage": performance_data.get("memory_usage", 0.0),
            "disk_usage": performance_data.get("disk_usage", 0.0)
        }

class MLQualityPredictor:
    """Machine learning models for quality prediction."""
    
    def __init__(self):
        self.models = {
            "quality_classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            "score_regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "risk_classifier": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scalers = {name: StandardScaler() for name in self.models.keys()}
        self.is_trained = {name: False for name in self.models.keys()}
        self.training_data = []
        
    def add_training_sample(
        self, 
        features: Dict[str, float], 
        quality_passed: bool, 
        quality_score: float,
        risk_level: str
    ):
        """Add a training sample to the dataset."""
        
        risk_mapping = {"low": 0, "medium": 1, "high": 2}
        
        sample = {
            "features": features,
            "quality_passed": quality_passed,
            "quality_score": quality_score,
            "risk_level": risk_mapping.get(risk_level.lower(), 1)
        }
        
        self.training_data.append(sample)
        
        # Retrain models if we have enough samples
        if len(self.training_data) % 10 == 0 and len(self.training_data) >= 20:
            self.train_models()
    
    def train_models(self):
        """Train all ML models with accumulated data."""
        
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for ML models")
            return
        
        # Prepare feature matrix
        feature_names = list(self.training_data[0]["features"].keys())
        X = np.array([[sample["features"][name] for name in feature_names] 
                      for sample in self.training_data])
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Prepare target variables
        y_quality = np.array([sample["quality_passed"] for sample in self.training_data])
        y_score = np.array([sample["quality_score"] for sample in self.training_data])
        y_risk = np.array([sample["risk_level"] for sample in self.training_data])
        
        try:
            # Train quality classifier
            X_scaled_clf = self.scalers["quality_classifier"].fit_transform(X)
            self.models["quality_classifier"].fit(X_scaled_clf, y_quality)
            self.is_trained["quality_classifier"] = True
            
            # Train score regressor
            X_scaled_reg = self.scalers["score_regressor"].fit_transform(X)
            self.models["score_regressor"].fit(X_scaled_reg, y_score)
            self.is_trained["score_regressor"] = True
            
            # Train risk classifier
            X_scaled_risk = self.scalers["risk_classifier"].fit_transform(X)
            self.models["risk_classifier"].fit(X_scaled_risk, y_risk)
            self.is_trained["risk_classifier"] = True
            
            logger.info(f"ML models trained successfully with {len(self.training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def predict_quality(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict quality metrics using trained models."""
        
        predictions = {}
        
        try:
            # Prepare feature vector
            if not self.training_data:
                # Use default feature names if no training data
                feature_names = list(features.keys())
            else:
                feature_names = list(self.training_data[0]["features"].keys())
            
            X = np.array([[features.get(name, 0) for name in feature_names]])
            X = np.nan_to_num(X)
            
            # Quality classification
            if self.is_trained["quality_classifier"]:
                X_scaled = self.scalers["quality_classifier"].transform(X)
                quality_pred = self.models["quality_classifier"].predict(X_scaled)[0]
                quality_proba = self.models["quality_classifier"].predict_proba(X_scaled)[0]
                predictions["quality_passed"] = bool(quality_pred)
                predictions["quality_confidence"] = float(np.max(quality_proba))
            else:
                predictions["quality_passed"] = True
                predictions["quality_confidence"] = 0.5
            
            # Score regression
            if self.is_trained["score_regressor"]:
                X_scaled = self.scalers["score_regressor"].transform(X)
                score_pred = self.models["score_regressor"].predict(X_scaled)[0]
                predictions["predicted_score"] = float(max(0, min(1, score_pred)))
            else:
                predictions["predicted_score"] = 0.75
            
            # Risk classification
            if self.is_trained["risk_classifier"]:
                X_scaled = self.scalers["risk_classifier"].transform(X)
                risk_pred = self.models["risk_classifier"].predict(X_scaled)[0]
                risk_proba = self.models["risk_classifier"].predict_proba(X_scaled)[0]
                risk_mapping = {0: "low", 1: "medium", 2: "high"}
                predictions["risk_level"] = risk_mapping[risk_pred]
                predictions["risk_confidence"] = float(np.max(risk_proba))
            else:
                predictions["risk_level"] = "medium"
                predictions["risk_confidence"] = 0.5
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            predictions = {
                "quality_passed": True,
                "quality_confidence": 0.5,
                "predicted_score": 0.75,
                "risk_level": "medium",
                "risk_confidence": 0.5
            }
        
        return predictions

class AdvancedMLQualityValidator:
    """Advanced ML-powered quality validation system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.feature_extractor = FeatureExtractor()
        self.ml_predictor = MLQualityPredictor()
        self.quality_gates = self._load_quality_gates(config_path)
        self.validation_history = []
        self.trend_window = 10
        
    def _load_quality_gates(self, config_path: Optional[Path]) -> List[QualityGateConfig]:
        """Load quality gate configuration."""
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    return [QualityGateConfig(**gate) for gate in config_data["gates"]]
            except Exception as e:
                logger.warning(f"Could not load quality gate config: {e}")
        
        # Default quality gates
        return [
            QualityGateConfig(
                gate_name="test_coverage",
                threshold=0.85,
                weight=0.25,
                critical=True,
                feature_set=["test_coverage", "test_pass_rate", "test_count"]
            ),
            QualityGateConfig(
                gate_name="code_quality",
                threshold=0.80,
                weight=0.20,
                critical=False,
                feature_set=["complexity_score", "lines_of_code", "function_count"]
            ),
            QualityGateConfig(
                gate_name="performance",
                threshold=0.90,
                weight=0.25,
                critical=True,
                feature_set=["response_time_p95", "throughput", "error_rate"]
            ),
            QualityGateConfig(
                gate_name="security",
                threshold=0.95,
                weight=0.30,
                critical=True,
                feature_set=["security_score"]
            )
        ]
    
    async def validate_quality(
        self,
        code_path: Path,
        test_results: Dict[str, Any],
        performance_data: Dict[str, Any],
        security_results: Dict[str, Any]
    ) -> QualityReport:
        """Run comprehensive quality validation."""
        
        logger.info("Starting advanced ML quality validation")
        
        # Extract features
        code_features = self.feature_extractor.extract_code_features(code_path)
        test_features = self.feature_extractor.extract_test_features(test_results)
        perf_features = self.feature_extractor.extract_performance_features(performance_data)
        
        # Combine all features
        all_features = {**code_features, **test_features, **perf_features}
        all_features["security_score"] = security_results.get("overall_score", 0.8)
        
        # Get ML predictions
        ml_predictions = self.ml_predictor.predict_quality(all_features)
        
        # Validate each quality gate
        gate_results = []
        for gate in self.quality_gates:
            result = await self._validate_gate(gate, all_features, ml_predictions)
            gate_results.append(result)
        
        # Calculate overall quality metrics
        quality_metrics = self._calculate_quality_metrics(gate_results, all_features)
        
        # Determine overall pass/fail
        critical_gates_passed = all(
            result.passed for result in gate_results 
            if any(gate.critical for gate in self.quality_gates if gate.gate_name == result.gate_name)
        )
        overall_passed = critical_gates_passed and quality_metrics.overall_quality_score >= 0.75
        
        # Generate insights and recommendations
        ml_insights = self._generate_ml_insights(ml_predictions, gate_results)
        recommendations = self._generate_recommendations(gate_results, quality_metrics)
        risk_assessment = self._assess_risk(ml_predictions, gate_results)
        
        # Update trends
        trends = self._update_trends(quality_metrics)
        
        # Create quality report
        report = QualityReport(
            timestamp=time.time(),
            overall_passed=overall_passed,
            quality_metrics=quality_metrics,
            gate_results=gate_results,
            ml_insights=ml_insights,
            trends=trends,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )
        
        # Update ML training data
        self.ml_predictor.add_training_sample(
            all_features, 
            overall_passed, 
            quality_metrics.overall_quality_score,
            risk_assessment.get("overall_risk", "medium")
        )
        
        # Store validation history
        self.validation_history.append(report)
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]
        
        logger.info(f"Quality validation completed: {'PASSED' if overall_passed else 'FAILED'}")
        
        return report
    
    async def _validate_gate(
        self,
        gate: QualityGateConfig,
        features: Dict[str, float],
        ml_predictions: Dict[str, Any]
    ) -> ValidationResult:
        """Validate individual quality gate."""
        
        if gate.gate_name == "test_coverage":
            score = features.get("test_coverage", 0.0) / 100.0
        elif gate.gate_name == "code_quality":
            # Inverse relationship for complexity
            complexity_normalized = 1.0 / (1.0 + features.get("complexity_score", 0) / 1000)
            lines_normalized = min(1.0, 10000 / max(1, features.get("lines_of_code", 1)))
            score = (complexity_normalized + lines_normalized) / 2
        elif gate.gate_name == "performance":
            # Composite performance score
            throughput_score = min(1.0, features.get("throughput", 0) / 1000)
            latency_score = max(0.0, 1.0 - features.get("response_time_p95", 0))
            error_score = max(0.0, 1.0 - features.get("error_rate", 0))
            score = (throughput_score + latency_score + error_score) / 3
        elif gate.gate_name == "security":
            score = features.get("security_score", 0.8)
        else:
            score = 0.75  # Default score
        
        # Use ML prediction if confident enough
        if ml_predictions.get("quality_confidence", 0) > gate.confidence_required:
            ml_score = ml_predictions.get("predicted_score", score)
            score = 0.7 * score + 0.3 * ml_score  # Blend traditional and ML scores
        
        passed = score >= gate.threshold
        confidence = ml_predictions.get("quality_confidence", 0.5)
        
        # Generate recommendations
        recommendations = []
        if not passed:
            if gate.gate_name == "test_coverage":
                recommendations.append("Increase test coverage by adding unit tests")
            elif gate.gate_name == "performance":
                recommendations.append("Optimize performance by improving algorithms or caching")
            elif gate.gate_name == "security":
                recommendations.append("Address security vulnerabilities found in scan")
            else:
                recommendations.append(f"Improve {gate.gate_name} to meet quality threshold")
        
        return ValidationResult(
            gate_name=gate.gate_name,
            passed=passed,
            score=score,
            threshold=gate.threshold,
            confidence=confidence,
            ml_prediction=ml_predictions,
            recommendations=recommendations
        )
    
    def _calculate_quality_metrics(
        self, 
        gate_results: List[ValidationResult],
        features: Dict[str, float]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        # Individual metrics
        accuracy_score = features.get("test_coverage", 85) / 100.0
        precision_score = max(0.8, 1.0 - features.get("error_rate", 0.02))
        recall_score = min(1.0, features.get("throughput", 500) / 1000)
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        # Derived metrics
        coverage_percentage = features.get("test_coverage", 85)
        performance_score = 1.0 - features.get("response_time_p95", 0.1)
        security_score = features.get("security_score", 0.8)
        
        # Maintainability based on code structure
        complexity_factor = 1.0 / (1.0 + features.get("complexity_score", 100) / 1000)
        maintainability_score = complexity_factor * 0.6 + (features.get("avg_functions_per_file", 5) / 20) * 0.4
        
        # Reliability based on test results and error rates
        reliability_score = (accuracy_score + precision_score) / 2
        
        # Scalability based on performance characteristics
        scalability_score = min(1.0, (features.get("throughput", 500) / 1000) * 0.7 + performance_score * 0.3)
        
        # Overall quality score (weighted)
        gate_weights = {result.gate_name: 0.25 for result in gate_results}  # Equal weights
        overall_quality_score = sum(
            result.score * gate_weights.get(result.gate_name, 0.25) 
            for result in gate_results
        )
        
        # Confidence level
        confidence_level = np.mean([result.confidence for result in gate_results])
        
        return QualityMetrics(
            accuracy_score=accuracy_score,
            precision_score=precision_score,
            recall_score=recall_score,
            f1_score=f1_score,
            coverage_percentage=coverage_percentage,
            performance_score=performance_score,
            security_score=security_score,
            maintainability_score=maintainability_score,
            reliability_score=reliability_score,
            scalability_score=scalability_score,
            overall_quality_score=overall_quality_score,
            confidence_level=confidence_level
        )
    
    def _generate_ml_insights(
        self,
        ml_predictions: Dict[str, Any],
        gate_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """Generate ML-powered insights."""
        
        insights = {
            "model_confidence": ml_predictions.get("quality_confidence", 0.5),
            "predicted_quality": ml_predictions.get("quality_passed", True),
            "risk_assessment": ml_predictions.get("risk_level", "medium"),
            "trend_prediction": "stable"  # Would be based on historical data
        }
        
        # Analyze patterns in gate results
        failed_gates = [result.gate_name for result in gate_results if not result.passed]
        if failed_gates:
            insights["failure_pattern"] = f"Common failures in: {', '.join(failed_gates)}"
        
        return insights
    
    def _generate_recommendations(
        self,
        gate_results: List[ValidationResult],
        quality_metrics: QualityMetrics
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Collect recommendations from gate results
        for result in gate_results:
            if result.recommendations:
                recommendations.extend(result.recommendations)
        
        # Overall recommendations based on metrics
        if quality_metrics.coverage_percentage < 85:
            recommendations.append("Priority: Increase test coverage to meet minimum requirements")
        
        if quality_metrics.performance_score < 0.8:
            recommendations.append("Consider performance optimization strategies")
        
        if quality_metrics.security_score < 0.9:
            recommendations.append("Address security vulnerabilities as high priority")
        
        if quality_metrics.maintainability_score < 0.7:
            recommendations.append("Refactor code to reduce complexity and improve maintainability")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_risk(
        self,
        ml_predictions: Dict[str, Any],
        gate_results: List[ValidationResult]
    ) -> Dict[str, float]:
        """Assess deployment and operational risk."""
        
        # Base risk from ML predictions
        risk_level = ml_predictions.get("risk_level", "medium")
        risk_mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
        base_risk = risk_mapping.get(risk_level, 0.5)
        
        # Adjust risk based on gate failures
        failed_critical_gates = sum(
            1 for result in gate_results 
            if not result.passed and any(
                gate.critical for gate in self.quality_gates 
                if gate.gate_name == result.gate_name
            )
        )
        
        critical_risk = failed_critical_gates * 0.3
        
        # Calculate component risks
        deployment_risk = min(1.0, base_risk + critical_risk)
        operational_risk = base_risk * 0.8  # Slightly lower than deployment risk
        maintenance_risk = base_risk * 0.6   # Lower for maintenance
        
        overall_risk_score = deployment_risk * 0.5 + operational_risk * 0.3 + maintenance_risk * 0.2
        
        if overall_risk_score < 0.3:
            overall_risk = "low"
        elif overall_risk_score < 0.7:
            overall_risk = "medium"
        else:
            overall_risk = "high"
        
        return {
            "deployment_risk": deployment_risk,
            "operational_risk": operational_risk,
            "maintenance_risk": maintenance_risk,
            "overall_risk_score": overall_risk_score,
            "overall_risk": overall_risk
        }
    
    def _update_trends(self, current_metrics: QualityMetrics) -> Dict[str, List[float]]:
        """Update quality trends over time."""
        
        trends = {
            "quality_score": [],
            "coverage": [],
            "performance": [],
            "security": []
        }
        
        # Get recent history
        recent_history = self.validation_history[-self.trend_window:]
        
        for report in recent_history:
            trends["quality_score"].append(report.quality_metrics.overall_quality_score)
            trends["coverage"].append(report.quality_metrics.coverage_percentage)
            trends["performance"].append(report.quality_metrics.performance_score)
            trends["security"].append(report.quality_metrics.security_score)
        
        # Add current metrics
        trends["quality_score"].append(current_metrics.overall_quality_score)
        trends["coverage"].append(current_metrics.coverage_percentage)
        trends["performance"].append(current_metrics.performance_score)
        trends["security"].append(current_metrics.security_score)
        
        return trends

# Global validator instance
_global_validator = None
_validator_lock = threading.Lock()

def get_global_ml_quality_validator() -> AdvancedMLQualityValidator:
    """Get or create global ML quality validator."""
    global _global_validator
    
    if _global_validator is None:
        with _validator_lock:
            if _global_validator is None:
                _global_validator = AdvancedMLQualityValidator()
    
    return _global_validator

async def initialize_ml_quality_validation(
    config_path: Optional[Path] = None
) -> AdvancedMLQualityValidator:
    """Initialize ML quality validation system."""
    
    global _global_validator
    with _validator_lock:
        _global_validator = AdvancedMLQualityValidator(config_path)
    
    logger.info("Advanced ML quality validation system initialized")
    return _global_validator