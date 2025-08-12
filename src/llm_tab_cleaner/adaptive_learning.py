"""Adaptive Real-Time Learning System for Data Cleaning.

Implements next-generation adaptive learning that continuously improves
data cleaning performance through:
- Online learning from user feedback
- Meta-learning across datasets
- Continuous model adaptation
- Real-time performance optimization
- Automated hyperparameter tuning

Based on cutting-edge research in online learning, meta-learning,
and adaptive systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import threading
import queue
import pickle
from collections import deque, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSignal:
    """User feedback signal for adaptive learning."""
    sample_id: str
    predicted_quality: float
    actual_quality: float
    user_rating: Optional[float]
    cleaning_actions: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def prediction_error(self) -> float:
        """Compute prediction error."""
        return abs(self.predicted_quality - self.actual_quality)
    
    @property
    def feedback_weight(self) -> float:
        """Compute feedback weight based on confidence and recency."""
        # Recency weight (decay over time)
        age_hours = (time.time() - self.timestamp) / 3600
        recency_weight = np.exp(-age_hours / 24)  # Decay over 24 hours
        
        # User rating weight
        rating_weight = self.user_rating or 0.5
        
        return recency_weight * rating_weight


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance."""
    learning_rate: float
    accuracy_improvement: float
    convergence_rate: float
    adaptation_speed: float
    stability_score: float
    memory_usage_mb: float
    processing_time_ms: float
    feedback_utilization: float


class OnlineLearner(ABC):
    """Abstract base class for online learning algorithms."""
    
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Update model with new data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        pass


class AdaptiveSGDLearner(OnlineLearner):
    """Adaptive Stochastic Gradient Descent learner with momentum."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        adaptive_lr: bool = True,
        l1_ratio: float = 0.15,
        l2_ratio: float = 0.85
    ):
        self.base_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.adaptive_lr = adaptive_lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        
        self.model = SGDClassifier(
            loss='log_loss',
            learning_rate='constant',
            eta0=learning_rate,
            alpha=0.01,
            l1_ratio=l1_ratio,
            penalty='elasticnet',
            random_state=42
        )
        
        self.is_fitted = False
        self.feature_count = None
        self.performance_history = deque(maxlen=100)
        self.lr_adaptation_window = deque(maxlen=20)
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Update model with new batch of data."""
        if not self.is_fitted:
            self.model.partial_fit(X, y, classes=[0, 1], sample_weight=sample_weight)
            self.is_fitted = True
            self.feature_count = X.shape[1]
        else:
            # Adapt learning rate if enabled
            if self.adaptive_lr and len(self.performance_history) > 10:
                self._adapt_learning_rate(X, y, sample_weight)
            
            self.model.partial_fit(X, y, sample_weight=sample_weight)
        
        # Track performance
        if len(X) > 0:
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions, sample_weight=sample_weight)
            self.performance_history.append(accuracy)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            return np.ones((X.shape[0], 2)) * 0.5
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from model coefficients."""
        if not self.is_fitted:
            return np.array([])
        return np.abs(self.model.coef_[0])
    
    def _adapt_learning_rate(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]):
        """Adapt learning rate based on recent performance."""
        recent_performance = list(self.performance_history)[-10:]
        
        if len(recent_performance) >= 10:
            # Check if performance is improving
            early_avg = np.mean(recent_performance[:5])
            late_avg = np.mean(recent_performance[5:])
            
            improvement = late_avg - early_avg
            
            if improvement > 0.01:
                # Performance improving, maintain or slightly increase LR
                self.current_learning_rate = min(
                    self.current_learning_rate * 1.05,
                    self.base_learning_rate * 2
                )
            elif improvement < -0.01:
                # Performance degrading, decrease LR
                self.current_learning_rate *= 0.95
            
            # Update model learning rate
            self.model.eta0 = self.current_learning_rate
            self.lr_adaptation_window.append(self.current_learning_rate)


class MetaLearner:
    """Meta-learner that learns optimal strategies across datasets."""
    
    def __init__(self, base_learners: Optional[List[OnlineLearner]] = None):
        self.base_learners = base_learners or [
            AdaptiveSGDLearner(learning_rate=0.01),
            AdaptiveSGDLearner(learning_rate=0.1),
            AdaptiveSGDLearner(learning_rate=0.001)
        ]
        
        self.meta_features = {}
        self.strategy_performance = defaultdict(list)
        self.current_best_learner = 0
        self.ensemble_weights = np.ones(len(self.base_learners)) / len(self.base_learners)
        
    def extract_meta_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Extract meta-features from dataset."""
        features = {}
        
        # Dataset characteristics
        features['n_samples'] = X.shape[0]
        features['n_features'] = X.shape[1]
        features['class_balance'] = np.mean(y) if len(y) > 0 else 0.5
        
        # Feature statistics
        if X.shape[0] > 0:
            features['mean_feature_std'] = np.mean(np.std(X, axis=0))
            features['mean_feature_skew'] = np.mean([
                abs(np.mean((col - np.mean(col))**3) / (np.std(col)**3 + 1e-8))
                for col in X.T
            ])
            features['feature_correlation'] = np.mean(np.abs(np.corrcoef(X.T)))
        else:
            features['mean_feature_std'] = 0.0
            features['mean_feature_skew'] = 0.0
            features['feature_correlation'] = 0.0
        
        # Label characteristics
        if len(y) > 1:
            label_changes = np.sum(np.diff(y) != 0) / max(len(y) - 1, 1)
            features['label_noise'] = label_changes
        else:
            features['label_noise'] = 0.0
        
        return features
    
    def select_best_learner(self, meta_features: Dict[str, float]) -> int:
        """Select best learner based on meta-features."""
        if not self.strategy_performance:
            return self.current_best_learner
        
        # Simple strategy: use learner with best recent performance
        recent_performance = {}
        for learner_idx, performance_list in self.strategy_performance.items():
            if performance_list:
                recent_performance[learner_idx] = np.mean(performance_list[-10:])
        
        if recent_performance:
            best_learner = max(recent_performance, key=recent_performance.get)
            self.current_best_learner = best_learner
        
        return self.current_best_learner
    
    def update_ensemble_weights(self, performances: List[float]):
        """Update ensemble weights based on performance."""
        if len(performances) == len(self.base_learners):
            # Softmax weighting
            exp_perf = np.exp(np.array(performances) - np.max(performances))
            self.ensemble_weights = exp_perf / np.sum(exp_perf)
    
    def get_ensemble_prediction(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction from all learners."""
        predictions = []
        
        for learner in self.base_learners:
            pred_proba = learner.predict_proba(X)
            predictions.append(pred_proba[:, 1])  # Positive class probability
        
        if predictions:
            weighted_pred = np.average(predictions, axis=0, weights=self.ensemble_weights)
            return (weighted_pred > 0.5).astype(int)
        else:
            return np.zeros(X.shape[0])


class AdaptiveLearningSystem:
    """Complete adaptive learning system for data cleaning."""
    
    def __init__(
        self,
        adaptation_rate: float = 0.1,
        feedback_buffer_size: int = 1000,
        min_feedback_for_update: int = 10,
        performance_window: int = 100,
        enable_meta_learning: bool = True
    ):
        self.adaptation_rate = adaptation_rate
        self.feedback_buffer_size = feedback_buffer_size
        self.min_feedback_for_update = min_feedback_for_update
        self.performance_window = performance_window
        self.enable_meta_learning = enable_meta_learning
        
        # Learning components
        self.primary_learner = AdaptiveSGDLearner()
        self.meta_learner = MetaLearner() if enable_meta_learning else None
        
        # Feedback management
        self.feedback_queue = queue.Queue(maxsize=feedback_buffer_size)
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)
        self.performance_history = deque(maxlen=performance_window)
        
        # Threading for async processing
        self.processing_thread = None
        self.is_running = False
        self.update_lock = threading.Lock()
        
        # Metrics tracking
        self.adaptation_metrics = []
        self.feature_importance_history = deque(maxlen=50)
        
    def start_adaptive_learning(self):
        """Start the adaptive learning process."""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_feedback_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Adaptive learning system started")
    
    def stop_adaptive_learning(self):
        """Stop the adaptive learning process."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Adaptive learning system stopped")
    
    def add_feedback(self, feedback: FeedbackSignal):
        """Add feedback signal for learning."""
        try:
            self.feedback_queue.put_nowait(feedback)
        except queue.Full:
            logger.warning("Feedback queue full, dropping oldest feedback")
            try:
                self.feedback_queue.get_nowait()
                self.feedback_queue.put_nowait(feedback)
            except queue.Empty:
                pass
    
    def predict_quality(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict data quality with confidence estimates."""
        with self.update_lock:
            if self.meta_learner and self.enable_meta_learning:
                predictions = self.meta_learner.get_ensemble_prediction(features)
                # Get probabilities for confidence
                probabilities = []
                for learner in self.meta_learner.base_learners:
                    prob = learner.predict_proba(features)
                    probabilities.append(prob[:, 1])
                
                if probabilities:
                    avg_prob = np.mean(probabilities, axis=0)
                    confidence = 1 - 2 * np.abs(avg_prob - 0.5)  # Distance from 0.5
                else:
                    confidence = np.ones(len(predictions)) * 0.5
            else:
                predictions = self.primary_learner.predict(features)
                prob = self.primary_learner.predict_proba(features)
                confidence = np.max(prob, axis=1)
        
        return predictions, confidence
    
    def get_feature_importance(self) -> np.ndarray:
        """Get current feature importance."""
        with self.update_lock:
            if self.meta_learner and self.enable_meta_learning:
                # Average importance across ensemble
                importances = []
                for learner in self.meta_learner.base_learners:
                    imp = learner.get_feature_importance()
                    if len(imp) > 0:
                        importances.append(imp)
                
                if importances:
                    return np.mean(importances, axis=0)
                else:
                    return np.array([])
            else:
                return self.primary_learner.get_feature_importance()
    
    def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get current adaptation metrics."""
        with self.update_lock:
            # Compute metrics
            recent_performance = list(self.performance_history)[-20:] if self.performance_history else [0.5]
            
            if len(recent_performance) > 10:
                early_perf = np.mean(recent_performance[:len(recent_performance)//2])
                late_perf = np.mean(recent_performance[len(recent_performance)//2:])
                accuracy_improvement = late_perf - early_perf
            else:
                accuracy_improvement = 0.0
            
            # Learning rate
            if hasattr(self.primary_learner, 'current_learning_rate'):
                learning_rate = self.primary_learner.current_learning_rate
            else:
                learning_rate = 0.01
            
            # Convergence rate (how quickly performance stabilizes)
            if len(recent_performance) > 5:
                convergence_rate = 1.0 / (1.0 + np.std(recent_performance))
            else:
                convergence_rate = 0.5
            
            # Adaptation speed (feedback processing rate)
            feedback_count = len(self.feedback_buffer)
            adaptation_speed = min(feedback_count / max(self.feedback_buffer_size, 1), 1.0)
            
            # Stability score
            if len(self.feature_importance_history) > 5:
                importance_arrays = list(self.feature_importance_history)
                if importance_arrays and len(importance_arrays[0]) > 0:
                    importance_matrix = np.array(importance_arrays)
                    stability_score = 1.0 / (1.0 + np.mean(np.std(importance_matrix, axis=0)))
                else:
                    stability_score = 0.5
            else:
                stability_score = 0.5
            
            return AdaptationMetrics(
                learning_rate=learning_rate,
                accuracy_improvement=accuracy_improvement,
                convergence_rate=convergence_rate,
                adaptation_speed=adaptation_speed,
                stability_score=stability_score,
                memory_usage_mb=self._estimate_memory_usage(),
                processing_time_ms=0.0,  # Would need to track actual processing times
                feedback_utilization=min(feedback_count / max(self.min_feedback_for_update, 1), 1.0)
            )
    
    def _process_feedback_loop(self):
        """Main feedback processing loop (runs in separate thread)."""
        while self.is_running:
            try:
                # Collect feedback batch
                feedback_batch = []
                
                # Get feedback from queue (blocking with timeout)
                try:
                    feedback = self.feedback_queue.get(timeout=1.0)
                    feedback_batch.append(feedback)
                    self.feedback_buffer.append(feedback)
                    
                    # Try to get more feedback (non-blocking)
                    while len(feedback_batch) < self.min_feedback_for_update:
                        try:
                            feedback = self.feedback_queue.get_nowait()
                            feedback_batch.append(feedback)
                            self.feedback_buffer.append(feedback)
                        except queue.Empty:
                            break
                
                except queue.Empty:
                    continue
                
                # Process feedback if we have enough
                if len(feedback_batch) >= self.min_feedback_for_update:
                    self._update_models(feedback_batch)
                
            except Exception as e:
                logger.error(f"Error in feedback processing loop: {e}")
                time.sleep(1.0)
    
    def _update_models(self, feedback_batch: List[FeedbackSignal]):
        """Update models with feedback batch."""
        with self.update_lock:
            try:
                # Prepare training data
                features = []
                labels = []
                weights = []
                
                for feedback in feedback_batch:
                    # Extract features (simplified - would need actual feature extraction)
                    sample_features = np.array([
                        feedback.predicted_quality,
                        feedback.prediction_error,
                        len(feedback.cleaning_actions),
                        feedback.feedback_weight,
                        time.time() - feedback.timestamp
                    ])
                    
                    features.append(sample_features)
                    labels.append(1 if feedback.actual_quality > 0.7 else 0)
                    weights.append(feedback.feedback_weight)
                
                if features:
                    X = np.array(features)
                    y = np.array(labels)
                    sample_weights = np.array(weights)
                    
                    # Update primary learner
                    self.primary_learner.partial_fit(X, y, sample_weights)
                    
                    # Update meta-learner
                    if self.meta_learner and self.enable_meta_learning:
                        # Extract meta-features
                        meta_features = self.meta_learner.extract_meta_features(X, y)
                        
                        # Update all base learners
                        performances = []
                        for i, learner in enumerate(self.meta_learner.base_learners):
                            learner.partial_fit(X, y, sample_weights)
                            
                            # Evaluate performance
                            predictions = learner.predict(X)
                            accuracy = accuracy_score(y, predictions, sample_weight=sample_weights)
                            performances.append(accuracy)
                            
                            # Track performance
                            self.meta_learner.strategy_performance[i].append(accuracy)
                        
                        # Update ensemble weights
                        self.meta_learner.update_ensemble_weights(performances)
                        
                        # Track overall performance
                        ensemble_pred = self.meta_learner.get_ensemble_prediction(X)
                        ensemble_accuracy = accuracy_score(y, ensemble_pred, sample_weight=sample_weights)
                        self.performance_history.append(ensemble_accuracy)
                    else:
                        # Track primary learner performance
                        predictions = self.primary_learner.predict(X)
                        accuracy = accuracy_score(y, predictions, sample_weight=sample_weights)
                        self.performance_history.append(accuracy)
                    
                    # Update feature importance history
                    current_importance = self.get_feature_importance()
                    if len(current_importance) > 0:
                        self.feature_importance_history.append(current_importance)
                    
                    logger.info(f"Updated models with {len(feedback_batch)} feedback samples")
                
            except Exception as e:
                logger.error(f"Error updating models: {e}")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Simplified memory estimation
        base_size = 1.0  # Base system size
        
        # Feedback buffer
        feedback_size = len(self.feedback_buffer) * 0.001  # ~1KB per feedback
        
        # Model sizes (rough estimates)
        model_size = 0.5  # Models are relatively small
        
        # History buffers
        history_size = (len(self.performance_history) + 
                       len(self.feature_importance_history)) * 0.0001
        
        return base_size + feedback_size + model_size + history_size
    
    def save_state(self, filepath: str):
        """Save the current state of the adaptive learning system."""
        state = {
            'primary_learner': self.primary_learner,
            'meta_learner': self.meta_learner,
            'feedback_buffer': list(self.feedback_buffer),
            'performance_history': list(self.performance_history),
            'feature_importance_history': list(self.feature_importance_history),
            'adaptation_metrics': self.adaptation_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Adaptive learning state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load a previously saved state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.primary_learner = state['primary_learner']
        self.meta_learner = state['meta_learner']
        self.feedback_buffer = deque(state['feedback_buffer'], maxlen=self.feedback_buffer_size)
        self.performance_history = deque(state['performance_history'], maxlen=self.performance_window)
        self.feature_importance_history = deque(state['feature_importance_history'], maxlen=50)
        self.adaptation_metrics = state['adaptation_metrics']
        
        logger.info(f"Adaptive learning state loaded from {filepath}")


def create_adaptive_system(
    enable_meta_learning: bool = True,
    adaptation_rate: float = 0.1,
    feedback_buffer_size: int = 1000
) -> AdaptiveLearningSystem:
    """Create a complete adaptive learning system for data cleaning.
    
    Args:
        enable_meta_learning: Whether to enable meta-learning across datasets
        adaptation_rate: Rate of adaptation to new feedback
        feedback_buffer_size: Size of feedback buffer
        
    Returns:
        Configured adaptive learning system
    """
    system = AdaptiveLearningSystem(
        adaptation_rate=adaptation_rate,
        feedback_buffer_size=feedback_buffer_size,
        enable_meta_learning=enable_meta_learning
    )
    
    logger.info("Created adaptive learning system")
    return system