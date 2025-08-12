"""Federated Learning for Data Quality - Research Module.

Implements federated learning approaches for collaborative data quality improvement
without sharing sensitive data. Based on cutting-edge research in federated ML
and privacy-preserving data cleaning.

Research Areas:
- Federated Data Quality Assessment
- Privacy-Preserving Rule Learning
- Collaborative Anomaly Detection
- Secure Multi-Party Data Cleaning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning system."""
    max_clients: int = 10
    min_clients: int = 3
    rounds: int = 50
    client_sample_fraction: float = 0.8
    learning_rate: float = 0.01
    privacy_budget: float = 1.0  # Differential privacy budget
    noise_multiplier: float = 1.1
    max_gradient_norm: float = 1.0
    secure_aggregation: bool = True
    homomorphic_encryption: bool = False
    model_compression: bool = True
    byzantine_robust: bool = True
    communication_rounds: int = 5


@dataclass
class ClientUpdate:
    """Update from a federated client."""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    training_loss: float
    accuracy: float
    gradient_norm: float
    timestamp: float
    privacy_spent: float
    
    def __post_init__(self):
        """Add computed fields."""
        self.weight_hash = self._compute_weight_hash()
    
    def _compute_weight_hash(self) -> str:
        """Compute hash of model weights for integrity verification."""
        weight_str = json.dumps(
            {k: v.tolist() for k, v in self.model_weights.items()},
            sort_keys=True
        )
        return hashlib.sha256(weight_str.encode()).hexdigest()[:16]


@dataclass
class FederatedMetrics:
    """Metrics from federated training round."""
    round_number: int
    participating_clients: int
    average_accuracy: float
    accuracy_std: float
    average_loss: float
    loss_std: float
    convergence_score: float
    communication_cost: float
    privacy_spent: float
    byzantine_attacks_detected: int
    model_divergence: float


class PrivacyMechanism:
    """Privacy-preserving mechanisms for federated learning."""
    
    @staticmethod
    def add_gaussian_noise(
        weights: np.ndarray,
        noise_multiplier: float,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """Add Gaussian noise for differential privacy."""
        noise_scale = noise_multiplier * sensitivity
        noise = np.random.normal(0, noise_scale, weights.shape)
        return weights + noise
    
    @staticmethod
    def clip_gradients(gradients: np.ndarray, max_norm: float) -> np.ndarray:
        """Clip gradients to bound sensitivity."""
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > max_norm:
            return gradients * (max_norm / grad_norm)
        return gradients
    
    @staticmethod
    def local_differential_privacy(
        value: float,
        epsilon: float,
        sensitivity: float = 1.0
    ) -> float:
        """Apply local differential privacy to a value."""
        noise_scale = sensitivity / epsilon
        noise = np.random.laplace(0, noise_scale)
        return value + noise


class SecureAggregation:
    """Secure aggregation for federated learning."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_keys = {}
        self.masked_updates = {}
    
    def generate_keys(self, client_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate key pair for a client."""
        private_key = np.random.randint(0, 2**16, size=100)
        public_key = np.random.randint(0, 2**16, size=100)
        self.client_keys[client_id] = (private_key, public_key)
        return private_key, public_key
    
    def mask_update(self, client_id: str, update: np.ndarray) -> np.ndarray:
        """Mask client update with cryptographic noise."""
        if client_id not in self.client_keys:
            raise ValueError(f"Keys not generated for client {client_id}")
        
        private_key, _ = self.client_keys[client_id]
        # Simple masking (in practice, use proper cryptographic protocols)
        mask = np.random.RandomState(int(private_key[0])).randint(
            -1000, 1000, size=update.shape
        )
        masked = update + mask
        self.masked_updates[client_id] = masked
        return masked
    
    def aggregate_masked_updates(self) -> np.ndarray:
        """Aggregate masked updates securely."""
        if not self.masked_updates:
            raise ValueError("No masked updates to aggregate")
        
        # Sum all masked updates
        total = sum(self.masked_updates.values())
        
        # Remove masks (simplified - in practice, use threshold schemes)
        total_mask = sum(
            np.random.RandomState(int(keys[0][0])).randint(
                -1000, 1000, size=total.shape
            )
            for keys in self.client_keys.values()
        )
        
        return total - total_mask


class ByzantineDefense:
    """Defense mechanisms against Byzantine attacks."""
    
    @staticmethod
    def coordinate_wise_median(updates: List[np.ndarray]) -> np.ndarray:
        """Aggregate using coordinate-wise median."""
        stacked = np.stack(updates, axis=0)
        return np.median(stacked, axis=0)
    
    @staticmethod
    def trimmed_mean(
        updates: List[np.ndarray],
        trim_ratio: float = 0.2
    ) -> np.ndarray:
        """Aggregate using trimmed mean."""
        stacked = np.stack(updates, axis=0)
        num_trim = int(len(updates) * trim_ratio)
        
        # Sort along each coordinate and trim extremes
        sorted_updates = np.sort(stacked, axis=0)
        if num_trim > 0:
            trimmed = sorted_updates[num_trim:-num_trim]
        else:
            trimmed = sorted_updates
        
        return np.mean(trimmed, axis=0)
    
    @staticmethod
    def detect_byzantine_clients(
        updates: List[ClientUpdate],
        threshold: float = 2.0
    ) -> List[str]:
        """Detect potentially Byzantine clients based on update statistics."""
        if len(updates) < 3:
            return []  # Need at least 3 clients for detection
        
        byzantine_clients = []
        
        # Analyze gradient norms
        grad_norms = [update.gradient_norm for update in updates]
        median_norm = np.median(grad_norms)
        mad = np.median(np.abs(grad_norms - median_norm))
        
        for update in updates:
            # Modified Z-score using MAD
            if mad > 0:
                z_score = 0.6745 * (update.gradient_norm - median_norm) / mad
                if abs(z_score) > threshold:
                    byzantine_clients.append(update.client_id)
                    logger.warning(f"Byzantine client detected: {update.client_id} (z-score: {z_score:.2f})")
        
        return byzantine_clients


class FederatedDataQualityClient:
    """Client for federated data quality learning."""
    
    def __init__(
        self,
        client_id: str,
        config: FederatedConfig,
        privacy_mechanism: Optional[PrivacyMechanism] = None
    ):
        self.client_id = client_id
        self.config = config
        self.privacy_mechanism = privacy_mechanism or PrivacyMechanism()
        self.local_model = None
        self.local_data = None
        self.privacy_spent = 0.0
        self.training_history = []
    
    def load_local_data(self, data: pd.DataFrame, quality_labels: np.ndarray):
        """Load local training data."""
        self.local_data = data
        self.quality_labels = quality_labels
        logger.info(f"Client {self.client_id} loaded {len(data)} samples")
    
    def initialize_model(self, global_model_weights: Optional[Dict[str, np.ndarray]] = None):
        """Initialize local model."""
        # Simple logistic regression for data quality classification
        self.local_model = LogisticRegression(random_state=42)
        
        if global_model_weights and self.local_data is not None:
            # Initialize with global weights (simplified)
            self.local_model.fit(
                self.local_data.select_dtypes(include=[np.number]).fillna(0),
                self.quality_labels
            )
    
    def local_training_round(
        self,
        global_weights: Optional[Dict[str, np.ndarray]] = None
    ) -> ClientUpdate:
        """Perform local training round."""
        if self.local_data is None or self.local_model is None:
            raise ValueError("Client not properly initialized")
        
        start_time = time.time()
        
        # Prepare features
        X = self.local_data.select_dtypes(include=[np.number]).fillna(0)
        y = self.quality_labels
        
        # Update model with global weights if provided
        if global_weights:
            self._update_model_weights(global_weights)
        
        # Train locally
        self.local_model.fit(X, y)
        
        # Get predictions and compute metrics
        predictions = self.local_model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        # Extract model weights (simplified)
        model_weights = {
            "coef": self.local_model.coef_,
            "intercept": self.local_model.intercept_
        }
        
        # Apply privacy mechanisms
        if self.config.privacy_budget > 0:
            for key, weights in model_weights.items():
                # Clip gradients
                clipped = self.privacy_mechanism.clip_gradients(
                    weights, self.config.max_gradient_norm
                )
                
                # Add noise
                noisy_weights = self.privacy_mechanism.add_gaussian_noise(
                    clipped, self.config.noise_multiplier
                )
                model_weights[key] = noisy_weights
            
            # Update privacy budget
            privacy_cost = self.config.noise_multiplier / len(X)
            self.privacy_spent += privacy_cost
        
        # Compute gradient norm
        total_weights = np.concatenate([w.flatten() for w in model_weights.values()])
        gradient_norm = np.linalg.norm(total_weights)
        
        training_time = time.time() - start_time
        
        update = ClientUpdate(
            client_id=self.client_id,
            model_weights=model_weights,
            num_samples=len(X),
            training_loss=0.0,  # Simplified
            accuracy=accuracy,
            gradient_norm=gradient_norm,
            timestamp=time.time(),
            privacy_spent=self.privacy_spent
        )
        
        self.training_history.append({
            "round": len(self.training_history),
            "accuracy": accuracy,
            "training_time": training_time,
            "privacy_spent": self.privacy_spent
        })
        
        logger.info(f"Client {self.client_id} training complete. Accuracy: {accuracy:.3f}")
        return update
    
    def _update_model_weights(self, global_weights: Dict[str, np.ndarray]):
        """Update local model with global weights."""
        if "coef" in global_weights:
            self.local_model.coef_ = global_weights["coef"]
        if "intercept" in global_weights:
            self.local_model.intercept_ = global_weights["intercept"]


class FederatedDataQualityServer:
    """Server for federated data quality learning."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients = {}
        self.global_model_weights = None
        self.round_history = []
        self.secure_aggregator = None
        self.byzantine_defense = ByzantineDefense()
        
        if config.secure_aggregation:
            self.secure_aggregator = SecureAggregation(config.max_clients)
    
    def register_client(self, client: FederatedDataQualityClient):
        """Register a new client."""
        if len(self.clients) >= self.config.max_clients:
            raise ValueError("Maximum number of clients reached")
        
        self.clients[client.client_id] = client
        
        if self.secure_aggregator:
            private_key, public_key = self.secure_aggregator.generate_keys(client.client_id)
            logger.info(f"Client {client.client_id} registered with secure aggregation")
        else:
            logger.info(f"Client {client.client_id} registered")
    
    def federated_training(self) -> List[FederatedMetrics]:
        """Run complete federated training process."""
        if len(self.clients) < self.config.min_clients:
            raise ValueError(f"Need at least {self.config.min_clients} clients")
        
        logger.info(f"Starting federated training with {len(self.clients)} clients")
        metrics_history = []
        
        for round_num in range(self.config.rounds):
            round_metrics = self._training_round(round_num)
            metrics_history.append(round_metrics)
            
            # Check convergence
            if (len(metrics_history) > 5 and 
                round_metrics.convergence_score < 0.01):
                logger.info(f"Converged after {round_num + 1} rounds")
                break
        
        return metrics_history
    
    def _training_round(self, round_num: int) -> FederatedMetrics:
        """Execute a single training round."""
        logger.info(f"=== Federated Round {round_num + 1} ===")
        
        # Select clients for this round
        selected_clients = self._select_clients()
        
        # Collect updates from clients
        client_updates = []
        
        with ThreadPoolExecutor(max_workers=min(len(selected_clients), 10)) as executor:
            future_to_client = {
                executor.submit(
                    client.local_training_round,
                    self.global_model_weights
                ): client_id
                for client_id, client in selected_clients.items()
            }
            
            for future in as_completed(future_to_client):
                try:
                    update = future.result(timeout=60)
                    client_updates.append(update)
                except Exception as e:
                    client_id = future_to_client[future]
                    logger.error(f"Client {client_id} failed: {e}")
        
        # Detect Byzantine clients
        byzantine_clients = []
        if self.config.byzantine_robust:
            byzantine_clients = self.byzantine_defense.detect_byzantine_clients(
                client_updates
            )
            # Remove Byzantine updates
            client_updates = [
                update for update in client_updates
                if update.client_id not in byzantine_clients
            ]
        
        # Aggregate updates
        if client_updates:
            self.global_model_weights = self._aggregate_updates(client_updates)
        
        # Compute round metrics
        metrics = self._compute_round_metrics(
            round_num, client_updates, byzantine_clients
        )
        
        self.round_history.append(metrics)
        
        logger.info(
            f"Round {round_num + 1} complete. "
            f"Participants: {metrics.participating_clients}, "
            f"Avg Accuracy: {metrics.average_accuracy:.3f}, "
            f"Byzantine Detected: {metrics.byzantine_attacks_detected}"
        )
        
        return metrics
    
    def _select_clients(self) -> Dict[str, FederatedDataQualityClient]:
        """Select clients for training round."""
        num_select = min(
            len(self.clients),
            max(
                self.config.min_clients,
                int(len(self.clients) * self.config.client_sample_fraction)
            )
        )
        
        selected_ids = np.random.choice(
            list(self.clients.keys()),
            size=num_select,
            replace=False
        )
        
        return {client_id: self.clients[client_id] for client_id in selected_ids}
    
    def _aggregate_updates(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate client updates into global model."""
        if not updates:
            return self.global_model_weights
        
        # Extract weights from updates
        weight_lists = {key: [] for key in updates[0].model_weights.keys()}
        sample_counts = []
        
        for update in updates:
            sample_counts.append(update.num_samples)
            for key, weights in update.model_weights.items():
                weight_lists[key].append(weights)
        
        # Aggregate using Byzantine-robust method if enabled
        aggregated_weights = {}
        
        for key, weights_list in weight_lists.items():
            if self.config.byzantine_robust and len(weights_list) >= 3:
                # Use trimmed mean for robustness
                aggregated = self.byzantine_defense.trimmed_mean(weights_list)
            else:
                # Weighted average by number of samples
                weights_array = np.array(weights_list)
                sample_weights = np.array(sample_counts) / sum(sample_counts)
                aggregated = np.average(weights_array, axis=0, weights=sample_weights)
            
            aggregated_weights[key] = aggregated
        
        return aggregated_weights
    
    def _compute_round_metrics(
        self,
        round_num: int,
        updates: List[ClientUpdate],
        byzantine_clients: List[str]
    ) -> FederatedMetrics:
        """Compute metrics for the training round."""
        if not updates:
            return FederatedMetrics(
                round_number=round_num,
                participating_clients=0,
                average_accuracy=0.0,
                accuracy_std=0.0,
                average_loss=0.0,
                loss_std=0.0,
                convergence_score=1.0,
                communication_cost=0.0,
                privacy_spent=0.0,
                byzantine_attacks_detected=0,
                model_divergence=0.0
            )
        
        accuracies = [update.accuracy for update in updates]
        losses = [update.training_loss for update in updates]
        
        # Compute convergence score
        if len(self.round_history) > 0:
            prev_accuracy = self.round_history[-1].average_accuracy
            current_accuracy = np.mean(accuracies)
            convergence_score = abs(current_accuracy - prev_accuracy)
        else:
            convergence_score = 1.0
        
        # Estimate communication cost (simplified)
        total_params = sum(
            np.prod(weights.shape)
            for update in updates
            for weights in update.model_weights.values()
        )
        communication_cost = total_params * 4  # 4 bytes per float32
        
        # Total privacy spent
        total_privacy = sum(update.privacy_spent for update in updates)
        
        # Model divergence (simplified)
        if len(updates) > 1:
            weight_matrices = []
            for update in updates:
                flat_weights = np.concatenate([
                    w.flatten() for w in update.model_weights.values()
                ])
                weight_matrices.append(flat_weights)
            weight_matrix = np.array(weight_matrices)
            model_divergence = np.std(weight_matrix, axis=0).mean()
        else:
            model_divergence = 0.0
        
        return FederatedMetrics(
            round_number=round_num,
            participating_clients=len(updates),
            average_accuracy=np.mean(accuracies),
            accuracy_std=np.std(accuracies),
            average_loss=np.mean(losses),
            loss_std=np.std(losses),
            convergence_score=convergence_score,
            communication_cost=communication_cost,
            privacy_spent=total_privacy,
            byzantine_attacks_detected=len(byzantine_clients),
            model_divergence=model_divergence
        )


def create_federated_data_quality_system(
    config: Optional[FederatedConfig] = None
) -> Tuple[FederatedDataQualityServer, List[FederatedDataQualityClient]]:
    """Create a complete federated data quality system for research.
    
    Returns:
        Tuple of (server, list of clients) ready for federated learning
    """
    config = config or FederatedConfig()
    
    # Create server
    server = FederatedDataQualityServer(config)
    
    # Create clients
    clients = []
    for i in range(config.max_clients):
        client_id = f"client_{i:03d}"
        client = FederatedDataQualityClient(client_id, config)
        clients.append(client)
        server.register_client(client)
    
    logger.info(f"Created federated system with {len(clients)} clients")
    return server, clients