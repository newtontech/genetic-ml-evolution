"""
Surrogate Model for Architecture Performance Prediction

This module implements a surrogate model that predicts the performance of
neural network architectures without actually training them.

Features:
- Architecture encoding (Transformer, CNN, Multimodal)
- Ensemble of ML models (Random Forest, Gradient Boosting, MLP)
- Integration with cache system to avoid re-evaluation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import logging

from .cache_system import ArchitectureCache

logger = logging.getLogger(__name__)


class SurrogateModel:
    """
    Surrogate model for predicting architecture performance.
    
    Uses an ensemble of machine learning models to predict the fitness/performance
    of neural network architectures without actually training them.
    
    Attributes:
        model_type (str): Type of model to use ("rf", "gb", "mlp", or "ensemble")
        cache (ArchitectureCache): Optional cache system for storing predictions
        training_data (list): List of (architecture, fitness) tuples
        is_fitted (bool): Whether the model has been trained
        best_model_name (str): Name of the best performing model
    
    Example:
        >>> model = SurrogateModel(model_type="ensemble")
        >>> model.add_training_point({"type": "transformer", "num_layers": 6}, 85.0)
        >>> model.fit()
        >>> prediction = model.predict({"type": "transformer", "num_layers": 4})
    """
    
    # Activation function options for one-hot encoding
    TRANSFORMER_ACTIVATIONS = ["relu", "gelu", "silu"]
    CNN_ACTIVATIONS = ["relu", "leaky_relu", "silu"]
    CNN_POOLING_TYPES = ["max", "avg", "adaptive"]
    MULTIMODAL_FUSION_TYPES = ["concat", "attention", "bilinear", "cross"]
    
    def __init__(
        self, 
        model_type: str = "ensemble",
        cache_db_path: Optional[str] = None
    ):
        """
        Initialize the surrogate model.
        
        Args:
            model_type: Type of model to use
                - "rf": Random Forest only
                - "gb": Gradient Boosting only
                - "mlp": Multi-Layer Perceptron only
                - "ensemble": Use all three and select the best (default)
            cache_db_path: Path to SQLite cache database. If provided, 
                            enables caching of predictions.
        """
        self.model_type = model_type
        self.cache = ArchitectureCache(cache_db_path) if cache_db_path else None
        self.training_data: List[Tuple[Dict[str, Any], float]] = []
        self.is_fitted = False
        self.best_model_name: Optional[str] = None
        self.best_score = float('-inf')
        
        # Initialize models
        self.models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
    
    def _one_hot_encode(self, value: str, options: List[str]) -> List[int]:
        """
        One-hot encode a categorical value.
        
        Args:
            value: The value to encode
            options: List of possible options
            
        Returns:
            One-hot encoded list (1 at the position of value, 0 elsewhere)
        """
        encoding = [0] * len(options)
        if value in options:
            encoding[options.index(value)] = 1
        return encoding
    
    def _encode_transformer(self, arch: Dict[str, Any]) -> List[float]:
        """
        Encode a Transformer architecture into a feature vector.
        
        Features (11 total):
        - num_layers (1)
        - hidden_size / 1000 (1)
        - num_heads (1)
        - ffn_dim / 3000 (1)
        - dropout (1)
        - vocab_size / 100000 (1)
        - max_seq_len / 1000 (1)
        - activation one-hot (4: relu, gelu, silu, other)
        
        Args:
            arch: Architecture dictionary
            
        Returns:
            Feature vector of length 11
        """
        features = []
        
        # Numerical features (normalized)
        features.append(arch.get("num_layers", 6))
        features.append(arch.get("hidden_size", 512) / 1000.0)
        features.append(arch.get("num_heads", 8))
        features.append(arch.get("ffn_dim", 2048) / 3000.0)
        features.append(arch.get("dropout", 0.1))
        features.append(arch.get("vocab_size", 50257) / 100000.0)
        features.append(arch.get("max_seq_len", 512) / 1000.0)
        
        # Activation function (one-hot)
        activation = arch.get("activation", "gelu")
        features.extend(self._one_hot_encode(activation, self.TRANSFORMER_ACTIVATIONS))
        
        # Pad to 4 if needed (for unknown activations)
        while len(features) < 11:
            features.append(0)
        
        return features[:11]
    
    def _encode_cnn(self, arch: Dict[str, Any]) -> List[float]:
        """
        Encode a CNN architecture into a feature vector.
        
        Features (14 total):
        - num_blocks (1)
        - base_channels / 200 (1)
        - kernel_size (1)
        - stride (1)
        - use_batch_norm (1, as float)
        - num_classes / 100 (1)
        - input_channels (1)
        - input_size / 100 (1)
        - activation one-hot (3: relu, leaky_relu, silu)
        - pooling one-hot (3: max, avg, adaptive)
        
        Args:
            arch: Architecture dictionary
            
        Returns:
            Feature vector of length 14
        """
        features = []
        
        # Numerical features
        features.append(arch.get("num_blocks", 4))
        features.append(arch.get("base_channels", 64) / 200.0)
        features.append(arch.get("kernel_size", 3))
        features.append(arch.get("stride", 1))
        features.append(1.0 if arch.get("use_batch_norm", True) else 0.0)
        features.append(arch.get("num_classes", 10) / 100.0)
        features.append(arch.get("input_channels", 3))
        features.append(arch.get("input_size", 32) / 100.0)
        
        # Activation function (one-hot)
        activation = arch.get("activation", "relu")
        features.extend(self._one_hot_encode(activation, self.CNN_ACTIVATIONS))
        
        # Pooling type (one-hot)
        pooling = arch.get("pooling", "max")
        features.extend(self._one_hot_encode(pooling, self.CNN_POOLING_TYPES))
        
        return features
    
    def _encode_multimodal(self, arch: Dict[str, Any]) -> List[float]:
        """
        Encode a Multimodal architecture into a feature vector.
        
        Features (12 total):
        - vision num_blocks (1)
        - vision base_channels / 200 (1)
        - text num_layers (1)
        - text hidden_size / 1000 (1)
        - fusion_dim / 1000 (1)
        - projection_dim / 1000 (1)
        - temperature (1)
        - use_contrastive (1, as float)
        - fusion_type one-hot (4: concat, attention, bilinear, cross)
        
        Args:
            arch: Architecture dictionary
            
        Returns:
            Feature vector of length 12
        """
        features = []
        
        vision_encoder = arch.get("vision_encoder", {})
        text_encoder = arch.get("text_encoder", {})
        
        # Vision encoder features
        features.append(vision_encoder.get("num_blocks", 3))
        features.append(vision_encoder.get("base_channels", 32) / 200.0)
        
        # Text encoder features
        features.append(text_encoder.get("num_layers", 4))
        features.append(text_encoder.get("hidden_size", 256) / 1000.0)
        
        # Fusion features
        features.append(arch.get("fusion_dim", 512) / 1000.0)
        features.append(arch.get("projection_dim", 256) / 1000.0)
        features.append(arch.get("temperature", 0.1))
        features.append(1.0 if arch.get("use_contrastive", True) else 0.0)
        
        # Fusion type (one-hot)
        fusion_type = arch.get("fusion_type", "attention")
        features.extend(self._one_hot_encode(fusion_type, self.MULTIMODAL_FUSION_TYPES))
        
        return features
    
    def _architecture_to_features(self, architecture: Dict[str, Any]) -> np.ndarray:
        """
        Convert an architecture dictionary to a feature vector.
        
        Args:
            architecture: Architecture configuration dictionary
            
        Returns:
            Numpy array of features
        """
        arch_type = architecture.get("type", "unknown")
        
        if arch_type == "transformer":
            features = self._encode_transformer(architecture)
        elif arch_type == "cnn":
            features = self._encode_cnn(architecture)
        elif arch_type == "multimodal":
            features = self._encode_multimodal(architecture)
        else:
            # Unknown type: return default feature vector
            logger.warning(f"Unknown architecture type: {arch_type}")
            features = [0.0] * 20
        
        return np.array(features, dtype=np.float64)
    
    def add_training_point(
        self, 
        architecture: Dict[str, Any], 
        fitness: float
    ) -> None:
        """
        Add a training data point.
        
        Args:
            architecture: Architecture configuration
            fitness: Fitness/performance score (0-100 scale)
        """
        self.training_data.append((architecture, fitness))
        self.is_fitted = False  # Reset fitted status when new data is added
    
    def fit(self) -> bool:
        """
        Train the surrogate model on collected training data.
        
        Returns:
            True if training succeeded, False if insufficient data
        """
        if len(self.training_data) < 5:
            logger.warning(
                f"Insufficient training data: {len(self.training_data)} samples. "
                f"Need at least 5 samples."
            )
            return False
        
        # Extract features and labels
        X = np.array([
            self._architecture_to_features(arch) 
            for arch, _ in self.training_data
        ])
        y = np.array([fitness for _, fitness in self.training_data])
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Use cross-validation to evaluate
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
                model_scores[name] = -scores.mean()  # Convert to positive MSE
                logger.debug(f"Model {name} CV MSE: {model_scores[name]:.4f}")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                model_scores[name] = float('inf')
        
        if not model_scores:
            logger.error("All models failed to train")
            return False
        
        # Select best model
        self.best_model_name = min(model_scores, key=model_scores.get)
        self.best_score = model_scores[self.best_model_name]
        
        # Fit the best model on all data
        if self.model_type == "ensemble":
            # Use the best model from ensemble
            self.models[self.best_model_name].fit(X_scaled, y)
        else:
            # Use the specified model type
            if self.model_type in self.models:
                self.models[self.model_type].fit(X_scaled, y)
                self.best_model_name = self.model_type
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return False
        
        self.is_fitted = True
        logger.info(
            f"Surrogate model trained successfully. "
            f"Best model: {self.best_model_name} (MSE: {self.best_score:.4f})"
        )
        
        return True
    
    def predict(self, architecture: Dict[str, Any]) -> Optional[float]:
        """
        Predict the performance of an architecture.
        
        First checks the cache if available, then uses the surrogate model.
        
        Args:
            architecture: Architecture configuration
            
        Returns:
            Predicted fitness score, Clamps result to [0, 100] range.
            Returns None if model is not fitted.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.lookup(architecture)
            if cached is not None:
                logger.debug(f"Cache hit for architecture prediction")
                return cached.get("fitness")
        
        if not self.is_fitted:
            logger.warning("Model not fitted yet. Call fit() first.")
            return None
        
        # Extract features
        features = self._architecture_to_features(architecture)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict using the best model
        prediction = self.models[self.best_model_name].predict(features_scaled)[0]
        
        # Clamp to valid range
        prediction = float(np.clip(prediction, 0.0, 100.0))
        
        return prediction
    
    def predict_batch(
        self, 
        architectures: List[Dict[str, Any]]
    ) -> List[Optional[float]]:
        """
        Predict performance for multiple architectures.
        
        Args:
            architectures: List of architecture configurations
            
        Returns:
            List of predicted fitness scores (None for uncached if model not fitted)
        """
        if not self.is_fitted:
            # Try cache only
            if self.cache:
                return [
                    self.cache.lookup(arch).get("fitness") if self.cache.lookup(arch) else None
                    for arch in architectures
                ]
            return [None] * len(architectures)
        
        predictions = []
        uncached_indices = []
        uncached_archs = []
        
        # Check cache for all architectures
        for i, arch in enumerate(architectures):
            if self.cache:
                cached = self.cache.lookup(arch)
                if cached is not None:
                    predictions.append(cached.get("fitness"))
                    continue
            
            predictions.append(None)
            uncached_indices.append(i)
            uncached_archs.append(arch)
        
        # Predict uncached architectures
        if uncached_archs:
            features = np.array([
                self._architecture_to_features(arch) 
                for arch in uncached_archs
            ])
            features_scaled = self.scaler.transform(features)
            uncached_preds = self.models[self.best_model_name].predict(features_scaled)
            uncached_preds = np.clip(uncached_preds, 0.0, 100.0)
            
            # Fill in predictions
            for idx, pred in zip(uncached_indices, uncached_preds):
                predictions[idx] = float(pred)
        
        return predictions
    
    def store_prediction(
        self, 
        architecture: Dict[str, Any], 
        fitness: float,
        evaluation_time: Optional[float] = None
    ) -> bool:
        """
        Store a prediction result in the cache.
        
        Args:
            architecture: Architecture configuration
            fitness: Actual fitness score
            evaluation_time: Time taken for evaluation (optional)
            
        Returns:
            True if stored successfully, False if cache not available
        """
        if not self.cache:
            logger.warning("No cache configured. Cannot store prediction.")
            return False
        
        metrics = {"fitness": fitness}
        return self.cache.store(architecture, metrics, evaluation_time)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary, or empty dict if cache not available
        """
        if not self.cache:
            return {}
        
        return self.cache.get_statistics()
