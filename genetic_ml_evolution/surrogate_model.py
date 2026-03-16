"""
Surrogate Model - 代理模型
用于快速预测架构性能，减少实际训练次数
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class SurrogateModel:
    """代理模型 - 预测架构性能"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: List[tuple] = []
        self.is_fitted = False
        self.best_model_name: Optional[str] = None
        self.best_score = float('-inf')
    
    def _architecture_to_features(self, architecture: Dict[str, Any]) -> np.ndarray:
        """将架构转换为特征向量"""
        encoders = {
            "transformer": self._encode_transformer,
            "cnn": self._encode_cnn,
            "multimodal": self._encode_multimodal,
        }
        
        arch_type = architecture.get("type", "unknown")
        encoder = encoders.get(arch_type, lambda _: [0.0] * 20)
        return np.array(encoder(architecture))
    
    def _one_hot_encode(self, value: str, options: List[str]) -> List[int]:
        """One-hot编码"""
        return [1 if value == opt else 0 for opt in options]
    
    def _encode_transformer(self, arch: Dict[str, Any]) -> List[float]:
        """编码Transformer架构特征"""
        features = [
            arch.get("num_layers", 6),
            arch.get("hidden_size", 512) / 1000.0,
            arch.get("num_heads", 8),
            arch.get("ffn_dim", 2048) / 3000.0,
            arch.get("dropout", 0.1),
            arch.get("vocab_size", 50257) / 100000.0,
            arch.get("max_seq_len", 512) / 1000.0,
        ]
        features.extend(self._one_hot_encode(arch.get("activation", ""), ["relu", "gelu", "silu", "other"]))
        return features
    
    def _encode_cnn(self, arch: Dict[str, Any]) -> List[float]:
        """编码CNN架构特征"""
        features = [
            arch.get("num_blocks", 4),
            arch.get("base_channels", 64) / 200.0,
            arch.get("kernel_size", 3),
            arch.get("stride", 1),
            1.0 if arch.get("use_batch_norm", False) else 0.0,
            arch.get("num_classes", 10) / 100.0,
            arch.get("input_channels", 3),
            arch.get("input_size", 32) / 100.0,
        ]
        features.extend(self._one_hot_encode(arch.get("activation", ""), ["relu", "leaky_relu", "other"]))
        features.extend(self._one_hot_encode(arch.get("pooling", ""), ["max", "avg", "other"]))
        return features
    
    def _encode_multimodal(self, arch: Dict[str, Any]) -> List[float]:
        """编码多模态架构特征"""
        vision = arch.get("vision_encoder", {})
        text = arch.get("text_encoder", {})
        
        features = [
            vision.get("num_blocks", 4),
            vision.get("base_channels", 64) / 200.0,
            text.get("num_layers", 6),
            text.get("hidden_size", 512) / 1000.0,
            arch.get("fusion_dim", 512) / 1000.0,
            arch.get("projection_dim", 256) / 1000.0,
            arch.get("temperature", 0.07),
            1.0 if arch.get("use_contrastive", False) else 0.0,
        ]
        features.extend(self._one_hot_encode(arch.get("fusion_type", ""), ["concat", "attention", "bilinear", "other"]))
        return features
    
    def add_training_point(self, architecture: Dict[str, Any], fitness: float) -> None:
        """添加训练样本"""
        features = self._architecture_to_features(architecture)
        self.training_data.append((features, fitness))
        self.is_fitted = False
    
    def fit(self) -> bool:
        """训练代理模型 - 包含多种模型比较"""
        if len(self.training_data) < 5:
            return False
        
        X = np.array([x[0] for x in self.training_data])
        y = np.array([x[1] for x in self.training_data])
        
        if len(X) < 10:
            return False
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers["default"] = scaler
        
        # 尝试多种模型
        models_to_try = {
            "rf": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            "gb": GradientBoostingRegressor(n_estimators=50, random_state=42),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
        
        self.best_score = float('-inf')
        self.best_model_name = None
        
        for name, model in models_to_try.items():
            try:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_val_scaled, y_val)
                self.models[name] = model
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model_name = name
            except Exception:
                continue
        
        self.is_fitted = self.best_model_name is not None
        return self.is_fitted
    
    def predict(self, architecture: Dict[str, Any]) -> Optional[float]:
        """预测架构性能"""
        if not self.is_fitted or self.best_model_name is None:
            return None
        
        features = self._architecture_to_features(architecture)
        features_scaled = self.scalers["default"].transform(features.reshape(1, -1))
        
        model = self.models[self.best_model_name]
        prediction = model.predict(features_scaled)[0]
        
        return max(0.0, min(100.0, prediction))
    
    def predict_batch(self, architectures: List[Dict[str, Any]]) -> List[Optional[float]]:
        """批量预测"""
        if not self.is_fitted:
            return [None] * len(architectures)
        
        features_list = [self._architecture_to_features(arch) for arch in architectures]
        X = np.array(features_list)
        X_scaled = self.scalers["default"].transform(X)
        
        if self.best_model_name is None:
            return [None] * len(architectures)
        model = self.models[self.best_model_name]
        predictions = model.predict(X_scaled)
        
        return [max(0.0, min(100.0, p)) for p in predictions]
