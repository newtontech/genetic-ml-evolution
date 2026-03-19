"""
Unit Tests for Surrogate Model Architecture Encoding
测试代理模型的架构编码函数
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from genetic_ml_evolution.surrogate_model import SurrogateModel


class TestArchitectureEncoding:
    """测试架构编码函数"""
    
    @pytest.fixture
    def surrogate_model(self):
        """创建代理模型实例"""
        return SurrogateModel(model_type="ensemble")
    
    @pytest.fixture
    def transformer_arch(self) -> Dict[str, Any]:
        """示例 Transformer 架构"""
        return {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
            "activation": "gelu",
            "vocab_size": 50257,
            "max_seq_len": 512
        }
    
    @pytest.fixture
    def cnn_arch(self) -> Dict[str, Any]:
        """示例 CNN 架构"""
        return {
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "use_batch_norm": True,
            "activation": "relu",
            "pooling": "max",
            "num_classes": 10,
            "input_channels": 3,
            "input_size": 32
        }
    
    @pytest.fixture
    def multimodal_arch(self) -> Dict[str, Any]:
        """示例多模态架构"""
        return {
            "type": "multimodal",
            "vision_encoder": {
                "type": "cnn",
                "num_blocks": 3,
                "base_channels": 32,
                "kernel_size": 3
            },
            "text_encoder": {
                "type": "transformer",
                "num_layers": 4,
                "hidden_size": 256,
                "num_heads": 4
            },
            "fusion_type": "attention",
            "fusion_dim": 512,
            "projection_dim": 256,
            "temperature": 0.1,
            "use_contrastive": True
        }
    
    # ==================== One-Hot Encoding Tests ====================
    
    def test_one_hot_encode_basic(self, surrogate_model):
        """测试基本的 one-hot 编码"""
        options = ["relu", "gelu", "silu"]
        
        # 测试第一个选项
        result = surrogate_model._one_hot_encode("relu", options)
        assert result == [1, 0, 0]
        
        # 测试中间选项
        result = surrogate_model._one_hot_encode("gelu", options)
        assert result == [0, 1, 0]
        
        # 测试最后一个选项
        result = surrogate_model._one_hot_encode("silu", options)
        assert result == [0, 0, 1]
    
    def test_one_hot_encode_non_existent(self, surrogate_model):
        """测试不存在的选项"""
        options = ["relu", "gelu", "silu"]
        result = surrogate_model._one_hot_encode("tanh", options)
        
        # 应该返回全零
        assert result == [0, 0, 0]
    
    def test_one_hot_encode_empty_options(self, surrogate_model):
        """测试空选项列表"""
        result = surrogate_model._one_hot_encode("relu", [])
        assert result == []
    
    def test_one_hot_encode_single_option(self, surrogate_model):
        """测试单个选项"""
        result = surrogate_model._one_hot_encode("relu", ["relu"])
        assert result == [1]
        
        result = surrogate_model._one_hot_encode("gelu", ["relu"])
        assert result == [0]
    
    # ==================== Transformer Encoding Tests ====================
    
    def test_encode_transformer_basic(self, surrogate_model, transformer_arch):
        """测试 Transformer 架构的基本编码"""
        features = surrogate_model._encode_transformer(transformer_arch)
        
        # 检查返回类型
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
        
        # 检查特征数量（7个数值特征 + 4个one-hot = 11个）
        assert len(features) == 11
        
        # 验证数值特征的值
        assert features[0] == 6  # num_layers
        assert features[1] == pytest.approx(0.512, rel=0.01)  # hidden_size / 1000
        assert features[2] == 8  # num_heads
        assert features[3] == pytest.approx(0.683, rel=0.01)  # ffn_dim / 3000
        assert features[4] == 0.1  # dropout
        assert features[5] == pytest.approx(0.50257, rel=0.01)  # vocab_size / 100000
        assert features[6] == pytest.approx(0.512, rel=0.01)  # max_seq_len / 1000
        
        # 验证 one-hot 编码（gelu）
        assert features[7] == 0  # relu
        assert features[8] == 1  # gelu
        assert features[9] == 0  # silu
        assert features[10] == 0  # other
    
    def test_encode_transformer_different_activations(self, surrogate_model):
        """测试不同激活函数的编码"""
        base_arch = {
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 1024,
            "dropout": 0.2,
            "vocab_size": 30000,
            "max_seq_len": 256
        }
        
        # 测试 relu
        arch_relu = {**base_arch, "activation": "relu"}
        features_relu = surrogate_model._encode_transformer(arch_relu)
        assert features_relu[7:11] == [1, 0, 0, 0]
        
        # 测试 gelu
        arch_gelu = {**base_arch, "activation": "gelu"}
        features_gelu = surrogate_model._encode_transformer(arch_gelu)
        assert features_gelu[7:11] == [0, 1, 0, 0]
        
        # 测试 silu
        arch_silu = {**base_arch, "activation": "silu"}
        features_silu = surrogate_model._encode_transformer(arch_silu)
        assert features_silu[7:11] == [0, 0, 1, 0]
        
        # 测试不在列表中的激活函数（应该返回全零）
        arch_other = {**base_arch, "activation": "tanh"}
        features_other = surrogate_model._encode_transformer(arch_other)
        assert features_other[7:11] == [0, 0, 0, 0]
    
    def test_encode_transformer_missing_fields(self, surrogate_model):
        """测试缺少字段的 Transformer 架构"""
        # 只有 type 字段
        minimal_arch = {"type": "transformer"}
        features = surrogate_model._encode_transformer(minimal_arch)
        
        # 应该使用默认值
        assert len(features) == 11
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_encode_transformer_extreme_values(self, surrogate_model):
        """测试极端值的 Transformer 架构"""
        # 最大值
        max_arch = {
            "type": "transformer",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072,
            "dropout": 0.5,
            "vocab_size": 100000,
            "max_seq_len": 1024
        }
        features_max = surrogate_model._encode_transformer(max_arch)
        assert len(features_max) == 11
        
        # 最小值
        min_arch = {
            "type": "transformer",
            "num_layers": 2,
            "hidden_size": 128,
            "num_heads": 2,
            "ffn_dim": 512,
            "dropout": 0.1,
            "vocab_size": 10000,
            "max_seq_len": 128
        }
        features_min = surrogate_model._encode_transformer(min_arch)
        assert len(features_min) == 11
    
    def test_encode_transformer_consistency(self, surrogate_model, transformer_arch):
        """测试编码的一致性"""
        # 多次编码同一架构应该得到相同结果
        features1 = surrogate_model._encode_transformer(transformer_arch)
        features2 = surrogate_model._encode_transformer(transformer_arch)
        
        assert features1 == features2
    
    # ==================== CNN Encoding Tests ====================
    
    def test_encode_cnn_basic(self, surrogate_model, cnn_arch):
        """测试 CNN 架构的基本编码"""
        features = surrogate_model._encode_cnn(cnn_arch)
        
        # 检查返回类型
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
        
        # 检查特征数量（8个数值特征 + 3个激活函数 + 3个池化 = 14个）
        assert len(features) == 14
        
        # 验证数值特征
        assert features[0] == 4  # num_blocks
        assert features[1] == pytest.approx(0.32, rel=0.01)  # base_channels / 200
        assert features[2] == 3  # kernel_size
        assert features[3] == 1  # stride
        assert features[4] == 1.0  # use_batch_norm (True)
        assert features[5] == pytest.approx(0.1, rel=0.01)  # num_classes / 100
        assert features[6] == 3  # input_channels
        assert features[7] == pytest.approx(0.32, rel=0.01)  # input_size / 100
        
        # 验证激活函数 one-hot (relu)
        assert features[8:11] == [1, 0, 0]
        
        # 验证池化 one-hot (max)
        assert features[11:14] == [1, 0, 0]
    
    def test_encode_cnn_without_batch_norm(self, surrogate_model):
        """测试没有 BatchNorm 的 CNN"""
        arch = {
            "type": "cnn",
            "num_blocks": 3,
            "base_channels": 32,
            "kernel_size": 5,
            "stride": 2,
            "use_batch_norm": False,
            "activation": "leaky_relu",
            "pooling": "avg"
        }
        
        features = surrogate_model._encode_cnn(arch)
        
        # use_batch_norm 应该是 0.0
        assert features[4] == 0.0
        
        # 验证激活函数 (leaky_relu)
        assert features[8:11] == [0, 1, 0]
        
        # 验证池化 (avg)
        assert features[11:14] == [0, 1, 0]
    
    def test_encode_cnn_different_activations(self, surrogate_model):
        """测试不同激活函数的 CNN 编码"""
        base_arch = {
            "type": "cnn",
            "num_blocks": 2,
            "base_channels": 16,
            "kernel_size": 3,
            "use_batch_norm": True
        }
        
        # relu
        arch_relu = {**base_arch, "activation": "relu"}
        features_relu = surrogate_model._encode_cnn(arch_relu)
        assert features_relu[8:11] == [1, 0, 0]
        
        # leaky_relu
        arch_leaky = {**base_arch, "activation": "leaky_relu"}
        features_leaky = surrogate_model._encode_cnn(arch_leaky)
        assert features_leaky[8:11] == [0, 1, 0]
        
        # 不在列表中的激活函数（应该返回全零）
        arch_other = {**base_arch, "activation": "swish"}
        features_other = surrogate_model._encode_cnn(arch_other)
        assert features_other[8:11] == [0, 0, 0]
    
    def test_encode_cnn_different_pooling(self, surrogate_model):
        """测试不同池化类型的 CNN 编码"""
        base_arch = {
            "type": "cnn",
            "num_blocks": 2,
            "base_channels": 16,
            "kernel_size": 3,
            "activation": "relu"
        }
        
        # max
        arch_max = {**base_arch, "pooling": "max"}
        features_max = surrogate_model._encode_cnn(arch_max)
        assert features_max[11:14] == [1, 0, 0]
        
        # avg
        arch_avg = {**base_arch, "pooling": "avg"}
        features_avg = surrogate_model._encode_cnn(arch_avg)
        assert features_avg[11:14] == [0, 1, 0]
        
        # adaptive (也是一个有效的池化类型)
        arch_adaptive = {**base_arch, "pooling": "adaptive"}
        features_adaptive = surrogate_model._encode_cnn(arch_adaptive)
        assert features_adaptive[11:14] == [0, 0, 1]
        
        # 不在列表中的池化类型（应该返回全零）
        arch_other = {**base_arch, "pooling": "global_avg"}
        features_other = surrogate_model._encode_cnn(arch_other)
        assert features_other[11:14] == [0, 0, 0]
    
    def test_encode_cnn_missing_fields(self, surrogate_model):
        """测试缺少字段的 CNN 架构"""
        minimal_arch = {"type": "cnn"}
        features = surrogate_model._encode_cnn(minimal_arch)
        
        assert len(features) == 14
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_encode_cnn_consistency(self, surrogate_model, cnn_arch):
        """测试 CNN 编码的一致性"""
        features1 = surrogate_model._encode_cnn(cnn_arch)
        features2 = surrogate_model._encode_cnn(cnn_arch)
        
        assert features1 == features2
    
    # ==================== Multimodal Encoding Tests ====================
    
    def test_encode_multimodal_basic(self, surrogate_model, multimodal_arch):
        """测试多模态架构的基本编码"""
        features = surrogate_model._encode_multimodal(multimodal_arch)
        
        # 检查返回类型
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
        
        # 检查特征数量（8个数值特征 + 4个融合类型 = 12个）
        assert len(features) == 12
        
        # 验证数值特征
        assert features[0] == 3  # vision num_blocks
        assert features[1] == pytest.approx(0.16, rel=0.01)  # vision base_channels / 200
        assert features[2] == 4  # text num_layers
        assert features[3] == pytest.approx(0.256, rel=0.01)  # text hidden_size / 1000
        assert features[4] == pytest.approx(0.512, rel=0.01)  # fusion_dim / 1000
        assert features[5] == pytest.approx(0.256, rel=0.01)  # projection_dim / 1000
        assert features[6] == pytest.approx(0.1, rel=0.01)  # temperature
        assert features[7] == 1.0  # use_contrastive (True)
        
        # 验证融合类型 one-hot (attention)
        assert features[8:12] == [0, 1, 0, 0]
    
    def test_encode_multimodal_without_contrastive(self, surrogate_model):
        """测试没有对比学习的多模态架构"""
        arch = {
            "type": "multimodal",
            "vision_encoder": {"num_blocks": 2, "base_channels": 16},
            "text_encoder": {"num_layers": 2, "hidden_size": 128},
            "fusion_type": "concat",
            "fusion_dim": 256,
            "use_contrastive": False
        }
        
        features = surrogate_model._encode_multimodal(arch)
        
        # use_contrastive 应该是 0.0
        assert features[7] == 0.0
        
        # 验证融合类型 (concat)
        assert features[8:12] == [1, 0, 0, 0]
    
    def test_encode_multimodal_different_fusion_types(self, surrogate_model):
        """测试不同融合类型的多模态架构"""
        base_arch = {
            "type": "multimodal",
            "vision_encoder": {"num_blocks": 2, "base_channels": 16},
            "text_encoder": {"num_layers": 2, "hidden_size": 128},
            "fusion_dim": 256
        }
        
        # concat
        arch_concat = {**base_arch, "fusion_type": "concat"}
        features_concat = surrogate_model._encode_multimodal(arch_concat)
        assert features_concat[8:12] == [1, 0, 0, 0]
        
        # attention
        arch_attention = {**base_arch, "fusion_type": "attention"}
        features_attention = surrogate_model._encode_multimodal(arch_attention)
        assert features_attention[8:12] == [0, 1, 0, 0]
        
        # bilinear
        arch_bilinear = {**base_arch, "fusion_type": "bilinear"}
        features_bilinear = surrogate_model._encode_multimodal(arch_bilinear)
        assert features_bilinear[8:12] == [0, 0, 1, 0]
        
        # 不在列表中的融合类型（应该返回全零）
        arch_other = {**base_arch, "fusion_type": "gated"}
        features_other = surrogate_model._encode_multimodal(arch_other)
        assert features_other[8:12] == [0, 0, 0, 0]
    
    def test_encode_multimodal_missing_fields(self, surrogate_model):
        """测试缺少字段的多模态架构"""
        minimal_arch = {
            "type": "multimodal",
            "vision_encoder": {},
            "text_encoder": {}
        }
        features = surrogate_model._encode_multimodal(minimal_arch)
        
        assert len(features) == 12
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_encode_multimodal_consistency(self, surrogate_model, multimodal_arch):
        """测试多模态编码的一致性"""
        features1 = surrogate_model._encode_multimodal(multimodal_arch)
        features2 = surrogate_model._encode_multimodal(multimodal_arch)
        
        assert features1 == features2
    
    # ==================== Architecture to Features Tests ====================
    
    def test_architecture_to_features_transformer(self, surrogate_model, transformer_arch):
        """测试将 Transformer 架构转换为特征向量"""
        features = surrogate_model._architecture_to_features(transformer_arch)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float64
        assert len(features) == 11
    
    def test_architecture_to_features_cnn(self, surrogate_model, cnn_arch):
        """测试将 CNN 架构转换为特征向量"""
        features = surrogate_model._architecture_to_features(cnn_arch)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float64
        assert len(features) == 14
    
    def test_architecture_to_features_multimodal(self, surrogate_model, multimodal_arch):
        """测试将多模态架构转换为特征向量"""
        features = surrogate_model._architecture_to_features(multimodal_arch)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float64
        assert len(features) == 12
    
    def test_architecture_to_features_unknown_type(self, surrogate_model):
        """测试未知类型的架构"""
        arch = {"type": "unknown", "param": 123}
        features = surrogate_model._architecture_to_features(arch)
        
        # 应该返回默认的特征向量（20个0）
        assert isinstance(features, np.ndarray)
        assert len(features) == 20
        assert all(f == 0.0 for f in features)
    
    def test_architecture_to_features_no_type(self, surrogate_model):
        """测试没有 type 字段的架构"""
        arch = {"param": 123}
        features = surrogate_model._architecture_to_features(arch)
        
        # 应该返回默认的特征向量
        assert isinstance(features, np.ndarray)
        assert len(features) == 20
    
    # ==================== Feature Normalization Tests ====================
    
    def test_feature_normalization_transformer(self, surrogate_model):
        """测试 Transformer 特征的归一化"""
        # 大值架构
        large_arch = {
            "type": "transformer",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072,
            "vocab_size": 100000,
            "max_seq_len": 1024
        }
        
        features = surrogate_model._encode_transformer(large_arch)
        
        # 归一化后的值应该在合理范围内
        assert all(0 <= f <= 12 for f in features[:7])  # 数值特征
        assert all(f in [0, 1] for f in features[7:])  # one-hot
    
    def test_feature_normalization_cnn(self, surrogate_model):
        """测试 CNN 特征的归一化"""
        # 大值架构
        large_arch = {
            "type": "cnn",
            "num_blocks": 8,
            "base_channels": 128,
            "kernel_size": 7,
            "stride": 2,
            "num_classes": 100,
            "input_size": 64
        }
        
        features = surrogate_model._encode_cnn(large_arch)
        
        # 归一化后的值应该在合理范围内
        assert all(0 <= f <= 8 for f in features[:8])  # 数值特征
        assert all(f in [0, 1] for f in features[8:])  # one-hot
    
    # ==================== Edge Cases Tests ====================
    
    def test_empty_architecture(self, surrogate_model):
        """测试空架构"""
        arch = {}
        features = surrogate_model._architecture_to_features(arch)
        
        # 应该返回默认特征向量
        assert isinstance(features, np.ndarray)
        assert len(features) == 20
    
    def test_partial_transformer_architecture(self, surrogate_model):
        """测试部分字段的 Transformer 架构"""
        arch = {
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256
            # 缺少其他字段
        }
        
        features = surrogate_model._encode_transformer(arch)
        
        assert len(features) == 11
        assert features[0] == 4
        assert features[1] == pytest.approx(0.256, rel=0.01)
        # 其他字段应该使用默认值
        assert isinstance(features[2], (int, float))
    
    def test_negative_values(self, surrogate_model):
        """测试负值（理论上不应该出现，但测试健壮性）"""
        arch = {
            "type": "transformer",
            "num_layers": -1,
            "hidden_size": -100,
            "dropout": -0.1
        }
        
        # 应该不会崩溃
        features = surrogate_model._encode_transformer(arch)
        assert len(features) == 11
    
    def test_float_values_for_int_fields(self, surrogate_model):
        """测试整数字段使用浮点数"""
        arch = {
            "type": "transformer",
            "num_layers": 6.5,
            "num_heads": 8.0
        }
        
        features = surrogate_model._encode_transformer(arch)
        assert len(features) == 11
        assert isinstance(features[0], (int, float))
        assert isinstance(features[2], (int, float))


class TestSurrogateModelTraining:
    """测试代理模型的训练和预测"""
    
    @pytest.fixture
    def surrogate_model(self):
        """创建代理模型实例"""
        return SurrogateModel(model_type="ensemble")
    
    @pytest.fixture
    def training_data(self):
        """生成训练数据"""
        data = []
        for i in range(20):
            arch = {
                "type": "transformer",
                "num_layers": (i % 6) + 2,
                "hidden_size": 128 * ((i % 4) + 1),
                "num_heads": (i % 4) + 2,
                "ffn_dim": 512 * ((i % 4) + 1),
                "dropout": 0.1 + (i % 5) * 0.05
            }
            fitness = 70.0 + (i % 20)
            data.append((arch, fitness))
        return data
    
    def test_add_training_point(self, surrogate_model):
        """测试添加训练样本"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        fitness = 85.0
        
        surrogate_model.add_training_point(arch, fitness)
        
        assert len(surrogate_model.training_data) == 1
        assert surrogate_model.is_fitted == False
    
    def test_fit_with_insufficient_data(self, surrogate_model):
        """测试数据不足时的训练"""
        # 只添加 3 个样本（少于 5 个）
        for i in range(3):
            arch = {"type": "transformer", "num_layers": i + 2}
            surrogate_model.add_training_point(arch, 80.0 + i)
        
        result = surrogate_model.fit()
        
        # 应该返回 False（数据不足）
        assert result == False
        assert surrogate_model.is_fitted == False
    
    def test_fit_with_sufficient_data(self, surrogate_model, training_data):
        """测试数据充足时的训练"""
        # 添加训练数据
        for arch, fitness in training_data:
            surrogate_model.add_training_point(arch, fitness)
        
        result = surrogate_model.fit()
        
        # 应该成功训练
        assert result == True
        assert surrogate_model.is_fitted == True
        assert surrogate_model.best_model_name is not None
    
    def test_predict_without_training(self, surrogate_model):
        """测试未训练时的预测"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        result = surrogate_model.predict(arch)
        
        # 应该返回 None（未训练）
        assert result is None
    
    def test_predict_after_training(self, surrogate_model, training_data):
        """测试训练后的预测"""
        # 添加训练数据并训练
        for arch, fitness in training_data:
            surrogate_model.add_training_point(arch, fitness)
        
        surrogate_model.fit()
        
        # 预测新架构
        new_arch = {
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 1024,
            "dropout": 0.2
        }
        
        prediction = surrogate_model.predict(new_arch)
        
        # 应该返回有效预测
        assert prediction is not None
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 100.0
    
    def test_predict_batch(self, surrogate_model, training_data):
        """测试批量预测"""
        # 添加训练数据并训练
        for arch, fitness in training_data:
            surrogate_model.add_training_point(arch, fitness)
        
        surrogate_model.fit()
        
        # 批量预测
        architectures = [
            {"type": "transformer", "num_layers": i, "hidden_size": 256}
            for i in range(5)
        ]
        
        predictions = surrogate_model.predict_batch(architectures)
        
        assert len(predictions) == 5
        assert all(p is not None for p in predictions)
        assert all(isinstance(p, float) for p in predictions)
    
    def test_different_architecture_types_in_training(self, surrogate_model):
        """测试训练数据包含不同架构类型"""
        architectures = [
            ({"type": "transformer", "num_layers": 6}, 85.0),
            ({"type": "cnn", "num_blocks": 4}, 80.0),
            ({"type": "multimodal", "vision_encoder": {}, "text_encoder": {}}, 75.0)
        ]
        
        for arch, fitness in architectures:
            surrogate_model.add_training_point(arch, fitness)
        
        # 数据不足，无法训练
        result = surrogate_model.fit()
        assert result == False
    
    def test_model_selection(self, surrogate_model, training_data):
        """测试模型选择（应该选择最佳模型）"""
        # 添加训练数据
        for arch, fitness in training_data:
            surrogate_model.add_training_point(arch, fitness)
        
        surrogate_model.fit()
        
        # 应该选择了最佳模型
        assert surrogate_model.best_model_name in ["rf", "gb", "mlp"]
        assert surrogate_model.best_score > float('-inf')


class TestEncodingRobustness:
    """测试编码函数的健壮性"""
    
    @pytest.fixture
    def surrogate_model(self):
        return SurrogateModel()
    
    def test_encoding_with_extra_fields(self, surrogate_model):
        """测试包含额外字段的架构"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "extra_field": "should_be_ignored",
            "another_extra": 123
        }
        
        # 应该不会崩溃
        features = surrogate_model._architecture_to_features(arch)
        assert len(features) == 11
    
    def test_encoding_with_nested_structures(self, surrogate_model):
        """测试包含嵌套结构的架构"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "nested": {
                "level1": {
                    "level2": "value"
                }
            }
        }
        
        # 应该不会崩溃
        features = surrogate_model._architecture_to_features(arch)
        assert len(features) == 11
    
    def test_encoding_reproducibility(self, surrogate_model):
        """测试编码的可重现性"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8
        }
        
        # 编码 100 次，结果应该完全一致
        results = [surrogate_model._encode_transformer(arch) for _ in range(100)]
        
        first = results[0]
        for result in results[1:]:
            assert result == first
    
    def test_encoding_deterministic(self, surrogate_model):
        """测试编码的确定性（不受外部状态影响）"""
        arch = {
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64
        }
        
        # 多次创建模型实例
        features_list = []
        for _ in range(5):
            model = SurrogateModel()
            features = model._encode_cnn(arch)
            features_list.append(features)
        
        # 所有结果应该相同
        for features in features_list[1:]:
            assert features == features_list[0]


class TestFeatureVectorProperties:
    """测试特征向量的数学性质"""
    
    @pytest.fixture
    def surrogate_model(self):
        return SurrogateModel()
    
    def test_feature_vector_is_numeric(self, surrogate_model):
        """测试特征向量只包含数值"""
        architectures = [
            {"type": "transformer", "num_layers": 6, "hidden_size": 512},
            {"type": "cnn", "num_blocks": 4, "base_channels": 64},
            {"type": "multimodal", "vision_encoder": {}, "text_encoder": {}}
        ]
        
        for arch in architectures:
            features = surrogate_model._architecture_to_features(arch)
            assert all(isinstance(f, (int, float, np.number)) for f in features)
    
    def test_feature_vector_no_nan(self, surrogate_model):
        """测试特征向量不包含 NaN"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        features = surrogate_model._architecture_to_features(arch)
        
        assert not any(np.isnan(f) for f in features)
    
    def test_feature_vector_no_inf(self, surrogate_model):
        """测试特征向量不包含无穷大"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        features = surrogate_model._architecture_to_features(arch)
        
        assert not any(np.isinf(f) for f in features)
    
    def test_feature_dimensions_consistency(self, surrogate_model):
        """测试同一类型架构的特征维度一致性"""
        transformer_archs = [
            {"type": "transformer", "num_layers": i, "hidden_size": 256 * (i + 1)}
            for i in range(10)
        ]
        
        feature_lengths = [
            len(surrogate_model._architecture_to_features(arch))
            for arch in transformer_archs
        ]
        
        # 所有 Transformer 架构应该产生相同长度的特征向量
        assert len(set(feature_lengths)) == 1
        assert feature_lengths[0] == 11
    
    def test_different_types_different_dimensions(self, surrogate_model):
        """测试不同类型架构的特征维度可能不同"""
        transformer_features = surrogate_model._architecture_to_features(
            {"type": "transformer"}
        )
        cnn_features = surrogate_model._architecture_to_features(
            {"type": "cnn"}
        )
        multimodal_features = surrogate_model._architecture_to_features(
            {"type": "multimodal", "vision_encoder": {}, "text_encoder": {}}
        )
        
        # 不同类型的特征维度应该不同
        assert len(transformer_features) != len(cnn_features)
        assert len(cnn_features) != len(multimodal_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=genetic_ml_evolution.surrogate_model", "--cov-report=term-missing"])
