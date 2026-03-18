"""
Unit Tests for SQLite Cache System
测试架构性能缓存系统
"""

import pytest
import tempfile
import os
from typing import Dict, Any

from genetic_ml_evolution.cache_system import ArchitectureCache


class TestArchitectureCache:
    """测试架构缓存系统"""
    
    @pytest.fixture
    def temp_db(self):
        """创建临时数据库文件"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
    
    @pytest.fixture
    def cache(self, temp_db):
        """创建缓存实例"""
        return ArchitectureCache(db_path=temp_db)
    
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
            "activation": "gelu"
        }
    
    @pytest.fixture
    def cnn_arch(self) -> Dict[str, Any]:
        """示例 CNN 架构"""
        return {
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64,
            "kernel_size": 3,
            "activation": "relu"
        }
    
    @pytest.fixture
    def multimodal_arch(self) -> Dict[str, Any]:
        """示例多模态架构"""
        return {
            "type": "multimodal",
            "vision_encoder": {"num_blocks": 3, "base_channels": 32},
            "text_encoder": {"num_layers": 4, "hidden_size": 256},
            "fusion_type": "attention"
        }
    
    # ==================== Basic Operations Tests ====================
    
    def test_store_and_lookup(self, cache, transformer_arch):
        """测试存储和查询"""
        metrics = {"accuracy": 0.85, "loss": 0.15}
        
        # 存储
        result = cache.store(transformer_arch, metrics)
        assert result is True
        
        # 查询
        cached = cache.lookup(transformer_arch)
        assert cached is not None
        assert cached["accuracy"] == pytest.approx(0.85, rel=0.01)
        assert cached["loss"] == pytest.approx(0.15, rel=0.01)
    
    def test_lookup_non_existent(self, cache):
        """查询不存在的架构"""
        arch = {"type": "transformer", "num_layers": 999}
        result = cache.lookup(arch)
        assert result is None
    
    def test_exists(self, cache, transformer_arch):
        """测试存在性检查"""
        metrics = {"accuracy": 0.8}
        
        assert cache.exists(transformer_arch) is False
        cache.store(transformer_arch, metrics)
        assert cache.exists(transformer_arch) is True
    
    def test_duplicate_store(self, cache, transformer_arch):
        """测试重复存储（应该返回 False）"""
        metrics = {"accuracy": 0.8}
        
        # 第一次存储
        result1 = cache.store(transformer_arch, metrics)
        assert result1 is True
        
        # 重复存储
        result2 = cache.store(transformer_arch, metrics)
        assert result2 is False
    
    # ==================== Multiple Architecture Types Tests ====================
    
    def test_store_different_types(self, cache, transformer_arch, cnn_arch, multimodal_arch):
        """测试存储不同类型的架构"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        cache.store(cnn_arch, {"accuracy": 0.80})
        cache.store(multimodal_arch, {"accuracy": 0.75})
        
        # 验证都能查询到
        assert cache.lookup(transformer_arch)["accuracy"] == pytest.approx(0.85, rel=0.01)
        assert cache.lookup(cnn_arch)["accuracy"] == pytest.approx(0.80, rel=0.01)
        assert cache.lookup(multimodal_arch)["accuracy"] == pytest.approx(0.75, rel=0.01)
    
    def test_get_by_type(self, cache, transformer_arch, cnn_arch):
        """测试按类型查询"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        cache.store(cnn_arch, {"accuracy": 0.80})
        
        # 查询 Transformer 类型
        transformer_results = cache.get_by_type("transformer")
        assert len(transformer_results) == 1
        assert transformer_results[0][1]["accuracy"] == pytest.approx(0.85, rel=0.01)
        
        # 查询 CNN 类型
        cnn_results = cache.get_by_type("cnn")
        assert len(cnn_results) == 1
        assert cnn_results[0][1]["accuracy"] == pytest.approx(0.80, rel=0.01)
    
    def test_get_by_type_with_limit(self, cache):
        """测试按类型查询并限制数量"""
        # 存储 5 个 Transformer 架构
        for i in range(5):
            arch = {"type": "transformer", "num_layers": i + 2}
            cache.store(arch, {"accuracy": 0.8 + i * 0.01})
        
        # 限制返回 3 个
        results = cache.get_by_type("transformer", limit=3)
        assert len(results) == 3
    
    # ==================== Top Performing Tests ====================
    
    def test_get_top_performing(self, cache):
        """测试获取最佳性能架构"""
        # 存储多个架构
        for i in range(10):
            arch = {"type": "transformer", "num_layers": i + 2}
            cache.store(arch, {"accuracy": 0.7 + i * 0.02})
        
        # 获取 top 5
        top_5 = cache.get_top_performing(metric="accuracy", limit=5)
        assert len(top_5) == 5
        
        # 验证是按 accuracy 降序排列
        for i in range(len(top_5) - 1):
            assert top_5[i][1]["accuracy"] >= top_5[i + 1][1]["accuracy"]
    
    def test_get_top_performing_by_type(self, cache):
        """测试按类型获取最佳性能架构"""
        # Transformer 架构
        for i in range(5):
            arch = {"type": "transformer", "num_layers": i + 2}
            cache.store(arch, {"accuracy": 0.7 + i * 0.02})
        
        # CNN 架构
        for i in range(5):
            arch = {"type": "cnn", "num_blocks": i + 2}
            cache.store(arch, {"accuracy": 0.6 + i * 0.03})
        
        # 只查询 Transformer
        top_transformers = cache.get_top_performing(
            metric="accuracy", 
            limit=3, 
            arch_type="transformer"
        )
        assert len(top_transformers) == 3
    
    # ==================== Statistics Tests ====================
    
    def test_get_statistics(self, cache, transformer_arch, cnn_arch):
        """测试获取统计信息"""
        cache.store(transformer_arch, {"accuracy": 0.85}, evaluation_time=10.5)
        cache.store(cnn_arch, {"accuracy": 0.80}, evaluation_time=8.0)
        
        stats = cache.get_statistics()
        
        assert stats["total_entries"] == 2
        assert stats["entries_by_type"]["transformer"] == 1
        assert stats["entries_by_type"]["cnn"] == 1
        assert stats["average_evaluation_time"] == pytest.approx(9.25, rel=0.01)
    
    def test_cache_hits_and_misses(self, cache, transformer_arch):
        """测试缓存命中和未命中"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        
        # 命中
        cache.lookup(transformer_arch)
        cache.lookup(transformer_arch)
        
        # 未命中
        cache.lookup({"type": "unknown"})
        
        stats = cache.get_statistics()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["hit_rate_percent"] == pytest.approx(66.67, rel=0.1)
    
    def test_len(self, cache, transformer_arch, cnn_arch):
        """测试 len() 方法"""
        assert len(cache) == 0
        cache.store(transformer_arch, {"accuracy": 0.85})
        assert len(cache) == 1
        cache.store(cnn_arch, {"accuracy": 0.80})
        assert len(cache) == 2
    
    # ==================== Clear and Delete Tests ====================
    
    def test_clear(self, cache, transformer_arch, cnn_arch):
        """测试清空缓存"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        cache.store(cnn_arch, {"accuracy": 0.80})
        
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
    
    def test_delete(self, cache, transformer_arch):
        """测试删除单个架构"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        assert cache.exists(transformer_arch) is True
        
        result = cache.delete(transformer_arch)
        assert result is True
        assert cache.exists(transformer_arch) is False
    
    def test_delete_non_existent(self, cache):
        """测试删除不存在的架构"""
        arch = {"type": "unknown"}
        result = cache.delete(arch)
        assert result is False
    
    # ==================== Import/Export Tests ====================
    
    def test_export_and_import_json(self, cache, temp_db, transformer_arch, cnn_arch):
        """测试导出和导入 JSON"""
        cache.store(transformer_arch, {"accuracy": 0.85}, evaluation_time=10.5)
        cache.store(cnn_arch, {"accuracy": 0.80}, evaluation_time=8.0)
        
        # 导出
        export_path = temp_db.replace(".db", "_export.json")
        cache.export_to_json(export_path)
        
        # 创建新缓存并导入
        new_db = temp_db.replace(".db", "_new.db")
        new_cache = ArchitectureCache(db_path=new_db)
        imported = new_cache.import_from_json(export_path)
        
        assert imported == 2
        assert len(new_cache) == 2
        assert new_cache.exists(transformer_arch) is True
        
        # 清理
        os.remove(export_path)
        os.remove(new_db)
    
    def test_import_with_overwrite(self, cache, temp_db, transformer_arch):
        """测试导入时覆盖现有数据"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        
        # 导出
        export_path = temp_db.replace(".db", "_export.json")
        cache.export_to_json(export_path)
        
        # 清空并重新导入
        cache.clear()
        imported = cache.import_from_json(export_path)
        
        assert imported == 1
        assert len(cache) == 1
        
        os.remove(export_path)
    
    # ==================== Evaluation Time Tests ====================
    
    def test_evaluation_time_storage(self, cache, transformer_arch):
        """测试评估时间存储"""
        metrics = {"accuracy": 0.85}
        eval_time = 12.5
        
        cache.store(transformer_arch, metrics, evaluation_time=eval_time)
        
        # 查询（验证存储成功）
        result = cache.lookup(transformer_arch)
        assert result is not None
    
    def test_average_evaluation_time(self, cache):
        """测试平均评估时间计算"""
        for i in range(5):
            arch = {"type": "transformer", "num_layers": i + 2}
            cache.store(arch, {"accuracy": 0.8}, evaluation_time=float(i + 1) * 10)
        
        stats = cache.get_statistics()
        assert stats["average_evaluation_time"] == 30.0
    
    # ==================== Access Tracking Tests ====================
    
    def test_access_count_tracking(self, cache, transformer_arch):
        """测试访问次数跟踪"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        
        # 多次查询
        for _ in range(5):
            cache.lookup(transformer_arch)
        
        # 验证访问次数（通过统计数据间接验证）
        stats = cache.get_statistics()
        assert stats["cache_hits"] == 5
    
    # ==================== Edge Cases Tests ====================
    
    def test_empty_architecture(self, cache):
        """测试空架构"""
        arch = {}
        metrics = {"accuracy": 0.5}
        
        result = cache.store(arch, metrics)
        assert result is True
        
        cached = cache.lookup(arch)
        assert cached is not None
    
    def test_complex_metrics(self, cache, transformer_arch):
        """测试复杂的性能指标"""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "inference_time": 0.025,
            "model_size_mb": 45.2
        }
        
        cache.store(transformer_arch, metrics)
        cached = cache.lookup(transformer_arch)
        
        assert cached["accuracy"] == pytest.approx(0.85, rel=0.01)
        assert cached["precision"] == pytest.approx(0.82, rel=0.01)
        assert cached["inference_time"] == pytest.approx(0.025, rel=0.01)
    
    def test_special_characters_in_architecture(self, cache):
        """测试架构中包含特殊字符"""
        arch = {
            "type": "transformer",
            "name": "模型-测试_v1.0",
            "description": "这是一个测试架构！@#$%"
        }
        metrics = {"accuracy": 0.8}
        
        result = cache.store(arch, metrics)
        assert result is True
        
        cached = cache.lookup(arch)
        assert cached is not None
        assert cached["accuracy"] == pytest.approx(0.8, rel=0.01)
    
    def test_very_large_architecture(self, cache):
        """测试大型架构配置"""
        arch = {
            "type": "transformer",
            "layers": [{"hidden_size": 512, "num_heads": 8} for _ in range(100)],
            "metadata": {f"param_{i}": i for i in range(100)}
        }
        metrics = {"accuracy": 0.8}
        
        result = cache.store(arch, metrics)
        assert result is True
        
        cached = cache.lookup(arch)
        assert cached is not None
    
    # ==================== Context Manager Tests ====================
    
    def test_context_manager(self, temp_db):
        """测试上下文管理器"""
        with ArchitectureCache(db_path=temp_db) as cache:
            arch = {"type": "transformer"}
            cache.store(arch, {"accuracy": 0.8})
            assert len(cache) == 1
        
        # 退出上下文后应该关闭连接
    
    # ==================== String Representation Tests ====================
    
    def test_repr(self, cache, transformer_arch):
        """测试字符串表示"""
        cache.store(transformer_arch, {"accuracy": 0.85})
        
        repr_str = repr(cache)
        assert "ArchitectureCache" in repr_str
        assert "entries=1" in repr_str


class TestCacheIntegration:
    """测试缓存与代理模型的集成"""
    
    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)
    
    def test_cache_with_surrogate_model(self, temp_db):
        """测试缓存与代理模型集成"""
        from genetic_ml_evolution.surrogate_model import SurrogateModel
        
        # 创建带缓存的代理模型
        model = SurrogateModel(model_type="rf", cache_db_path=temp_db)
        
        arch = {"type": "transformer", "num_layers": 6}
        
        # 添加训练数据
        for i in range(10):
            train_arch = {"type": "transformer", "num_layers": i + 2}
            model.add_training_point(train_arch, 70.0 + i * 2)
        
        # 训练
        model.fit()
        
        # 预测（应该使用模型）
        prediction = model.predict(arch)
        assert prediction is not None
        
        # 存储预测结果
        model.store_prediction(arch, 85.0)
        
        # 再次查询（应该使用缓存）
        stats_before = model.get_cache_statistics()
        cached_pred = model.predict(arch)
        stats_after = model.get_cache_statistics()
        
        assert stats_after["cache_hits"] > stats_before["cache_hits"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
