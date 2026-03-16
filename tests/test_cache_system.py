"""
Unit Tests for Cache System
测试 SQLite 缓存系统的各项功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time
import threading
import concurrent.futures

from genetic_ml_evolution.cache_system import CacheSystem


class TestCacheSystem:
    """测试缓存系统的基础功能"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """创建缓存实例"""
        return CacheSystem(cache_dir=temp_cache_dir)
    
    @pytest.fixture
    def sample_architecture(self):
        """示例架构配置"""
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
    
    def test_init_database(self, temp_cache_dir):
        """测试数据库初始化"""
        cache = CacheSystem(cache_dir=temp_cache_dir)
        
        # 检查数据库文件是否创建
        db_path = Path(temp_cache_dir) / "architecture_cache.db"
        assert db_path.exists()
        
        # 检查表是否创建
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='architecture_cache'")
        result = cursor.fetchone()
        assert result is not None
        conn.close()
    
    def test_put_and_get(self, cache, sample_architecture):
        """测试基本的存取操作"""
        fitness = 85.5
        
        # 存储架构评估结果
        cache.put(sample_architecture, fitness)
        
        # 读取评估结果
        cached_fitness = cache.get(sample_architecture)
        
        assert cached_fitness is not None
        assert cached_fitness == fitness
    
    def test_get_non_existent(self, cache):
        """测试获取不存在的缓存"""
        arch = {"type": "unknown", "param": 123}
        result = cache.get(arch)
        assert result is None
    
    def test_get_detailed(self, cache, sample_architecture):
        """测试获取详细评估结果"""
        fitness = 85.5
        accuracy = 88.2
        loss = 0.123
        training_time = 120.5
        memory_usage = 2048.0
        
        # 存储详细结果
        cache.put(
            sample_architecture, 
            fitness,
            accuracy=accuracy,
            loss=loss,
            training_time=training_time,
            memory_usage=memory_usage
        )
        
        # 获取详细结果
        result = cache.get_detailed(sample_architecture)
        
        assert result is not None
        assert result['fitness'] == fitness
        assert result['accuracy'] == accuracy
        assert result['loss'] == loss
        assert result['training_time'] == training_time
        assert result['memory_usage'] == memory_usage
    
    def test_update_existing_cache(self, cache, sample_architecture):
        """测试更新现有缓存"""
        # 第一次存储
        cache.put(sample_architecture, 80.0)
        
        # 更新缓存
        cache.put(sample_architecture, 85.0)
        
        # 应该返回更新后的值
        cached_fitness = cache.get(sample_architecture)
        assert cached_fitness == 85.0
    
    def test_cache_key_consistency(self, cache):
        """测试缓存键的一致性"""
        arch1 = {"a": 1, "b": 2, "c": 3}
        arch2 = {"c": 3, "b": 2, "a": 1}  # 相同内容，不同顺序
        
        cache.put(arch1, 90.0)
        
        # 应该能读取到，因为键是基于排序后的 JSON
        cached_fitness = cache.get(arch2)
        assert cached_fitness == 90.0
    
    def test_memory_cache(self, cache, sample_architecture):
        """测试内存缓存"""
        fitness = 85.5
        cache.put(sample_architecture, fitness)
        
        # 应该在内存缓存中
        key = cache._get_cache_key(sample_architecture)
        assert key in cache.memory_cache
        assert cache.memory_cache[key]['fitness'] == fitness
        
        # 统计信息应该正确
        stats = cache.get_stats()
        assert stats['memory_entries'] == 1
    
    def test_cache_hit_miss_stats(self, cache, sample_architecture):
        """测试缓存命中和未命中统计"""
        # 第一次查询（未命中）
        result = cache.get(sample_architecture)
        assert result is None
        
        stats = cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0
        
        # 存储数据
        cache.put(sample_architecture, 85.0)
        
        # 第二次查询（命中）
        result = cache.get(sample_architecture)
        assert result == 85.0
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_get_top_performers(self, cache):
        """测试获取表现最好的架构"""
        # 存储多个架构
        architectures = [
            {"type": "transformer", "num_layers": i, "hidden_size": 512}
            for i in range(5)
        ]
        
        for i, arch in enumerate(architectures):
            cache.put(arch, fitness=80.0 + i)
        
        # 获取前 3 个
        top_3 = cache.get_top_performers(limit=3)
        
        assert len(top_3) == 3
        assert top_3[0]['fitness'] == 84.0  # 最高分
        assert top_3[1]['fitness'] == 83.0
        assert top_3[2]['fitness'] == 82.0
    
    def test_clear_cache(self, cache, sample_architecture):
        """测试清空缓存"""
        cache.put(sample_architecture, 85.0)
        
        # 清空缓存
        cache.clear()
        
        # 应该读取不到
        result = cache.get(sample_architecture)
        assert result is None
        
        # 统计信息应该重置
        stats = cache.get_stats()
        assert stats['memory_entries'] == 0
        assert stats['db_entries'] == 0
    
    def test_persistence(self, temp_cache_dir, sample_architecture):
        """测试缓存持久化"""
        # 创建缓存并存储数据
        cache1 = CacheSystem(cache_dir=temp_cache_dir)
        cache1.put(sample_architecture, 85.0)
        
        # 创建新的缓存实例（应该能读取之前的数据）
        cache2 = CacheSystem(cache_dir=temp_cache_dir)
        result = cache2.get(sample_architecture)
        
        assert result == 85.0
    
    def test_export_import_json(self, cache, temp_cache_dir):
        """测试导出和导入 JSON"""
        # 存储多个架构
        architectures = [
            {"type": "transformer", "num_layers": i, "hidden_size": 512}
            for i in range(3)
        ]
        
        for i, arch in enumerate(architectures):
            cache.put(
                arch, 
                fitness=80.0 + i,
                accuracy=85.0 + i,
                loss=0.1 + i * 0.01
            )
        
        # 导出到 JSON
        output_path = Path(temp_cache_dir) / "export.json"
        cache.export_to_json(str(output_path))
        
        assert output_path.exists()
        
        # 清空缓存
        cache.clear()
        
        # 从 JSON 导入
        count = cache.import_from_json(str(output_path))
        
        assert count == 3
        
        # 验证数据
        result = cache.get(architectures[0])
        assert result == 80.0
    
    def test_concurrent_access(self, cache):
        """测试并发访问"""
        def write_cache(index):
            arch = {"type": "test", "index": index}
            cache.put(arch, fitness=float(index))
            return cache.get(arch)
        
        # 使用线程池并发写入
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_cache, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # 所有操作应该成功
        assert len(results) == 100
        assert all(r is not None for r in results)
    
    def test_vacuum(self, cache, sample_architecture):
        """测试数据库清理"""
        # 存储一些数据
        for i in range(10):
            arch = {"type": "test", "index": i}
            cache.put(arch, fitness=float(i))
        
        # 清空部分数据
        cache.clear()
        
        # 清理数据库碎片
        cache.vacuum()
        
        # 应该不会出错
        stats = cache.get_stats()
        assert stats['db_entries'] == 0
    
    def test_repr(self, cache):
        """测试字符串表示"""
        repr_str = repr(cache)
        assert "CacheSystem" in repr_str
        assert "db_entries" in repr_str
        assert "hit_rate" in repr_str
    
    def test_len(self, cache):
        """测试长度获取"""
        assert len(cache) == 0
        
        # 存储数据
        for i in range(5):
            arch = {"type": "test", "index": i}
            cache.put(arch, fitness=float(i))
        
        assert len(cache) == 5


class TestCacheSystemAdvanced:
    """测试缓存系统的高级功能"""
    
    @pytest.fixture
    def cache(self):
        """创建缓存实例"""
        temp_dir = tempfile.mkdtemp()
        cache = CacheSystem(cache_dir=temp_dir)
        yield cache
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_different_architecture_types(self, cache):
        """测试不同类型的架构"""
        architectures = [
            {
                "type": "transformer",
                "num_layers": 6,
                "hidden_size": 512,
                "num_heads": 8
            },
            {
                "type": "cnn",
                "num_blocks": 4,
                "base_channels": 64,
                "kernel_size": 3
            },
            {
                "type": "multimodal",
                "vision_encoder": {"num_blocks": 3},
                "text_encoder": {"num_layers": 4}
            }
        ]
        
        # 存储所有类型
        for i, arch in enumerate(architectures):
            cache.put(arch, fitness=80.0 + i)
        
        # 验证所有类型都能正确读取
        for i, arch in enumerate(architectures):
            result = cache.get(arch)
            assert result == 80.0 + i
    
    def test_large_dataset(self, cache):
        """测试大数据集"""
        # 存储 1000 个架构
        for i in range(1000):
            arch = {
                "type": "transformer",
                "num_layers": (i % 10) + 2,
                "hidden_size": 128 * ((i % 6) + 1),
                "index": i
            }
            cache.put(arch, fitness=50.0 + (i % 50))
        
        # 验证数据库条目数
        stats = cache.get_stats()
        assert stats['db_entries'] == 1000
        
        # 验证能正确读取（使用实际存储的架构）
        # 当 i=32 时：num_layers=4, hidden_size=384
        test_arch = {
            "type": "transformer",
            "num_layers": 4,  # (32 % 10) + 2 = 4
            "hidden_size": 384,  # 128 * ((32 % 6) + 1) = 384
            "index": 32
        }
        result = cache.get(test_arch)
        assert result == 50.0 + (32 % 50)  # 82.0
    
    def test_special_characters_in_architecture(self, cache):
        """测试架构中的特殊字符"""
        arch = {
            "type": "test",
            "name": "架构-测试_123!@#$%^&*()",
            "unicode": "中文日本語한국어"
        }
        
        fitness = 90.0
        cache.put(arch, fitness)
        
        result = cache.get(arch)
        assert result == fitness
    
    def test_extreme_values(self, cache):
        """测试极端值"""
        arch = {"type": "test", "value": 0}
        
        # 测试最大值
        cache.put(arch, fitness=100.0)
        assert cache.get(arch) == 100.0
        
        # 测试最小值
        cache.put(arch, fitness=0.0)
        assert cache.get(arch) == 0.0
    
    def test_cache_statistics_accuracy(self, cache):
        """测试缓存统计的准确性"""
        # 存储一些数据
        for i in range(10):
            arch = {"type": "test", "index": i}
            cache.put(arch, fitness=float(i))
        
        # 重置统计
        cache.hit_count = 0
        cache.miss_count = 0
        
        # 执行查询
        for i in range(20):
            if i < 10:
                # 前 10 次应该命中
                arch = {"type": "test", "index": i}
                result = cache.get(arch)
                assert result is not None
            else:
                # 后 10 次应该未命中
                arch = {"type": "test", "index": 100 + i}
                result = cache.get(arch)
                assert result is None
        
        # 验证统计
        stats = cache.get_stats()
        assert stats['hits'] == 10
        assert stats['misses'] == 10
        assert stats['total'] == 20
        assert stats['hit_rate'] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
