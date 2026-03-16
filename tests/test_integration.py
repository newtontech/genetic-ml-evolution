"""
Integration Tests for Cache System with Evolution Engine
测试缓存系统与进化引擎的集成
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from genetic_ml_evolution import CacheSystem, EvolutionEngine, ModelEvaluator


class TestCacheIntegration:
    """测试缓存系统与其他模块的集成"""
    
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
    def evaluator(self):
        """创建评估器实例"""
        return ModelEvaluator(gpu_memory=16, dataset="imdb")
    
    def test_cache_with_evaluator(self, cache, evaluator):
        """测试缓存与评估器的集成"""
        architecture = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1
        }
        
        # 第一次评估（缓存未命中）
        fitness1 = evaluator.evaluate(architecture)
        cache.put(architecture, fitness1)
        
        # 第二次评估（应该从缓存读取）
        cached_fitness = cache.get(architecture)
        assert cached_fitness == fitness1
    
    def test_cache_with_evolution_engine(self, temp_cache_dir):
        """测试缓存与进化引擎的集成"""
        # 创建进化引擎（小规模测试）
        engine = EvolutionEngine(
            gpu_memory=16,
            task_type="language",
            dataset="imdb",
            population_size=5,
            generations=2
        )
        
        # 创建缓存系统
        cache = CacheSystem(cache_dir=temp_cache_dir)
        
        # 修改评估函数以使用缓存
        original_evaluate = engine._evaluate_individual
        
        def evaluate_with_cache(individual):
            # 检查缓存
            cached_fitness = cache.get(individual.architecture)
            if cached_fitness is not None:
                individual.fitness = cached_fitness
                individual.evaluated = True
                return cached_fitness
            
            # 执行评估
            fitness = original_evaluate(individual)
            
            # 存储到缓存
            cache.put(individual.architecture, fitness)
            
            return fitness
        
        # 替换评估函数
        engine._evaluate_individual = evaluate_with_cache
        
        # 运行进化
        best_individual = engine.evolve()
        
        # 验证结果
        assert best_individual is not None
        assert best_individual.fitness is not None
        assert best_individual.fitness > 0
        
        # 验证缓存中有数据
        stats = cache.get_stats()
        assert stats['db_entries'] > 0
        print(f"缓存统计: {stats}")
    
    def test_cache_reuse_across_evolution_runs(self, temp_cache_dir):
        """测试缓存在多次进化运行中的存储和查询"""
        cache = CacheSystem(cache_dir=temp_cache_dir)
        
        # 第一次进化
        engine1 = EvolutionEngine(
            gpu_memory=16,
            task_type="image",
            dataset="cifar10",
            population_size=5,
            generations=2
        )
        
        original_evaluate = engine1._evaluate_individual
        evaluation_count = [0]  # 使用列表以便在闭包中修改
        
        def evaluate_with_cache_count(individual):
            # 检查缓存
            cached_fitness = cache.get(individual.architecture)
            if cached_fitness is not None:
                individual.fitness = cached_fitness
                individual.evaluated = True
                return cached_fitness
            
            # 执行评估
            evaluation_count[0] += 1
            fitness = original_evaluate(individual)
            
            # 存储到缓存
            cache.put(individual.architecture, fitness)
            
            return fitness
        
        engine1._evaluate_individual = evaluate_with_cache_count
        
        # 运行第一次进化
        best1 = engine1.evolve()
        first_evaluations = evaluation_count[0]
        
        # 获取缓存统计
        stats = cache.get_stats()
        print(f"第一次进化后缓存统计: {stats}")
        
        # 验证缓存存储了数据
        assert stats['db_entries'] > 0
        assert stats['misses'] == first_evaluations
        
        # 重新查询已缓存的架构（应该命中）
        if len(engine1.population) > 0:
            test_arch = engine1.population[0].architecture
            cached_fitness = cache.get(test_arch)
            assert cached_fitness is not None
            
            # 再次获取统计，应该有命中
            stats_after_hit = cache.get_stats()
            print(f"查询后缓存统计: {stats_after_hit}")
            assert stats_after_hit['hits'] > 0
    
    def test_cache_persistence_across_sessions(self, temp_cache_dir):
        """测试缓存在不同会话间的持久化"""
        architecture = {
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64,
            "kernel_size": 3
        }
        fitness = 85.5
        
        # 第一个会话：存储数据
        cache1 = CacheSystem(cache_dir=temp_cache_dir)
        cache1.put(architecture, fitness)
        
        # 第二个会话：读取数据
        cache2 = CacheSystem(cache_dir=temp_cache_dir)
        cached_fitness = cache2.get(architecture)
        
        assert cached_fitness == fitness
    
    def test_cache_with_different_architecture_types(self, temp_cache_dir):
        """测试缓存支持不同类型的架构"""
        cache = CacheSystem(cache_dir=temp_cache_dir)
        evaluator = ModelEvaluator(gpu_memory=16)
        
        architectures = [
            {
                "type": "transformer",
                "num_layers": 6,
                "hidden_size": 512,
                "num_heads": 8,
                "ffn_dim": 2048,
                "dropout": 0.1,
                "activation": "gelu",
                "vocab_size": 50257,
                "max_seq_len": 512
            },
            {
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
            },
            {
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
        ]
        
        # 评估并缓存所有类型的架构
        for arch in architectures:
            fitness = evaluator.evaluate(arch)
            cache.put(arch, fitness)
        
        # 验证所有类型都能从缓存读取
        for arch in architectures:
            cached_fitness = cache.get(arch)
            assert cached_fitness is not None
            assert cached_fitness > 0
        
        # 验证缓存统计
        stats = cache.get_stats()
        assert stats['db_entries'] == 3
        print(f"缓存统计: {stats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
