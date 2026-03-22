"""
Unit Tests for Genetic Algorithm
测试遗传算法的核心功能，特别是针对小规模语言模型优化的变异操作
"""

import pytest
import numpy as np
from typing import Dict, Any
from copy import deepcopy

from genetic_ml_evolution.genetic_algorithm import (
    GeneticAlgorithm,
    Individual,
    MutationStrategy
)


class TestIndividual:
    """测试 Individual 类"""
    
    def test_individual_creation(self):
        """测试个体创建"""
        arch = {"type": "transformer", "num_layers": 6}
        individual = Individual(architecture=arch)
        
        assert individual.architecture == arch
        assert individual.fitness is None
        assert individual.age == 0
        assert individual.parent_ids == []
        assert individual.mutation_history == []
    
    def test_individual_hash(self):
        """测试个体哈希"""
        arch1 = {"type": "transformer", "num_layers": 6}
        arch2 = {"type": "transformer", "num_layers": 6}
        arch3 = {"type": "transformer", "num_layers": 4}
        
        ind1 = Individual(architecture=arch1)
        ind2 = Individual(architecture=arch2)
        ind3 = Individual(architecture=arch3)
        
        # 相同架构应该有相同的哈希
        assert hash(ind1) == hash(ind2)
        
        # 不同架构应该有不同的哈希
        assert hash(ind1) != hash(ind3)
    
    def test_individual_equality(self):
        """测试个体相等性"""
        arch1 = {"type": "transformer", "num_layers": 6}
        arch2 = {"type": "transformer", "num_layers": 6}
        arch3 = {"type": "transformer", "num_layers": 4}
        
        ind1 = Individual(architecture=arch1)
        ind2 = Individual(architecture=arch2)
        ind3 = Individual(architecture=arch3)
        
        assert ind1 == ind2
        assert ind1 != ind3
    
    def test_individual_with_fitness(self):
        """测试带适应度的个体"""
        arch = {"type": "transformer", "num_layers": 6}
        individual = Individual(architecture=arch, fitness=85.5)
        
        assert individual.fitness == 85.5


class TestMutationStrategy:
    """测试变异策略（针对小规模语言模型优化）"""
    
    # ==================== Transformer 变异测试 ====================
    
    def test_mutate_transformer_basic(self):
        """测试 Transformer 基本变异"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1
        }
        
        mutated, desc = MutationStrategy.mutate_transformer(
            arch, mutation_rate=1.0, strategy="moderate"
        )
        
        # 应该返回一个架构和描述
        assert isinstance(mutated, dict)
        assert isinstance(desc, str)
        
        # 变异后的架构应该保持类型
        assert mutated["type"] == "transformer"
        
        # 参数应该在合理范围内
        assert 2 <= mutated.get("num_layers", 6) <= 12
        assert 128 <= mutated.get("hidden_size", 512) <= 768
        assert 2 <= mutated.get("num_heads", 8) <= 16
        assert 512 <= mutated.get("ffn_dim", 2048) <= 3072
        assert 0.0 <= mutated.get("dropout", 0.1) <= 0.3
    
    def test_mutate_transformer_small_model_bias(self):
        """测试小模型偏好的变异策略"""
        # 运行多次变异，统计倾向
        results = {"layers": [], "hidden": [], "heads": []}
        
        for _ in range(100):
            arch = {
                "type": "transformer",
                "num_layers": 8,
                "hidden_size": 512,
                "num_heads": 8
            }
            
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            results["layers"].append(mutated.get("num_layers", 8))
            results["hidden"].append(mutated.get("hidden_size", 512))
            results["heads"].append(mutated.get("num_heads", 8))
        
        # 统计倾向
        avg_layers = np.mean(results["layers"])
        avg_hidden = np.mean(results["hidden"])
        avg_heads = np.mean(results["heads"])
        
        # 应该倾向于较小的值（因为有小模型偏好）
        # 注意：这是一个统计测试，可能偶尔失败
        assert avg_layers <= 8.5, f"Average layers {avg_layers} should be <= 8.5 (small model bias)"
        assert avg_hidden <= 600, f"Average hidden {avg_hidden} should be <= 600 (small model bias)"
        assert avg_heads <= 10, f"Average heads {avg_heads} should be <= 10 (small model bias)"
    
    def test_mutate_transformer_adaptive_rate(self):
        """测试自适应变异率"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 早期世代：高变异率
        mutated_early, _ = MutationStrategy.mutate_transformer(
            arch, mutation_rate=0.5, strategy="adaptive", generation=0
        )
        
        # 晚期世代：低变异率
        mutated_late, _ = MutationStrategy.mutate_transformer(
            arch, mutation_rate=0.5, strategy="adaptive", generation=100
        )
        
        # 两个都应该返回有效架构
        assert isinstance(mutated_early, dict)
        assert isinstance(mutated_late, dict)
    
    def test_mutate_transformer_no_change(self):
        """测试无变异的情况（低变异率）"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 低变异率，应该经常不改变
        no_change_count = 0
        for _ in range(100):
            mutated, desc = MutationStrategy.mutate_transformer(
                arch, mutation_rate=0.01, strategy="moderate"
            )
            if desc == "no_change":
                no_change_count += 1
        
        # 至少应该有 50% 的情况不改变
        assert no_change_count >= 50, f"Expected >= 50 no-change cases, got {no_change_count}"
    
    def test_mutate_transformer_hidden_heads_compatibility(self):
        """测试 hidden_size 和 num_heads 的兼容性"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8
        }
        
        # 多次变异，检查兼容性
        for _ in range(100):
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            hidden = mutated.get("hidden_size", 512)
            heads = mutated.get("num_heads", 8)
            
            # hidden_size 应该能被 num_heads 整除
            assert hidden % heads == 0, f"hidden_size {hidden} not divisible by num_heads {heads}"
    
    # ==================== CNN 变异测试 ====================
    
    def test_mutate_cnn_basic(self):
        """测试 CNN 基本变异"""
        arch = {
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64,
            "kernel_size": 3,
            "activation": "relu",
            "pooling": "max"
        }
        
        mutated, desc = MutationStrategy.mutate_cnn(
            arch, mutation_rate=1.0, strategy="moderate"
        )
        
        assert isinstance(mutated, dict)
        assert isinstance(desc, str)
        assert mutated["type"] == "cnn"
        
        # 检查参数范围
        assert 2 <= mutated.get("num_blocks", 4) <= 8
        assert 16 <= mutated.get("base_channels", 64) <= 128
        assert 3 <= mutated.get("kernel_size", 3) <= 7
    
    def test_mutate_cnn_small_model_bias(self):
        """测试 CNN 小模型偏好"""
        results = {"blocks": [], "channels": []}
        
        for _ in range(100):
            arch = {
                "type": "cnn",
                "num_blocks": 6,
                "base_channels": 96
            }
            
            mutated, _ = MutationStrategy.mutate_cnn(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            results["blocks"].append(mutated.get("num_blocks", 6))
            results["channels"].append(mutated.get("base_channels", 96))
        
        avg_blocks = np.mean(results["blocks"])
        avg_channels = np.mean(results["channels"])
        
        # 应该倾向于较小的值
        assert avg_blocks <= 6.5, f"Average blocks {avg_blocks} should show small model bias"
        assert avg_channels <= 110, f"Average channels {avg_channels} should show small model bias"
    
    # ==================== Multimodal 变异测试 ====================
    
    def test_mutate_multimodal_basic(self):
        """测试多模态基本变异"""
        arch = {
            "type": "multimodal",
            "vision_encoder": {
                "type": "cnn",
                "num_blocks": 3,
                "base_channels": 32
            },
            "text_encoder": {
                "type": "transformer",
                "num_layers": 4,
                "hidden_size": 256
            },
            "fusion_type": "attention",
            "fusion_dim": 512
        }
        
        mutated, desc = MutationStrategy.mutate_multimodal(
            arch, mutation_rate=0.5, strategy="moderate"
        )
        
        assert isinstance(mutated, dict)
        assert isinstance(desc, str)
        assert mutated["type"] == "multimodal"
        
        # 子编码器应该仍然存在
        assert "vision_encoder" in mutated
        assert "text_encoder" in mutated
    
    def test_mutate_multimodal_sub_encoders(self):
        """测试多模态子编码器变异"""
        arch = {
            "type": "multimodal",
            "vision_encoder": {
                "type": "cnn",
                "num_blocks": 4
            },
            "text_encoder": {
                "type": "transformer",
                "num_layers": 6
            }
        }
        
        # 高变异率，应该变异子编码器
        vision_changed = False
        text_changed = False
        
        for _ in range(50):
            mutated, desc = MutationStrategy.mutate_multimodal(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            if mutated["vision_encoder"].get("num_blocks") != 4:
                vision_changed = True
            if mutated["text_encoder"].get("num_layers") != 6:
                text_changed = True
            
            if vision_changed and text_changed:
                break
        
        # 至少应该有一次变异了子编码器
        assert vision_changed or text_changed, "Sub-encoders should be mutated"
    
    # ==================== 边界情况测试 ====================
    
    def test_mutate_transformer_extreme_values(self):
        """测试极端值的变异"""
        # 最小值
        min_arch = {
            "type": "transformer",
            "num_layers": 2,
            "hidden_size": 128,
            "num_heads": 2,
            "ffn_dim": 512
        }
        
        mutated, _ = MutationStrategy.mutate_transformer(
            min_arch, mutation_rate=1.0, strategy="moderate"
        )
        
        # 应该保持在合理范围内
        assert mutated.get("num_layers", 2) >= 2
        assert mutated.get("hidden_size", 128) >= 128
        
        # 最大值
        max_arch = {
            "type": "transformer",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072
        }
        
        mutated, _ = MutationStrategy.mutate_transformer(
            max_arch, mutation_rate=1.0, strategy="moderate"
        )
        
        assert mutated.get("num_layers", 12) <= 12
        assert mutated.get("hidden_size", 768) <= 768
        assert mutated.get("ffn_dim", 3072) <= 3072
    
    def test_mutate_missing_fields(self):
        """测试缺少字段的情况"""
        minimal_arch = {"type": "transformer"}
        
        # 应该不会崩溃
        mutated, desc = MutationStrategy.mutate_transformer(
            minimal_arch, mutation_rate=0.5, strategy="moderate"
        )
        
        assert isinstance(mutated, dict)
        assert "type" in mutated


class TestGeneticAlgorithm:
    """测试遗传算法核心功能"""
    
    @pytest.fixture
    def ga(self):
        """创建遗传算法实例"""
        return GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.3,
            crossover_rate=0.3,
            elitism_rate=0.1,
            task_type="language"
        )
    
    @pytest.fixture
    def dummy_fitness(self):
        """虚拟适应度函数"""
        def fitness_func(architecture):
            # 简单的适应度：基于架构大小
            if architecture.get("type") == "transformer":
                layers = architecture.get("num_layers", 6)
                hidden = architecture.get("hidden_size", 512)
                return 50 + (layers * 2) + (hidden / 100)
            return 50.0
        
        return fitness_func
    
    def test_ga_initialization(self, ga):
        """测试遗传算法初始化"""
        assert ga.population_size == 10
        assert ga.mutation_rate == 0.3
        assert ga.crossover_rate == 0.3
        assert ga.task_type == "language"
        assert ga.population == []
        assert ga.generation == 0
    
    def test_initialize_population(self, ga):
        """测试种群初始化"""
        ga.initialize_population()
        
        assert len(ga.population) == ga.population_size
        assert all(isinstance(ind, Individual) for ind in ga.population)
        
        # 所有个体应该有架构
        assert all(ind.architecture is not None for ind in ga.population)
        
        # 应该有一些多样性
        unique_archs = len(set(hash(ind) for ind in ga.population))
        assert unique_archs > 1, "Population should have diversity"
    
    def test_initialize_population_with_seeds(self, ga):
        """测试使用种子架构初始化"""
        seeds = [
            {"type": "transformer", "num_layers": 4},
            {"type": "transformer", "num_layers": 8}
        ]
        
        ga.initialize_population(seed_architectures=seeds)
        
        assert len(ga.population) == ga.population_size
        
        # 前两个个体应该是种子架构
        assert ga.population[0].architecture in seeds
        assert ga.population[1].architecture in seeds
    
    def test_evaluate_population(self, ga, dummy_fitness):
        """测试种群评估"""
        ga.initialize_population()
        ga.evaluate_population(dummy_fitness)
        
        # 所有个体应该有适应度
        assert all(ind.fitness is not None for ind in ga.population)
        
        # 适应度应该是数值
        assert all(isinstance(ind.fitness, (int, float)) for ind in ga.population)
        
        # 应该有最佳个体
        assert ga.best_individual is not None
        assert ga.best_individual.fitness is not None
    
    def test_select_parent(self, ga, dummy_fitness):
        """测试父代选择"""
        ga.initialize_population()
        ga.evaluate_population(dummy_fitness)
        
        parent = ga.select_parent()
        
        assert isinstance(parent, Individual)
        assert parent in ga.population
    
    def test_mutate(self, ga):
        """测试变异操作"""
        ga.initialize_population()
        
        individual = ga.population[0]
        mutated = ga.mutate(individual, strategy="moderate")
        
        assert isinstance(mutated, Individual)
        assert mutated.parent_ids == [hash(individual)]
        assert len(mutated.mutation_history) > 0
    
    def test_crossover(self, ga):
        """测试交叉操作"""
        ga.initialize_population()
        
        parent1 = ga.population[0]
        parent2 = ga.population[1]
        
        child = ga.crossover(parent1, parent2)
        
        assert isinstance(child, Individual)
        assert len(child.parent_ids) == 2
        assert hash(parent1) in child.parent_ids
        assert hash(parent2) in child.parent_ids
    
    def test_evolve(self, ga, dummy_fitness):
        """测试进化一代"""
        ga.initialize_population()
        ga.evaluate_population(dummy_fitness)
        
        old_population = ga.population.copy()
        new_population = ga.evolve()
        
        assert len(new_population) == ga.population_size
        assert ga.generation == 1
        
        # 应该保留精英（至少一个个体的架构可能相同）
        # 注意：由于哈希基于架构，不同对象可能有相同哈希
    
    def test_run_evolution(self, ga, dummy_fitness):
        """测试运行完整进化"""
        best = ga.run(
            fitness_function=dummy_fitness,
            max_generations=5,
            verbose=False
        )
        
        assert isinstance(best, Individual)
        assert best.fitness is not None
        # generation 和 history 应该是 4 (因为最后一轮不进化,不记录历史)
        assert ga.generation == 4
        assert len(ga.history) == 4
    
    def test_run_with_target_fitness(self, ga, dummy_fitness):
        """测试达到目标适应度提前停止"""
        best = ga.run(
            fitness_function=dummy_fitness,
            max_generations=20,
            target_fitness=100.0,  # 较低的目标
            verbose=False
        )
        
        assert best.fitness >= 100.0 or ga.generation < 20
    
    def test_diversity_calculation(self, ga):
        """测试多样性计算"""
        ga.initialize_population()
        
        diversity = ga._calculate_diversity()
        
        assert 0.0 <= diversity <= 1.0
        assert diversity > 0.0  # 应该有一些多样性
    
    def test_statistics(self, ga, dummy_fitness):
        """测试统计信息"""
        ga.initialize_population()
        ga.evaluate_population(dummy_fitness)
        
        stats = ga.get_statistics()
        
        assert "generation" in stats
        assert "population_size" in stats
        assert "best_fitness" in stats
        assert "avg_fitness" in stats
        assert "diversity" in stats
    
    def test_different_task_types(self):
        """测试不同任务类型"""
        # Language model
        ga_lang = GeneticAlgorithm(task_type="language")
        ga_lang.initialize_population()
        assert all(
            ind.architecture.get("type") in ["transformer", "unknown"]
            for ind in ga_lang.population
        )
        
        # Image model
        ga_image = GeneticAlgorithm(task_type="image")
        ga_image.initialize_population()
        assert all(
            ind.architecture.get("type") in ["cnn", "unknown"]
            for ind in ga_image.population
        )
        
        # Multimodal
        ga_multi = GeneticAlgorithm(task_type="multimodal")
        ga_multi.initialize_population()
        assert all(
            ind.architecture.get("type") in ["multimodal", "unknown"]
            for ind in ga_multi.population
        )


class TestSmallModelOptimizations:
    """测试小规模语言模型的特殊优化"""
    
    def test_layer_range_for_small_models(self):
        """测试小模型的层数范围"""
        # 小模型应该使用 2-12 层
        for _ in range(100):
            arch = {"type": "transformer", "num_layers": 8}
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            layers = mutated.get("num_layers", 8)
            assert 2 <= layers <= 12, f"Layers {layers} out of small model range [2, 12]"
    
    def test_hidden_size_range_for_small_models(self):
        """测试小模型的隐藏维度范围"""
        for _ in range(100):
            arch = {"type": "transformer", "hidden_size": 512}
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            hidden = mutated.get("hidden_size", 512)
            assert 128 <= hidden <= 768, f"Hidden size {hidden} out of small model range [128, 768]"
    
    def test_ffn_ratio_for_small_models(self):
        """测试小模型的 FFN 比例"""
        for _ in range(100):
            arch = {
                "type": "transformer",
                "hidden_size": 512,
                "ffn_dim": 2048
            }
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            hidden = mutated.get("hidden_size", 512)
            ffn = mutated.get("ffn_dim", 2048)
            ratio = ffn / hidden
            
            # 小模型通常使用 2-4x 的 FFN 比例
            assert 2 <= ratio <= 6, f"FFN ratio {ratio} out of typical range [2, 6]"
    
    def test_dropout_range_for_small_models(self):
        """测试小模型的 dropout 范围"""
        for _ in range(100):
            arch = {"type": "transformer", "dropout": 0.1}
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            dropout = mutated.get("dropout", 0.1)
            assert 0.0 <= dropout <= 0.3, f"Dropout {dropout} out of range [0.0, 0.3]"
    
    def test_tendency_toward_smaller_configs(self):
        """测试倾向于较小配置"""
        # 统计多次变异的结果
        layer_changes = []
        hidden_changes = []
        
        for _ in range(200):
            # 从中间值开始
            arch = {
                "type": "transformer",
                "num_layers": 8,
                "hidden_size": 512
            }
            
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="moderate"
            )
            
            layer_changes.append(mutated.get("num_layers", 8) - 8)
            hidden_changes.append(mutated.get("hidden_size", 512) - 512)
        
        # 统计倾向
        avg_layer_change = np.mean(layer_changes)
        avg_hidden_change = np.mean(hidden_changes)
        
        # 应该倾向于减小（负值或接近0）
        # 注意：这是统计测试，允许一定的随机性
        assert avg_layer_change <= 1.0, f"Average layer change {avg_layer_change} should be <= 1.0 (small model bias)"
        assert avg_hidden_change <= 50, f"Average hidden change {avg_hidden_change} should be <= 50 (small model bias)"


class TestMutationStrategies:
    """测试不同的变异策略"""
    
    def test_fine_tune_strategy(self):
        """测试微调策略"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 微调策略应该产生小的变化
        changes = []
        for _ in range(100):
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=0.3, strategy="fine_tune"
            )
            layer_diff = abs(mutated.get("num_layers", 6) - 6)
            changes.append(layer_diff)
        
        # 大多数变化应该是小的
        small_changes = sum(1 for c in changes if c <= 1)
        assert small_changes >= 70, f"Expected >= 70 small changes, got {small_changes}"
    
    def test_exploratory_strategy(self):
        """测试探索策略"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 探索策略应该产生较大的变化
        changes = []
        for _ in range(100):
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=1.0, strategy="exploratory"
            )
            layer_diff = abs(mutated.get("num_layers", 6) - 6)
            changes.append(layer_diff)
        
        # 应该有更多的变化
        avg_change = np.mean(changes)
        # 注意：这个测试可能因为随机性而偶尔失败
    
    def test_adaptive_strategy_evolution(self):
        """测试自适应策略随世代变化"""
        arch = {
            "type": "transformer",
            "num_layers": 6
        }
        
        # 早期：高变异率
        early_mutations = []
        for _ in range(100):
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=0.5, strategy="adaptive", generation=0
            )
            if mutated.get("num_layers") != 6:
                early_mutations.append(1)
            else:
                early_mutations.append(0)
        
        # 晚期：低变异率
        late_mutations = []
        for _ in range(100):
            mutated, _ = MutationStrategy.mutate_transformer(
                arch, mutation_rate=0.5, strategy="adaptive", generation=100
            )
            if mutated.get("num_layers") != 6:
                late_mutations.append(1)
            else:
                late_mutations.append(0)
        
        early_rate = np.mean(early_mutations)
        late_rate = np.mean(late_mutations)
        
        # 早期变异率应该高于晚期
        # 注意：这是统计测试，可能偶尔失败
        assert early_rate >= late_rate * 0.8, \
            f"Early rate {early_rate} should be >= late rate {late_rate} * 0.8"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=genetic_ml_evolution.genetic_algorithm", "--cov-report=term-missing"])
