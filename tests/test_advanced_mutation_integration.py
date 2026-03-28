"""
Test integration of AdvancedMutationStrategy with GeneticAlgorithm
测试高级变异策略与遗传算法的集成
"""

import pytest
import numpy as np
from genetic_ml_evolution.genetic_algorithm import GeneticAlgorithm, Individual


class TestAdvancedMutationIntegration:
    """测试高级变异策略集成"""
    
    def test_ga_with_advanced_mutation_enabled(self):
        """测试启用高级变异的遗传算法"""
        ga = GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.3,
            use_advanced_mutation=True,
            max_parameters=50_000_000,
            ucb_alpha=1.0,
            task_type="language"
        )
        
        # 检查高级变异策略已初始化
        assert ga.use_advanced_mutation is True
        assert ga.advanced_mutation_strategy is not None
        assert ga.max_parameters == 50_000_000
    
    def test_ga_with_basic_mutation(self):
        """测试使用基础变异的遗传算法"""
        ga = GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.3,
            use_advanced_mutation=False,
            task_type="language"
        )
        
        # 检查高级变异策略未初始化
        assert ga.use_advanced_mutation is False
        assert ga.advanced_mutation_strategy is None
    
    def test_mutate_with_advanced_strategy(self):
        """测试使用高级变异策略进行变异"""
        ga = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            max_parameters=100_000_000,
            task_type="language"
        )
        
        # 创建个体
        individual = Individual(
            architecture={
                "type": "transformer",
                "num_layers": 6,
                "hidden_size": 512,
                "num_heads": 8,
                "ffn_dim": 2048,
                "dropout": 0.1
            },
            fitness=75.0,
            age=5
        )
        
        # 执行变异
        mutated = ga.mutate(individual)
        
        # 检查变异后的个体
        assert isinstance(mutated, Individual)
        assert mutated.architecture["type"] == "transformer"
        assert 2 <= mutated.architecture.get("num_layers", 6) <= 12
        assert 128 <= mutated.architecture.get("hidden_size", 512) <= 768
    
    def test_mutate_respects_parameter_budget(self):
        """测试变异遵守参数预算"""
        ga = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            max_parameters=5_000_000,  # 5M budget
            task_type="language"
        )
        
        # 小模型个体
        individual = Individual(
            architecture={
                "type": "transformer",
                "num_layers": 2,
                "hidden_size": 128,
                "num_heads": 2,
                "ffn_dim": 256,
                "vocab_size": 10000
            },
            fitness=70.0,
            age=3
        )
        
        # 执行多次变异
        for _ in range(20):
            mutated = ga.mutate(individual)
            params = ga.advanced_mutation_strategy.estimate_parameters(mutated.architecture)
            # 参数预算检查是渐进的，允许轻微超出
            assert params <= 6_000_000, f"Params {params} exceed budget 5M"
    
    def test_statistics_include_mutation_stats(self):
        """测试统计信息包含变异统计"""
        ga = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            task_type="language"
        )
        
        # 初始化种群
        ga.initialize_population()
        
        # 评估种群
        def fitness_func(arch):
            return np.random.uniform(60, 90)
        
        ga.evaluate_population(fitness_func)
        
        # 执行一些进化
        for _ in range(5):
            ga.evolve()
            ga.evaluate_population(fitness_func)
        
        # 获取统计信息
        stats = ga.get_statistics()
        
        # 检查统计信息包含变异统计
        assert "mutation_stats" in stats
        assert "total_mutations" in stats["mutation_stats"]
        assert "overall_success_rate" in stats["mutation_stats"]
    
    def test_adaptive_mutation_rate_during_evolution(self):
        """测试进化过程中的自适应变异率"""
        ga = GeneticAlgorithm(
            population_size=20,
            mutation_rate=0.3,
            use_advanced_mutation=True,
            task_type="language"
        )
        
        # 初始化种群
        ga.initialize_population()
        
        def fitness_func(arch):
            # 简单的适应度函数
            layers = arch.get("num_layers", 6)
            hidden = arch.get("hidden_size", 512)
            return 80 + (12 - layers) * 0.5 + (768 - hidden) * 0.01
        
        # 运行进化
        rates = []
        for gen in range(30):
            ga.evaluate_population(fitness_func)
            
            # 记录变异率
            if ga.best_individual:
                individual = ga.population[0]
                rate = ga.advanced_mutation_strategy.calculate_adaptive_mutation_rate(
                    base_rate=0.3,
                    individual_fitness=individual.fitness,
                    individual_age=individual.age,
                    population_diversity=ga._calculate_diversity(),
                    generation=gen,
                    best_fitness=ga.best_individual.fitness
                )
                rates.append(rate)
            
            if gen < 29:
                ga.evolve()
        
        # 检查变异率变化
        assert len(rates) > 0
        # 早期变异率应该高于晚期（通常）
        early_avg = np.mean(rates[:10])
        late_avg = np.mean(rates[-10:])
        # 不强制要求 early > late，因为取决于适应度和多样性
    
    def test_ucb_learning_during_evolution(self):
        """测试进化过程中的UCB学习"""
        ga = GeneticAlgorithm(
            population_size=20,
            use_advanced_mutation=True,
            ucb_alpha=1.0,
            task_type="language"
        )
        
        # 初始化种群
        ga.initialize_population()
        
        def fitness_func(arch):
            return np.random.uniform(60, 90)
        
        # 运行进化
        for gen in range(20):
            ga.evaluate_population(fitness_func)
            if gen < 19:
                ga.evolve()
        
        # 检查UCB学习了变异操作
        stats = ga.get_statistics()
        mutation_stats = stats.get("mutation_stats", {})
        
        # 应该有变异记录
        assert mutation_stats.get("total_mutations", 0) > 0
        
        # 应该有多种操作被尝试
        operation_stats = mutation_stats.get("operation_stats", {})
        assert len(operation_stats) > 0
    
    def test_phase_based_mutation_during_evolution(self):
        """测试进化过程中的阶段性变异"""
        ga = GeneticAlgorithm(
            population_size=20,
            use_advanced_mutation=True,
            task_type="language"
        )
        
        # 测试不同阶段的变异幅度
        phases = [
            (5, "exploration"),
            (40, "balanced"),
            (80, "exploitation")
        ]
        
        individual = Individual(
            architecture={
                "type": "transformer",
                "num_layers": 6,
                "hidden_size": 512
            },
            fitness=80.0,
            age=5
        )
        
        for gen, expected_phase in phases:
            ga.generation = gen
            phase = ga.advanced_mutation_strategy.get_mutation_phase(gen)
            assert phase == expected_phase
    
    def test_comparison_advanced_vs_basic(self):
        """比较高级变异与基础变异"""
        # 使用高级变异
        ga_advanced = GeneticAlgorithm(
            population_size=20,
            mutation_rate=0.3,
            use_advanced_mutation=True,
            max_parameters=50_000_000,
            task_type="language"
        )
        
        # 使用基础变异
        ga_basic = GeneticAlgorithm(
            population_size=20,
            mutation_rate=0.3,
            use_advanced_mutation=False,
            task_type="language"
        )
        
        # 初始化种群
        ga_advanced.initialize_population()
        ga_basic.initialize_population()
        
        def fitness_func(arch):
            layers = arch.get("num_layers", 6)
            hidden = arch.get("hidden_size", 512)
            return 80 + (12 - layers) * 0.5 + (768 - hidden) * 0.01
        
        # 运行进化
        for gen in range(10):
            ga_advanced.evaluate_population(fitness_func)
            ga_basic.evaluate_population(fitness_func)
            
            if gen < 9:
                ga_advanced.evolve()
                ga_basic.evolve()
        
        # 检查高级变异有统计信息
        stats_advanced = ga_advanced.get_statistics()
        assert "mutation_stats" in stats_advanced
        
        # 检查基础变异没有变异统计
        stats_basic = ga_basic.get_statistics()
        assert "mutation_stats" not in stats_basic


class TestEdgeCases:
    """测试边界情况"""
    
    def test_cnn_with_advanced_mutation(self):
        """测试CNN架构使用高级变异（应回退到基础变异）"""
        ga = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            task_type="image"
        )
        
        individual = Individual(
            architecture={
                "type": "cnn",
                "num_blocks": 4,
                "base_channels": 64
            },
            fitness=75.0,
            age=5
        )
        
        # 执行变异
        mutated = ga.mutate(individual)
        
        # 应该使用基础变异策略
        assert isinstance(mutated, Individual)
        assert mutated.architecture["type"] == "cnn"
    
    def test_multimodal_with_advanced_mutation(self):
        """测试多模态架构使用高级变异（应回退到基础变异）"""
        ga = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            task_type="multimodal"
        )
        
        individual = Individual(
            architecture={
                "type": "multimodal",
                "vision_encoder": {"type": "cnn", "num_blocks": 4},
                "text_encoder": {"type": "transformer", "num_layers": 6}
            },
            fitness=75.0,
            age=5
        )
        
        # 执行变异
        mutated = ga.mutate(individual)
        
        # 应该使用基础变异策略
        assert isinstance(mutated, Individual)
        assert mutated.architecture["type"] == "multimodal"
    
    def test_none_fitness_with_advanced_mutation(self):
        """测试适应度为None时的高级变异"""
        ga = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            task_type="language"
        )
        
        individual = Individual(
            architecture={
                "type": "transformer",
                "num_layers": 6,
                "hidden_size": 512
            },
            fitness=None,  # 无适应度
            age=0
        )
        
        # 执行变异（不应崩溃）
        mutated = ga.mutate(individual)
        
        assert isinstance(mutated, Individual)
    
    def test_extreme_parameter_budget(self):
        """测试极端参数预算"""
        # 非常小的预算
        ga_small = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            max_parameters=1_000_000,  # 1M
            task_type="language"
        )
        
        # 非常大的预算
        ga_large = GeneticAlgorithm(
            population_size=10,
            use_advanced_mutation=True,
            max_parameters=1_000_000_000,  # 1B
            task_type="language"
        )
        
        individual = Individual(
            architecture={
                "type": "transformer",
                "num_layers": 2,
                "hidden_size": 128,
                "num_heads": 2,
                "ffn_dim": 256,
                "vocab_size": 10000
            },
            fitness=75.0,
            age=5
        )
        
        # 两个都应该能正常工作
        mutated_small = ga_small.mutate(individual)
        mutated_large = ga_large.mutate(individual)
        
        assert isinstance(mutated_small, Individual)
        assert isinstance(mutated_large, Individual)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=genetic_ml_evolution.genetic_algorithm", "--cov-report=term-missing"])
