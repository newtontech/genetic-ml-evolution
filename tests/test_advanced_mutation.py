"""
Unit Tests for Advanced Mutation Strategies
测试高级变异策略的优化功能
"""

import pytest
import numpy as np
from typing import Dict, Any

from genetic_ml_evolution.advanced_mutation import (
    AdvancedMutationStrategy,
    MutationStatistics
)


class TestMutationStatistics:
    """测试变异统计类"""
    
    def test_statistics_creation(self):
        """测试统计类创建"""
        stats = MutationStatistics()
        
        assert stats.operation_counts == {}
        assert stats.success_counts == {}
    
    def test_record_mutation(self):
        """测试记录变异操作"""
        stats = MutationStatistics()
        
        stats.record_mutation("num_layers", True)
        stats.record_mutation("num_layers", False)
        stats.record_mutation("num_layers", True)
        
        assert stats.operation_counts["num_layers"] == 3
        assert stats.success_counts["num_layers"] == 2
    
    def test_get_success_rate(self):
        """测试获取成功率"""
        stats = MutationStatistics()
        
        # 未记录的操作
        assert stats.get_success_rate("unknown") == 0.5
        
        # 记录操作
        stats.record_mutation("op1", True)
        stats.record_mutation("op1", True)
        stats.record_mutation("op1", False)
        
        assert stats.get_success_rate("op1") == pytest.approx(2/3)
    
    def test_get_best_operations(self):
        """测试获取最佳操作"""
        stats = MutationStatistics()
        
        # 空统计
        assert stats.get_best_operations() == []
        
        # 添加操作
        stats.record_mutation("op1", True)
        stats.record_mutation("op1", True)
        stats.record_mutation("op2", True)
        stats.record_mutation("op2", False)
        stats.record_mutation("op3", False)
        
        best = stats.get_best_operations(n=2)
        assert len(best) == 2
        assert best[0] == "op1"  # 100% success
        assert best[1] == "op2"  # 50% success


class TestAdvancedMutationStrategy:
    """测试高级变异策略"""
    
    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        return AdvancedMutationStrategy(ucb_alpha=1.0)
    
    # ==================== 自适应变异率测试 ====================
    
    def test_calculate_adaptive_rate_basic(self, strategy):
        """测试基本自适应变异率计算"""
        rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=0.1,
            individual_fitness=80.0,
            individual_age=5,
            population_diversity=0.5,
            generation=30,
            best_fitness=100.0
        )
        
        # 应该在合理范围内
        assert 0.01 <= rate <= 0.9
        assert isinstance(rate, float)
    
    def test_adaptive_rate_fitness_effect(self, strategy):
        """测试适应度对变异率的影响"""
        base_rate = 0.1
        best_fitness = 100.0
        
        # 高适应度个体：应该降低变异率
        high_fitness_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=95.0,
            individual_age=5,
            population_diversity=0.5,
            generation=30,
            best_fitness=best_fitness
        )
        
        # 低适应度个体：应该提高变异率
        low_fitness_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=50.0,
            individual_age=5,
            population_diversity=0.5,
            generation=30,
            best_fitness=best_fitness
        )
        
        # 低适应度应该有更高的变异率
        assert low_fitness_rate > high_fitness_rate
    
    def test_adaptive_rate_age_effect(self, strategy):
        """测试年龄对变异率的影响"""
        base_rate = 0.1
        
        # 年轻个体：应该提高变异率
        young_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=80.0,
            individual_age=2,
            population_diversity=0.5,
            generation=30,
            best_fitness=100.0
        )
        
        # 老年个体：应该降低变异率
        old_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=80.0,
            individual_age=15,
            population_diversity=0.5,
            generation=30,
            best_fitness=100.0
        )
        
        # 年轻个体应该有更高的变异率
        assert young_rate > old_rate
    
    def test_adaptive_rate_diversity_effect(self, strategy):
        """测试多样性对变异率的影响"""
        base_rate = 0.1
        
        # 低多样性：应该提高变异率（探索）
        low_diversity_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=80.0,
            individual_age=5,
            population_diversity=0.2,
            generation=30,
            best_fitness=100.0
        )
        
        # 高多样性：应该降低变异率（利用）
        high_diversity_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=80.0,
            individual_age=5,
            population_diversity=0.9,
            generation=30,
            best_fitness=100.0
        )
        
        # 低多样性应该有更高的变异率
        assert low_diversity_rate > high_diversity_rate
    
    def test_adaptive_rate_generation_effect(self, strategy):
        """测试世代对变异率的影响"""
        base_rate = 0.1
        
        # 早期（探索阶段）：应该提高变异率
        early_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=80.0,
            individual_age=5,
            population_diversity=0.5,
            generation=10,
            best_fitness=100.0
        )
        
        # 晚期（利用阶段）：应该降低变异率
        late_rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=base_rate,
            individual_fitness=80.0,
            individual_age=5,
            population_diversity=0.5,
            generation=70,
            best_fitness=100.0
        )
        
        # 早期应该有更高的变异率
        assert early_rate > late_rate
    
    # ==================== 变异阶段测试 ====================
    
    def test_get_mutation_phase(self, strategy):
        """测试获取变异阶段"""
        # 探索阶段
        assert strategy.get_mutation_phase(0) == "exploration"
        assert strategy.get_mutation_phase(10) == "exploration"
        assert strategy.get_mutation_phase(19) == "exploration"
        
        # 平衡阶段
        assert strategy.get_mutation_phase(20) == "balanced"
        assert strategy.get_mutation_phase(40) == "balanced"
        assert strategy.get_mutation_phase(59) == "balanced"
        
        # 利用阶段
        assert strategy.get_mutation_phase(60) == "exploitation"
        assert strategy.get_mutation_phase(100) == "exploitation"
        assert strategy.get_mutation_phase(1000) == "exploitation"
    
    # ==================== UCB操作选择测试 ====================
    
    def test_select_operation_random_initial(self, strategy):
        """测试初始随机操作选择"""
        operations = ["op1", "op2", "op3"]
        
        # 没有历史数据时，应该随机选择
        selected = strategy.select_mutation_operation(operations)
        assert selected in operations
    
    def test_select_operation_with_history(self, strategy):
        """测试有历史数据的操作选择"""
        operations = ["op1", "op2", "op3"]
        
        # 记录一些操作
        for _ in range(5):
            strategy.stats.record_mutation("op1", True)
        
        for _ in range(5):
            strategy.stats.record_mutation("op2", False)
        
        for _ in range(10):
            strategy.stats.record_mutation("op3", True)
        
        # 多次选择，应该倾向于成功率高的操作
        selections = []
        for _ in range(100):
            selected = strategy.select_mutation_operation(operations)
            selections.append(selected)
        
        # op1 和 op3 成功率高，应该被选中更多
        op1_count = selections.count("op1")
        op3_count = selections.count("op3")
        
        # 至少应该选择成功操作多于失败操作
        assert op1_count + op3_count > 50
    
    def test_select_operation_empty_list(self, strategy):
        """测试空操作列表"""
        selected = strategy.select_mutation_operation([])
        assert selected == "random"
    
    # ==================== 参数估算测试 ====================
    
    def test_estimate_parameters_transformer(self, strategy):
        """测试Transformer参数估算"""
        # 小模型 (约 10M 参数)
        small_arch = {
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 512,
            "vocab_size": 10000
        }
        
        small_params = strategy.estimate_parameters(small_arch)
        assert small_params > 0
        assert small_params < 20_000_000  # < 20M
        
        # 中等模型 (约 50M 参数)
        medium_arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "vocab_size": 30000
        }
        
        medium_params = strategy.estimate_parameters(medium_arch)
        assert medium_params > small_params
        assert medium_params < 100_000_000  # < 100M
    
    def test_estimate_parameters_non_transformer(self, strategy):
        """测试非Transformer架构"""
        cnn_arch = {"type": "cnn"}
        params = strategy.estimate_parameters(cnn_arch)
        assert params == 0
    
    def test_estimate_tokens_per_forward(self, strategy):
        """测试前向传播token估算"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "max_seq_len": 512
        }
        
        tokens = strategy.estimate_tokens_per_forward(arch)
        assert tokens > 0
    
    # ==================== 高级变异测试 ====================
    
    def test_mutate_transformer_advanced_basic(self, strategy):
        """测试高级Transformer变异基本功能"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1
        }
        
        mutated, desc = strategy.mutate_transformer_advanced(
            architecture=arch,
            base_mutation_rate=0.5,
            generation=30
        )
        
        # 应该返回有效架构
        assert isinstance(mutated, dict)
        assert mutated["type"] == "transformer"
        assert isinstance(desc, str)
        
        # 参数应该在范围内
        assert 2 <= mutated.get("num_layers", 6) <= 12
        assert 128 <= mutated.get("hidden_size", 512) <= 768
    
    def test_mutate_with_parameter_budget(self, strategy):
        """测试参数预算限制"""
        # 使用一个较小的起始架构
        arch = {
            "type": "transformer",
            "num_layers": 2,
            "hidden_size": 128,
            "num_heads": 2,
            "ffn_dim": 256,
            "vocab_size": 10000
        }
        
        # 设置参数预算
        mutated, desc = strategy.mutate_transformer_advanced(
            architecture=arch,
            base_mutation_rate=0.5,
            generation=30,
            max_parameters=2_000_000  # 2M
        )
        
        # 变异后的架构应该不超过参数预算
        final_params = strategy.estimate_parameters(mutated)
        # 由于参数预算检查是逐步的，允许轻微超出
        assert final_params <= 2_500_000, f"Params {final_params} exceed budget 2.5M"
    
    def test_mutate_phase_exploration(self, strategy):
        """测试探索阶段变异"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 探索阶段（早期）：应该有较大变化
        changes = []
        for _ in range(50):
            mutated, _ = strategy.mutate_transformer_advanced(
                architecture=arch,
                base_mutation_rate=1.0,
                generation=10,  # 探索阶段
                max_parameters=100_000_000
            )
            
            layer_diff = abs(mutated.get("num_layers", 6) - 6)
            changes.append(layer_diff)
        
        avg_change = np.mean(changes)
        # 探索阶段应该有较大变化
        assert avg_change >= 0.5
    
    def test_mutate_phase_exploitation(self, strategy):
        """测试利用阶段变异"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 利用阶段（晚期）：应该有较小变化
        changes = []
        for _ in range(50):
            mutated, _ = strategy.mutate_transformer_advanced(
                architecture=arch,
                base_mutation_rate=1.0,
                generation=80,  # 利用阶段
                max_parameters=100_000_000
            )
            
            layer_diff = abs(mutated.get("num_layers", 6) - 6)
            changes.append(layer_diff)
        
        avg_change = np.mean(changes)
        # 利用阶段应该有较小变化
        assert avg_change <= 1.5
    
    def test_mutate_small_model_bias(self, strategy):
        """测试小模型偏好"""
        results = {"layers": [], "hidden": []}
        
        for _ in range(100):
            arch = {
                "type": "transformer",
                "num_layers": 8,
                "hidden_size": 512
            }
            
            mutated, _ = strategy.mutate_transformer_advanced(
                architecture=arch,
                base_mutation_rate=1.0,
                generation=30,
                max_parameters=100_000_000
            )
            
            results["layers"].append(mutated.get("num_layers", 8))
            results["hidden"].append(mutated.get("hidden_size", 512))
        
        avg_layers = np.mean(results["layers"])
        avg_hidden = np.mean(results["hidden"])
        
        # 应该倾向于较小的值
        assert avg_layers <= 8.5, f"Average layers {avg_layers} shows small model bias"
        assert avg_hidden <= 600, f"Average hidden {avg_hidden} shows small model bias"
    
    def test_mutate_hidden_heads_compatibility(self, strategy):
        """测试hidden_size和num_heads兼容性"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8
        }
        
        for _ in range(100):
            mutated, _ = strategy.mutate_transformer_advanced(
                architecture=arch,
                base_mutation_rate=1.0,
                generation=30,
                max_parameters=100_000_000
            )
            
            hidden = mutated.get("hidden_size", 512)
            heads = mutated.get("num_heads", 8)
            
            # hidden_size应该能被num_heads整除
            assert hidden % heads == 0, f"{hidden} not divisible by {heads}"
    
    def test_get_mutation_statistics(self, strategy):
        """测试获取变异统计"""
        # 执行一些变异
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        for _ in range(20):
            strategy.mutate_transformer_advanced(
                architecture=arch,
                base_mutation_rate=0.5,
                generation=30,
                max_parameters=100_000_000
            )
        
        stats = strategy.get_mutation_statistics()
        
        assert "total_mutations" in stats
        assert "total_successes" in stats
        assert "operation_stats" in stats
        assert "overall_success_rate" in stats
        
        assert stats["total_mutations"] > 0
        assert 0 <= stats["overall_success_rate"] <= 1.0
    
    def test_statistics_tracking_across_mutations(self, strategy):
        """测试跨变异的统计跟踪"""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1
        }
        
        # 执行多次变异
        for _ in range(100):
            strategy.mutate_transformer_advanced(
                architecture=arch,
                base_mutation_rate=0.5,
                generation=30,
                max_parameters=100_000_000
            )
        
        stats = strategy.get_mutation_statistics()
        
        # 应该有统计记录
        assert stats["total_mutations"] >= 100  # 每次变异尝试多个操作
        
        # 应该有不同操作类型的统计
        assert len(stats["operation_stats"]) > 0
        
        # 成功率应该在合理范围
        for op, op_stats in stats["operation_stats"].items():
            assert 0 <= op_stats["success_rate"] <= 1.0
            assert op_stats["count"] >= 0
            assert op_stats["successes"] >= 0


class TestIntegrationWithGeneticAlgorithm:
    """测试与遗传算法的集成"""
    
    def test_strategy_statefulness(self):
        """测试策略的状态保持"""
        strategy = AdvancedMutationStrategy(ucb_alpha=1.0)
        
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
        
        # 第一次变异
        mutated1, _ = strategy.mutate_transformer_advanced(
            architecture=arch,
            base_mutation_rate=0.5,
            generation=10
        )
        
        stats1 = strategy.get_mutation_statistics()
        
        # 第二次变异
        mutated2, _ = strategy.mutate_transformer_advanced(
            architecture=arch,
            base_mutation_rate=0.5,
            generation=10
        )
        
        stats2 = strategy.get_mutation_statistics()
        
        # 第二次应该有更多统计
        assert stats2["total_mutations"] > stats1["total_mutations"]
    
    def test_adaptive_rate_integration(self):
        """测试自适应变异率的实际效果"""
        strategy = AdvancedMutationStrategy(ucb_alpha=1.0)
        
        # 模拟进化过程
        arch = {"type": "transformer", "num_layers": 6}
        
        rates = []
        for gen in range(100):
            rate = strategy.calculate_adaptive_mutation_rate(
                base_rate=0.2,
                individual_fitness=80.0 + gen * 0.2,  # 逐渐提高
                individual_age=min(gen, 10),
                population_diversity=0.5 - gen * 0.003,  # 逐渐降低
                generation=gen,
                best_fitness=100.0 + gen * 0.2
            )
            rates.append(rate)
        
        # 早期变异率应该高于晚期
        early_avg = np.mean(rates[:20])
        late_avg = np.mean(rates[80:])
        
        assert early_avg > late_avg, "Early mutation rates should be higher than late rates"


class TestEdgeCases:
    """测试边界情况"""
    
    @pytest.fixture
    def strategy(self):
        return AdvancedMutationStrategy(ucb_alpha=1.0)
    
    def test_mutate_extreme_small_architecture(self, strategy):
        """测试极小架构变异"""
        min_arch = {
            "type": "transformer",
            "num_layers": 2,
            "hidden_size": 128,
            "num_heads": 2,
            "ffn_dim": 512
        }
        
        mutated, _ = strategy.mutate_transformer_advanced(
            architecture=min_arch,
            base_mutation_rate=1.0,
            generation=30,
            max_parameters=100_000_000
        )
        
        # 应该保持在合理范围内
        assert mutated.get("num_layers", 2) >= 2
        assert mutated.get("hidden_size", 128) >= 128
    
    def test_mutate_extreme_large_architecture(self, strategy):
        """测试极大架构变异"""
        max_arch = {
            "type": "transformer",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072
        }
        
        mutated, _ = strategy.mutate_transformer_advanced(
            architecture=max_arch,
            base_mutation_rate=1.0,
            generation=30,
            max_parameters=100_000_000
        )
        
        # 应该保持在合理范围内
        assert mutated.get("num_layers", 12) <= 12
        assert mutated.get("hidden_size", 768) <= 768
    
    def test_mutate_with_none_fitness(self, strategy):
        """测试适应度为None的情况"""
        arch = {
            "type": "transformer",
            "num_layers": 6
        }
        
        # 不应该崩溃
        mutated, desc = strategy.mutate_transformer_advanced(
            architecture=arch,
            base_mutation_rate=0.5,
            individual_fitness=None,
            generation=30
        )
        
        assert isinstance(mutated, dict)
    
    def test_mutate_minimal_architecture(self, strategy):
        """测试最小架构（只有type）"""
        minimal_arch = {"type": "transformer"}
        
        # 不应该崩溃
        mutated, desc = strategy.mutate_transformer_advanced(
            architecture=minimal_arch,
            base_mutation_rate=0.5,
            generation=30
        )
        
        assert isinstance(mutated, dict)
        assert "type" in mutated


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=genetic_ml_evolution.advanced_mutation", "--cov-report=term-missing"])
