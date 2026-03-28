"""
SLM-Optimized Mutation Operators - Usage Example
演示如何使用针对小规模语言模型优化的变异操作

This example demonstrates:
1. Basic usage of SLMOptimizedMutation
2. Resource-aware mutations
3. Semantic-aware mutations
4. Adaptive mutation strategies
5. Integration with genetic algorithm
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from genetic_ml_evolution.slm_optimized_mutation import (
    SLMOptimizedMutation,
    ResourceEstimator,
    SemanticAnalyzer,
    create_slm_mutation_operator,
)
from genetic_ml_evolution.genetic_operators import ArchitectureGene
import json


def example_basic_usage():
    """示例 1: 基础用法"""
    print("=" * 80)
    print("示例 1: 基础用法")
    print("=" * 80)
    
    # 创建变异操作器
    mutator = create_slm_mutation_operator(
        max_params=100_000_000,  # 100M 参数上限
        max_memory_gb=20.0,      # 20GB 显存上限
        enable_semantic_analysis=True,
        enable_history_learning=True,
        verbose=True
    )
    
    # 定义一个 Transformer 架构
    architecture = {
        "type": "transformer",
        "num_layers": 6,
        "hidden_size": 512,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
        "vocab_size": 50000,
        "max_seq_len": 512,
    }
    
    print(f"\n原始架构:")
    print(json.dumps(architecture, indent=2))
    
    # 估算资源
    params = ResourceEstimator.estimate_transformer_params(architecture)
    memory = ResourceEstimator.estimate_memory_gb(architecture)
    
    print(f"\n资源估算:")
    print(f"  参数量: {params / 1e6:.2f}M")
    print(f"  显存需求: {memory:.2f}GB")
    
    # 执行变异
    mutated, desc = mutator.mutate(architecture, fitness=50.0, strategy="moderate")
    
    print(f"\n变异后架构:")
    print(json.dumps(mutated, indent=2))
    print(f"\n变异描述: {desc}")
    
    # 估算新资源
    new_params = ResourceEstimator.estimate_transformer_params(mutated)
    new_memory = ResourceEstimator.estimate_memory_gb(mutated)
    
    print(f"\n新资源估算:")
    print(f"  参数量: {new_params / 1e6:.2f}M ({(new_params - params) / 1e6:+.2f}M)")
    print(f"  显存需求: {new_memory:.2f}GB ({new_memory - memory:+.2f}GB)")


def example_semantic_analysis():
    """示例 2: 语义分析"""
    print("\n\n" + "=" * 80)
    print("示例 2: 语义分析")
    print("=" * 80)
    
    # 分析一个不平衡的架构
    unbalanced_arch = {
        "num_layers": 12,
        "hidden_size": 256,
        "num_heads": 12,
        "ffn_dim": 512,  # 只有 2x hidden，太小
        "dropout": 0.05,
    }
    
    print(f"\n不平衡架构:")
    print(json.dumps(unbalanced_arch, indent=2))
    
    # 语义分析
    analysis = SemanticAnalyzer.analyze_transformer_semantics(unbalanced_arch)
    
    print(f"\n语义分析结果:")
    print(f"  深度: {analysis['depth']}")
    print(f"  宽度: {analysis['width']}")
    print(f"  FFN 比例: {analysis['ffn_ratio']:.2f}x")
    print(f"  Head 大小: {analysis['head_size']}")
    print(f"  平衡分数: {analysis['balance_score']:.1f}/100")
    
    if analysis['issues']:
        print(f"\n问题:")
        for issue in analysis['issues']:
            print(f"  - {issue}")
    
    if analysis['recommendations']:
        print(f"\n建议:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    # 获取改进建议
    suggestions = SemanticAnalyzer.suggest_improvements(unbalanced_arch, analysis)
    
    print(f"\n具体改进建议:")
    for suggestion in suggestions[:3]:  # 只显示前 3 个
        print(f"  [{suggestion['priority']}] {suggestion['description']}")
        if 'target_value' in suggestion:
            print(f"    建议值: {suggestion['target_value']}")


def example_adaptive_strategies():
    """示例 3: 自适应策略"""
    print("\n\n" + "=" * 80)
    print("示例 3: 自适应策略")
    print("=" * 80)
    
    mutator = SLMOptimizedMutation(verbose=False)
    
    base_architecture = {
        "type": "transformer",
        "num_layers": 6,
        "hidden_size": 512,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
    }
    
    # 不同适应度的自适应策略
    fitness_levels = [
        (85, "高性能 - 保守策略"),
        (50, "中等性能 - 适度策略"),
        (30, "低性能 - 激进策略"),
    ]
    
    print(f"\n基础架构:")
    print(json.dumps(base_architecture, indent=2))
    
    for fitness, description in fitness_levels:
        mutated, desc = mutator.mutate(
            base_architecture, 
            fitness=fitness, 
            strategy="adaptive"
        )
        
        print(f"\n{description} (fitness={fitness}):")
        print(f"  变异: {desc}")
        
        # 统计变化
        changes = sum(
            1 for k in base_architecture.keys()
            if base_architecture[k] != mutated.get(k)
        )
        print(f"  变化参数数: {changes}")


def example_resource_constraints():
    """示例 4: 资源约束"""
    print("\n\n" + "=" * 80)
    print("示例 4: 资源约束")
    print("=" * 80)
    
    # 创建一个严格的资源约束
    strict_mutator = SLMOptimizedMutation(
        max_params=50_000_000,  # 只有 50M
        max_memory_gb=10.0,     # 只有 10GB
        verbose=False
    )
    
    large_architecture = {
        "type": "transformer",
        "num_layers": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "ffn_dim": 3072,
        "vocab_size": 100000,
    }
    
    print(f"\n大型架构:")
    print(json.dumps(large_architecture, indent=2))
    
    params = ResourceEstimator.estimate_transformer_params(large_architecture)
    print(f"\n参数量: {params / 1e6:.2f}M")
    print(f"预算: {strict_mutator.max_params / 1e6:.2f}M")
    
    # 尝试变异
    print(f"\n尝试变异（预算约束）:")
    for i in range(5):
        mutated, desc = strict_mutator.mutate(large_architecture, strategy="moderate")
        new_params = ResourceEstimator.estimate_transformer_params(mutated)
        
        within_budget = strict_mutator._is_within_budget(mutated)
        status = "✓" if within_budget else "✗"
        
        print(f"  {i+1}. {desc}")
        print(f"     参数量: {new_params / 1e6:.2f}M {status}")


def example_integration_with_genetic_algorithm():
    """示例 5: 与遗传算法集成"""
    print("\n\n" + "=" * 80)
    print("示例 5: 与遗传算法集成")
    print("=" * 80)
    
    from genetic_ml_evolution.genetic_algorithm import GeneticAlgorithm
    
    # 创建使用 SLM 优化的遗传算法
    mutator = SLMOptimizedMutation(
        max_params=100_000_000,
        enable_semantic_analysis=True,
        verbose=False
    )
    
    # 简单的适应度函数
    def fitness_function(architecture):
        """简化的适应度函数"""
        if architecture.get("type") != "transformer":
            return 50.0
        
        layers = architecture.get("num_layers", 6)
        hidden = architecture.get("hidden_size", 512)
        dropout = architecture.get("dropout", 0.1)
        
        # 基础分数
        score = 60.0
        
        # 层数影响
        if 4 <= layers <= 8:
            score += 15
        else:
            score += 5
        
        # 隐藏维度影响
        if 256 <= hidden <= 512:
            score += 10
        else:
            score += 5
        
        # Dropout 影响
        if 0.1 <= dropout <= 0.2:
            score += 10
        else:
            score += 3
        
        import random
        score += random.uniform(-2, 2)
        
        return min(100, max(0, score))
    
    # 自定义变异函数（使用 SLM 优化）
    def custom_mutate(individual, ga_instance):
        """使用 SLM 优化的变异"""
        mutated_arch, desc = mutator.mutate(
            individual.architecture,
            fitness=individual.fitness,
            strategy="adaptive"
        )
        
        # 记录结果
        if individual.fitness is not None:
            # 稍后会评估新的适应度
            pass
        
        # 创建新个体
        from genetic_ml_evolution.genetic_algorithm import Individual
        return Individual(architecture=mutated_arch)
    
    print(f"\n运行遗传算法（使用 SLM 优化的变异）:")
    print(f"  种群大小: 20")
    print(f"  迭代次数: 10")
    
    # 注意：这里展示如何集成，实际运行需要更多时间
    print(f"\n集成方式:")
    print(f"  1. 创建 SLMOptimizedMutation 实例")
    print(f"  2. 在遗传算法中使用自定义变异函数")
    print(f"  3. 利用语义分析和资源约束")
    
    # 获取统计信息
    stats = mutator.get_statistics()
    print(f"\n变异统计:")
    print(f"  总变异次数: {stats['total_mutations']}")
    print(f"  成功率: {stats['success_rate']:.2%}")


def example_block_mutations():
    """示例 6: 块变异"""
    print("\n\n" + "=" * 80)
    print("示例 6: 块变异")
    print("=" * 80)
    
    mutator = SLMOptimizedMutation(verbose=False)
    
    architecture = {
        "type": "transformer",
        "num_layers": 6,
        "hidden_size": 512,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
    }
    
    print(f"\n原始架构:")
    print(json.dumps(architecture, indent=2))
    
    print(f"\n块变异示例:")
    
    for i in range(5):
        block_result = mutator._block_mutation(architecture)
        
        if block_result:
            mutated, desc = block_result
            print(f"\n{i+1}. {desc}")
            
            # 显示关键变化
            if "layers" in desc:
                old_layers = architecture["num_layers"]
                new_layers = mutated["num_layers"]
                print(f"   层数: {old_layers} → {new_layers}")
            
            if "hidden" in desc:
                old_hidden = architecture["hidden_size"]
                new_hidden = mutated["hidden_size"]
                print(f"   隐藏维度: {old_hidden} → {new_hidden}")


def main():
    """运行所有示例"""
    example_basic_usage()
    example_semantic_analysis()
    example_adaptive_strategies()
    example_resource_constraints()
    example_integration_with_genetic_algorithm()
    example_block_mutations()
    
    print("\n\n" + "=" * 80)
    print("所有示例完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
