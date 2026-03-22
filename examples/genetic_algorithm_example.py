"""
Genetic Algorithm Example for Small-Scale Language Models
演示如何使用遗传算法优化小规模语言模型的架构

这个示例展示了:
1. 如何创建遗传算法实例
2. 如何定义适应度函数
3. 如何运行进化过程
4. 如何分析结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from genetic_ml_evolution import GeneticAlgorithm, SurrogateModel
import tempfile
import json


def simple_fitness_function(architecture):
    """
    简单的适应度函数示例
    
    在实际应用中,这里应该是:
    1. 根据架构构建模型
    2. 在数据集上训练模型
    3. 返回验证集性能
    
    为了演示,我们使用一个简化的评估方法
    """
    if architecture.get("type") != "transformer":
        return 50.0  # 非语言模型的基础分数
    
    # 基于架构大小的启发式评估
    layers = architecture.get("num_layers", 6)
    hidden = architecture.get("hidden_size", 512)
    heads = architecture.get("num_heads", 8)
    ffn = architecture.get("ffn_dim", 2048)
    dropout = architecture.get("dropout", 0.1)
    
    # 基础分数
    base_score = 60.0
    
    # 层数影响 (太少不好,太多也不好)
    layer_score = min(layers, 8) * 2  # 鼓励 8 层左右
    
    # 隐藏维度影响 (适中最好)
    if 256 <= hidden <= 512:
        hidden_score = 10.0
    else:
        hidden_score = 5.0
    
    # 注意力头数影响
    if heads in [4, 8]:
        head_score = 8.0
    else:
        head_score = 4.0
    
    # FFN 比例影响
    ffn_ratio = ffn / hidden if hidden > 0 else 0
    if 2 <= ffn_ratio <= 4:
        ffn_score = 6.0
    else:
        ffn_score = 3.0
    
    # Dropout 影响 (0.1-0.2 最好)
    if 0.1 <= dropout <= 0.2:
        dropout_score = 5.0
    else:
        dropout_score = 2.0
    
    # 总分
    total_score = base_score + layer_score + hidden_score + head_score + ffn_score + dropout_score
    
    # 添加一些随机性 (模拟真实训练的波动)
    import random
    noise = random.uniform(-2.0, 2.0)
    
    return min(100.0, max(0.0, total_score + noise))


def advanced_fitness_with_surrogate(architecture, surrogate_model):
    """
    使用代理模型的适应度函数
    
    如果代理模型可用,先预测,只在必要时真实评估
    """
    # 尝试使用代理模型预测
    if surrogate_model and surrogate_model.is_fitted:
        prediction = surrogate_model.predict(architecture)
        if prediction is not None:
            print(f"  Using surrogate prediction: {prediction:.2f}")
            return prediction
    
    # 真实评估
    real_fitness = simple_fitness_function(architecture)
    
    # 更新代理模型
    if surrogate_model:
        surrogate_model.add_training_point(architecture, real_fitness)
        if len(surrogate_model.training_data) >= 5:
            surrogate_model.fit()
    
    return real_fitness


def main():
    print("=" * 80)
    print("遗传算法优化小规模语言模型架构 - 示例")
    print("=" * 80)
    
    # ==================== 示例 1: 基础进化 ====================
    print("\n【示例 1】基础遗传算法进化")
    print("-" * 80)
    
    ga = GeneticAlgorithm(
        population_size=20,
        mutation_rate=0.3,
        crossover_rate=0.3,
        elitism_rate=0.1,
        task_type="language"
    )
    
    print(f"配置:")
    print(f"  种群大小: {ga.population_size}")
    print(f"  变异率: {ga.mutation_rate}")
    print(f"  交叉率: {ga.crossover_rate}")
    print(f"  精英保留率: {ga.elitism_rate}")
    
    # 运行进化
    best_individual = ga.run(
        fitness_function=simple_fitness_function,
        max_generations=10,
        verbose=True
    )
    
    print(f"\n最佳架构:")
    print(f"  架构: {json.dumps(best_individual.architecture, indent=4)}")
    print(f"  适应度: {best_individual.fitness:.2f}")
    print(f"  年龄: {best_individual.age} 代")
    
    # ==================== 示例 2: 使用代理模型加速 ====================
    print("\n\n【示例 2】使用代理模型加速进化")
    print("-" * 80)
    
    # 创建代理模型
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_db = f.name
    
    surrogate = SurrogateModel(model_type="ensemble", cache_db_path=cache_db)
    
    ga_with_surrogate = GeneticAlgorithm(
        population_size=20,
        mutation_rate=0.3,
        task_type="language",
        surrogate_model=surrogate
    )
    
    # 创建带代理模型的适应度函数
    def fitness_with_surrogate(arch):
        return advanced_fitness_with_surrogate(arch, surrogate)
    
    best_with_surrogate = ga_with_surrogate.run(
        fitness_function=fitness_with_surrogate,
        max_generations=10,
        verbose=True
    )
    
    print(f"\n最佳架构 (使用代理模型):")
    print(f"  架构: {json.dumps(best_with_surrogate.architecture, indent=4)}")
    print(f"  适应度: {best_with_surrogate.fitness:.2f}")
    
    # 缓存统计
    if surrogate.cache:
        cache_stats = surrogate.get_cache_statistics()
        print(f"\n缓存统计:")
        print(f"  总条目: {cache_stats.get('total_entries', 0)}")
        print(f"  命中率: {cache_stats.get('hit_rate_percent', 0):.2f}%")
    else:
        print(f"\n缓存统计: 未启用缓存")
    
    # 清理
    os.remove(cache_db)
    
    # ==================== 示例 3: 自定义种子架构 ====================
    print("\n\n【示例 3】从种子架构开始进化")
    print("-" * 80)
    
    # 定义一些种子架构 (基于已知的好架构)
    seed_architectures = [
        {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
            "activation": "gelu"
        },
        {
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 1024,
            "dropout": 0.1,
            "activation": "gelu"
        },
        {
            "type": "transformer",
            "num_layers": 8,
            "hidden_size": 384,
            "num_heads": 6,
            "ffn_dim": 1536,
            "dropout": 0.15,
            "activation": "gelu"
        }
    ]
    
    ga_seeded = GeneticAlgorithm(
        population_size=20,
        mutation_rate=0.2,  # 较低的变异率,更依赖种子
        task_type="language"
    )
    
    # 使用种子初始化
    ga_seeded.initialize_population(seed_architectures=seed_architectures)
    
    best_seeded = ga_seeded.run(
        fitness_function=simple_fitness_function,
        max_generations=10,
        verbose=True
    )
    
    print(f"\n最佳架构 (从种子进化):")
    print(f"  架构: {json.dumps(best_seeded.architecture, indent=4)}")
    print(f"  适应度: {best_seeded.fitness:.2f}")
    
    # ==================== 示例 4: 分析进化历史 ====================
    print("\n\n【示例 4】进化历史分析")
    print("-" * 80)
    
    ga_analysis = GeneticAlgorithm(
        population_size=15,
        mutation_rate=0.3,
        task_type="language"
    )
    
    ga_analysis.run(
        fitness_function=simple_fitness_function,
        max_generations=15,
        verbose=False
    )
    
    # 分析历史
    print(f"进化历史 (共 {len(ga_analysis.history)} 代):")
    print(f"\n{'代数':<8} {'最佳适应度':<15} {'平均适应度':<15} {'多样性':<10}")
    print("-" * 50)
    
    for record in ga_analysis.history:
        print(
            f"{record['generation']:<8} "
            f"{record['best_fitness']:<15.2f} "
            f"{record['avg_fitness']:<15.2f} "
            f"{record['population_diversity']:<10.2%}"
        )
    
    # 统计分析
    best_fitnesses = [r['best_fitness'] for r in ga_analysis.history]
    avg_fitnesses = [r['avg_fitness'] for r in ga_analysis.history]
    
    print(f"\n统计摘要:")
    print(f"  最佳适应度范围: {min(best_fitnesses):.2f} - {max(best_fitnesses):.2f}")
    print(f"  平均适应度范围: {min(avg_fitnesses):.2f} - {max(avg_fitnesses):.2f}")
    print(f"  进化提升: {best_fitnesses[-1] - best_fitnesses[0]:.2f}")
    
    # ==================== 示例 5: 小模型约束优化 ====================
    print("\n\n【示例 5】小规模模型约束优化")
    print("-" * 80)
    
    def constrained_fitness(architecture):
        """
        带约束的适应度函数
        惩罚过大的模型
        """
        base_fitness = simple_fitness_function(architecture)
        
        if architecture.get("type") != "transformer":
            return base_fitness
        
        # 计算参数量 (粗略估计)
        layers = architecture.get("num_layers", 6)
        hidden = architecture.get("hidden_size", 512)
        vocab = architecture.get("vocab_size", 50000)
        
        # 简化的参数量估计
        # Embedding: vocab * hidden
        # Each layer: ~ 4 * hidden^2
        param_count = vocab * hidden + layers * 4 * hidden * hidden
        
        # 小模型目标: < 100M 参数
        target_params = 100_000_000
        
        # 惩罚过大的模型
        if param_count > target_params:
            penalty = (param_count / target_params - 1) * 10
            base_fitness -= penalty
        
        return max(0, base_fitness)
    
    ga_constrained = GeneticAlgorithm(
        population_size=20,
        mutation_rate=0.3,
        task_type="language"
    )
    
    best_constrained = ga_constrained.run(
        fitness_function=constrained_fitness,
        max_generations=10,
        verbose=True
    )
    
    # 估计参数量
    arch = best_constrained.architecture
    layers = arch.get("num_layers", 6)
    hidden = arch.get("hidden_size", 512)
    vocab = arch.get("vocab_size", 50000)
    param_count = vocab * hidden + layers * 4 * hidden * hidden
    
    print(f"\n最佳小模型架构:")
    print(f"  层数: {layers}")
    print(f"  隐藏维度: {hidden}")
    print(f"  估计参数量: {param_count / 1_000_000:.2f}M")
    print(f"  适应度: {best_constrained.fitness:.2f}")
    
    print("\n" + "=" * 80)
    print("示例完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
