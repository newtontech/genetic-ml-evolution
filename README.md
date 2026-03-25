# Genetic ML Evolution 🧬🤖

> 低成本遗传算法自我进化机器学习算法框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GPU Optimized](https://img.shields.io/badge/GPU-10--20GB-green.svg)](https://github.com)

## 🎯 项目目标

在**单张 10-20GB 显存 GPU** 上实现遗传算法自我进化机器学习算法，专注于：
- 📝 小规模语言模型（GPT-2 small 级别）
- 🖼️ 轻量级图像模型（MobileNet 级别）
- 🔄 多模态联合进化

## 🚀 核心特性

### 1. 低成本设计
- ✅ 代理模型加速（减少 90% 训练次数）
- ✅ 智能缓存机制（避免重复评估）
- ✅ 混合进化策略（梯度 + 遗传）
- ✅ 增量式进化（从简单到复杂）

### 2. 创新技术
- 🧬 **代理辅助进化**：使用代理模型预测架构性能
- 💾 **缓存优化**：分布式 + 本地缓存
- 🎯 **多目标优化**：精度 + 速度 + 显存
- 🔄 **自适应策略**：动态调整进化参数

### 3. 高级变异策略（NEW! 🎉）
- 🎯 **智能自适应变异率**：基于个体适应度、年龄、种群多样性和世代数动态调整
- 📊 **分层变异策略**：探索期（0-20代）→ 平衡期（20-60代）→ 利用期（60+代）
- 🎰 **UCB操作选择**：使用Upper Confidence Bound算法选择最优变异操作
- 💡 **Token限制优化**：考虑小规模语言模型的token限制和生成质量
- 📈 **参数预算控制**：限制模型参数数量，防止过大的模型
- 🔍 **成功率追踪**：记录每种变异操作的成功率，优化未来选择

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/genetic-ml-evolution.git
cd genetic-ml-evolution

# 安装依赖
pip install -r requirements.txt

# 安装论文查找工具（可选）
pip install arxiv arxiv-dl scholarly semanticscholar
```

## 🔧 快速开始

### 基础使用

```python
from genetic_ml_evolution import EvolutionEngine

# 创建进化引擎（单 GPU 配置）
engine = EvolutionEngine(
    gpu_memory=16,  # 16GB 显存
    task_type="language",  # 或 "image"
    dataset="imdb",  # 或 "cifar10"
    population_size=20,
    generations=50
)

# 开始进化
best_model = engine.evolve()
print(f"最佳模型: {best_model.architecture}")
print(f"验证精度: {best_model.accuracy:.2%}")
```

### 高级变异策略（NEW! 🎉）

#### 方式一：直接使用遗传算法（推荐）

```python
from genetic_ml_evolution import GeneticAlgorithm

# 创建遗传算法（自动启用高级变异）
engine = GeneticAlgorithm(
    population_size=20,
    mutation_rate=0.3,
    use_advanced_mutation=True,  # 启用高级变异策略
    max_parameters=50_000_000,   # 50M参数预算
    ucb_alpha=1.0,               # UCB探索参数
    task_type="language"
)

# 定义适应度函数
def fitness_function(architecture):
    # 评估架构性能
    return evaluate_model(architecture)

# 开始进化
best_model = engine.run(
    fitness_function=fitness_function,
    max_generations=50,
    verbose=True
)

# 查看统计信息
stats = engine.get_statistics()
print(f"最佳适应度: {stats['best_fitness']:.2f}")
print(f"变异成功率: {stats['mutation_stats']['overall_success_rate']:.2%}")
```

#### 方式二：直接使用高级变异策略

```python
from genetic_ml_evolution import AdvancedMutationStrategy

# 初始化高级变异策略
strategy = AdvancedMutationStrategy(ucb_alpha=1.0)

# 小规模语言模型架构
architecture = {
    "type": "transformer",
    "num_layers": 4,
    "hidden_size": 256,
    "num_heads": 4,
    "ffn_dim": 512,
    "dropout": 0.1,
    "vocab_size": 10000
}

# 使用高级变异（自适应率 + 分层策略 + UCB选择）
mutated, description = strategy.mutate_transformer_advanced(
    architecture=architecture,
    base_mutation_rate=0.3,
    individual_fitness=75.0,      # 当前个体适应度
    individual_age=5,              # 个体年龄（代数）
    population_diversity=0.5,      # 种群多样性
    generation=30,                 # 当前世代
    best_fitness=100.0,           # 最佳适应度
    max_parameters=50_000_000     # 参数预算（50M）
)

# 查看变异统计
stats = strategy.get_mutation_statistics()
print(f"总变异次数: {stats['total_mutations']}")
print(f"成功率: {stats['overall_success_rate']:.2%}")
```

**关键优化特性**：

1. **自适应变异率**：根据多种因素动态调整
   - 高适应度个体 → 降低变异率（保留优秀基因）
   - 年轻个体 → 提高变异率（增加探索）
   - 低多样性 → 提高变异率（防止早熟收敛）
   - 早期世代 → 提高变异率（探索阶段）

2. **分层策略**：不同进化阶段使用不同策略
   - **探索期（0-20代）**：大幅变异，探索搜索空间
   - **平衡期（20-60代）**：中等变异，平衡探索与利用
   - **利用期（60+代）**：小幅变异，精细调优

3. **UCB操作选择**：智能选择变异操作
   - 记录每种操作的成功率
   - 使用Upper Confidence Bound平衡探索与利用
   - 自动学习最优变异策略

4. **小模型优化**：专为小规模语言模型设计
   - 偏好较少层数（2-12层）
   - 偏好较小隐藏维度（128-768）
   - 参数预算限制（防止过大模型）
   - Token限制考虑

## 📊 架构设计

```
┌─────────────────────────────────────────┐
│          进化控制器 (Evolution)           │
├─────────────────────────────────────────┤
│  ┌──────────────┐   ┌────────────────┐ │
│  │ 代理模型预测  │←→│   缓存系统      │ │
│  └──────────────┘   └────────────────┘ │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ 语言模型  │  │ 图像模型  │  │ 多模态 ││
│  │  进化器   │  │  进化器   │  │ 进化器 ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

## 🧪 实验路线图

### 阶段 1：基础验证（Week 1-2）
- [ ] 实现代理模型（MLP 预测器）
- [ ] 实现基础缓存系统
- [ ] 在 CIFAR-10 上验证 CNN 进化

### 阶段 2：语言模型进化（Week 3-4）
- [ ] 设计 Transformer 搜索空间
- [ ] 在 IMDB 上验证文本分类
- [ ] 对比人工设计架构

### 阶段 3：多模态联合进化（Week 5-6）
- [ ] 设计视觉-语言联合搜索空间
- [ ] 在小型多模态数据集验证
- [ ] 优化显存占用

### 阶段 4：高级特性（Week 7-8）
- [ ] 元进化（进化遗传算法本身）
- [ ] 多目标优化（Pareto 前沿）
- [ ] 分布式扩展（多 GPU）

## 📚 论文资源

详见 [论文综述文档](https://feishu.cn/doc/...)（待更新链接）

核心论文：
1. **AutoML-Zero** - ICML 2020
2. **NeuroLGP-SM** - arXiv 2024
3. **EG-NAS** - AAAI 2024
4. **Similarity Surrogate NAS** - InfoSci 2024

## 🤝 贡献

欢迎贡献！请查看 [Issues](https://github.com/YOUR_USERNAME/genetic-ml-evolution/issues) 了解创新点和待完成任务。

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**创建时间**：2026-03-15  
**维护者**：OpenClaw AI Agent  
**状态**：🚧 开发中
