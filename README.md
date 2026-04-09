# Genetic ML Evolution 🧬🤖

> 低成本遗传算法自我进化机器学习算法框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GPU Optimized](https://img.shields.io/badge/GPU-10--20GB-green.svg)](https://github.com)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)]()

## 🎯 项目简介

**Genetic ML Evolution** 是一个基于遗传算法的神经架构搜索（NAS）框架，专门为 **小规模语言模型（SLM）** 和 **轻量级图像模型** 优化设计。核心目标是在 **单张 10-20GB 显存 GPU** 上，通过进化算法自动搜索最优的神经网络架构。

### 核心特性

- 🧬 **代理辅助进化** — 使用 ML 代理模型预测架构性能，减少 90% 真实训练次数
- 💾 **智能缓存系统** — SQLite 持久化缓存，避免重复评估相同架构
- 🎯 **SLM 优化变异** — 针对小模型设计的语义感知、资源约束变异策略
- 📊 **多目标优化** — 精度 + 参数量 + 显存占用的综合权衡
- 🔄 **自适应策略** — 根据进化阶段动态调整变异率和探索/利用比例
- 🏗️ **多架构支持** — Transformer、CNN、多模态联合进化

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/genetic-ml-evolution.git
cd genetic-ml-evolution

# 安装依赖
pip install -r requirements.txt
```

### 核心依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | ≥2.0.0 | 深度学习框架 |
| transformers | ≥4.30.0 | 语言模型支持 |
| scikit-learn | ≥1.3.0 | 代理模型 |
| numpy | ≥1.24.0 | 数值计算 |
| deap / pymoo | ≥1.4.0 | 进化算法 |

## 🚀 快速开始

### 1. 基础进化

```python
from genetic_ml_evolution import GeneticAlgorithm
import random

# 定义适应度函数（实际使用时替换为真实训练评估）
def fitness_function(architecture):
    # 根据架构参数评分
    layers = architecture.get("num_layers", 6)
    hidden = architecture.get("hidden_size", 512)
    score = 60 + min(layers, 8) * 2
    if 256 <= hidden <= 512:
        score += 10
    return score + random.uniform(-2, 2)

# 创建并运行遗传算法
ga = GeneticAlgorithm(
    population_size=20,
    mutation_rate=0.3,
    crossover_rate=0.3,
    task_type="language"  # "language", "image", "multimodal"
)

best = ga.run(fitness_function, max_generations=50)
print(f"最佳架构: {best.architecture}")
print(f"适应度: {best.fitness:.2f}")
```

### 2. 使用代理模型加速

```python
from genetic_ml_evolution import GeneticAlgorithm, SurrogateModel

# 创建代理模型（RF + GBDT + MLP 集成）
surrogate = SurrogateModel(model_type="ensemble")

def fitness_with_surrogate(architecture, surrogate=surrogate):
    if surrogate.is_fitted:
        prediction = surrogate.predict(architecture)
        if prediction is not None:
            return prediction
    real_fitness = fitness_function(architecture)
    surrogate.add_training_point(architecture, real_fitness)
    if len(surrogate.training_data) >= 5:
        surrogate.fit()
    return real_fitness

ga = GeneticAlgorithm(
    population_size=20,
    surrogate_model=surrogate,
    task_type="language"
)
best = ga.run(fitness_with_surrogate, max_generations=50)
```

### 3. 使用进化引擎（推荐）

```python
from genetic_ml_evolution import EvolutionEngine, EvolutionConfig

config = EvolutionConfig()
config.population_size = 20
config.generations = 50
config.task_type = "language"
config.use_cache = True
config.cache_db_path = "cache.db"
config.use_surrogate = True

engine = EvolutionEngine(config)
result = engine.evolve(fitness_function=fitness_function)

print(f"最佳适应度: {result['best_fitness']:.2f}")
print(f"最佳架构: {result['best_architecture']}")
print(f"总耗时: {result['total_time']:.2f}s")
```

### 4. 从种子架构开始

```python
seed_architectures = [
    {"type": "transformer", "num_layers": 6, "hidden_size": 512,
     "num_heads": 8, "ffn_dim": 2048, "dropout": 0.1, "activation": "gelu"},
    {"type": "transformer", "num_layers": 4, "hidden_size": 256,
     "num_heads": 4, "ffn_dim": 1024, "dropout": 0.1, "activation": "gelu"},
]

ga = GeneticAlgorithm(population_size=20, mutation_rate=0.2, task_type="language")
ga.initialize_population(seed_architectures=seed_architectures)
best = ga.run(fitness_function, max_generations=30)
```

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────┐
│              EvolutionEngine (进化引擎)               │
├──────────────────┬──────────────────────────────────┤
│  GeneticAlgorithm│       EvolutionConfig             │
│  ├─ Individual   │  population_size / mutation_rate  │
│  ├─ Mutation     │  crossover_rate / elitism_rate    │
│  ├─ Crossover    │  generations / early_stopping     │
│  └─ Selection    │  task_type / use_cache            │
├──────────────────┼──────────────────────────────────┤
│  SLMOptimizedMutation  │   SurrogateModel             │
│  ├─ ResourceEstimator  │   ├─ RandomForest            │
│  ├─ SemanticAnalyzer   │   ├─ GradientBoosting        │
│  └─ AdaptiveStrategy   │   └─ MLP                     │
├──────────────────┼──────────────────────────────────┤
│  SLMutationOperators   │   SLMCrossoverOperators      │
│  SLMSelectionOperators │   ArchitectureCache (SQLite)  │
└──────────────────┴──────────────────────────────────┘
```

## 📖 核心模块说明

### `genetic_algorithm.py` — 遗传算法核心

主要类：

| 类 | 说明 |
|---|---|
| `GeneticAlgorithm` | 遗传算法主类，管理种群进化流程 |
| `Individual` | 个体数据类，包含架构、适应度、年龄、变异历史 |
| `MutationStrategy` | 变异策略，支持 fine_tune / moderate / exploratory / guided / adaptive |

关键方法：

```python
ga = GeneticAlgorithm(population_size=20, task_type="language")

ga.initialize_population(seed_architectures=[...])  # 初始化种群
ga.evaluate_population(fitness_function)             # 评估适应度
ga.evolve()                                          # 进化一代
best = ga.run(fitness_function, max_generations=50)  # 完整运行
stats = ga.get_statistics()                           # 获取统计
```

### `genetic_operators.py` — 遗传算子

针对 SLM 优化的选择、交叉、变异算子：

| 类 | 说明 |
|---|---|
| `ArchitectureGene` | 架构基因，支持深拷贝和适应度追踪 |
| `SLMutationOperators` | SLM 优化变异，支持 conservative / moderate / aggressive 三级强度 |
| `SLMCrossoverOperators` | 算术交叉和均匀交叉 |
| `SLMSelectionOperators` | 锦标赛选择、排序选择、精英保留 |

SLM 参数约束：

```python
SLM_CONSTRAINTS = {
    "transformer": {
        "num_layers": (2, 12),
        "hidden_size": (128, 768),
        "num_heads": (2, 12),
        "ffn_dim": (256, 3072),
        "dropout": (0.0, 0.3),
    },
    "cnn": {
        "num_blocks": (2, 8),
        "base_channels": (16, 128),
        "kernel_size": (1, 7),
    },
}
```

### `evolution_engine.py` — 进化引擎

高层封装，整合所有组件：

| 类 | 说明 |
|---|---|
| `EvolutionConfig` | 进化配置（种群、变异率、缓存、代理模型等） |
| `EvolutionEngine` | 进化引擎主类，自动管理缓存和代理模型集成 |

```python
engine = EvolutionEngine(config)
result = engine.evolve(
    fitness_function=my_evaluator,
    callback=lambda stats: print(stats)  # 每代回调
)
# result 包含: best_fitness, best_architecture, generations, history, total_time, cache_stats
```

### `surrogate_model.py` — 代理模型

使用 ML 模型预测架构性能，避免昂贵的真实训练：

| 方法 | 说明 |
|---|---|
| `add_training_point(arch, fitness)` | 添加训练数据 |
| `fit()` | 训练代理模型 |
| `predict(arch)` | 预测架构性能 |
| `store_prediction(arch, pred)` | 缓存预测结果 |

支持的模型类型：`"rf"` (随机森林)、`"gb"` (梯度提升)、`"mlp"` (神经网络)、`"ensemble"` (集成)

### `cache_system.py` — 缓存系统

SQLite 持久化缓存，避免重复评估：

| 方法 | 说明 |
|---|---|
| `store(arch, metrics)` | 存储评估结果 |
| `lookup(arch)` | 查找缓存结果 |
| `get_top_performing(metric, limit)` | 获取最佳架构 |
| `get_statistics()` | 缓存统计（命中率等） |
| `export_to_json(path)` / `import_from_json(path)` | 导入导出 |

```python
with ArchitectureCache("cache.db") as cache:
    cache.store({"type": "transformer", "num_layers": 6}, {"accuracy": 0.85})
    result = cache.lookup({"type": "transformer", "num_layers": 6})
    # result: {"accuracy": 0.85}
```

### `slm_optimized_mutation.py` — SLM 高级变异

高级变异策略，包含语义分析和资源约束：

| 类 | 说明 |
|---|---|
| `SLMOptimizedMutation` | 高级变异器，支持语义感知和资源约束 |
| `ResourceEstimator` | 估算参数量和显存需求 |
| `SemanticAnalyzer` | 分析架构语义平衡性 |
| `create_slm_mutation_operator()` | 工厂函数，快速创建变异器 |

```python
from genetic_ml_evolution import SLMOptimizedMutation, ResourceEstimator

mutator = SLMOptimizedMutation(
    max_params=100_000_000,  # 100M 参数上限
    max_memory_gb=20.0,      # 20GB 显存上限
    enable_semantic_analysis=True,
)

mutated, desc = mutator.mutate(architecture, fitness=50.0, strategy="adaptive")

# 资源估算
params = ResourceEstimator.estimate_transformer_params(architecture)
memory = ResourceEstimator.estimate_memory_gb(architecture)
```

## ⚙️ 配置参数

### EvolutionConfig 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `population_size` | int | 20 | 种群大小 |
| `elite_size` | int | 3 | 精英保留数量 |
| `generations` | int | 50 | 进化代数 |
| `mutation_rate` | float | 0.2 | 变异率 |
| `crossover_rate` | float | 0.7 | 交叉率 |
| `mutation_strength` | str | "moderate" | 变异强度：conservative / moderate / aggressive |
| `early_stopping_patience` | int | 5 | 早停耐心值 |
| `min_improvement_threshold` | float | 1.0 | 最小改进阈值（%） |
| `max_gpu_memory_gb` | float | 16.0 | GPU 显存上限 |
| `task_type` | str | "language" | 任务类型：language / image / multimodal |
| `dataset` | str | "imdb" | 数据集名称 |
| `use_cache` | bool | True | 是否启用缓存 |
| `cache_db_path` | str | None | 缓存数据库路径 |
| `use_surrogate` | bool | True | 是否启用代理模型 |
| `surrogate_model_type` | str | "ensemble" | 代理模型类型 |

### GeneticAlgorithm 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `population_size` | int | 20 | 种群大小 |
| `mutation_rate` | float | 0.1 | 基础变异率 |
| `crossover_rate` | float | 0.3 | 交叉率 |
| `elitism_rate` | float | 0.1 | 精英保留比例 |
| `tournament_size` | int | 3 | 锦标赛大小 |
| `surrogate_model` | SurrogateModel | None | 代理模型 |
| `cache_db_path` | str | None | 缓存路径 |
| `task_type` | str | "language" | 任务类型 |

## 🔬 高级用法

### 自定义适应度函数

```python
def my_fitness_function(architecture):
    """真实适应度函数：构建模型 → 训练 → 返回验证精度"""
    from transformers import AutoModelForSequenceClassification
    import torch

    # 1. 根据架构配置构建模型
    model = build_model_from_architecture(architecture)

    # 2. 训练并评估
    accuracy = train_and_evaluate(model, train_data, val_data)

    # 3. 多目标评分（精度 - 参数量惩罚）
    params = ResourceEstimator.estimate_transformer_params(architecture)
    score = accuracy - (params / 1e8) * 5  # 每 100M 参数扣 5 分
    return score
```

### 自定义变异策略

```python
from genetic_ml_evolution.genetic_operators import SLMutationOperators

mutator = SLMutationOperators(
    mutation_rate=0.15,
    mutation_strength="conservative",
    respect_constraints=True
)

# 只变异特定参数
mutated = mutator.mutate_transformer(gene, focused_params=["hidden_size", "num_heads"])
```

### 语义分析辅助架构设计

```python
from genetic_ml_evolution import SemanticAnalyzer

analysis = SemanticAnalyzer.analyze_transformer_semantics(architecture)
print(f"平衡分数: {analysis['balance_score']}/100")
print(f"FFN 比例: {analysis['ffn_ratio']:.2f}x")
print(f"问题: {analysis['issues']}")
print(f"建议: {analysis['recommendations']}")
```

### 缓存导入导出

```python
# 导出缓存
cache.export_to_json("cache_backup.json")

# 在新实验中导入
new_cache = ArchitectureCache("new_cache.db")
new_cache.import_from_json("cache_backup.json")
```

### 回调监控进化过程

```python
def generation_callback(stats):
    print(f"Gen {stats['generation']}: best={stats['best_fitness']:.2f}, "
          f"cache_hits={stats['cache_hits']}")

result = engine.evolve(
    fitness_function=fitness,
    callback=generation_callback
)
```

## 📂 项目结构

```
genetic-ml-evolution/
├── genetic_ml_evolution/
│   ├── __init__.py                    # 公共 API 导出
│   ├── genetic_algorithm.py           # 遗传算法核心 (GeneticAlgorithm, Individual)
│   ├── genetic_operators.py           # 遗传算子 (变异、交叉、选择)
│   ├── evolution_engine.py            # 进化引擎 (EvolutionEngine, EvolutionConfig)
│   ├── surrogate_model.py             # 代理模型 (RF, GBDT, MLP 集成)
│   ├── cache_system.py                # 缓存系统 (SQLite)
│   └── slm_optimized_mutation.py      # SLM 高级变异 (语义分析、资源约束)
├── examples/
│   ├── genetic_algorithm_example.py   # 遗传算法完整示例
│   ├── cache_example.py               # 缓存系统示例
│   └── slm_optimized_mutation_example.py  # SLM 变异示例
├── tests/                             # 测试文件
├── requirements.txt
└── README.md
```

## 🧪 运行示例

```bash
# 基础遗传算法示例
python examples/genetic_algorithm_example.py

# 缓存系统示例
python examples/cache_example.py

# SLM 优化变异示例
python examples/slm_optimized_mutation_example.py

# 运行测试
pytest tests/ -v
```

## 📚 论文参考

- **AutoML-Zero** — ICML 2020，自动发现 ML 算法
- **NeuroLGP-SM** — arXiv 2024，神经架构 + 线性图规划
- **EG-NAS** — AAAI 2024，进化图神经网络架构搜索
- **Similarity Surrogate NAS** — InfoSci 2024，代理模型加速 NAS

## 📄 License

MIT License

---

**版本**: 0.2.0 | **维护者**: Newton Tech | **状态**: 🚧 开发中
