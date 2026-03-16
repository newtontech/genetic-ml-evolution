# SQLite 缓存系统使用指南

## 概述

SQLite 缓存系统用于存储和检索架构评估结果，避免重复评估相同的架构，从而节省时间和计算资源。

## 核心特性

- **SQLite 数据库存储**: 持久化存储架构评估结果
- **内存缓存加速**: 热点数据缓存在内存中，提升查询速度
- **线程安全**: 支持多线程并发访问
- **丰富的评估指标**: 支持存储适应度、准确率、损失、训练时间、显存使用等多种指标
- **统计功能**: 提供缓存命中率、条目数等统计信息
- **导入导出**: 支持 JSON 格式的导入导出

## 快速开始

### 基本使用

```python
from genetic_ml_evolution import CacheSystem

# 创建缓存系统实例
cache = CacheSystem(cache_dir=".cache/genetic_ml")

# 定义架构配置
architecture = {
    "type": "transformer",
    "num_layers": 6,
    "hidden_size": 512,
    "num_heads": 8,
    "ffn_dim": 2048,
    "dropout": 0.1
}

# 存储评估结果
cache.put(
    architecture, 
    fitness=85.5,
    accuracy=88.2,
    loss=0.123,
    training_time=120.5,
    memory_usage=2048.0
)

# 查询评估结果
fitness = cache.get(architecture)
print(f"适应度: {fitness}")  # 输出: 适应度: 85.5

# 获取详细评估结果
detailed_result = cache.get_detailed(architecture)
print(f"详细结果: {detailed_result}")
# 输出: {
#   'fitness': 85.5,
#   'accuracy': 88.2,
#   'loss': 0.123,
#   'training_time': 120.5,
#   'memory_usage': 2048.0
# }
```

### 获取表现最好的架构

```python
# 获取表现最好的前 10 个架构
top_architectures = cache.get_top_performers(limit=10)

for i, arch_data in enumerate(top_architectures, 1):
    print(f"#{i} 适应度: {arch_data['fitness']:.2f}")
    print(f"   架构: {arch_data['architecture']}")
```

### 缓存统计

```python
# 获取缓存统计信息
stats = cache.get_stats()
print(f"命中次数: {stats['hits']}")
print(f"未命中次数: {stats['misses']}")
print(f"命中率: {stats['hit_rate']:.2%}")
print(f"内存缓存条目: {stats['memory_entries']}")
print(f"数据库条目: {stats['db_entries']}")
print(f"数据库大小: {stats['db_size_mb']:.2f} MB")
```

### 导入导出

```python
# 导出到 JSON 文件
cache.export_to_json("cache_backup.json")

# 从 JSON 文件导入
cache.import_from_json("cache_backup.json", overwrite=True)
```

### 缓存管理

```python
# 清空缓存
cache.clear()

# 清理数据库碎片（释放空间）
cache.vacuum()

# 获取缓存条目数
print(f"缓存条目数: {len(cache)}")

# 打印缓存信息
print(cache)  # 输出: CacheSystem(db_entries=100, memory_entries=100, hit_rate=75.00%)
```

## 集成到进化流程

```python
from genetic_ml_evolution import EvolutionEngine, CacheSystem, ModelEvaluator

# 创建缓存系统
cache = CacheSystem(cache_dir=".cache/genetic_ml")

# 创建模型评估器
evaluator = ModelEvaluator(gpu_memory=16, dataset="imdb")

# 创建进化引擎
engine = EvolutionEngine(
    gpu_memory=16,
    task_type="language",
    dataset="imdb",
    population_size=20,
    generations=50
)

# 在评估个体时使用缓存
def evaluate_with_cache(individual):
    # 先检查缓存
    cached_fitness = cache.get(individual.architecture)
    
    if cached_fitness is not None:
        print(f"缓存命中！适应度: {cached_fitness}")
        return cached_fitness
    
    # 缓存未命中，执行实际评估
    print("缓存未命中，执行评估...")
    fitness = evaluator.evaluate(individual.architecture)
    
    # 存储到缓存
    cache.put(individual.architecture, fitness)
    
    return fitness

# 使用缓存的评估函数
engine._evaluate_individual = evaluate_with_cache

# 运行进化
best_model = engine.evolve()
print(f"最佳模型适应度: {best_model.fitness}")
```

## 性能优化建议

### 1. 内存缓存大小

内存缓存会存储所有访问过的架构。对于大规模搜索空间，建议定期清理：

```python
# 清理内存缓存（保留数据库）
cache.memory_cache.clear()
```

### 2. 批量操作

对于大量数据的导入导出，使用批量操作：

```python
# 批量存储
architectures = [...]  # 大量架构列表
for arch, fitness in architectures:
    cache.put(arch, fitness)

# 导出所有数据
cache.export_to_json("all_architectures.json")
```

### 3. 定期维护

```python
# 定期清理数据库碎片
if len(cache) > 10000:
    cache.vacuum()
```

## 数据库表结构

```sql
CREATE TABLE architecture_cache (
    cache_key TEXT PRIMARY KEY,          -- 缓存键（架构哈希）
    architecture TEXT NOT NULL,          -- 架构配置（JSON）
    fitness REAL NOT NULL,               -- 适应度
    accuracy REAL,                       -- 准确率
    loss REAL,                          -- 损失值
    training_time REAL,                  -- 训练时间（秒）
    memory_usage REAL,                   -- 显存使用（MB）
    created_at TEXT NOT NULL,            -- 创建时间
    updated_at TEXT NOT NULL             -- 更新时间
);

-- 索引
CREATE INDEX idx_fitness ON architecture_cache(fitness DESC);
CREATE INDEX idx_created_at ON architecture_cache(created_at DESC);
```

## 线程安全性

缓存系统使用线程锁确保并发安全：

```python
from concurrent.futures import ThreadPoolExecutor

def evaluate_architecture(index):
    arch = {"type": "test", "index": index}
    
    # 检查缓存
    cached = cache.get(arch)
    if cached is not None:
        return cached
    
    # 模拟评估
    fitness = some_evaluation_function(arch)
    
    # 存储到缓存
    cache.put(arch, fitness)
    return fitness

# 多线程并发评估
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(evaluate_architecture, range(100)))
```

## 常见问题

### Q: 如何备份缓存数据？

A: 使用导出功能：
```python
cache.export_to_json("backup.json")
```

### Q: 如何在不同机器之间共享缓存？

A: 复制 SQLite 数据库文件或导出为 JSON：
```python
# 方法1: 复制 .cache/genetic_ml/architecture_cache.db 文件

# 方法2: 导出为 JSON
cache.export_to_json("shared_cache.json")

# 在另一台机器上导入
cache.import_from_json("shared_cache.json")
```

### Q: 如何查看缓存效率？

A: 查看统计信息：
```python
stats = cache.get_stats()
print(f"命中率: {stats['hit_rate']:.2%}")
```

### Q: 数据库文件太大怎么办？

A: 清理并压缩数据库：
```python
cache.clear()  # 清空数据
cache.vacuum()  # 压缩数据库文件
```

## API 参考

### `CacheSystem(cache_dir=".cache/genetic_ml")`

创建缓存系统实例。

**参数:**
- `cache_dir`: 缓存目录路径

### `put(architecture, fitness, accuracy=None, loss=None, training_time=None, memory_usage=None)`

存储架构评估结果。

**参数:**
- `architecture`: 架构配置字典（必需）
- `fitness`: 适应度分数（必需）
- `accuracy`: 准确率（可选）
- `loss`: 损失值（可选）
- `training_time`: 训练时间（可选）
- `memory_usage`: 显存使用（可选）

### `get(architecture) -> Optional[float]`

获取架构的适应度。

**返回:**
- 适应度分数，若缓存未命中返回 `None`

### `get_detailed(architecture) -> Optional[Dict]`

获取详细的评估结果。

**返回:**
- 包含所有指标的字典，若缓存未命中返回 `None`

### `get_top_performers(limit=10) -> List[Dict]`

获取表现最好的 N 个架构。

### `get_stats() -> Dict`

获取缓存统计信息。

### `clear()`

清空缓存。

### `vacuum()`

清理数据库碎片。

### `export_to_json(output_path)`

导出缓存到 JSON 文件。

### `import_from_json(input_path, overwrite=False) -> int`

从 JSON 文件导入缓存。

**返回:**
- 导入的条目数量

## 更新日志

### v0.1.0 (2026-03-17)
- 实现基于 SQLite 的缓存系统
- 支持内存缓存加速
- 支持多种评估指标存储
- 支持导入导出功能
- 完整的单元测试覆盖
