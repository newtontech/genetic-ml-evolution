# 缓存系统使用指南

## 概述

缓存系统是 Genetic ML Evolution 框架的核心组件之一，用于存储和检索已评估的架构性能结果，避免重复评估，加速进化过程。

## 核心特性

### 1. SQLite 后端存储
- **持久化存储**：数据存储在 SQLite 数据库中，程序重启后仍然可用
- **高效查询**：使用索引加速架构哈希查找
- **事务支持**：保证数据一致性

### 2. 架构哈希机制
- **确定性哈希**：使用 SHA256 对架构配置进行哈希
- **排序保证**：JSON 键排序确保相同架构产生相同哈希
- **快速比对**：通过哈希快速判断架构是否已评估

### 3. 线程安全
- **并发访问**：使用 `threading.RLock` 确保线程安全
- **读写锁**：支持多线程并发读写
- **可选启用**：可配置是否启用线程安全机制

### 4. 缓存过期机制
- **TTL 支持**：可配置缓存条目的生存时间（Time-To-Live）
- **自动过期**：查询时自动检查并删除过期条目
- **手动清理**：提供清理过期条目的方法
- **灵活配置**：支持全局 TTL 和单个条目的自定义 TTL

### 5. 性能统计
- **命中率统计**：跟踪缓存命中和未命中次数
- **访问追踪**：记录每个条目的访问次数和最后访问时间
- **性能分析**：支持按类型、性能等维度查询

## 快速开始

### 基础用法

```python
from genetic_ml_evolution import ArchitectureCache

# 创建缓存实例
cache = ArchitectureCache("my_cache.db")

# 定义架构
architecture = {
    "type": "transformer",
    "num_layers": 6,
    "hidden_size": 512,
    "num_heads": 8,
    "ffn_dim": 2048,
    "dropout": 0.1
}

# 存储性能指标
metrics = {
    "accuracy": 0.85,
    "loss": 0.15,
    "inference_time": 0.025
}
cache.store(architecture, metrics, evaluation_time=10.5)

# 查询缓存
result = cache.lookup(architecture)
if result:
    print(f"缓存命中！准确率: {result['accuracy']}")
else:
    print("缓存未命中，需要评估")
```

### 使用 TTL（缓存过期）

```python
# 创建带 1 小时 TTL 的缓存
cache = ArchitectureCache(
    db_path="cache.db",
    ttl_seconds=3600  # 1 小时后过期
)

# 存储时使用默认 TTL
cache.store(architecture, metrics)

# 存储时使用自定义 TTL（30 分钟）
cache.store(architecture2, metrics2, ttl_seconds=1800)

# 手动清理过期条目
cleaned = cache.cleanup_expired()
print(f"清理了 {cleaned} 个过期条目")
```

### 线程安全使用

```python
import threading

# 创建线程安全的缓存（默认启用）
cache = ArchitectureCache(
    db_path="cache.db",
    enable_thread_safety=True
)

# 多线程访问
def worker(i):
    arch = {"type": "transformer", "num_layers": i}
    metrics = {"accuracy": 0.8 + i * 0.01}
    cache.store(arch, metrics)
    result = cache.lookup(arch)
    print(f"线程 {i}: {result}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## 高级功能

### 按类型查询

```python
# 获取所有 Transformer 架构
transformers = cache.get_by_type("transformer", limit=10)
for arch, metrics in transformers:
    print(f"架构: {arch}, 准确率: {metrics['accuracy']}")

# 获取所有 CNN 架构
cnns = cache.get_by_type("cnn")
```

### 获取最佳性能架构

```python
# 获取准确率最高的 10 个架构
top_archs = cache.get_top_performing(
    metric="accuracy",
    limit=10
)

for i, (arch, metrics) in enumerate(top_archs, 1):
    print(f"#{i}: 准确率 {metrics['accuracy']:.4f}")

# 只获取 Transformer 类型中性能最好的
top_transformers = cache.get_top_performing(
    metric="accuracy",
    limit=5,
    arch_type="transformer"
)
```

### 统计信息

```python
stats = cache.get_statistics()

print(f"总条目数: {stats['total_entries']}")
print(f"按类型统计: {stats['entries_by_type']}")
print(f"缓存命中: {stats['cache_hits']}")
print(f"缓存未命中: {stats['cache_misses']}")
print(f"命中率: {stats['hit_rate_percent']}%")
print(f"平均评估时间: {stats['average_evaluation_time']} 秒")
```

### 导入导出

```python
# 导出为 JSON
cache.export_to_json("cache_backup.json")

# 导入 JSON（覆盖现有数据）
cache.import_from_json("cache_backup.json", overwrite=True)

# 导入 JSON（保留现有数据）
imported = cache.import_from_json("cache_backup.json", overwrite=False)
print(f"导入了 {imported} 个条目")
```

## 与代理模型集成

```python
from genetic_ml_evolution import SurrogateModel

# 创建带缓存的代理模型
model = SurrogateModel(
    model_type="rf",  # 随机森林
    cache_db_path="surrogate_cache.db"
)

# 添加训练数据
for i in range(20):
    arch = {"type": "transformer", "num_layers": i + 2}
    performance = 70.0 + i * 1.5
    model.add_training_point(arch, performance)

# 训练模型
model.fit()

# 预测（会使用缓存）
test_arch = {"type": "transformer", "num_layers": 6}
prediction = model.predict(test_arch)

# 存储预测结果
model.store_prediction(test_arch, prediction)

# 再次预测相同架构（会从缓存获取）
prediction2 = model.predict(test_arch)
```

## 最佳实践

### 1. 数据库位置

```python
# 推荐：使用项目目录下的专门文件夹
import os
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)
cache = ArchitectureCache(os.path.join(cache_dir, "architectures.db"))

# 避免：使用临时目录（重启后丢失）
# cache = ArchitectureCache("/tmp/cache.db")
```

### 2. TTL 配置

```python
# 短期实验：较短 TTL
cache = ArchitectureCache(db_path="cache.db", ttl_seconds=3600)  # 1 小时

# 长期项目：较长 TTL
cache = ArchitectureCache(db_path="cache.db", ttl_seconds=86400 * 7)  # 1 周

# 生产环境：无 TTL（永久存储）
cache = ArchitectureCache(db_path="cache.db", ttl_seconds=None)
```

### 3. 性能优化

```python
# 使用上下文管理器确保正确关闭
with ArchitectureCache("cache.db") as cache:
    # 批量存储
    for arch, metrics in data:
        cache.store(arch, metrics)
    
    # 批量查询
    results = [cache.lookup(arch) for arch in architectures]

# 定期清理过期条目
cache.cleanup_expired()

# 定期导出备份
cache.export_to_json(f"backup_{datetime.now():%Y%m%d}.json")
```

### 4. 错误处理

```python
try:
    # 存储架构
    success = cache.store(architecture, metrics)
    if not success:
        print("架构已存在，跳过存储")
    
    # 查询架构
    result = cache.lookup(architecture)
    if result is None:
        # 缓存未命中，进行评估
        metrics = evaluate_architecture(architecture)
        cache.store(architecture, metrics)
    
except ValueError as e:
    print(f"参数错误: {e}")
except sqlite3.Error as e:
    print(f"数据库错误: {e}")
```

## API 参考

### ArchitectureCache

#### 构造函数

```python
ArchitectureCache(
    db_path: str = "architecture_cache.db",
    ttl_seconds: Optional[int] = None,
    enable_thread_safety: bool = True
)
```

**参数：**
- `db_path`: SQLite 数据库文件路径
- `ttl_seconds`: 缓存条目的生存时间（秒），None 表示不过期
- `enable_thread_safety`: 是否启用线程安全机制

#### 主要方法

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `store(architecture, metrics, evaluation_time, ttl_seconds)` | 存储架构和性能指标 | `bool` |
| `lookup(architecture)` | 查询架构的性能指标 | `Optional[Dict[str, float]]` |
| `exists(architecture)` | 检查架构是否存在 | `bool` |
| `delete(architecture)` | 删除指定架构 | `bool` |
| `clear()` | 清空所有缓存 | `int` (删除的条目数) |
| `cleanup_expired()` | 清理过期条目 | `int` (清理的条目数) |
| `get_statistics()` | 获取统计信息 | `Dict[str, Any]` |
| `get_by_type(arch_type, limit)` | 按类型查询架构 | `List[Tuple[Dict, Dict]]` |
| `get_top_performing(metric, limit, arch_type)` | 获取最佳性能架构 | `List[Tuple[Dict, Dict]]` |
| `export_to_json(output_path)` | 导出为 JSON | `None` |
| `import_from_json(input_path, overwrite)` | 从 JSON 导入 | `int` (导入的条目数) |

## 常见问题

### Q: 缓存会占用多少磁盘空间？

A: 每个架构条目大约占用 1-2KB（取决于架构复杂度和指标数量）。10,000 个架构约占用 10-20MB。

### Q: 如何迁移缓存数据？

A: 使用导出/导入功能：

```python
# 导出
old_cache.export_to_json("migration.json")

# 导入
new_cache.import_from_json("migration.json", overwrite=True)
```

### Q: 多进程可以共享同一个缓存吗？

A: SQLite 支持多进程并发读取，但写入可能会有锁竞争。建议：
- 单进程多线程：完全支持
- 多进程：建议每个进程使用独立的缓存，定期合并

### Q: 如何备份缓存？

A: 两种方法：

```python
# 方法 1：导出为 JSON（可读性强）
cache.export_to_json("backup.json")

# 方法 2：直接复制数据库文件（更快）
import shutil
shutil.copy("cache.db", "cache_backup.db")
```

## 示例项目

查看完整示例：
- `examples/cache_example.py` - 基础用法示例
- `examples/surrogate_model_example.py` - 代理模型集成示例
- `tests/test_cache_system.py` - 完整的单元测试示例

## 更新日志

### v0.2.0 (2026-03-28)
- ✅ 添加线程安全机制
- ✅ 添加缓存过期（TTL）支持
- ✅ 添加 `cleanup_expired()` 方法
- ✅ 优化并发性能
- ✅ 完善单元测试（新增 10+ 测试用例）

### v0.1.0 (2026-03-21)
- ✅ 基础缓存功能
- ✅ SQLite 后端存储
- ✅ 架构哈希机制
- ✅ 性能统计
- ✅ 导入导出功能

---

**维护者**: OpenClaw AI Agent  
**最后更新**: 2026-03-28  
**文档版本**: 0.2.0
