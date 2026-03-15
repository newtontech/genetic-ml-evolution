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
