"""
SLM-Optimized Mutation Operators
针对小规模语言模型优化的变异操作

This module implements advanced mutation strategies optimized for Small Language Models (SLM)
with a focus on efficiency, semantic awareness, and resource constraints.

Key Features:
1. Structured Mutation - Maintain architectural coherence
2. Semantic-Aware Mutation - Understand parameter relationships
3. Resource-Aware Mutation - Respect computational budget
4. Performance-Adaptive Mutation - Adjust strategy based on fitness
5. History-Guided Mutation - Learn from past successes/failures
6. Progressive Scheduling - Dynamic mutation across generations
7. PEFT-Aware Mutation - LoRA/QLoRA-friendly architecture mutations
8. GQA Support - Grouped Query Attention in mutation space
9. Quantization-Aware - Consider 4bit/8bit quantization constraints
10. Multi-Objective Scoring - Balance accuracy, params, and latency
"""

import random
import copy
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class MutationStrategy(Enum):
    """变异策略枚举"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    PROGRESSIVE_EARLY = "progressive_early"
    PROGRESSIVE_MID = "progressive_mid"
    PROGRESSIVE_LATE = "progressive_late"
    PEFT_FOCUSED = "peft_focused"
    QUANT_AWARE = "quant_aware"


class QuantizationMode(Enum):
    """量化模式"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"


class AttentionType(Enum):
    """注意力类型"""
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"


@dataclass
class MutationRecord:
    """记录单次变异的历史"""
    parent_arch: Dict[str, Any]
    child_arch: Dict[str, Any]
    parent_fitness: Optional[float]
    child_fitness: Optional[float]
    improvement: float
    mutation_type: str
    generation: int


@dataclass
class MutationStatistics:
    """变异统计信息"""
    total_mutations: int = 0
    successful_mutations: int = 0  # improvement > 0
    failed_mutations: int = 0  # improvement <= 0
    mutation_history: List[MutationRecord] = field(default_factory=list)
    
    # 按变异类型统计
    by_type: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"success": 0, "failure": 0}))
    
    # 按代统计（用于 progressive scheduling）
    by_generation: Dict[int, Dict[str, Any]] = field(default_factory=lambda: defaultdict(lambda: {"count": 0, "avg_improvement": 0.0}))
    
    def record_mutation(self, record: MutationRecord):
        """记录一次变异"""
        self.total_mutations += 1
        self.mutation_history.append(record)
        
        if record.improvement > 0:
            self.successful_mutations += 1
            self.by_type[record.mutation_type]["success"] += 1
        else:
            self.failed_mutations += 1
            self.by_type[record.mutation_type]["failure"] += 1
        
        # 更新代统计
        gen_stats = self.by_generation[record.generation]
        gen_stats["count"] += 1
        n = gen_stats["count"]
        gen_stats["avg_improvement"] = (
            (gen_stats["avg_improvement"] * (n - 1) + record.improvement) / n
        )
    
    def get_success_rate(self, mutation_type: Optional[str] = None) -> float:
        """获取变异成功率"""
        if mutation_type:
            stats = self.by_type[mutation_type]
            total = stats["success"] + stats["failure"]
            return stats["success"] / total if total > 0 else 0.0
        else:
            return self.successful_mutations / self.total_mutations if self.total_mutations > 0 else 0.0
    
    def get_recent_trend(self, window: int = 5) -> float:
        """获取最近 window 代的平均改进趋势"""
        if not self.mutation_history:
            return 0.0
        recent = [r for r in self.mutation_history[-window * 10:]]
        if not recent:
            return 0.0
        return sum(r.improvement for r in recent) / len(recent)


class ResourceEstimator:
    """资源估算器 - 支持 GQA 和量化感知估算"""
    
    @staticmethod
    def estimate_transformer_params(arch: Dict[str, Any]) -> int:
        """估算 Transformer 参数量（支持 GQA 和 SwiGLU）"""
        vocab_size = arch.get("vocab_size", 50000)
        num_layers = arch.get("num_layers", 6)
        hidden_size = arch.get("hidden_size", 512)
        num_heads = arch.get("num_heads", 8)
        ffn_dim = arch.get("ffn_dim", 2048)
        num_kv_heads = arch.get("num_kv_heads", num_heads)  # GQA support
        
        # Embedding
        embedding_params = vocab_size * hidden_size
        
        # Attention with GQA awareness
        head_dim = hidden_size // num_heads
        attention_params_per_layer = (
            hidden_size * hidden_size +                      # Q
            hidden_size * head_dim * num_kv_heads +          # K
            hidden_size * head_dim * num_kv_heads +          # V
            hidden_size * hidden_size                        # O
        )
        
        # FFN with SwiGLU support
        activation = arch.get("activation", "gelu")
        if activation in ("swiglu", "silu"):
            ffn_params_per_layer = 3 * hidden_size * ffn_dim
        else:
            ffn_params_per_layer = 2 * hidden_size * ffn_dim
        
        layernorm_params_per_layer = 2 * hidden_size
        layer_params = (attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer) * num_layers
        
        return embedding_params + layer_params
    
    @staticmethod
    def estimate_memory_gb(arch: Dict[str, Any], batch_size: int = 32) -> float:
        """估算显存需求（GB）- 支持量化和 PEFT"""
        num_layers = arch.get("num_layers", 6)
        hidden_size = arch.get("hidden_size", 512)
        max_seq_len = arch.get("max_seq_len", 512)
        
        total_params = ResourceEstimator.estimate_transformer_params(arch)
        
        # Quantization reduces parameter memory
        quant_mode = arch.get("quantization", "none")
        bytes_per_param = {"int4": 0.5, "int8": 1.0}.get(quant_mode, 4.0)
        
        # PEFT mode reduces trainable parameters
        peft_mode = arch.get("peft_mode", "none")
        if peft_mode in ("lora", "qlora"):
            lora_rank = arch.get("lora_rank", 8 if peft_mode == "lora" else 16)
            trainable_ratio = min((2 * hidden_size * lora_rank * num_layers) / total_params, 0.1)
        else:
            trainable_ratio = 1.0
        
        param_memory = total_params * bytes_per_param / (1024**3)
        trainable_memory = total_params * trainable_ratio * 4 / (1024**3)
        activation_memory = batch_size * max_seq_len * hidden_size * num_layers * 4 / (1024**3)
        gradient_memory = trainable_memory
        optimizer_memory = trainable_memory * 2
        
        return param_memory + trainable_memory + activation_memory + gradient_memory + optimizer_memory
    
    @staticmethod
    def estimate_latency_ms(arch: Dict[str, Any], seq_len: int = 128) -> float:
        """估算推理延迟（ms）- 用于多目标优化"""
        num_layers = arch.get("num_layers", 6)
        hidden_size = arch.get("hidden_size", 512)
        num_heads = arch.get("num_heads", 8)
        ffn_dim = arch.get("ffn_dim", 2048)
        num_kv_heads = arch.get("num_kv_heads", num_heads)
        
        head_dim = hidden_size // num_heads
        attn_flops = 2 * seq_len * (hidden_size * head_dim + hidden_size * head_dim * num_kv_heads + hidden_size * hidden_size)
        ffn_flops = 2 * seq_len * 2 * hidden_size * ffn_dim
        total_flops_per_token = num_layers * (attn_flops + ffn_flops)
        latency_per_token = total_flops_per_token / 10e12 * 1000  # ms @ 10 TFLOPS
        
        return max(1.0, min(latency_per_token * seq_len, 10000.0))


class SemanticAnalyzer:
    """语义分析器 - 理解参数间的语义关系"""
    
    @staticmethod
    def analyze_transformer_semantics(arch: Dict[str, Any]) -> Dict[str, Any]:
        """分析 Transformer 架构的语义特征"""
        num_layers = arch.get("num_layers", 6)
        hidden_size = arch.get("hidden_size", 512)
        num_heads = arch.get("num_heads", 8)
        ffn_dim = arch.get("ffn_dim", 2048)
        dropout = arch.get("dropout", 0.1)
        
        analysis = {
            "depth": "shallow" if num_layers <= 4 else "medium" if num_layers <= 8 else "deep",
            "width": "narrow" if hidden_size <= 256 else "medium" if hidden_size <= 512 else "wide",
            "ffn_ratio": ffn_dim / hidden_size if hidden_size > 0 else 0,
            "head_size": hidden_size // num_heads if num_heads > 0 else 0,
            "regularization": "light" if dropout < 0.1 else "moderate" if dropout < 0.2 else "heavy",
            "balance_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # 检查平衡性
        balance_score = 100.0
        
        # FFN 比例检查
        if not (2.0 <= analysis["ffn_ratio"] <= 4.0):
            balance_score -= 20
            analysis["issues"].append(f"FFN ratio ({analysis['ffn_ratio']:.1f}x) outside optimal range [2-4]")
            analysis["recommendations"].append(f"Adjust FFN dim to {int(hidden_size * 3)} (3x hidden)")
        
        # Head size 检查
        if analysis["head_size"] < 32:
            balance_score -= 15
            analysis["issues"].append(f"Head size ({analysis['head_size']}) too small")
            analysis["recommendations"].append("Reduce num_heads or increase hidden_size")
        elif analysis["head_size"] > 128:
            balance_score -= 10
            analysis["issues"].append(f"Head size ({analysis['head_size']}) too large")
            analysis["recommendations"].append("Increase num_heads or reduce hidden_size")
        
        # 深度-宽度平衡
        if analysis["depth"] == "shallow" and analysis["width"] == "narrow":
            balance_score -= 25
            analysis["issues"].append("Both shallow and narrow - may lack capacity")
            analysis["recommendations"].append("Increase either layers or hidden_size")
        
        if analysis["depth"] == "deep" and analysis["width"] == "wide":
            balance_score -= 20
            analysis["issues"].append("Both deep and wide - may overfit or exceed budget")
            analysis["recommendations"].append("Reduce either layers or hidden_size")
        
        # Dropout 与深度关系
        if analysis["depth"] == "deep" and analysis["regularization"] == "light":
            balance_score -= 15
            analysis["issues"].append("Deep network with light regularization")
            analysis["recommendations"].append("Increase dropout to 0.15-0.2")
        
        analysis["balance_score"] = max(0, balance_score)
        
        return analysis
    
    @staticmethod
    def suggest_improvements(arch: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于语义分析建议改进"""
        suggestions = []
        
        for rec in analysis.get("recommendations", []):
            if "FFN" in rec:
                # FFN 调整建议
                hidden = arch.get("hidden_size", 512)
                suggestions.append({
                    "type": "adjust_ffn",
                    "description": rec,
                    "target_value": int(hidden * 3),
                    "priority": "high"
                })
            
            elif "head" in rec.lower():
                # Head 调整建议
                hidden = arch.get("hidden_size", 512)
                current_heads = arch.get("num_heads", 8)
                head_size = hidden // current_heads
                
                if head_size < 32:
                    # 减少头数
                    new_heads = max(2, hidden // 64)  # target head_size = 64
                    suggestions.append({
                        "type": "reduce_heads",
                        "description": rec,
                        "target_value": new_heads,
                        "priority": "medium"
                    })
                elif head_size > 128:
                    # 增加头数
                    new_heads = min(12, hidden // 64)
                    suggestions.append({
                        "type": "increase_heads",
                        "description": rec,
                        "target_value": new_heads,
                        "priority": "medium"
                    })
            
            elif "dropout" in rec.lower():
                # Dropout 调整建议
                suggestions.append({
                    "type": "adjust_dropout",
                    "description": rec,
                    "target_value": 0.15,
                    "priority": "low"
                })
            
            elif "layers" in rec.lower() or "depth" in rec.lower():
                # 层数调整建议
                current_layers = arch.get("num_layers", 6)
                if analysis["depth"] == "shallow":
                    suggestions.append({
                        "type": "increase_layers",
                        "description": rec,
                        "target_value": current_layers + 2,
                        "priority": "high"
                    })
                else:
                    suggestions.append({
                        "type": "decrease_layers",
                        "description": rec,
                        "target_value": max(2, current_layers - 2),
                        "priority": "high"
                    })
        
        return suggestions


class SLMOptimizedMutation:
    """
    针对 SLM 优化的变异操作
    
    集成了多种优化策略:
    1. 结构化变异 - 保持架构一致性
    2. 语义感知变异 - 理解参数关系
    3. 资源感知变异 - 考虑计算预算
    4. 性能自适应变异 - 根据适应度调整策略
    5. 历史引导变异 - 学习成功模式
    6. 渐进式调度 - 跨代动态调整变异强度
    7. PEFT 感知 - LoRA/QLoRA 友好的架构变异
    8. GQA 支持 - 在变异空间中探索分组查询注意力
    9. 量化感知 - 考虑 4bit/8bit 量化约束
    10. 多目标评分 - 平衡精度、参数量和延迟
    """
    
    def __init__(
        self,
        max_params: int = 100_000_000,  # 100M 参数上限
        max_memory_gb: float = 20.0,  # 20GB 显存上限
        max_latency_ms: float = 500.0,  # 500ms 推理延迟上限
        enable_semantic_analysis: bool = True,
        enable_history_learning: bool = True,
        enable_gqa: bool = True,  # 启用 GQA 变异
        enable_peft_mutation: bool = True,  # 启用 PEFT 模式变异
        enable_quant_aware: bool = True,  # 启用量化感知
        enable_progressive: bool = True,  # 启用渐进式调度
        total_generations: int = 50,  # 总代数（用于渐进式调度）
        verbose: bool = False
    ):
        self.max_params = max_params
        self.max_memory_gb = max_memory_gb
        self.max_latency_ms = max_latency_ms
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_history_learning = enable_history_learning
        self.enable_gqa = enable_gqa
        self.enable_peft_mutation = enable_peft_mutation
        self.enable_quant_aware = enable_quant_aware
        self.enable_progressive = enable_progressive
        self.total_generations = total_generations
        self.verbose = verbose
        
        # 工具类
        self.resource_estimator = ResourceEstimator()
        self.semantic_analyzer = SemanticAnalyzer()
        
        # 统计和历史
        self.statistics = MutationStatistics()
        self.generation = 0
    
    def mutate(
        self,
        architecture: Dict[str, Any],
        fitness: Optional[float] = None,
        strategy: str = "adaptive"
    ) -> Tuple[Dict[str, Any], str]:
        """
        执行优化的变异
        
        Args:
            architecture: 原始架构
            fitness: 当前适应度（用于自适应策略）
            strategy: 变异策略 (adaptive/conservative/moderate/aggressive)
            
        Returns:
            (变异后的架构, 变异描述)
        """
        arch_type = architecture.get("type", "transformer")
        
        if arch_type == "transformer":
            return self.mutate_transformer(architecture, fitness, strategy)
        elif arch_type == "cnn":
            return self.mutate_cnn(architecture, fitness, strategy)
        elif arch_type == "multimodal":
            return self.mutate_multimodal(architecture, fitness, strategy)
        else:
            logger.warning(f"Unknown architecture type: {arch_type}")
            return copy.deepcopy(architecture), "no_change"
    
    def mutate_transformer(
        self,
        architecture: Dict[str, Any],
        fitness: Optional[float] = None,
        strategy: str = "adaptive"
    ) -> Tuple[Dict[str, Any], str]:
        """
        Transformer 架构的优化变异
        """
        mutated = copy.deepcopy(architecture)
        mutations = []
        
        # 1. 语义分析
        semantic_analysis = None
        if self.enable_semantic_analysis:
            semantic_analysis = self.semantic_analyzer.analyze_transformer_semantics(mutated)
        
        # 2. 选择变异策略
        if strategy == "adaptive" and fitness is not None:
            strategy = self._select_adaptive_strategy(fitness)
        
        # 3. 生成候选变异
        candidates = self._generate_mutation_candidates(
            mutated, 
            strategy, 
            semantic_analysis
        )
        
        # 4. 过滤无效变异（资源检查）
        valid_candidates = [
            (arch, desc) for arch, desc in candidates
            if self._is_within_budget(arch)
        ]
        
        if not valid_candidates:
            # 回退到保守变异
            return self._conservative_mutate_transformer(mutated)
        
        # 5. 选择最佳变异
        best_mutation = self._select_best_mutation(valid_candidates, semantic_analysis)
        
        if best_mutation:
            mutated, desc = best_mutation
            return mutated, desc
        else:
            return mutated, "no_change"
    
    def _select_adaptive_strategy(self, fitness: float) -> str:
        """根据适应度和进化阶段选择变异策略"""
        # 先检查渐进式调度
        progressive = self._get_progressive_strategy()
        
        # 结合适应度和阶段
        if fitness >= 80:
            # 高性能：以渐进式为主，偏向保守
            return "conservative" if progressive != "aggressive" else "moderate"
        elif fitness >= 50:
            # 中等：跟随渐进式
            return progressive
        else:
            # 低性能：适度激进
            return progressive if progressive == "aggressive" else "moderate"
    
    def _generate_mutation_candidates(
        self,
        arch: Dict[str, Any],
        strategy: str,
        semantic_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], str]]:
        """生成多个变异候选"""
        candidates = []
        
        # 基础变异
        base_candidate = copy.deepcopy(arch)
        mutations = []
        
        # 根据策略确定变异强度
        if strategy == "conservative":
            mutation_rate = 0.1
            layer_delta_range = [-1, 0, 1]
            hidden_delta_range = [-64, 0, 64]
        elif strategy == "moderate":
            mutation_rate = 0.2
            layer_delta_range = [-2, -1, 0, 1, 2]
            hidden_delta_range = [-128, -64, 0, 64, 128]
        else:  # aggressive
            mutation_rate = 0.3
            layer_delta_range = [-3, -2, -1, 0, 1, 2, 3]
            hidden_delta_range = [-256, -128, -64, 0, 64, 128, 256]
        
        # 1. 层数变异
        if random.random() < mutation_rate:
            old_layers = arch.get("num_layers", 6)
            delta = random.choice(layer_delta_range)
            new_layers = max(2, min(12, old_layers + delta))
            
            if new_layers != old_layers:
                base_candidate["num_layers"] = new_layers
                mutations.append(f"layers:{old_layers}→{new_layers}")
        
        # 2. 隐藏维度变异
        if random.random() < mutation_rate:
            old_hidden = arch.get("hidden_size", 512)
            delta = random.choice(hidden_delta_range)
            new_hidden = max(128, min(768, old_hidden + delta))
            
            # 确保可被 num_heads 整除
            num_heads = base_candidate.get("num_heads", 8)
            new_hidden = (new_hidden // num_heads) * num_heads
            
            if new_hidden != old_hidden:
                base_candidate["hidden_size"] = new_hidden
                mutations.append(f"hidden:{old_hidden}→{new_hidden}")
                
                # 3. 同步调整 FFN dim（保持比例）
                old_ffn = arch.get("ffn_dim", 2048)
                ffn_ratio = old_ffn / old_hidden if old_hidden > 0 else 3.0
                
                # 优化 FFN 比例（倾向于 3x）
                if semantic_analysis and "ffn_ratio" in semantic_analysis:
                    target_ratio = 3.0  # 标准比例
                    ffn_ratio = target_ratio
                
                new_ffn = int(new_hidden * ffn_ratio)
                new_ffn = max(256, min(3072, new_ffn))
                base_candidate["ffn_dim"] = new_ffn
                mutations.append(f"ffn:{old_ffn}→{new_ffn}")
        
        # 4. 注意力头变异
        if random.random() < mutation_rate:
            hidden_size = base_candidate.get("hidden_size", 512)
            old_heads = arch.get("num_heads", 8)
            
            # 找到所有有效的头数
            valid_heads = [h for h in [2, 4, 6, 8, 12] if hidden_size % h == 0]
            
            if valid_heads:
                # 倾向于选择 head_size 在 32-96 之间的
                preferred_heads = [
                    h for h in valid_heads 
                    if 32 <= hidden_size // h <= 96
                ]
                
                if preferred_heads:
                    new_heads = random.choice(preferred_heads)
                else:
                    new_heads = random.choice(valid_heads)
                
                if new_heads != old_heads:
                    base_candidate["num_heads"] = new_heads
                    mutations.append(f"heads:{old_heads}→{new_heads}")
        
        # 5. Dropout 变异
        if random.random() < mutation_rate:
            old_dropout = arch.get("dropout", 0.1)
            
            # 根据网络深度调整 dropout
            num_layers = base_candidate.get("num_layers", 6)
            if num_layers >= 8:
                # 深层网络需要更强的正则化
                dropout_range = [0.15, 0.2, 0.25]
            elif num_layers <= 4:
                # 浅层网络
                dropout_range = [0.05, 0.1, 0.15]
            else:
                dropout_range = [0.1, 0.15, 0.2]
            
            new_dropout = random.choice(dropout_range)
            
            if new_dropout != old_dropout:
                base_candidate["dropout"] = new_dropout
                mutations.append(f"dropout:{old_dropout:.2f}→{new_dropout:.2f}")
        
        # 6. 基于语义分析的优化变异
        if semantic_analysis and semantic_analysis.get("issues"):
            suggestions = self.semantic_analyzer.suggest_improvements(arch, semantic_analysis)
            for suggestion in suggestions[:1]:
                if suggestion["priority"] == "high" and random.random() < 0.5:
                    if suggestion["type"] == "adjust_ffn":
                        old_ffn = base_candidate.get("ffn_dim", 2048)
                        base_candidate["ffn_dim"] = suggestion["target_value"]
                        mutations.append(f"ffn_optimized:{old_ffn}→{suggestion['target_value']}")
        
        # 7. GQA 变异（分组查询注意力）
        mutations.extend(self._mutate_gqa(base_candidate))
        
        # 8. PEFT 配置变异
        mutations.extend(self._mutate_peft_config(base_candidate))
        
        # 9. 量化配置变异
        mutations.extend(self._mutate_quantization(base_candidate))
        
        if mutations:
            candidates.append((base_candidate, "; ".join(mutations)))
        
        # 生成额外的结构化变异候选
        if strategy in ["moderate", "aggressive"]:
            # 块变异：同时调整相关参数
            block_candidate = self._block_mutation(arch)
            if block_candidate:
                candidates.append(block_candidate)
        
        return candidates
    
    def _block_mutation(self, arch: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        块变异：同时调整相关的参数组
        
        例如：
        - 增加层数 + 减小 hidden_size（保持参数量）
        - 增加深度 + 增加 dropout（防止过拟合）
        """
        block_type = random.choice(["scale_depth", "scale_width", "balance"])
        mutated = copy.deepcopy(arch)
        mutations = []
        
        if block_type == "scale_depth":
            # 调整深度相关参数
            old_layers = mutated.get("num_layers", 6)
            delta = random.choice([-2, 2])
            new_layers = max(2, min(12, old_layers + delta))
            
            if new_layers != old_layers:
                mutated["num_layers"] = new_layers
                mutations.append(f"layers:{old_layers}→{new_layers}")
                
                # 同步调整 dropout
                if new_layers > old_layers:
                    # 增加深度，增加正则化
                    old_dropout = mutated.get("dropout", 0.1)
                    new_dropout = min(0.3, old_dropout + 0.05)
                    mutated["dropout"] = new_dropout
                    mutations.append(f"dropout:{old_dropout:.2f}→{new_dropout:.2f}")
        
        elif block_type == "scale_width":
            # 调整宽度相关参数
            old_hidden = mutated.get("hidden_size", 512)
            scale_factor = random.choice([0.75, 1.25])
            new_hidden = int(max(128, min(768, old_hidden * scale_factor)))
            
            # 确保可被 num_heads 整除
            num_heads = mutated.get("num_heads", 8)
            new_hidden = (new_hidden // num_heads) * num_heads
            
            if new_hidden != old_hidden:
                mutated["hidden_size"] = new_hidden
                mutations.append(f"hidden:{old_hidden}→{new_hidden}")
                
                # 同步调整 FFN
                old_ffn = mutated.get("ffn_dim", 2048)
                new_ffn = int(new_hidden * 3)  # 标准 3x 比例
                mutated["ffn_dim"] = new_ffn
                mutations.append(f"ffn:{old_ffn}→{new_ffn}")
        
        elif block_type == "balance":
            # 平衡深度和宽度
            current_params = self.resource_estimator.estimate_transformer_params(mutated)
            
            if current_params > self.max_params * 0.8:
                # 接近上限，减小规模
                if mutated.get("num_layers", 6) > 4:
                    mutated["num_layers"] -= 1
                    mutations.append("balance:reduce_layers")
                elif mutated.get("hidden_size", 512) > 256:
                    mutated["hidden_size"] = max(128, mutated["hidden_size"] - 64)
                    mutations.append("balance:reduce_hidden")
        
        if mutations:
            return mutated, "block:" + "; ".join(mutations)
        else:
            return None
    
    def _get_progressive_strategy(self) -> str:
        """根据进化阶段返回渐进式策略"""
        if not self.enable_progressive or self.total_generations <= 0:
            return "moderate"
        progress = self.generation / self.total_generations
        if progress < 0.2:
            return "aggressive"
        elif progress < 0.5:
            return "moderate"
        else:
            return "conservative"
    
    def _mutate_gqa(self, arch: Dict[str, Any]) -> List[str]:
        """
        GQA (Grouped Query Attention) 变异
        
        在 num_heads 和 num_kv_heads 之间建立有意义的变异关系。
        GQA 用较少的 KV heads 减少 KV cache 内存，同时保持模型质量。
        """
        mutations = []
        if not self.enable_gqa:
            return mutations
        
        num_heads = arch.get("num_heads", 8)
        num_kv_heads = arch.get("num_kv_heads", num_heads)
        hidden_size = arch.get("hidden_size", 512)
        
        valid_kv_heads = [h for h in range(1, num_heads + 1) if num_heads % h == 0]
        if not valid_kv_heads:
            return mutations
        
        if random.random() < 0.3:
            preferred_kv = [h for h in valid_kv_heads if 2 <= num_heads // h <= 8]
            new_kv = random.choice(preferred_kv) if preferred_kv else random.choice(valid_kv_heads)
            
            if new_kv != num_kv_heads:
                arch["num_kv_heads"] = new_kv
                if new_kv == 1:
                    arch["attention_type"] = "mqa"
                elif new_kv < num_heads:
                    arch["attention_type"] = "gqa"
                else:
                    arch["attention_type"] = "mha"
                mutations.append(f"kv_heads:{num_kv_heads}→{new_kv}({arch['attention_type']})")
        
        return mutations
    
    def _mutate_peft_config(self, arch: Dict[str, Any]) -> List[str]:
        """
        PEFT (Parameter-Efficient Fine-Tuning) 配置变异
        
        探索 LoRA/QLoRA 的 rank、alpha 和 target_modules。
        """
        mutations = []
        if not self.enable_peft_mutation:
            return mutations
        
        if random.random() < 0.2:
            current_mode = arch.get("peft_mode", "none")
            modes = ["none", "lora", "qlora"]
            other_modes = [m for m in modes if m != current_mode]
            new_mode = random.choice(other_modes)
            arch["peft_mode"] = new_mode
            mutations.append(f"peft_mode:{current_mode}→{new_mode}")
            
            if new_mode == "lora" and "lora_rank" not in arch:
                arch["lora_rank"] = 8
                arch["lora_alpha"] = 16
            elif new_mode == "qlora" and "lora_rank" not in arch:
                arch["lora_rank"] = 16
                arch["lora_alpha"] = 32
                arch["quantization"] = "int4"
        
        if arch.get("peft_mode") in ("lora", "qlora") and random.random() < 0.3:
            valid_ranks = [4, 8, 16, 32, 64]
            old_rank = arch.get("lora_rank", 8)
            new_rank = random.choice(valid_ranks)
            if new_rank != old_rank:
                arch["lora_rank"] = new_rank
                arch["lora_alpha"] = random.choice([new_rank, int(new_rank * 1.5), new_rank * 2])
                mutations.append(f"lora_rank:{old_rank}→{new_rank}(alpha={arch['lora_alpha']})")
        
        return mutations
    
    def _mutate_quantization(self, arch: Dict[str, Any]) -> List[str]:
        """量化配置变异"""
        mutations = []
        if not self.enable_quant_aware:
            return mutations
        
        if random.random() < 0.15:
            current_quant = arch.get("quantization", "none")
            quant_options = ["none", "int8", "int4"]
            other_options = [q for q in quant_options if q != current_quant]
            new_quant = random.choice(other_options)
            
            if arch.get("peft_mode") == "qlora":
                new_quant = "int4"
            
            if new_quant != current_quant:
                arch["quantization"] = new_quant
                mutations.append(f"quantization:{current_quant}→{new_quant}")
        
        return mutations
    
    def _multi_objective_score(self, arch: Dict[str, Any], fitness: Optional[float] = None) -> float:
        """多目标评分：平衡精度、参数量和延迟"""
        params = self.resource_estimator.estimate_transformer_params(arch)
        memory = self.resource_estimator.estimate_memory_gb(arch)
        latency = self.resource_estimator.estimate_latency_ms(arch)
        
        param_score = max(0.0, 1.0 - params / self.max_params)
        memory_score = max(0.0, 1.0 - memory / self.max_memory_gb)
        latency_score = max(0.0, 1.0 - latency / self.max_latency_ms)
        efficiency = 0.4 * param_score + 0.3 * memory_score + 0.3 * latency_score
        
        if fitness is not None:
            accuracy_score = min(fitness / 100.0, 1.0)
            return 0.6 * accuracy_score + 0.4 * efficiency
        
        return efficiency
    
    def _conservative_mutate_transformer(self, arch: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """保守变异：最小化改动（支持 GQA/PEFT/量化）"""
        mutated = copy.deepcopy(arch)
        mutations = []
        
        # Dropout 微调
        if random.random() < 0.3:
            old_dropout = mutated.get("dropout", 0.1)
            delta = random.uniform(-0.02, 0.02)
            new_dropout = max(0.05, min(0.25, old_dropout + delta))
            mutated["dropout"] = round(new_dropout, 2)
            if new_dropout != old_dropout:
                mutations.append(f"dropout:{old_dropout:.2f}→{new_dropout:.2f}")
        
        # GQA 微调
        mutations.extend(self._mutate_gqa(mutated))
        
        # 量化微调
        mutations.extend(self._mutate_quantization(mutated))
        
        if mutations:
            return mutated, "conservative:" + "; ".join(mutations)
        else:
            return mutated, "no_change"
    
    def _is_within_budget(self, arch: Dict[str, Any]) -> bool:
        """检查架构是否在资源预算内（参数量、显存、延迟）"""
        param_count = self.resource_estimator.estimate_transformer_params(arch)
        if param_count > self.max_params:
            return False
        
        memory_gb = self.resource_estimator.estimate_memory_gb(arch)
        if memory_gb > self.max_memory_gb:
            return False
        
        latency_ms = self.resource_estimator.estimate_latency_ms(arch)
        if latency_ms > self.max_latency_ms:
            return False
        
        return True
    
    def _select_best_mutation(
        self,
        candidates: List[Tuple[Dict[str, Any], str]],
        semantic_analysis: Optional[Dict[str, Any]] = None,
        fitness: Optional[float] = None
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """选择最佳的变异候选（使用多目标评分）"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        scored_candidates = []
        for arch, desc in candidates:
            # 使用多目标评分
            score = self._multi_objective_score(arch, fitness) * 100
            
            # 语义平衡加分
            if semantic_analysis:
                analysis = self.semantic_analyzer.analyze_transformer_semantics(arch)
                balance_score = analysis.get("balance_score", 50) / 100.0
                score += balance_score * 10
            
            # 变异幅度评分（适度变异更好）
            mutation_count = desc.count("→")
            if mutation_count == 2:
                score += 3
            elif mutation_count <= 4:
                score += 1
            elif mutation_count > 6:
                score -= 5
            
            scored_candidates.append((score, arch, desc))
        
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return (scored_candidates[0][1], scored_candidates[0][2])
    
    def mutate_cnn(
        self,
        architecture: Dict[str, Any],
        fitness: Optional[float] = None,
        strategy: str = "adaptive"
    ) -> Tuple[Dict[str, Any], str]:
        """CNN 架构的优化变异（简化版）"""
        mutated = copy.deepcopy(architecture)
        mutations = []
        
        # 简化的 CNN 变异逻辑
        mutation_rate = 0.2
        
        if random.random() < mutation_rate:
            old_blocks = mutated.get("num_blocks", 4)
            delta = random.choice([-1, 0, 1])
            new_blocks = max(2, min(8, old_blocks + delta))
            
            if new_blocks != old_blocks:
                mutated["num_blocks"] = new_blocks
                mutations.append(f"blocks:{old_blocks}→{new_blocks}")
        
        if mutations:
            return mutated, "; ".join(mutations)
        else:
            return mutated, "no_change"
    
    def mutate_multimodal(
        self,
        architecture: Dict[str, Any],
        fitness: Optional[float] = None,
        strategy: str = "adaptive"
    ) -> Tuple[Dict[str, Any], str]:
        """多模态架构的优化变异（简化版）"""
        mutated = copy.deepcopy(architecture)
        mutations = []
        
        # 变异视觉编码器
        if "vision_encoder" in mutated and random.random() < 0.3:
            vision_mutated, vision_desc = self.mutate_cnn(
                mutated["vision_encoder"],
                fitness,
                "conservative"
            )
            if vision_desc != "no_change":
                mutated["vision_encoder"] = vision_mutated
                mutations.append(f"vision:{vision_desc}")
        
        # 变异文本编码器
        if "text_encoder" in mutated and random.random() < 0.3:
            text_mutated, text_desc = self.mutate_transformer(
                mutated["text_encoder"],
                fitness,
                "conservative"
            )
            if text_desc != "no_change":
                mutated["text_encoder"] = text_mutated
                mutations.append(f"text:{text_desc}")
        
        if mutations:
            return mutated, "; ".join(mutations)
        else:
            return mutated, "no_change"
    
    def record_result(
        self,
        parent_arch: Dict[str, Any],
        child_arch: Dict[str, Any],
        parent_fitness: Optional[float],
        child_fitness: Optional[float],
        mutation_type: str
    ):
        """记录变异结果用于学习"""
        if not self.enable_history_learning:
            return
        
        improvement = 0.0
        if parent_fitness is not None and child_fitness is not None:
            improvement = child_fitness - parent_fitness
        
        record = MutationRecord(
            parent_arch=parent_arch,
            child_arch=child_arch,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            improvement=improvement,
            mutation_type=mutation_type,
            generation=self.generation
        )
        
        self.statistics.record_mutation(record)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取变异统计信息"""
        return {
            "total_mutations": self.statistics.total_mutations,
            "successful_mutations": self.statistics.successful_mutations,
            "failed_mutations": self.statistics.failed_mutations,
            "success_rate": self.statistics.get_success_rate(),
            "by_type": dict(self.statistics.by_type)
        }
    
    def advance_generation(self):
        """推进到下一代"""
        self.generation += 1


# 便捷函数
def create_slm_mutation_operator(
    max_params: int = 100_000_000,
    max_memory_gb: float = 20.0,
    **kwargs
) -> SLMOptimizedMutation:
    """创建 SLM 优化的变异操作器"""
    return SLMOptimizedMutation(
        max_params=max_params,
        max_memory_gb=max_memory_gb,
        **kwargs
    )
