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
"""

import random
import copy
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


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
    
    def get_success_rate(self, mutation_type: Optional[str] = None) -> float:
        """获取变异成功率"""
        if mutation_type:
            stats = self.by_type[mutation_type]
            total = stats["success"] + stats["failure"]
            return stats["success"] / total if total > 0 else 0.0
        else:
            return self.successful_mutations / self.total_mutations if self.total_mutations > 0 else 0.0


class ResourceEstimator:
    """资源估算器"""
    
    @staticmethod
    def estimate_transformer_params(arch: Dict[str, Any]) -> int:
        """估算 Transformer 参数量，支持 GQA 和 RoPE 等现代 SLM 技术"""
        vocab_size = arch.get("vocab_size", 50000)
        num_layers = arch.get("num_layers", 6)
        hidden_size = arch.get("hidden_size", 512)
        num_heads = arch.get("num_heads", 8)
        ffn_dim = arch.get("ffn_dim", 2048)
        max_seq_len = arch.get("max_seq_len", 512)
        num_kv_heads = arch.get("num_kv_heads", num_heads)  # GQA support
        
        # Embedding: vocab_size * hidden_size
        embedding_params = vocab_size * hidden_size
        
        # Self-attention with GQA support:
        # - Q projection: hidden_size * hidden_size
        # - K projection: hidden_size * (hidden_size / num_heads * num_kv_heads)
        # - V projection: same as K
        # - O projection: hidden_size * hidden_size
        head_dim = hidden_size // num_heads
        kv_dim = head_dim * num_kv_heads
        attention_params_per_layer = (hidden_size * hidden_size  # Q
                                      + 2 * hidden_size * kv_dim  # K, V
                                      + hidden_size * hidden_size)  # O
        
        # FFN: support SwiGLU (3 projections) vs standard (2 projections)
        ffn_type = arch.get("ffn_type", "standard")
        if ffn_type == "swiglu":
            # SwiGLU: gate_proj + up_proj + down_proj = 3 * hidden_size * ffn_dim
            ffn_params_per_layer = 3 * hidden_size * ffn_dim
        else:
            ffn_params_per_layer = 2 * hidden_size * ffn_dim
        
        layernorm_params_per_layer = 2 * hidden_size
        
        layer_params = (attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer) * num_layers
        
        # Output layer (usually tied with embeddings)
        output_params = 0
        
        total_params = embedding_params + layer_params + output_params
        
        return total_params
    
    @staticmethod
    def estimate_memory_gb(arch: Dict[str, Any], batch_size: int = 32, quant_bits: int = 16) -> float:
        """估算显存需求（GB），支持训练/推理和量化"""
        num_layers = arch.get("num_layers", 6)
        hidden_size = arch.get("hidden_size", 512)
        max_seq_len = arch.get("max_seq_len", 512)
        num_kv_heads = arch.get("num_kv_heads", arch.get("num_heads", 8))
        num_heads = arch.get("num_heads", 8)
        
        bytes_per_param = quant_bits // 8
        total_params = ResourceEstimator.estimate_transformer_params(arch)
        
        # Parameters
        param_memory = total_params * bytes_per_param / (1024**3)
        
        # Activations: more accurate with GQA (KV cache smaller)
        head_dim = hidden_size // num_heads
        # Q/V activations: batch * seq * hidden * layers * 2 (input + output)
        # K/V activations: batch * seq * head_dim * num_kv_heads * layers * 2
        activation_memory = (
            batch_size * max_seq_len * hidden_size * num_layers * 2  # Q, V
            + batch_size * max_seq_len * head_dim * num_kv_heads * num_layers * 2  # K, V (GQA)
        ) * 4 / (1024**3)
        
        # KV Cache for inference (not needed during training but useful for SLM)
        kv_cache_per_layer = 2 * num_kv_heads * head_dim * max_seq_len * 2  # K + V, fp16
        kv_cache_total = kv_cache_per_layer * num_layers * batch_size / (1024**3)
        
        # For training: gradients + optimizer
        gradient_memory = param_memory
        optimizer_memory = param_memory * 2  # Adam states
        
        # Training mode: params + activations + gradients + optimizer
        # Inference mode: params + KV cache
        total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory
        
        return total_memory
    
    @staticmethod
    def estimate_inference_memory_gb(arch: Dict[str, Any], batch_size: int = 1, seq_len: int = 512, quant_bits: int = 4) -> float:
        """估算推理显存需求（GB），考虑量化"""
        total_params = ResourceEstimator.estimate_transformer_params(arch)
        bytes_per_param = quant_bits // 8
        param_memory = total_params * bytes_per_param / (1024**3)
        
        num_layers = arch.get("num_layers", 6)
        num_heads = arch.get("num_heads", 8)
        num_kv_heads = arch.get("num_kv_heads", num_heads)
        head_dim = arch.get("hidden_size", 512) // num_heads
        
        # KV Cache: batch * seq * num_layers * num_kv_heads * head_dim * 2 (K+V) * 2 bytes (fp16)
        kv_cache = batch_size * seq_len * num_layers * num_kv_heads * head_dim * 2 * 2 / (1024**3)
        
        return param_memory + kv_cache


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
    """
    
    def __init__(
        self,
        max_params: int = 100_000_000,  # 100M 参数上限
        max_memory_gb: float = 20.0,  # 20GB 显存上限
        enable_semantic_analysis: bool = True,
        enable_history_learning: bool = True,
        verbose: bool = False,
        total_generations: int = 50,  # 总代数（用于自适应衰减）
    ):
        self.max_params = max_params
        self.max_memory_gb = max_memory_gb
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_history_learning = enable_history_learning
        self.verbose = verbose
        self.total_generations = total_generations
        
        # 工具类
        self.resource_estimator = ResourceEstimator()
        self.semantic_analyzer = SemanticAnalyzer()
        
        # 统计和历史
        self.statistics = MutationStatistics()
        self.generation = 0
        
        # SLM 专用的维度对齐策略（量化友好）
        self._alignment_multiples = [64, 32]  # 优先对齐到 64 的倍数
    
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
        """根据适应度和进化阶段选择变异策略（代数衰减）"""
        # 计算进化进度 (0.0 ~ 1.0)
        progress = min(1.0, self.generation / max(1, self.total_generations))
        
        # 基于适应度的基础策略
        if fitness >= 80:
            base_strategy = "conservative"
        elif fitness >= 50:
            base_strategy = "moderate"
        else:
            base_strategy = "aggressive"
        
        # 进化后期自动趋向保守（精炼阶段）
        if progress > 0.7:
            # 后 30% 代数：降低探索，增加利用
            if base_strategy == "aggressive":
                return "moderate"
            elif base_strategy == "moderate":
                return "conservative"
        elif progress < 0.2:
            # 前 20% 代数：鼓励探索
            if base_strategy == "conservative":
                return "moderate"
        
        return base_strategy
    
    @staticmethod
    def _align_dimension(value: int, multiples: List[int] = None) -> int:
        """将维度对齐到量化友好的倍数，优先最小倍数"""
        if multiples is None:
            multiples = [64, 32]
        for m in multiples:
            if value % m == 0:
                return value
        # 对齐到第一个倍数
        return (value // multiples[0] + 1) * multiples[0]
    
    def _get_best_mutation_type(self) -> Optional[str]:
        """基于历史数据选择成功率最高的变异类型"""
        if not self.enable_history_learning:
            return None
        
        best_type = None
        best_rate = 0.0
        min_samples = 3  # 至少 3 次样本才信任统计
        
        for mutation_type, stats in self.statistics.by_type.items():
            total = stats["success"] + stats["failure"]
            if total >= min_samples:
                rate = stats["success"] / total
                if rate > best_rate:
                    best_rate = rate
                    best_type = mutation_type
        
        return best_type
    
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
        
        # 2. 隐藏维度变异（量化友好对齐）
        if random.random() < mutation_rate:
            old_hidden = arch.get("hidden_size", 512)
            delta = random.choice(hidden_delta_range)
            new_hidden = max(128, min(768, old_hidden + delta))
            
            # 确保可被 num_heads 整除
            num_heads = base_candidate.get("num_heads", 8)
            new_hidden = (new_hidden // num_heads) * num_heads
            
            # 量化友好对齐
            new_hidden = self._align_dimension(new_hidden, self._alignment_multiples)
            
            if new_hidden != old_hidden:
                base_candidate["hidden_size"] = new_hidden
                mutations.append(f"hidden:{old_hidden}→{new_hidden}")
                
                # 3. 同步调整 FFN dim（保持比例，量化友好）
                old_ffn = arch.get("ffn_dim", 2048)
                ffn_ratio = old_ffn / old_hidden if old_hidden > 0 else 3.0
                
                # 优化 FFN 比例（倾向于 3x）
                if semantic_analysis and "ffn_ratio" in semantic_analysis:
                    target_ratio = 3.0  # 标准比例
                    ffn_ratio = target_ratio
                
                new_ffn = int(new_hidden * ffn_ratio)
                new_ffn = max(256, min(3072, new_ffn))
                # 量化友好对齐
                new_ffn = self._align_dimension(new_ffn, [256, 128])
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
        
        # 6. GQA (Grouped Query Attention) 变异 - SLM 关键优化
        if random.random() < mutation_rate:
            old_kv_heads = arch.get("num_kv_heads", arch.get("num_heads", 8))
            num_heads = base_candidate.get("num_heads", 8)
            hidden_size = base_candidate.get("hidden_size", 512)
            head_dim = hidden_size // num_heads
            
            # GQA: num_kv_heads 可以是 num_heads 的 1/N (N=1,2,4,8)
            # 常见配置: MHA (kv_heads=heads), GQA-4 (kv_heads=heads/4), MQA (kv_heads=1)
            gqa_ratios = [1, 2, 4, 8]  # num_heads / num_kv_heads
            valid_kv_heads = []
            for ratio in gqa_ratios:
                kv_h = num_heads // ratio
                if kv_h >= 1 and num_heads % ratio == 0:
                    valid_kv_heads.append(kv_h)
            
            # 倾向于更少的 KV heads（更高效的 SLM）
            if len(valid_kv_heads) > 1 and random.random() < 0.6:
                # 60% 概率减少 KV heads
                current_idx = valid_kv_heads.index(old_kv_heads) if old_kv_heads in valid_kv_heads else 0
                new_idx = min(current_idx + 1, len(valid_kv_heads) - 1)
                new_kv_heads = valid_kv_heads[new_idx]
            elif valid_kv_heads:
                new_kv_heads = random.choice(valid_kv_heads)
            else:
                new_kv_heads = old_kv_heads
            
            if new_kv_heads != old_kv_heads and new_kv_heads != num_heads:
                base_candidate["num_kv_heads"] = new_kv_heads
                gqa_type = "MQA" if new_kv_heads == 1 else f"GQA-{num_heads // new_kv_heads}"
                mutations.append(f"kv_heads:{old_kv_heads}→{new_kv_heads}({gqa_type})")
            elif new_kv_heads == num_heads and old_kv_heads != num_heads:
                # 回退到 MHA
                base_candidate["num_kv_heads"] = num_heads
                mutations.append(f"kv_heads:{old_kv_heads}→{num_heads}(MHA)")
        
        # 7. Activation function 变异
        if random.random() < mutation_rate * 0.5:  # 较低频率
            old_activation = arch.get("activation", "gelu")
            activations = ["gelu", "silu", "relu", "gelu_new"]
            # SLM 偏好 SiLU (Swish) 和 GELU
            slm_preferred = ["silu", "gelu"]
            if random.random() < 0.7:
                new_activation = random.choice(slm_preferred)
            else:
                new_activation = random.choice([a for a in activations if a != old_activation])
            
            if new_activation != old_activation:
                base_candidate["activation"] = new_activation
                mutations.append(f"activation:{old_activation}→{new_activation}")
        
        # 8. FFN type 变异 (standard vs SwiGLU)
        if random.random() < mutation_rate * 0.3:
            old_ffn_type = arch.get("ffn_type", "standard")
            ffn_types = ["standard", "swiglu"]
            # SLM 偏好 SwiGLU (LLaMA-style)
            new_ffn_type = "swiglu" if old_ffn_type == "standard" else "standard"
            if new_ffn_type != old_ffn_type:
                base_candidate["ffn_type"] = new_ffn_type
                # SwiGLU 使用 8/3 ≈ 2.67x 而非 4x
                if new_ffn_type == "swiglu":
                    hidden = base_candidate.get("hidden_size", 512)
                    new_ffn = self._align_dimension(int(hidden * 8 / 3), [256, 128])
                    base_candidate["ffn_dim"] = new_ffn
                mutations.append(f"ffn_type:{old_ffn_type}→{new_ffn_type}")
        
        # 9. 基于语义分析的优化变异
        if semantic_analysis and semantic_analysis.get("issues"):
            # 应用建议的改进
            suggestions = self.semantic_analyzer.suggest_improvements(arch, semantic_analysis)
            
            for suggestion in suggestions[:1]:  # 只应用最高优先级的建议
                if suggestion["priority"] == "high" and random.random() < 0.5:
                    if suggestion["type"] == "adjust_ffn":
                        old_ffn = base_candidate.get("ffn_dim", 2048)
                        base_candidate["ffn_dim"] = suggestion["target_value"]
                        mutations.append(f"ffn_optimized:{old_ffn}→{suggestion['target_value']}")
        
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
    
    def _conservative_mutate_transformer(self, arch: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """保守变异：最小化改动"""
        mutated = copy.deepcopy(arch)
        mutations = []
        
        # 只做非常小的调整
        if random.random() < 0.3:
            old_dropout = mutated.get("dropout", 0.1)
            delta = random.uniform(-0.02, 0.02)
            new_dropout = max(0.05, min(0.25, old_dropout + delta))
            mutated["dropout"] = round(new_dropout, 2)
            mutations.append(f"dropout:{old_dropout:.2f}→{new_dropout:.2f}")
        
        if mutations:
            return mutated, "conservative:" + "; ".join(mutations)
        else:
            return mutated, "no_change"
    
    def _is_within_budget(self, arch: Dict[str, Any]) -> bool:
        """检查架构是否在资源预算内"""
        # 检查参数量
        param_count = self.resource_estimator.estimate_transformer_params(arch)
        if param_count > self.max_params:
            if self.verbose:
                logger.debug(f"Architecture exceeds param budget: {param_count / 1e6:.2f}M > {self.max_params / 1e6:.2f}M")
            return False
        
        # 检查显存
        memory_gb = self.resource_estimator.estimate_memory_gb(arch)
        if memory_gb > self.max_memory_gb:
            if self.verbose:
                logger.debug(f"Architecture exceeds memory budget: {memory_gb:.2f}GB > {self.max_memory_gb:.2f}GB")
            return False
        
        return True
    
    def _select_best_mutation(
        self,
        candidates: List[Tuple[Dict[str, Any], str]],
        semantic_analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """选择最佳的变异候选（结合历史成功率和 GQA 效率）"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        best_history_type = self._get_best_mutation_type()
        
        scored_candidates = []
        for arch, desc in candidates:
            score = 0.0
            
            # 1. 资源效率评分
            params = self.resource_estimator.estimate_transformer_params(arch)
            resource_score = 1.0 - (params / self.max_params)
            score += resource_score * 20
            
            # 2. 语义平衡评分
            if semantic_analysis:
                analysis = self.semantic_analyzer.analyze_transformer_semantics(arch)
                balance_score = analysis.get("balance_score", 50) / 100.0
                score += balance_score * 30
            
            # 3. 变异幅度评分
            mutation_count = desc.count("→")
            if mutation_count == 1:
                score += 10
            elif mutation_count == 2:
                score += 15
            elif mutation_count <= 4:
                score += 5
            else:
                score -= 10
            
            # 4. 历史引导评分
            if best_history_type and best_history_type in desc:
                score += 15
            
            # 5. GQA/MQA 效率加分
            num_heads = arch.get("num_heads", 8)
            num_kv_heads = arch.get("num_kv_heads", num_heads)
            if num_kv_heads < num_heads:
                gqa_ratio = num_heads / num_kv_heads
                score += min(10, gqa_ratio * 3)
            
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
