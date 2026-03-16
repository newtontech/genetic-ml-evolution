"""
Evolution Engine - 进化控制器
管理遗传算法的核心进化流程
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    LANGUAGE = "language"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


@dataclass
class Individual:
    """进化个体"""
    architecture: Dict[str, Any]
    fitness: Optional[float] = None
    generation: int = 0
    evaluated: bool = False


@dataclass
class EvolutionConfig:
    """进化配置"""
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: int = 2
    tournament_size: int = 3


class EvolutionEngine:
    """进化引擎 - 管理遗传算法的完整生命周期"""
    
    def __init__(
        self,
        gpu_memory: int = 16,
        task_type: str = "language",
        dataset: str = "imdb",
        population_size: int = 20,
        generations: int = 50
    ):
        self.gpu_memory = gpu_memory
        self.task_type = TaskType(task_type)
        self.dataset = dataset
        self.config = EvolutionConfig(
            population_size=population_size,
            generations=generations
        )
        self.population: List[Individual] = []
        self.history: List[Dict] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        
    def initialize_population(self) -> None:
        """初始化种群 - 生成随机架构"""
        self.population = []
        for i in range(self.config.population_size):
            if self.task_type == TaskType.LANGUAGE:
                arch = self._create_language_architecture()
            elif self.task_type == TaskType.IMAGE:
                arch = self._create_image_architecture()
            else:
                arch = self._create_multimodal_architecture()
            
            individual = Individual(
                architecture=arch,
                generation=0,
                evaluated=False
            )
            self.population.append(individual)
    
    def _create_language_architecture(self) -> Dict[str, Any]:
        """创建语言模型架构"""
        return {
            "type": "transformer",
            "num_layers": random.randint(2, 12),
            "hidden_size": random.choice([128, 256, 512, 768]),
            "num_heads": random.choice([4, 8, 12]),
            "ffn_dim": random.choice([512, 1024, 2048, 3072]),
            "dropout": random.uniform(0.1, 0.5),
            "activation": random.choice(["relu", "gelu", "silu"]),
            "vocab_size": 50257,
            "max_seq_len": random.choice([128, 256, 512, 1024]),
        }
    
    def _create_image_architecture(self) -> Dict[str, Any]:
        """创建图像模型架构"""
        return {
            "type": "cnn",
            "num_blocks": random.randint(2, 8),
            "base_channels": random.choice([16, 32, 64, 128]),
            "kernel_size": random.choice([3, 5, 7]),
            "stride": random.choice([1, 2]),
            "use_batch_norm": random.choice([True, False]),
            "activation": random.choice(["relu", "leaky_relu", "swish"]),
            "pooling": random.choice(["max", "avg", "adaptive"]),
            "num_classes": 10,
            "input_channels": 3,
            "input_size": 32,
        }
    
    def _create_multimodal_architecture(self) -> Dict[str, Any]:
        """创建多模态架构"""
        return {
            "type": "multimodal",
            "vision_encoder": self._create_image_architecture(),
            "text_encoder": self._create_language_architecture(),
            "fusion_type": random.choice(["concat", "attention", "bilinear", "gated"]),
            "fusion_dim": random.choice([256, 512, 768]),
            "projection_dim": random.choice([128, 256, 512]),
            "temperature": random.uniform(0.07, 0.5),
            "use_contrastive": random.choice([True, False]),
        }
    
    def evolve(self) -> Individual:
        """主进化循环 - 运行完整遗传算法"""
        self.initialize_population()
        
        for gen in range(self.config.generations):
            self.generation = gen
            self._evaluate_population()
            self._update_best_individual()
            self._record_generation_stats(gen)
            
            if self._check_termination():
                break
            
            self.population = self._create_next_generation()
        
        return self.best_individual if self.best_individual else self.population[0]
    
    def _evaluate_population(self) -> None:
        """评估整个种群"""
        for individual in self.population:
            if not individual.evaluated:
                individual.fitness = self._evaluate_individual(individual)
                individual.evaluated = True
    
    def _update_best_individual(self) -> None:
        """更新最佳个体"""
        self.population.sort(
            key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
            reverse=True
        )
        
        current_best = self.population[0]
        if self.best_individual is None:
            self.best_individual = current_best
        elif current_best.fitness and current_best.fitness > (self.best_individual.fitness or 0):
            self.best_individual = current_best
    
    def _record_generation_stats(self, gen: int) -> None:
        """记录当前代统计信息"""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        self.history.append({
            "generation": gen,
            "best_fitness": self.population[0].fitness,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0,
            "worst_fitness": self.population[-1].fitness,
        })
    
    def _create_next_generation(self) -> List[Individual]:
        """创建下一代种群"""
        new_population = self.population[:self.config.elitism].copy()
        
        while len(new_population) < self.config.population_size:
            new_individuals = self._generate_offspring()
            for ind in new_individuals:
                if len(new_population) < self.config.population_size:
                    new_population.append(ind)
        
        return new_population
    
    def _generate_offspring(self) -> List[Individual]:
        """生成后代 - 根据概率选择交叉或变异"""
        if random.random() < self.config.crossover_rate:
            return self._perform_crossover()
        return [self._perform_mutation()]
    
    def _perform_crossover(self) -> List[Individual]:
        """执行交叉操作"""
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        crossover_methods = {
            TaskType.LANGUAGE: self._crossover_language,
            TaskType.IMAGE: self._crossover_image,
            TaskType.MULTIMODAL: self._crossover_multimodal,
        }
        method = crossover_methods.get(self.task_type, self._crossover_language)
        
        offspring1, offspring2 = method(parent1, parent2)
        return [offspring1, offspring2]
    
    def _perform_mutation(self) -> Individual:
        """执行变异操作"""
        parent = self._tournament_selection()
        
        mutation_methods = {
            TaskType.LANGUAGE: self._mutate_language,
            TaskType.IMAGE: self._mutate_image,
            TaskType.MULTIMODAL: self._mutate_multimodal,
        }
        method = mutation_methods.get(self.task_type, self._mutate_language)
        
        return method(parent)
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """评估个体适应度 - 路由到具体评估器"""
        arch = individual.architecture
        evaluators = {
            "transformer": self._evaluate_transformer_fitness,
            "cnn": self._evaluate_cnn_fitness,
            "multimodal": self._evaluate_multimodal_fitness,
        }
        
        evaluator = evaluators.get(arch["type"], lambda a: 0.0)
        fitness = evaluator(arch)
        
        # 添加随机噪声
        fitness += random.gauss(0, 2)
        return max(0.0, min(100.0, fitness))
    
    def _get_activation_multiplier(self, activation: str) -> float:
        """获取激活函数的复杂度乘数"""
        multipliers = {"gelu": 1.1, "silu": 1.2, "swish": 1.2, "leaky_relu": 1.1}
        return multipliers.get(activation, 1.0)
    
    def _get_seq_len_multiplier(self, seq_len: int) -> float:
        """获取序列长度的复杂度乘数"""
        if seq_len > 512:
            return 1.3
        elif seq_len < 256:
            return 0.8
        return 1.0
    
    def _apply_memory_penalty(self, complexity: float, thresholds: list) -> float:
        """应用显存限制惩罚 - thresholds: [(limit, penalty), ...]"""
        for limit, penalty in thresholds:
            if complexity > limit:
                return complexity * penalty
        return complexity
    
    def _evaluate_transformer_fitness(self, arch: Dict[str, Any]) -> float:
        """评估Transformer架构适应度"""
        complexity = arch["num_layers"] * arch["hidden_size"] * arch["num_heads"]
        
        # 应用各维度乘数
        if arch.get("ffn_dim", 0) > 2048:
            complexity *= 1.5
        if arch.get("dropout", 0) > 0.3:
            complexity *= 0.9
        
        complexity *= self._get_activation_multiplier(arch.get("activation", ""))
        complexity *= self._get_seq_len_multiplier(arch.get("max_seq_len", 512))
        
        # 显存惩罚
        if self.gpu_memory < 16:
            complexity = self._apply_memory_penalty(complexity, [(100000, 0.7), (50000, 0.85)])
        elif complexity > 200000:
            complexity *= 0.8
        
        return 100.0 - (complexity / 1000.0)
    
    def _evaluate_cnn_fitness(self, arch: Dict[str, Any]) -> float:
        """评估CNN架构适应度"""
        complexity = arch["num_blocks"] * arch["base_channels"] * (arch["kernel_size"] ** 2)
        
        # 应用各维度乘数
        if arch.get("use_batch_norm", False):
            complexity *= 1.1
        
        pooling_multipliers = {"adaptive": 1.15, "avg": 1.05}
        complexity *= pooling_multipliers.get(arch.get("pooling"), 1.0)
        complexity *= self._get_activation_multiplier(arch.get("activation", ""))
        
        if arch.get("stride", 1) > 1:
            complexity *= 0.9
        
        # 显存惩罚
        if self.gpu_memory < 12:
            complexity = self._apply_memory_penalty(complexity, [(50000, 0.6), (20000, 0.8)])
        elif self.gpu_memory < 20 and complexity > 100000:
            complexity *= 0.7
        
        return 100.0 - (complexity / 500.0)
    
    def _calculate_fusion_complexity(self, arch: Dict[str, Any]) -> float:
        """计算融合层复杂度"""
        fusion_type = arch.get("fusion_type", "concat")
        fusion_dim = arch.get("fusion_dim", 512)
        
        if fusion_type == "attention":
            return fusion_dim ** 2
        elif fusion_type == "bilinear":
            return fusion_dim * arch.get("projection_dim", 256)
        elif fusion_type == "gated":
            return fusion_dim * 2
        return fusion_dim
    
    def _evaluate_multimodal_fitness(self, arch: Dict[str, Any]) -> float:
        """评估多模态架构适应度"""
        vision = arch["vision_encoder"]
        text = arch["text_encoder"]
        
        vision_complexity = vision["num_blocks"] * vision["base_channels"]
        text_complexity = text["num_layers"] * text["hidden_size"]
        fusion_complexity = self._calculate_fusion_complexity(arch)
        
        complexity = vision_complexity + text_complexity + fusion_complexity
        
        if arch.get("use_contrastive", False):
            complexity *= 1.3
        if self.gpu_memory < 20:
            complexity *= 0.75
        
        return 100.0 - (complexity / 2000.0)
    
    def _tournament_selection(self) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
    
    def _crossover_language(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """语言模型交叉"""
        arch1 = p1.architecture.copy()
        arch2 = p2.architecture.copy()
        
        for key in ["num_layers", "hidden_size", "num_heads", "ffn_dim"]:
            if random.random() < 0.5:
                arch1[key], arch2[key] = arch2[key], arch1[key]
        
        return (
            Individual(arch1, generation=self.generation + 1),
            Individual(arch2, generation=self.generation + 1)
        )
    
    def _crossover_image(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """图像模型交叉"""
        arch1 = p1.architecture.copy()
        arch2 = p2.architecture.copy()
        
        for key in ["num_blocks", "base_channels", "kernel_size", "stride"]:
            if random.random() < 0.5:
                arch1[key], arch2[key] = arch2[key], arch1[key]
        
        return (
            Individual(arch1, generation=self.generation + 1),
            Individual(arch2, generation=self.generation + 1)
        )
    
    def _crossover_multimodal(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """多模态模型交叉"""
        arch1 = p1.architecture.copy()
        arch2 = p2.architecture.copy()
        
        # 分别交叉vision和text encoder
        v1 = arch1["vision_encoder"]
        v2 = arch2["vision_encoder"]
        t1 = arch1["text_encoder"]
        t2 = arch2["text_encoder"]
        
        if random.random() < 0.5:
            arch1["vision_encoder"], arch2["vision_encoder"] = v2, v1
        
        if random.random() < 0.5:
            arch1["text_encoder"], arch2["text_encoder"] = t2, t1
        
        if random.random() < 0.5:
            arch1["fusion_type"], arch2["fusion_type"] = arch2["fusion_type"], arch1["fusion_type"]
        
        return (
            Individual(arch1, generation=self.generation + 1),
            Individual(arch2, generation=self.generation + 1)
        )
    
    def _mutate_language(self, parent: Individual) -> Individual:
        """语言模型变异"""
        arch = parent.architecture.copy()
        
        mutation_type = random.choice(["layers", "hidden", "heads", "dropout", "activation"])
        
        if mutation_type == "layers":
            arch["num_layers"] = max(2, min(12, arch["num_layers"] + random.choice([-1, 1])))
        elif mutation_type == "hidden":
            arch["hidden_size"] = random.choice([128, 256, 512, 768])
        elif mutation_type == "heads":
            arch["num_heads"] = random.choice([2, 4, 8, 12])
        elif mutation_type == "dropout":
            arch["dropout"] = max(0.1, min(0.5, arch["dropout"] + random.uniform(-0.1, 0.1)))
        else:
            arch["activation"] = random.choice(["relu", "gelu", "silu"])
        
        return Individual(arch, generation=self.generation + 1)
    
    def _mutate_image(self, parent: Individual) -> Individual:
        """图像模型变异"""
        arch = parent.architecture.copy()
        
        mutation_type = random.choice(["blocks", "channels", "kernel", "norm", "activation"])
        
        if mutation_type == "blocks":
            arch["num_blocks"] = max(2, min(8, arch["num_blocks"] + random.choice([-1, 1])))
        elif mutation_type == "channels":
            arch["base_channels"] = random.choice([16, 32, 64, 128])
        elif mutation_type == "kernel":
            arch["kernel_size"] = random.choice([3, 5, 7])
        elif mutation_type == "norm":
            arch["use_batch_norm"] = not arch["use_batch_norm"]
        else:
            arch["activation"] = random.choice(["relu", "leaky_relu", "swish"])
        
        return Individual(arch, generation=self.generation + 1)
    
    def _mutate_multimodal(self, parent: Individual) -> Individual:
        """多模态模型变异"""
        arch = parent.architecture.copy()
        
        component = random.choice(["vision", "text", "fusion"])
        
        if component == "vision":
            mutation_type = random.choice(["blocks", "channels", "kernel"])
            if mutation_type == "blocks":
                arch["vision_encoder"]["num_blocks"] = max(2, min(8, arch["vision_encoder"]["num_blocks"] + random.choice([-1, 1])))
            elif mutation_type == "channels":
                arch["vision_encoder"]["base_channels"] = random.choice([16, 32, 64, 128])
            else:
                arch["vision_encoder"]["kernel_size"] = random.choice([3, 5, 7])
        elif component == "text":
            mutation_type = random.choice(["layers", "hidden", "heads"])
            if mutation_type == "layers":
                arch["text_encoder"]["num_layers"] = max(2, min(12, arch["text_encoder"]["num_layers"] + random.choice([-1, 1])))
            elif mutation_type == "hidden":
                arch["text_encoder"]["hidden_size"] = random.choice([128, 256, 512, 768])
            else:
                arch["text_encoder"]["num_heads"] = random.choice([2, 4, 8, 12])
        else:
            arch["fusion_type"] = random.choice(["concat", "attention", "bilinear", "gated"])
        
        return Individual(arch, generation=self.generation + 1)
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        if self.generation >= self.config.generations - 1:
            return True
        
        if len(self.history) >= 10:
            recent = self.history[-10:]
            best_fitness = [h["best_fitness"] for h in recent if h["best_fitness"] is not None]
            if len(best_fitness) == 10:
                if max(best_fitness) - min(best_fitness) < 0.1:
                    return True
        
        return False
