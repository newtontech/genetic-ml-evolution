"""
Advanced Mutation Strategies for Small-Scale Language Models

This module implements advanced mutation strategies optimized for small-scale language models,
with focus on:
1. Smart adaptive mutation rates (based on fitness, age, diversity)
2. Layered mutation strategies (different phases of evolution)
3. Intelligent mutation operation selection (based on success rates)
4. Token limit and generation quality considerations
"""

import numpy as np
import random
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .surrogate_model import SurrogateModel

logger = logging.getLogger(__name__)


@dataclass
class MutationStatistics:
    """Track mutation operation statistics"""
    operation_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    success_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_mutation(self, operation: str, success: bool):
        """Record a mutation operation"""
        self.operation_counts[operation] += 1
        if success:
            self.success_counts[operation] += 1
    
    def get_success_rate(self, operation: str) -> float:
        """Get success rate for an operation"""
        total = self.operation_counts[operation]
        if total == 0:
            return 0.5  # Default for unknown operations
        return self.success_counts[operation] / total
    
    def get_best_operations(self, n: int = 3) -> List[str]:
        """Get top n operations by success rate"""
        if not self.operation_counts:
            return []
        
        operations = list(self.operation_counts.keys())
        operations.sort(key=lambda op: self.get_success_rate(op), reverse=True)
        return operations[:n]


class AdvancedMutationStrategy:
    """
    Advanced mutation strategies optimized for small-scale language models.
    
    Key features:
    - Smart adaptive mutation rates based on individual fitness and population state
    - Layered mutation strategies (exploration → exploitation)
    - UCB-based mutation operation selection
    - Token efficiency considerations
    """
    
    # Small-scale LM parameter ranges (optimized for 10M-100M parameters)
    SLM_RANGES = {
        "num_layers": (2, 12),
        "hidden_size": (128, 768),
        "num_heads": (2, 12),
        "ffn_dim": (512, 3072),
        "dropout": (0.0, 0.3),
        "vocab_size": (10000, 100000),
        "max_seq_len": (128, 512)
    }
    
    # Mutation phases
    PHASES = {
        "exploration": (0, 20),    # Generations 0-20: high exploration
        "balanced": (20, 60),      # Generations 20-60: balanced
        "exploitation": (60, 9999) # Generations 60+: exploitation
    }
    
    def __init__(self, ucb_alpha: float = 1.0):
        """
        Initialize advanced mutation strategy.
        
        Args:
            ucb_alpha: UCB exploration parameter (higher = more exploration)
        """
        self.stats = MutationStatistics()
        self.ucb_alpha = ucb_alpha
    
    def calculate_adaptive_mutation_rate(
        self,
        base_rate: float,
        individual_fitness: Optional[float],
        individual_age: int,
        population_diversity: float,
        generation: int,
        best_fitness: Optional[float] = None
    ) -> float:
        """
        Calculate adaptive mutation rate based on multiple factors.
        
        Args:
            base_rate: Base mutation rate
            individual_fitness: Fitness of individual to mutate
            individual_age: Age of individual (generations survived)
            population_diversity: Current population diversity (0-1)
            generation: Current generation
            best_fitness: Best fitness in population
            
        Returns:
            Adjusted mutation rate
        """
        rate = base_rate
        
        # Factor 1: Individual fitness (high fitness = low mutation)
        if individual_fitness is not None and best_fitness is not None and best_fitness > 0:
            fitness_ratio = individual_fitness / best_fitness
            # Scale: 0.7x for best individual, 1.3x for worst
            fitness_factor = 1.0 - 0.3 * fitness_ratio
            rate *= max(0.7, min(1.3, fitness_factor))
        
        # Factor 2: Individual age (young = high mutation)
        # Young individuals (< 5 generations): 1.2x rate
        # Old individuals (> 10 generations): 0.8x rate
        if individual_age < 5:
            rate *= 1.2
        elif individual_age > 10:
            rate *= 0.8
        
        # Factor 3: Population diversity (low diversity = high mutation)
        if population_diversity < 0.3:
            rate *= 1.5  # Boost exploration when diversity is low
        elif population_diversity > 0.8:
            rate *= 0.8  # Reduce mutation when diversity is high
        
        # Factor 4: Generation phase
        if generation < 20:
            # Exploration phase: increase rate
            rate *= 1.2
        elif generation > 60:
            # Exploitation phase: decrease rate
            rate *= 0.7
        
        # Clamp to reasonable range
        return max(0.01, min(0.9, rate))
    
    def get_mutation_phase(self, generation: int) -> str:
        """
        Determine current mutation phase based on generation.
        
        Args:
            generation: Current generation number
            
        Returns:
            Phase name ("exploration", "balanced", "exploitation")
        """
        for phase, (start, end) in self.PHASES.items():
            if start <= generation < end:
                return phase
        return "exploitation"
    
    def select_mutation_operation(self, operations: List[str]) -> str:
        """
        Select mutation operation using UCB algorithm.
        
        Args:
            operations: List of available operation names
            
        Returns:
            Selected operation name
        """
        if not operations:
            return "random"
        
        # If we don't have enough data, use random selection
        total_ops = sum(self.stats.operation_counts.values())
        if total_ops < 10:
            return random.choice(operations)
        
        # UCB selection
        ucb_scores = {}
        for op in operations:
            success_rate = self.stats.get_success_rate(op)
            op_count = self.stats.operation_counts[op]
            
            # UCB formula: success_rate + alpha * sqrt(ln(total) / op_count)
            if op_count > 0:
                exploration_bonus = self.ucb_alpha * np.sqrt(
                    np.log(total_ops) / op_count
                )
                ucb_scores[op] = success_rate + exploration_bonus
            else:
                # Never tried operation: give high priority
                ucb_scores[op] = 1.0
        
        # Select operation with highest UCB score
        return max(operations, key=lambda op: ucb_scores.get(op, 0))
    
    def estimate_parameters(self, architecture: Dict[str, Any]) -> int:
        """
        Estimate number of parameters for a Transformer architecture.
        
        Args:
            architecture: Architecture configuration
            
        Returns:
            Estimated parameter count
        """
        if architecture.get("type") != "transformer":
            return 0
        
        L = architecture.get("num_layers", 6)
        H = architecture.get("hidden_size", 512)
        A = architecture.get("num_heads", 8)
        F = architecture.get("ffn_dim", 2048)
        V = architecture.get("vocab_size", 30000)
        
        # Embedding parameters (input + output sharing)
        embedding_params = V * H
        
        # Per-layer parameters:
        # - Self-attention: Q, K, V, O projections (4 * H * H)
        # - FFN: two layers (H * F + F * H)
        # - Layer norms: 2 per layer (4 * H)
        per_layer = 4 * H * H + 2 * H * F + 4 * H
        
        total = embedding_params + L * per_layer
        
        return total
    
    def estimate_tokens_per_forward(
        self,
        architecture: Dict[str, Any],
        sequence_length: int = 512
    ) -> int:
        """
        Estimate memory (in tokens) required for forward pass.
        
        Args:
            architecture: Architecture configuration
            sequence_length: Sequence length (default from architecture)
            
        Returns:
            Estimated token-equivalent memory
        """
        if architecture.get("type") != "transformer":
            return 0
        
        seq_len = architecture.get("max_seq_len", sequence_length)
        L = architecture.get("num_layers", 6)
        H = architecture.get("hidden_size", 512)
        A = architecture.get("num_heads", 8)
        
        # Memory for attention (seq_len^2 for attention matrix)
        attention_memory = seq_len * seq_len
        
        # Memory for activations (per layer)
        activation_memory = seq_len * H
        
        # Total per layer
        per_layer_memory = attention_memory + activation_memory
        
        return L * per_layer_memory
    
    def mutate_transformer_advanced(
        self,
        architecture: Dict[str, Any],
        base_mutation_rate: float = 0.1,
        individual_fitness: Optional[float] = None,
        individual_age: int = 0,
        population_diversity: float = 0.5,
        generation: int = 0,
        best_fitness: Optional[float] = None,
        surrogate_model: Optional[SurrogateModel] = None,
        max_parameters: int = 100_000_000  # 100M parameters
    ) -> Tuple[Dict[str, Any], str]:
        """
        Advanced mutation for Transformer with all optimizations.
        
        Args:
            architecture: Original architecture
            base_mutation_rate: Base mutation rate
            individual_fitness: Fitness of individual
            individual_age: Age of individual
            population_diversity: Population diversity
            generation: Current generation
            best_fitness: Best fitness in population
            surrogate_model: Optional surrogate model
            max_parameters: Maximum allowed parameters
            
        Returns:
            Tuple of (mutated architecture, mutation description)
        """
        from copy import deepcopy
        
        mutated = deepcopy(architecture)
        mutations = []
        
        # Calculate adaptive mutation rate
        mutation_rate = self.calculate_adaptive_mutation_rate(
            base_mutation_rate,
            individual_fitness,
            individual_age,
            population_diversity,
            generation,
            best_fitness
        )
        
        # Determine phase
        phase = self.get_mutation_phase(generation)
        
        # Define mutation operations
        operations = [
            "num_layers", "hidden_size", "num_heads", 
            "ffn_dim", "dropout", "activation"
        ]
        
        # Select operation using UCB
        selected_op = self.select_mutation_operation(operations)
        
        # Execute mutations
        current_params = self.estimate_parameters(mutated)
        
        # 1. Number of layers
        if selected_op == "num_layers" or random.random() < mutation_rate:
            old_value = mutated.get("num_layers", 6)
            
            # Phase-based mutation magnitude
            if phase == "exploration":
                deltas = [-3, -2, -1, 1, 2, 3]
            elif phase == "balanced":
                deltas = [-2, -1, 1, 2]
            else:  # exploitation
                deltas = [-1, 1]
            
            # Small model bias: prefer fewer layers
            if random.random() < 0.6:
                deltas = [d for d in deltas if d <= 0] or [-1, -1]
            
            delta = random.choice(deltas)
            new_value = max(2, min(12, old_value + delta))
            
            # Check parameter budget
            test_arch = deepcopy(mutated)
            test_arch["num_layers"] = new_value
            if self.estimate_parameters(test_arch) <= max_parameters:
                if new_value != old_value:
                    mutated["num_layers"] = new_value
                    mutations.append(f"layers:{old_value}→{new_value}")
                    self.stats.record_mutation("num_layers", True)
                else:
                    self.stats.record_mutation("num_layers", False)
        
        # 2. Hidden size
        if selected_op == "hidden_size" or random.random() < mutation_rate:
            old_value = mutated.get("hidden_size", 512)
            num_heads = mutated.get("num_heads", 8)
            
            # Phase-based deltas
            if phase == "exploration":
                deltas = [-256, -128, -64, 64, 128, 256]
            elif phase == "balanced":
                deltas = [-128, -64, 64, 128]
            else:
                deltas = [-64, 64]
            
            # Small model bias
            if random.random() < 0.6:
                deltas = [d for d in deltas if d <= 0] or [-64, -64]
            
            delta = random.choice(deltas)
            new_value = old_value + delta
            
            # Ensure divisible by num_heads
            new_value = max(128, min(768, new_value))
            new_value = (new_value // num_heads) * num_heads
            
            # Check parameter budget
            test_arch = deepcopy(mutated)
            test_arch["hidden_size"] = new_value
            if self.estimate_parameters(test_arch) <= max_parameters:
                if new_value != old_value:
                    mutated["hidden_size"] = new_value
                    mutations.append(f"hidden:{old_value}→{new_value}")
                    self.stats.record_mutation("hidden_size", True)
                else:
                    self.stats.record_mutation("hidden_size", False)
        
        # 3. Number of heads
        if selected_op == "num_heads" or random.random() < mutation_rate:
            old_value = mutated.get("num_heads", 8)
            hidden_size = mutated.get("hidden_size", 512)
            
            valid_heads = [h for h in [2, 4, 8, 12, 16] if hidden_size % h == 0]
            
            if valid_heads:
                if phase == "exploration":
                    candidates = valid_heads
                else:
                    # Prefer fewer heads for small models
                    if random.random() < 0.6:
                        candidates = [h for h in valid_heads if h <= old_value]
                        candidates = candidates if candidates else [min(valid_heads)]
                    else:
                        candidates = valid_heads
                
                new_value = random.choice(candidates)
                
                if new_value != old_value:
                    mutated["num_heads"] = new_value
                    mutations.append(f"heads:{old_value}→{new_value}")
                    self.stats.record_mutation("num_heads", True)
                else:
                    self.stats.record_mutation("num_heads", False)
        
        # 4. FFN dimension
        if selected_op == "ffn_dim" or random.random() < mutation_rate:
            old_value = mutated.get("ffn_dim", 2048)
            hidden_size = mutated.get("hidden_size", 512)
            
            # Small models: prefer 2-3x ratio
            if random.random() < 0.7:
                ratio = random.choice([2, 2, 3])
            else:
                ratio = random.choice([2, 3, 4])
            
            new_value = hidden_size * ratio
            new_value = max(512, min(3072, new_value))
            
            # Check parameter budget
            test_arch = deepcopy(mutated)
            test_arch["ffn_dim"] = new_value
            if self.estimate_parameters(test_arch) <= max_parameters:
                if new_value != old_value:
                    mutated["ffn_dim"] = new_value
                    mutations.append(f"ffn:{old_value}→{new_value}")
                    self.stats.record_mutation("ffn_dim", True)
                else:
                    self.stats.record_mutation("ffn_dim", False)
        
        # 5. Dropout
        if selected_op == "dropout" or random.random() < mutation_rate:
            old_value = mutated.get("dropout", 0.1)
            
            # Small perturbation
            delta = random.uniform(-0.05, 0.05)
            new_value = max(0.0, min(0.3, old_value + delta))
            new_value = round(new_value, 2)
            
            if new_value != old_value:
                mutated["dropout"] = new_value
                mutations.append(f"dropout:{old_value:.2f}→{new_value:.2f}")
                self.stats.record_mutation("dropout", True)
            else:
                self.stats.record_mutation("dropout", False)
        
        # 6. Activation
        if selected_op == "activation" or random.random() < mutation_rate * 0.5:
            activations = ["relu", "gelu", "silu"]
            old_value = mutated.get("activation", "gelu")
            
            # Prefer gelu for small models
            if random.random() < 0.5:
                new_value = "gelu"
            else:
                new_value = random.choice(activations)
            
            if new_value != old_value:
                mutated["activation"] = new_value
                mutations.append(f"activation:{old_value}→{new_value}")
                self.stats.record_mutation("activation", True)
            else:
                self.stats.record_mutation("activation", False)
        
        mutation_desc = "; ".join(mutations) if mutations else "no_change"
        return mutated, mutation_desc
    
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mutation operations.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_mutations": sum(self.stats.operation_counts.values()),
            "total_successes": sum(self.stats.success_counts.values()),
            "operation_stats": {}
        }
        
        for op in self.stats.operation_counts:
            stats["operation_stats"][op] = {
                "count": self.stats.operation_counts[op],
                "successes": self.stats.success_counts[op],
                "success_rate": self.stats.get_success_rate(op)
            }
        
        if stats["total_mutations"] > 0:
            stats["overall_success_rate"] = (
                stats["total_successes"] / stats["total_mutations"]
            )
        else:
            stats["overall_success_rate"] = 0.0
        
        return stats
