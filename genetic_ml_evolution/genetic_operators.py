"""
Genetic Operators for Neural Architecture Evolution
针对小规模语言模型优化的遗传算子

This module implements genetic operators (selection, crossover, mutation) 
optimized for Small Language Models (SLM) with limited computational resources.

Key optimizations for SLM:
- Simplified mutation operations to reduce complexity
- Resource-efficient parameter ranges (smaller hidden sizes, fewer layers)
- Gradual mutations for stable evolution
- Architecture constraints to ensure SLM compatibility
"""

import random
import copy
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


# SLM-specific constraints
SLM_CONSTRAINTS = {
    "transformer": {
        "num_layers": (2, 12),       # Limited layers for SLM
        "hidden_size": (128, 768),   # Smaller hidden dimensions
        "num_heads": (2, 12),        # Fewer attention heads
        "ffn_dim": (256, 3072),      # Smaller FFN dimensions
        "dropout": (0.0, 0.3),       # Reasonable dropout range
        "vocab_size": (1000, 100000), # Common vocabulary sizes
        "max_seq_len": (64, 512),    # Shorter sequences for efficiency
    },
    "cnn": {
        "num_blocks": (2, 8),        # Fewer blocks
        "base_channels": (16, 128),  # Smaller channel counts
        "kernel_size": (1, 7),       # Reasonable kernel sizes
        "stride": (1, 2),            # Limited stride options
        "num_classes": (2, 100),     # Classification tasks
        "input_channels": (1, 3),    # Image input channels
        "input_size": (16, 128),     # Input image size
    },
    "multimodal": {
        "vision_num_blocks": (2, 6),
        "vision_base_channels": (16, 64),
        "text_num_layers": (2, 8),
        "text_hidden_size": (128, 512),
        "fusion_dim": (64, 512),
        "projection_dim": (64, 256),
        "temperature": (0.01, 0.5),
    }
}

# SLM-optimized mutation rates (lower for stability)
SLM_MUTATION_RATES = {
    "conservative": 0.1,   # Very gradual changes
    "moderate": 0.2,       # Balanced exploration
    "aggressive": 0.3,     # More exploration
}


class ArchitectureGene:
    """
    Represents a single architecture as a "gene" for genetic operations.
    
    The gene encodes neural network architecture parameters that can be
    mutated, crossed over, and evaluated.
    """
    
    def __init__(self, architecture: Dict[str, Any]):
        """
        Initialize an architecture gene.
        
        Args:
            architecture: Architecture configuration dictionary
        """
        self.architecture = copy.deepcopy(architecture)
        self.fitness: Optional[float] = None
        self.age: int = 0  # Number of generations survived
    
    def __repr__(self) -> str:
        arch_type = self.architecture.get("type", "unknown")
        return f"ArchitectureGene(type={arch_type}, fitness={self.fitness})"
    
    def copy(self) -> 'ArchitectureGene':
        """Create a deep copy of this gene."""
        new_gene = ArchitectureGene(self.architecture)
        new_gene.fitness = self.fitness
        new_gene.age = self.age
        return new_gene


class SLMutationOperators:
    """
    Mutation operators optimized for Small Language Models.
    
    Key design principles:
    1. Gradual changes - avoid large jumps in architecture space
    2. Resource awareness - respect memory and compute constraints
    3. Semantic preservation - maintain architectural validity
    4. Efficiency - minimize computational overhead
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.2,
        mutation_strength: str = "moderate",
        respect_constraints: bool = True
    ):
        """
        Initialize SLM-optimized mutation operators.
        
        Args:
            mutation_rate: Base probability of mutation (0-1)
            mutation_strength: One of "conservative", "moderate", "aggressive"
            respect_constraints: Whether to enforce SLM constraints
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.respect_constraints = respect_constraints
        
        # Get the effective mutation rate
        self.effective_rate = SLM_MUTATION_RATES.get(mutation_strength, 0.2)
    
    def _clamp_value(
        self, 
        value: float, 
        min_val: float, 
        max_val: float,
        arch_type: str,
        param_name: str
    ) -> float:
        """
        Clamp a value to valid range for SLM.
        
        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            arch_type: Architecture type for constraints
            param_name: Parameter name for constraints
            
        Returns:
            Clamped value
        """
        if self.respect_constraints and arch_type in SLM_CONSTRAINTS:
            if param_name in SLM_CONSTRAINTS[arch_type]:
                min_val = max(min_val, SLM_CONSTRAINTS[arch_type][param_name][0])
                max_val = min(max_val, SLM_CONSTRAINTS[arch_type][param_name][1])
        
        return max(min_val, min(max_val, value))
    
    def _get_mutation_step(self, current_value: float, is_integer: bool = True) -> float:
        """
        Calculate a gradual mutation step for SLM.
        
        For SLM, we prefer small, gradual changes:
        - Integer params: +/- 1 or 2 steps
        - Float params: +/- 10-20% of current value
        
        Args:
            current_value: Current parameter value
            is_integer: Whether the parameter should be an integer
            
        Returns:
            Mutation step size
        """
        if is_integer:
            # For integer parameters, use small steps
            if current_value <= 4:
                return random.choice([-1, 0, 1])
            elif current_value <= 16:
                return random.choice([-2, -1, 0, 1, 2])
            else:
                return random.choice([-4, -2, 0, 2, 4])
        else:
            # For float parameters, use percentage-based changes
            percentage = random.uniform(-0.2, 0.2)  # +/- 20%
            return current_value * percentage
    
    def mutate_transformer(
        self, 
        gene: ArchitectureGene,
        focused_params: Optional[List[str]] = None
    ) -> ArchitectureGene:
        """
        Mutate a Transformer architecture with SLM optimizations.
        
        SLM-specific optimizations:
        - Prefer smaller hidden sizes and fewer layers
        - Gradual changes to avoid breaking pretrained weights
        - Maintain divisibility constraints (hidden_size % num_heads == 0)
        
        Args:
            gene: Architecture gene to mutate
            focused_params: Specific parameters to mutate (None = all eligible)
            
        Returns:
            Mutated architecture gene
        """
        mutated = gene.copy()
        arch = mutated.architecture
        
        # Parameters that can be mutated
        mutable_params = {
            "num_layers": (int, True),
            "hidden_size": (int, True),
            "num_heads": (int, True),
            "ffn_dim": (int, True),
            "dropout": (float, False),
            "max_seq_len": (int, True),
        }
        
        # Filter to focused params if specified
        if focused_params:
            mutable_params = {k: v for k, v in mutable_params.items() 
                            if k in focused_params}
        
        for param, (param_type, is_integer) in mutable_params.items():
            if random.random() < self.effective_rate:
                current = arch.get(param)
                if current is None:
                    continue
                
                # Get mutation step
                step = self._get_mutation_step(current, is_integer)
                
                # Apply mutation
                new_value = current + step
                
                # Clamp to valid range
                new_value = self._clamp_value(
                    new_value, 
                    1, 
                    float('inf'),
                    "transformer",
                    param
                )
                
                # Convert to correct type
                if param_type == int:
                    new_value = int(round(new_value))
                
                arch[param] = new_value
        
        # Ensure hidden_size is divisible by num_heads for SLM
        if "num_heads" in arch and "hidden_size" in arch:
            hidden = arch["hidden_size"]
            heads = arch["num_heads"]
            # Adjust hidden_size to be divisible by num_heads
            if hidden % heads != 0:
                # Round to nearest multiple
                arch["hidden_size"] = round(hidden / heads) * heads
        
        # Ensure ffn_dim is a reasonable multiple of hidden_size (2-4x)
        if "ffn_dim" in arch and "hidden_size" in arch:
            hidden = arch["hidden_size"]
            ffn = arch["ffn_dim"]
            ratio = ffn / hidden
            if ratio < 1.5 or ratio > 4.5:
                # Reset to 3x hidden_size (common in SLM)
                arch["ffn_dim"] = hidden * 3
        
        return mutated
    
    def mutate_cnn(
        self, 
        gene: ArchitectureGene,
        focused_params: Optional[List[str]] = None
    ) -> ArchitectureGene:
        """
        Mutate a CNN architecture with SLM optimizations.
        
        SLM-specific optimizations:
        - Limited channel expansion to save memory
        - Gradual kernel size changes
        - Preserve spatial dimensions where possible
        
        Args:
            gene: Architecture gene to mutate
            focused_params: Specific parameters to mutate (None = all eligible)
            
        Returns:
            Mutated architecture gene
        """
        mutated = gene.copy()
        arch = mutated.architecture
        
        mutable_params = {
            "num_blocks": (int, True),
            "base_channels": (int, True),
            "kernel_size": (int, True),
            "stride": (int, True),
        }
        
        if focused_params:
            mutable_params = {k: v for k, v in mutable_params.items() 
                            if k in focused_params}
        
        for param, (param_type, is_integer) in mutable_params.items():
            if random.random() < self.effective_rate:
                current = arch.get(param)
                if current is None:
                    continue
                
                step = self._get_mutation_step(current, is_integer)
                new_value = current + step
                
                new_value = self._clamp_value(
                    new_value, 
                    1, 
                    float('inf'),
                    "cnn",
                    param
                )
                
                if param_type == int:
                    new_value = int(round(new_value))
                
                arch[param] = new_value
        
        # Ensure base_channels follows typical pattern (powers of 2)
        if "base_channels" in arch:
            channels = arch["base_channels"]
            # Round to nearest power of 2
            arch["base_channels"] = 2 ** round(channels / 16) * 16
            arch["base_channels"] = self._clamp_value(
                arch["base_channels"], 16, 128, "cnn", "base_channels"
            )
        
        # Ensure kernel_size is odd (common in CNNs)
        if "kernel_size" in arch:
            k = arch["kernel_size"]
            if k % 2 == 0:
                arch["kernel_size"] = k + 1
        
        return mutated
    
    def mutate_multimodal(
        self, 
        gene: ArchitectureGene,
        focused_params: Optional[List[str]] = None
    ) -> ArchitectureGene:
        """
        Mutate a Multimodal architecture with SLM optimizations.
        
        SLM-specific optimizations:
        - Balance between vision and text components
        - Efficient fusion dimensions
        - Temperature scaling for contrastive learning
        
        Args:
            gene: Architecture gene to mutate
            focused_params: Specific parameters to mutate (None = all eligible)
            
        Returns:
            Mutated architecture gene
        """
        mutated = gene.copy()
        arch = mutated.architecture
        
        # Vision encoder mutations
        if "vision_encoder" in arch:
            vision = arch["vision_encoder"]
            if random.random() < self.effective_rate:
                if "num_blocks" in vision:
                    step = self._get_mutation_step(vision["num_blocks"], True)
                    vision["num_blocks"] = int(self._clamp_value(
                        vision["num_blocks"] + step, 2, 6, "multimodal", "vision_num_blocks"
                    ))
                if "base_channels" in vision and random.random() < 0.5:
                    step = self._get_mutation_step(vision["base_channels"], True)
                    vision["base_channels"] = int(self._clamp_value(
                        vision["base_channels"] + step, 16, 64, "multimodal", "vision_base_channels"
                    ))
        
        # Text encoder mutations
        if "text_encoder" in arch:
            text = arch["text_encoder"]
            if random.random() < self.effective_rate:
                if "num_layers" in text:
                    step = self._get_mutation_step(text["num_layers"], True)
                    text["num_layers"] = int(self._clamp_value(
                        text["num_layers"] + step, 2, 8, "multimodal", "text_num_layers"
                    ))
                if "hidden_size" in text and random.random() < 0.5:
                    step = self._get_mutation_step(text["hidden_size"], True)
                    text["hidden_size"] = int(self._clamp_value(
                        text["hidden_size"] + step, 128, 512, "multimodal", "text_hidden_size"
                    ))
        
        # Fusion parameters
        fusion_params = {
            "fusion_dim": (int, True),
            "projection_dim": (int, True),
            "temperature": (float, False),
        }
        
        if focused_params:
            fusion_params = {k: v for k, v in fusion_params.items() 
                           if k in focused_params}
        
        for param, (param_type, is_integer) in fusion_params.items():
            if param in arch and random.random() < self.effective_rate:
                current = arch[param]
                step = self._get_mutation_step(current, is_integer)
                new_value = current + step
                
                new_value = self._clamp_value(
                    new_value, 
                    1, 
                    float('inf'),
                    "multimodal",
                    param
                )
                
                if param_type == int:
                    new_value = int(round(new_value))
                
                arch[param] = new_value
        
        # Ensure fusion_dim >= projection_dim for efficiency
        if "fusion_dim" in arch and "projection_dim" in arch:
            if arch["fusion_dim"] < arch["projection_dim"]:
                arch["fusion_dim"] = arch["projection_dim"]
        
        # Mutate fusion type (less frequently)
        if random.random() < self.effective_rate * 0.3:  # 30% of mutation rate
            fusion_types = ["concat", "attention", "bilinear", "cross"]
            current = arch.get("fusion_type", "attention")
            other_types = [t for t in fusion_types if t != current]
            if other_types:
                arch["fusion_type"] = random.choice(other_types)
        
        return mutated
    
    def mutate(
        self, 
        gene: ArchitectureGene,
        focused_params: Optional[List[str]] = None
    ) -> ArchitectureGene:
        """
        Mutate an architecture gene based on its type.
        
        Args:
            gene: Architecture gene to mutate
            focused_params: Specific parameters to mutate
            
        Returns:
            Mutated architecture gene
        """
        arch_type = gene.architecture.get("type", "unknown")
        
        if arch_type == "transformer":
            return self.mutate_transformer(gene, focused_params)
        elif arch_type == "cnn":
            return self.mutate_cnn(gene, focused_params)
        elif arch_type == "multimodal":
            return self.mutate_multimodal(gene, focused_params)
        else:
            logger.warning(f"Unknown architecture type: {arch_type}")
            return gene.copy()


class SLMCrossoverOperators:
    """
    Crossover operators optimized for Small Language Models.
    
    Key design principles:
    1. Preserve good building blocks from both parents
    2. Maintain architectural validity
    3. Balance exploration and exploitation
    """
    
    def __init__(self, crossover_rate: float = 0.7):
        """
        Initialize SLM-optimized crossover operators.
        
        Args:
            crossover_rate: Probability of crossover (0-1)
        """
        self.crossover_rate = crossover_rate
    
    def _uniform_crossover(
        self, 
        parent1: Dict[str, Any], 
        parent2: Dict[str, Any],
        numeric_params: List[str]
    ) -> Dict[str, Any]:
        """
        Perform uniform crossover on numeric parameters.
        
        Args:
            parent1: First parent architecture
            parent2: Second parent architecture
            numeric_params: List of numeric parameter names
            
        Returns:
            Child architecture
        """
        child = copy.deepcopy(parent1)
        
        for param in numeric_params:
            if param in parent1 and param in parent2:
                if random.random() < 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            elif param in parent2:
                child[param] = parent2[param]
        
        return child
    
    def _arithmetic_crossover(
        self, 
        parent1: Dict[str, Any], 
        parent2: Dict[str, Any],
        numeric_params: List[str],
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform arithmetic crossover (blend) for numeric parameters.
        
        This is particularly useful for SLM as it creates intermediate
        values rather than just selecting from parents.
        
        Args:
            parent1: First parent architecture
            parent2: Second parent architecture
            numeric_params: List of numeric parameter names
            alpha: Blend factor (0.5 = average)
            
        Returns:
            Child architecture
        """
        child = copy.deepcopy(parent1)
        
        for param in numeric_params:
            if param in parent1 and param in parent2:
                val1 = parent1[param]
                val2 = parent2[param]
                
                # Blend values
                blended = alpha * val1 + (1 - alpha) * val2
                
                # Round if integer
                if isinstance(val1, int) and isinstance(val2, int):
                    blended = int(round(blended))
                
                child[param] = blended
        
        return child
    
    def crossover_transformer(
        self, 
        parent1: ArchitectureGene, 
        parent2: ArchitectureGene
    ) -> Tuple[ArchitectureGene, ArchitectureGene]:
        """
        Crossover two Transformer architectures.
        
        Args:
            parent1: First parent gene
            parent2: Second parent gene
            
        Returns:
            Tuple of two child genes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        numeric_params = [
            "num_layers", "hidden_size", "num_heads", 
            "ffn_dim", "dropout", "max_seq_len"
        ]
        
        # Use arithmetic crossover for SLM (creates intermediate architectures)
        child1_arch = self._arithmetic_crossover(
            parent1.architecture, parent2.architecture, numeric_params
        )
        child2_arch = self._arithmetic_crossover(
            parent2.architecture, parent1.architecture, numeric_params
        )
        
        # Fix divisibility constraint for both children
        for child_arch in [child1_arch, child2_arch]:
            if "num_heads" in child_arch and "hidden_size" in child_arch:
                hidden = child_arch["hidden_size"]
                heads = child_arch["num_heads"]
                if hidden % heads != 0:
                    child_arch["hidden_size"] = round(hidden / heads) * heads
        
        return ArchitectureGene(child1_arch), ArchitectureGene(child2_arch)
    
    def crossover_cnn(
        self, 
        parent1: ArchitectureGene, 
        parent2: ArchitectureGene
    ) -> Tuple[ArchitectureGene, ArchitectureGene]:
        """
        Crossover two CNN architectures.
        
        Args:
            parent1: First parent gene
            parent2: Second parent gene
            
        Returns:
            Tuple of two child genes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        numeric_params = [
            "num_blocks", "base_channels", "kernel_size", "stride"
        ]
        
        # Mix uniform and arithmetic crossover
        if random.random() < 0.5:
            child1_arch = self._uniform_crossover(
                parent1.architecture, parent2.architecture, numeric_params
            )
            child2_arch = self._uniform_crossover(
                parent2.architecture, parent1.architecture, numeric_params
            )
        else:
            child1_arch = self._arithmetic_crossover(
                parent1.architecture, parent2.architecture, numeric_params
            )
            child2_arch = self._arithmetic_crossover(
                parent2.architecture, parent1.architecture, numeric_params
            )
        
        # Fix constraints
        for child_arch in [child1_arch, child2_arch]:
            # Ensure odd kernel size
            if "kernel_size" in child_arch:
                k = child_arch["kernel_size"]
                if k % 2 == 0:
                    child_arch["kernel_size"] = k + 1
            
            # Round channels to power of 2
            if "base_channels" in child_arch:
                ch = child_arch["base_channels"]
                child_arch["base_channels"] = 2 ** round(ch / 16) * 16
        
        return ArchitectureGene(child1_arch), ArchitectureGene(child2_arch)
    
    def crossover_multimodal(
        self, 
        parent1: ArchitectureGene, 
        parent2: ArchitectureGene
    ) -> Tuple[ArchitectureGene, ArchitectureGene]:
        """
        Crossover two Multimodal architectures.
        
        Args:
            parent1: First parent gene
            parent2: Second parent gene
            
        Returns:
            Tuple of two child genes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_arch = copy.deepcopy(parent1.architecture)
        child2_arch = copy.deepcopy(parent2.architecture)
        
        # Crossover vision encoders
        if "vision_encoder" in parent1.architecture and "vision_encoder" in parent2.architecture:
            vision_params = ["num_blocks", "base_channels"]
            child1_arch["vision_encoder"] = self._arithmetic_crossover(
                parent1.architecture["vision_encoder"],
                parent2.architecture["vision_encoder"],
                vision_params
            )
            child2_arch["vision_encoder"] = self._arithmetic_crossover(
                parent2.architecture["vision_encoder"],
                parent1.architecture["vision_encoder"],
                vision_params
            )
        
        # Crossover text encoders
        if "text_encoder" in parent1.architecture and "text_encoder" in parent2.architecture:
            text_params = ["num_layers", "hidden_size"]
            child1_arch["text_encoder"] = self._arithmetic_crossover(
                parent1.architecture["text_encoder"],
                parent2.architecture["text_encoder"],
                text_params
            )
            child2_arch["text_encoder"] = self._arithmetic_crossover(
                parent2.architecture["text_encoder"],
                parent1.architecture["text_encoder"],
                text_params
            )
        
        # Crossover fusion parameters
        fusion_params = ["fusion_dim", "projection_dim", "temperature"]
        for param in fusion_params:
            if param in parent1.architecture and param in parent2.architecture:
                val1, val2 = parent1.architecture[param], parent2.architecture[param]
                child1_arch[param] = 0.5 * val1 + 0.5 * val2
                child2_arch[param] = 0.5 * val2 + 0.5 * val1
                
                if isinstance(val1, int):
                    child1_arch[param] = int(round(child1_arch[param]))
                    child2_arch[param] = int(round(child2_arch[param]))
        
        return ArchitectureGene(child1_arch), ArchitectureGene(child2_arch)
    
    def crossover(
        self, 
        parent1: ArchitectureGene, 
        parent2: ArchitectureGene
    ) -> Tuple[ArchitectureGene, ArchitectureGene]:
        """
        Crossover two architecture genes based on their type.
        
        Args:
            parent1: First parent gene
            parent2: Second parent gene
            
        Returns:
            Tuple of two child genes
        """
        arch_type = parent1.architecture.get("type", "unknown")
        
        # Ensure same type for crossover
        if parent2.architecture.get("type") != arch_type:
            logger.warning("Cannot crossover different architecture types")
            return parent1.copy(), parent2.copy()
        
        if arch_type == "transformer":
            return self.crossover_transformer(parent1, parent2)
        elif arch_type == "cnn":
            return self.crossover_cnn(parent1, parent2)
        elif arch_type == "multimodal":
            return self.crossover_multimodal(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()


class SLMSelectionOperators:
    """
    Selection operators optimized for Small Language Models.
    
    Key design principles:
    1. Maintain diversity in the population
    2. Balance exploration and exploitation
    3. Consider computational efficiency
    """
    
    @staticmethod
    def tournament_selection(
        population: List[ArchitectureGene],
        tournament_size: int = 3,
        selection_pressure: float = 0.7
    ) -> ArchitectureGene:
        """
        Tournament selection with configurable pressure.
        
        Args:
            population: Population of genes
            tournament_size: Number of individuals in tournament
            selection_pressure: Probability of selecting the best (0.5-1.0)
            
        Returns:
            Selected gene
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Sort by fitness (descending)
        tournament.sort(key=lambda x: x.fitness if x.fitness is not None else 0, reverse=True)
        
        # Select based on pressure
        if random.random() < selection_pressure:
            return tournament[0].copy()
        else:
            return random.choice(tournament).copy()
    
    @staticmethod
    def rank_selection(
        population: List[ArchitectureGene],
        selection_pressure: float = 1.5
    ) -> ArchitectureGene:
        """
        Rank-based selection.
        
        Args:
            population: Population of genes
            selection_pressure: Selection pressure (1.0 = uniform, 2.0 = strong)
            
        Returns:
            Selected gene
        """
        # Sort by fitness
        sorted_pop = sorted(
            population, 
            key=lambda x: x.fitness if x.fitness is not None else 0, 
            reverse=True
        )
        
        n = len(sorted_pop)
        
        # Calculate selection probabilities based on rank
        probs = []
        for i in range(n):
            # Linear ranking
            prob = (2 - selection_pressure) / n + \
                   (2 * i * (selection_pressure - 1)) / (n * (n - 1))
            probs.append(max(0, prob))
        
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / n] * n
        
        # Select
        return random.choices(sorted_pop, weights=probs, k=1)[0].copy()
    
    @staticmethod
    def elitism_selection(
        population: List[ArchitectureGene],
        elite_size: int = 2
    ) -> List[ArchitectureGene]:
        """
        Select top individuals for direct survival.
        
        Args:
            population: Population of genes
            elite_size: Number of elite individuals to select
            
        Returns:
            List of elite genes
        """
        sorted_pop = sorted(
            population, 
            key=lambda x: x.fitness if x.fitness is not None else 0, 
            reverse=True
        )
        
        return [g.copy() for g in sorted_pop[:elite_size]]
