"""
Genetic Algorithm for Neural Architecture Search

This module implements a genetic algorithm framework for evolving neural network
architectures, with special optimizations for small-scale language models.

Features:
- Population-based architecture evolution
- Adaptive mutation strategies for small models
- Surrogate model-guided mutations
- Multi-objective optimization (accuracy + efficiency)
- Integration with cache system
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from copy import deepcopy

from .surrogate_model import SurrogateModel
from .cache_system import ArchitectureCache
from .advanced_mutation import AdvancedMutationStrategy

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    Represents an individual (architecture) in the population.
    
    Attributes:
        architecture (Dict[str, Any]): Architecture configuration
        fitness (Optional[float]): Fitness score (0-100)
        age (int): Number of generations this individual has survived
        parent_ids (List[int]): IDs of parent individuals
        mutation_history (List[str]): History of mutations applied
    """
    architecture: Dict[str, Any]
    fitness: Optional[float] = None
    age: int = 0
    parent_ids: List[int] = None
    mutation_history: List[str] = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutation_history is None:
            self.mutation_history = []
    
    def __hash__(self):
        """Make individual hashable based on architecture"""
        arch_str = str(sorted(self.architecture.items()))
        return hash(arch_str)
    
    def __eq__(self, other):
        """Check equality based on architecture"""
        if not isinstance(other, Individual):
            return False
        return self.architecture == other.architecture


class MutationStrategy:
    """
    Mutation strategies optimized for small-scale language models.
    
    Small model considerations:
    - Prefer fewer layers (2-12)
    - Prefer smaller hidden dimensions (128-768)
    - Balance performance vs. parameter count
    - Avoid overly complex configurations
    """
    
    # Small-scale language model parameter ranges
    SMALL_LM_RANGES = {
        "num_layers": (2, 12),
        "hidden_size": (128, 768),
        "num_heads": (2, 12),
        "ffn_dim": (512, 3072),
        "dropout": (0.0, 0.3),
        "vocab_size": (10000, 100000),
        "max_seq_len": (128, 512)
    }
    
    # Small-scale CNN parameter ranges
    SMALL_CNN_RANGES = {
        "num_blocks": (2, 8),
        "base_channels": (16, 128),
        "kernel_size": (3, 7),
        "stride": (1, 2),
        "dropout": (0.0, 0.3)
    }
    
    # Mutation strategies with different aggressiveness
    MUTATION_TYPES = [
        "fine_tune",      # Small adjustments
        "moderate",       # Medium changes
        "exploratory",    # Large exploratory changes
        "guided",         # Surrogate model-guided
        "adaptive"        # Based on individual's age/performance
    ]
    
    @staticmethod
    def mutate_transformer(
        architecture: Dict[str, Any],
        mutation_rate: float = 0.1,
        strategy: str = "adaptive",
        surrogate_model: Optional[SurrogateModel] = None,
        generation: int = 0
    ) -> Tuple[Dict[str, Any], str]:
        """
        Mutate a Transformer architecture with small model optimizations.
        
        Args:
            architecture: Original architecture configuration
            mutation_rate: Probability of mutation for each parameter
            strategy: Mutation strategy type
            surrogate_model: Optional surrogate model for guided mutations
            generation: Current generation number (for adaptive strategies)
            
        Returns:
            Tuple of (mutated architecture, mutation description)
        """
        mutated = deepcopy(architecture)
        mutations = []
        
        # Adaptive mutation rate based on generation
        if strategy == "adaptive":
            # Decrease mutation rate over time (exploitation > exploration)
            mutation_rate = mutation_rate * (1.0 - min(generation / 100, 0.5))
        
        # 1. Number of layers mutation (small models: prefer fewer layers)
        if random.random() < mutation_rate:
            old_value = mutated.get("num_layers", 6)
            
            if strategy == "guided" and surrogate_model and surrogate_model.is_fitted:
                # Try multiple values and pick best predicted
                candidates = [old_value + delta for delta in [-2, -1, 0, 1, 2]]
                candidates = [
                    max(2, min(12, c)) for c in candidates  # Clamp to range
                ]
                best_value = old_value
                best_pred = surrogate_model.predict(mutated)
                
                for candidate in candidates:
                    test_arch = deepcopy(mutated)
                    test_arch["num_layers"] = candidate
                    pred = surrogate_model.predict(test_arch)
                    if pred and pred > best_pred:
                        best_pred = pred
                        best_value = candidate
                
                new_value = best_value
            else:
                # Small model bias: tend toward fewer layers
                if random.random() < 0.6:  # 60% chance to decrease or stay
                    delta = random.choice([-2, -1, 0, 0])
                else:
                    delta = random.choice([1, 2])
                
                new_value = max(2, min(12, old_value + delta))
            
            if new_value != old_value:
                mutated["num_layers"] = new_value
                mutations.append(f"layers:{old_value}→{new_value}")
        
        # 2. Hidden size mutation (small models: prefer smaller dimensions)
        if random.random() < mutation_rate:
            old_value = mutated.get("hidden_size", 512)
            
            # Small model bias: tend toward smaller dimensions
            if random.random() < 0.6:
                # Decrease or stay
                options = [-128, -64, 0, 0]
            else:
                # Increase
                options = [64, 128]
            
            delta = random.choice(options)
            new_value = max(128, min(768, old_value + delta))
            
            # Ensure hidden_size is divisible by num_heads
            num_heads = mutated.get("num_heads", 8)
            new_value = (new_value // num_heads) * num_heads
            
            if new_value != old_value:
                mutated["hidden_size"] = new_value
                mutations.append(f"hidden:{old_value}→{new_value}")
        
        # 3. Number of heads mutation
        if random.random() < mutation_rate:
            old_value = mutated.get("num_heads", 8)
            hidden_size = mutated.get("hidden_size", 512)
            
            # Must divide hidden_size evenly
            valid_heads = [h for h in [2, 4, 8, 12, 16] if hidden_size % h == 0]
            
            if valid_heads:
                if random.random() < 0.6:  # Prefer fewer heads for small models
                    smaller_heads = [h for h in valid_heads if h <= old_value]
                    new_value = random.choice(smaller_heads if smaller_heads else valid_heads)
                else:
                    new_value = random.choice(valid_heads)
                
                if new_value != old_value:
                    mutated["num_heads"] = new_value
                    mutations.append(f"heads:{old_value}→{new_value}")
        
        # 4. FFN dimension mutation (small models: prefer smaller FFN)
        if random.random() < mutation_rate:
            old_value = mutated.get("ffn_dim", 2048)
            hidden_size = mutated.get("hidden_size", 512)
            
            # Typical FFN ratios: 2x, 3x, 4x hidden_size
            ratios = [2, 3, 4]
            
            if random.random() < 0.6:  # Prefer smaller ratio for small models
                ratio = min(ratios)
            else:
                ratio = random.choice(ratios)
            
            new_value = hidden_size * ratio
            new_value = max(512, min(3072, new_value))
            
            if new_value != old_value:
                mutated["ffn_dim"] = new_value
                mutations.append(f"ffn:{old_value}→{new_value}")
        
        # 5. Dropout mutation
        if random.random() < mutation_rate:
            old_value = mutated.get("dropout", 0.1)
            
            # Small perturbation
            delta = random.uniform(-0.05, 0.05)
            new_value = max(0.0, min(0.3, old_value + delta))
            new_value = round(new_value, 2)
            
            if new_value != old_value:
                mutated["dropout"] = new_value
                mutations.append(f"dropout:{old_value:.2f}→{new_value:.2f}")
        
        # 6. Activation function mutation
        if random.random() < mutation_rate * 0.5:  # Lower rate for categorical
            activations = ["relu", "gelu", "silu"]
            old_value = mutated.get("activation", "gelu")
            
            # Prefer gelu for small models (good balance)
            if random.random() < 0.5:
                new_value = "gelu"
            else:
                new_value = random.choice(activations)
            
            if new_value != old_value:
                mutated["activation"] = new_value
                mutations.append(f"activation:{old_value}→{new_value}")
        
        mutation_desc = "; ".join(mutations) if mutations else "no_change"
        return mutated, mutation_desc
    
    @staticmethod
    def mutate_cnn(
        architecture: Dict[str, Any],
        mutation_rate: float = 0.1,
        strategy: str = "adaptive",
        surrogate_model: Optional[SurrogateModel] = None,
        generation: int = 0
    ) -> Tuple[Dict[str, Any], str]:
        """
        Mutate a CNN architecture with small model optimizations.
        
        Args:
            architecture: Original architecture configuration
            mutation_rate: Probability of mutation for each parameter
            strategy: Mutation strategy type
            surrogate_model: Optional surrogate model for guided mutations
            generation: Current generation number
            
        Returns:
            Tuple of (mutated architecture, mutation description)
        """
        mutated = deepcopy(architecture)
        mutations = []
        
        # Adaptive mutation rate
        if strategy == "adaptive":
            mutation_rate = mutation_rate * (1.0 - min(generation / 100, 0.5))
        
        # 1. Number of blocks mutation
        if random.random() < mutation_rate:
            old_value = mutated.get("num_blocks", 4)
            
            # Small model bias: prefer fewer blocks
            if random.random() < 0.6:
                delta = random.choice([-2, -1, 0, 0])
            else:
                delta = random.choice([1, 2])
            
            new_value = max(2, min(8, old_value + delta))
            
            if new_value != old_value:
                mutated["num_blocks"] = new_value
                mutations.append(f"blocks:{old_value}→{new_value}")
        
        # 2. Base channels mutation
        if random.random() < mutation_rate:
            old_value = mutated.get("base_channels", 64)
            
            # Small model bias: prefer fewer channels
            if random.random() < 0.6:
                delta = random.choice([-32, -16, 0, 0])
            else:
                delta = random.choice([16, 32])
            
            new_value = max(16, min(128, old_value + delta))
            
            if new_value != old_value:
                mutated["base_channels"] = new_value
                mutations.append(f"channels:{old_value}→{new_value}")
        
        # 3. Kernel size mutation
        if random.random() < mutation_rate:
            old_value = mutated.get("kernel_size", 3)
            new_value = random.choice([3, 5, 7])
            
            if new_value != old_value:
                mutated["kernel_size"] = new_value
                mutations.append(f"kernel:{old_value}→{new_value}")
        
        # 4. Activation mutation
        if random.random() < mutation_rate * 0.5:
            activations = ["relu", "leaky_relu", "silu"]
            old_value = mutated.get("activation", "relu")
            new_value = random.choice(activations)
            
            if new_value != old_value:
                mutated["activation"] = new_value
                mutations.append(f"activation:{old_value}→{new_value}")
        
        # 5. Pooling mutation
        if random.random() < mutation_rate * 0.5:
            pooling_types = ["max", "avg", "adaptive"]
            old_value = mutated.get("pooling", "max")
            new_value = random.choice(pooling_types)
            
            if new_value != old_value:
                mutated["pooling"] = new_value
                mutations.append(f"pooling:{old_value}→{new_value}")
        
        mutation_desc = "; ".join(mutations) if mutations else "no_change"
        return mutated, mutation_desc
    
    @staticmethod
    def mutate_multimodal(
        architecture: Dict[str, Any],
        mutation_rate: float = 0.1,
        strategy: str = "adaptive",
        surrogate_model: Optional[SurrogateModel] = None,
        generation: int = 0
    ) -> Tuple[Dict[str, Any], str]:
        """
        Mutate a multimodal architecture.
        
        Args:
            architecture: Original architecture configuration
            mutation_rate: Probability of mutation for each parameter
            strategy: Mutation strategy type
            surrogate_model: Optional surrogate model for guided mutations
            generation: Current generation number
            
        Returns:
            Tuple of (mutated architecture, mutation description)
        """
        mutated = deepcopy(architecture)
        mutations = []
        
        # Adaptive mutation rate
        if strategy == "adaptive":
            mutation_rate = mutation_rate * (1.0 - min(generation / 100, 0.5))
        
        # Mutate vision encoder
        if "vision_encoder" in mutated and random.random() < mutation_rate:
            vision_mutated, vision_desc = MutationStrategy.mutate_cnn(
                mutated["vision_encoder"],
                mutation_rate * 0.8,  # Lower rate for sub-components
                strategy,
                surrogate_model,
                generation
            )
            mutated["vision_encoder"] = vision_mutated
            if vision_desc != "no_change":
                mutations.append(f"vision:{vision_desc}")
        
        # Mutate text encoder
        if "text_encoder" in mutated and random.random() < mutation_rate:
            text_mutated, text_desc = MutationStrategy.mutate_transformer(
                mutated["text_encoder"],
                mutation_rate * 0.8,
                strategy,
                surrogate_model,
                generation
            )
            mutated["text_encoder"] = text_mutated
            if text_desc != "no_change":
                mutations.append(f"text:{text_desc}")
        
        # Fusion parameters
        if random.random() < mutation_rate:
            old_dim = mutated.get("fusion_dim", 512)
            
            if random.random() < 0.6:  # Prefer smaller dimensions
                delta = random.choice([-128, -64, 0])
            else:
                delta = random.choice([64, 128])
            
            new_dim = max(128, min(1024, old_dim + delta))
            
            if new_dim != old_dim:
                mutated["fusion_dim"] = new_dim
                mutations.append(f"fusion_dim:{old_dim}→{new_dim}")
        
        # Fusion type
        if random.random() < mutation_rate * 0.3:  # Low rate for fusion type
            fusion_types = ["concat", "attention", "bilinear", "cross"]
            old_value = mutated.get("fusion_type", "attention")
            new_value = random.choice(fusion_types)
            
            if new_value != old_value:
                mutated["fusion_type"] = new_value
                mutations.append(f"fusion_type:{old_value}→{new_value}")
        
        mutation_desc = "; ".join(mutations) if mutations else "no_change"
        return mutated, mutation_desc


class GeneticAlgorithm:
    """
    Genetic Algorithm for Neural Architecture Search.
    
    Implements population-based evolution with:
    - Tournament selection
    - Architecture-specific mutations
    - Surrogate model guidance
    - Elitism (preserve best individuals)
    - Diversity preservation
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.3,
        elitism_rate: float = 0.1,
        tournament_size: int = 3,
        surrogate_model: Optional[SurrogateModel] = None,
        cache_db_path: Optional[str] = None,
        task_type: str = "language",
        use_advanced_mutation: bool = True,
        max_parameters: int = 100_000_000,
        ucb_alpha: float = 1.0
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Base mutation rate
            crossover_rate: Crossover rate
            elitism_rate: Fraction of top individuals to preserve
            tournament_size: Size of tournament for selection
            surrogate_model: Optional surrogate model for guided mutations
            cache_db_path: Path to cache database
            task_type: Type of task ("language", "image", "multimodal")
            use_advanced_mutation: Whether to use advanced mutation strategies (default: True)
            max_parameters: Maximum number of parameters allowed (default: 100M)
            ucb_alpha: UCB exploration parameter for advanced mutation (default: 1.0)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.surrogate_model = surrogate_model
        self.task_type = task_type
        self.use_advanced_mutation = use_advanced_mutation
        self.max_parameters = max_parameters
        
        # Initialize cache
        self.cache = ArchitectureCache(cache_db_path) if cache_db_path else None
        
        # Initialize advanced mutation strategy if enabled
        if use_advanced_mutation:
            self.advanced_mutation_strategy = AdvancedMutationStrategy(ucb_alpha=ucb_alpha)
        else:
            self.advanced_mutation_strategy = None
        
        # Population and statistics
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []
    
    def initialize_population(self, seed_architectures: Optional[List[Dict]] = None):
        """
        Initialize the population with random or seed architectures.
        
        Args:
            seed_architectures: Optional list of seed architectures to include
        """
        self.population = []
        
        # Add seed architectures if provided
        if seed_architectures:
            for arch in seed_architectures[:self.population_size]:
                self.population.append(Individual(architecture=arch))
        
        # Fill remaining population with random architectures
        while len(self.population) < self.population_size:
            if self.task_type == "language":
                arch = self._random_transformer()
            elif self.task_type == "image":
                arch = self._random_cnn()
            else:  # multimodal
                arch = self._random_multimodal()
            
            individual = Individual(architecture=arch)
            
            # Avoid duplicates
            if individual not in self.population:
                self.population.append(individual)
    
    def _random_transformer(self) -> Dict[str, Any]:
        """Generate a random Transformer architecture optimized for small models."""
        # Bias toward smaller configurations
        num_layers = random.choice([2, 3, 4, 4, 6, 6, 8])  # More small values
        hidden_size = random.choice([128, 256, 256, 384, 512, 512])  # Prefer smaller
        num_heads = random.choice([2, 4, 4, 8])  # Prefer fewer heads
        
        # Ensure hidden_size is divisible by num_heads
        hidden_size = (hidden_size // num_heads) * num_heads
        
        # FFN dimension typically 2-4x hidden_size
        ffn_ratio = random.choice([2, 3, 4])
        ffn_dim = hidden_size * ffn_ratio
        
        return {
            "type": "transformer",
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "dropout": round(random.uniform(0.05, 0.2), 2),
            "activation": random.choice(["relu", "gelu", "gelu"]),  # Prefer gelu
            "vocab_size": random.choice([10000, 30000, 50000]),
            "max_seq_len": random.choice([128, 256, 512])
        }
    
    def _random_cnn(self) -> Dict[str, Any]:
        """Generate a random CNN architecture optimized for small models."""
        num_blocks = random.choice([2, 3, 4, 4, 5])  # Prefer fewer blocks
        base_channels = random.choice([16, 32, 32, 64, 64])  # Prefer fewer channels
        
        return {
            "type": "cnn",
            "num_blocks": num_blocks,
            "base_channels": base_channels,
            "kernel_size": random.choice([3, 3, 5]),  # Prefer smaller kernels
            "stride": random.choice([1, 1, 2]),
            "use_batch_norm": True,
            "activation": random.choice(["relu", "relu", "leaky_relu"]),
            "pooling": random.choice(["max", "avg", "adaptive"]),
            "dropout": round(random.uniform(0.0, 0.2), 2),
            "num_classes": 10,  # Default
            "input_channels": 3,
            "input_size": 32
        }
    
    def _random_multimodal(self) -> Dict[str, Any]:
        """Generate a random multimodal architecture."""
        return {
            "type": "multimodal",
            "vision_encoder": self._random_cnn(),
            "text_encoder": self._random_transformer(),
            "fusion_type": random.choice(["concat", "attention", "attention"]),
            "fusion_dim": random.choice([256, 384, 512]),
            "projection_dim": random.choice([128, 256]),
            "temperature": round(random.uniform(0.05, 0.2), 2),
            "use_contrastive": True
        }
    
    def evaluate_population(self, fitness_function: Callable[[Dict], float]):
        """
        Evaluate fitness for all individuals in population.
        
        Args:
            fitness_function: Function that takes architecture and returns fitness
        """
        for individual in self.population:
            if individual.fitness is None:
                # Check cache first
                if self.cache:
                    cached = self.cache.lookup(individual.architecture)
                    if cached and "fitness" in cached:
                        individual.fitness = cached["fitness"]
                        logger.debug(f"Cache hit for individual")
                        continue
                
                # Evaluate using fitness function
                individual.fitness = fitness_function(individual.architecture)
                
                # Cache the result
                if self.cache and individual.fitness is not None:
                    self.cache.store(
                        individual.architecture,
                        {"fitness": individual.fitness}
                    )
        
        # Update best individual
        valid_individuals = [ind for ind in self.population if ind.fitness is not None]
        if valid_individuals:
            self.best_individual = max(valid_individuals, key=lambda x: x.fitness)
    
    def select_parent(self) -> Individual:
        """
        Select a parent using tournament selection.
        
        Returns:
            Selected individual
        """
        # Tournament selection
        tournament = random.sample(
            self.population,
            min(self.tournament_size, len(self.population))
        )
        
        # Select best from tournament (higher fitness is better)
        valid_tournament = [ind for ind in tournament if ind.fitness is not None]
        
        if not valid_tournament:
            # If no valid individuals, select randomly
            return random.choice(tournament)
        
        return max(valid_tournament, key=lambda x: x.fitness)
    
    def mutate(
        self,
        individual: Individual,
        strategy: str = "adaptive"
    ) -> Individual:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            strategy: Mutation strategy
            
        Returns:
            Mutated individual
        """
        arch_type = individual.architecture.get("type", "transformer")
        
        # Use advanced mutation strategy if enabled
        if self.use_advanced_mutation and arch_type == "transformer":
            # Calculate population diversity for adaptive mutation
            diversity = self._calculate_diversity()
            
            # Get best fitness (or use individual's fitness if no best yet)
            best_fitness = self.best_individual.fitness if self.best_individual else individual.fitness
            
            mutated_arch, mutation_desc = self.advanced_mutation_strategy.mutate_transformer_advanced(
                architecture=individual.architecture,
                base_mutation_rate=self.mutation_rate,
                individual_fitness=individual.fitness,
                individual_age=individual.age,
                population_diversity=diversity,
                generation=self.generation,
                best_fitness=best_fitness,
                surrogate_model=self.surrogate_model,
                max_parameters=self.max_parameters
            )
        else:
            # Use basic mutation strategy
            if arch_type == "transformer":
                mutated_arch, mutation_desc = MutationStrategy.mutate_transformer(
                    individual.architecture,
                    self.mutation_rate,
                    strategy,
                    self.surrogate_model,
                    self.generation
                )
            elif arch_type == "cnn":
                mutated_arch, mutation_desc = MutationStrategy.mutate_cnn(
                    individual.architecture,
                    self.mutation_rate,
                    strategy,
                    self.surrogate_model,
                    self.generation
                )
            else:  # multimodal
                mutated_arch, mutation_desc = MutationStrategy.mutate_multimodal(
                    individual.architecture,
                    self.mutation_rate,
                    strategy,
                    self.surrogate_model,
                    self.generation
                )
        
        return Individual(
            architecture=mutated_arch,
            parent_ids=[hash(individual)],
            mutation_history=individual.mutation_history + [mutation_desc]
        )
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Individual:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        # Simple crossover: randomly select parameters from each parent
        child_arch = deepcopy(parent1.architecture)
        
        arch_type = child_arch.get("type")
        
        if arch_type == "transformer":
            # Crossover Transformer parameters
            crossover_params = ["num_layers", "hidden_size", "num_heads", 
                              "ffn_dim", "dropout", "activation"]
        elif arch_type == "cnn":
            # Crossover CNN parameters
            crossover_params = ["num_blocks", "base_channels", "kernel_size",
                              "stride", "activation", "pooling"]
        else:  # multimodal
            # For multimodal, crossover fusion parameters
            crossover_params = ["fusion_dim", "projection_dim", 
                              "fusion_type", "temperature"]
        
        for param in crossover_params:
            if random.random() < 0.5 and param in parent2.architecture:
                child_arch[param] = parent2.architecture[param]
        
        # For multimodal, also crossover sub-encoders
        if arch_type == "multimodal":
            if "vision_encoder" in parent1.architecture and "vision_encoder" in parent2.architecture:
                if random.random() < 0.5:
                    child_arch["vision_encoder"] = deepcopy(parent2.architecture["vision_encoder"])
            
            if "text_encoder" in parent1.architecture and "text_encoder" in parent2.architecture:
                if random.random() < 0.5:
                    child_arch["text_encoder"] = deepcopy(parent2.architecture["text_encoder"])
        
        return Individual(
            architecture=child_arch,
            parent_ids=[hash(parent1), hash(parent2)],
            mutation_history=[]
        )
    
    def evolve(self) -> List[Individual]:
        """
        Evolve the population for one generation.
        
        Returns:
            New population
        """
        new_population = []
        
        # Elitism: preserve top individuals
        sorted_population = sorted(
            [ind for ind in self.population if ind.fitness is not None],
            key=lambda x: x.fitness,
            reverse=True
        )
        
        num_elite = max(1, int(self.population_size * self.elitism_rate))
        elite = sorted_population[:num_elite]
        
        # Increase age of elite individuals
        for ind in elite:
            ind.age += 1
        
        new_population.extend(elite)
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = self.crossover(parent1, parent2)
                
                # Also mutate the child
                if random.random() < 0.5:
                    child = self.mutate(child)
                
                new_population.append(child)
            else:
                # Mutation only
                parent = self.select_parent()
                child = self.mutate(parent)
                new_population.append(child)
        
        # Update population and generation
        self.population = new_population
        self.generation += 1
        
        # Record history
        if self.best_individual:
            self.history.append({
                "generation": self.generation,
                "best_fitness": self.best_individual.fitness,
                "avg_fitness": np.mean([
                    ind.fitness for ind in self.population 
                    if ind.fitness is not None
                ]),
                "population_diversity": self._calculate_diversity()
            })
        
        return self.population
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity (ratio of unique architectures).
        
        Returns:
            Diversity score (0-1)
        """
        if not self.population:
            return 0.0
        
        unique_hashes = len(set(hash(ind) for ind in self.population))
        return unique_hashes / len(self.population)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current evolution statistics.
        
        Returns:
            Dictionary of statistics
        """
        valid_individuals = [
            ind for ind in self.population if ind.fitness is not None
        ]
        
        stats = {
            "generation": self.generation,
            "population_size": len(self.population),
            "evaluated": len(valid_individuals),
            "use_advanced_mutation": self.use_advanced_mutation,
            "max_parameters": self.max_parameters
        }
        
        if not valid_individuals:
            return stats
        
        fitnesses = [ind.fitness for ind in valid_individuals]
        
        stats.update({
            "best_fitness": max(fitnesses),
            "avg_fitness": np.mean(fitnesses),
            "min_fitness": min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "diversity": self._calculate_diversity(),
            "cache_stats": self.cache.get_statistics() if self.cache else None
        })
        
        # Add mutation statistics if using advanced mutation
        if self.use_advanced_mutation and self.advanced_mutation_strategy:
            stats["mutation_stats"] = self.advanced_mutation_strategy.get_mutation_statistics()
        
        return stats
    
    def run(
        self,
        fitness_function: Callable[[Dict], float],
        max_generations: int = 50,
        target_fitness: Optional[float] = None,
        verbose: bool = True
    ) -> Individual:
        """
        Run the genetic algorithm.
        
        Args:
            fitness_function: Function to evaluate architectures
            max_generations: Maximum number of generations
            target_fitness: Optional target fitness (stop early if reached)
            verbose: Print progress
            
        Returns:
            Best individual found
        """
        # Initialize population
        self.initialize_population()
        
        if verbose:
            logger.info(f"Starting evolution with population size {self.population_size}")
        
        for gen in range(max_generations):
            # Evaluate population
            self.evaluate_population(fitness_function)
            
            # Get statistics
            stats = self.get_statistics()
            
            if verbose:
                logger.info(
                    f"Generation {gen+1}/{max_generations} - "
                    f"Best: {stats.get('best_fitness', 0):.2f}, "
                    f"Avg: {stats.get('avg_fitness', 0):.2f}, "
                    f"Diversity: {stats.get('diversity', 0):.2%}"
                )
            
            # Check if target reached
            if target_fitness and self.best_individual:
                if self.best_individual.fitness >= target_fitness:
                    logger.info(f"Target fitness reached: {self.best_individual.fitness:.2f}")
                    break
            
            # Evolve to next generation (except last)
            if gen < max_generations - 1:
                self.evolve()
        
        return self.best_individual
