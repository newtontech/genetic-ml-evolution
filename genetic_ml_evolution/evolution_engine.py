"""
Evolution Engine for Neural Architecture Search
针对小规模语言模型优化的进化引擎

This module implements the main evolution engine that combines:
genetic operators with surrogate model and cache system.

Key features for SLM optimization:
- Adaptive population sizing based on GPU memory
- Resource-aware mutation rates
- Early stopping via surrogate predictions
- Progressive complexity from simple to complex architectures
- Performance caching to avoid redundant evaluations
"""

import random
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
import time

from .genetic_operators import (
    ArchitectureGene,
    SLMutationOperators,
    SLMCrossoverOperators,
    SLMSelectionOperators,
    tournament_selection,
    rank_selection,
    elitism_selection
)
from .surrogate_model import SurrogateModel
from .cache_system import ArchitectureCache

logger = logging.getLogger(__name__)


class EvolutionConfig:
    """Configuration for the evolution engine."""
    
    # Population settings
    population_size: int = 20
    elite_size: int = 3
    
    # Evolution parameters
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    mutation_strength: str = "moderate"  # conservative, moderate, aggressive
    
    # Early stopping
    early_stopping_patience: int = 5
    min_improvement_threshold: float = 1.0
    
    # GPU constraints
    max_gpu_memory_gb: float = 16.0
    
    # Task settings
    task_type: str = "language"  # language, image, multimodal
    dataset: str = "imdb"
    
    # Logging
    log_level: str = "INFO"
    
    # Callback for custom evaluation
    custom_evaluator: Optional[Callable] = None
    
    # Cache settings
    cache_db_path: Optional[str] = None
    use_cache: bool = True
    
    # Surrogate settings
    use_surrogate: bool = True
    surrogate_model_type: str = "ensemble"


class EvolutionEngine:
    """
    Main evolution engine for neural architecture search.
    
    Optimized for Small Language Models (SLM) with:
    - Reduced computational requirements
    - Adaptive mutation strategies
    - Efficient memory usage
    - Performance caching to avoid redundant evaluations
    """
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        """
        Initialize the evolution engine.
        
        Args:
            config: Evolution configuration. If None, uses defaults.
        """
        self.config = config or EvolutionConfig()
        
        # Initialize operators
        self.mutator = SLMutationOperators(
            mutation_rate=self.config.mutation_rate,
            mutation_strength=self.config.mutation_strength
        )
        self.crossover = SLMCrossoverOperators(
            crossover_rate=self.config.crossover_rate
        )
        self.selector = SLMSelectionOperators()
        
        # Initialize surrogate model
        self.surrogate = SurrogateModel(
            model_type=self.config.surrogate_model_type,
            cache_db_path=self.config.cache_db_path
        ) if self.config.use_surrogate else None
        
        # Initialize cache if configured
        if self.config.use_cache and self.config.cache_db_path:
            self.cache = ArchitectureCache(self.config.cache_db_path)
            logger.info(f"Cache system initialized at {self.config.cache_db_path}")
        else:
            self.cache = None
        
        # Population
        self.population: List[ArchitectureGene] = []
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_architecture: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        
        # Time tracking
        self.start_time = time.time()
        self.last_improvement_time = self.start_time
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"EvolutionEngine initialized with config: "
            f"population_size={self.config.population_size}, "
            f"generations={self.config.generations}, "
            f"task_type={self.config.task_type}, "
            f"cache_enabled={self.cache is not None}"
        )
    
    def _initialize_population(self) -> None:
        """Initialize the population with random architectures."""
        arch_type_map = {
            "language": self._random_transformer,
            "image": self._random_cnn,
            "multimodal": self._random_multimodal,
        }
        
        if self.config.task_type not in arch_type_map:
            raise ValueError(f"Unknown task type: {self.config.task_type}")
        
        generator = arch_type_map[self.config.task_type]
        
        for _ in range(self.config.population_size):
            arch = generator()
            gene = ArchitectureGene(arch)
            self.population.append(gene)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def _random_transformer(self) -> Dict[str, Any]:
        """Generate a random transformer architecture."""
        return {
            "type": "transformer",
            "num_layers": random.randint(2, 12),
            "hidden_size": random.choice([128, 256, 384, 512, 768]),
            "num_heads": random.choice([2, 4, 8, 12]),
            "ffn_dim": random.choice([256, 512, 1024, 2048, 3072]),
            "dropout": round(random.uniform(0, 0.3), 2) / 10,
            "activation": random.choice(["relu", "gelu", "silu"]),
            "vocab_size": random.choice([10000, 30000, 50000]),
            "max_seq_len": random.choice([128, 256, 512]),
        }
    
    def _random_cnn(self) -> Dict[str, Any]:
        """Generate a random CNN architecture."""
        return {
            "type": "cnn",
            "num_blocks": random.randint(2, 8),
            "base_channels": random.choice([16, 32, 64, 128]),
            "kernel_size": random.choice([3, 5, 7]),
            "stride": random.choice([1, 2]),
            "activation": random.choice(["relu", "leaky_relu", "silu"]),
            "pooling": random.choice(["max", "avg", "adaptive"]),
            "use_batch_norm": random.choice([True, False]),
        }
    
    def _random_multimodal(self) -> Dict[str, Any]:
        """Generate a random multimodal architecture."""
        return {
            "type": "multimodal",
            "vision_encoder": {
                "num_blocks": random.randint(2, 6),
                "base_channels": random.choice([16, 32, 64]),
            },
            "text_encoder": {
                "num_layers": random.randint(2, 8),
                "hidden_size": random.choice([128, 256, 384, 512]),
            },
            "fusion_dim": random.choice([128, 256, 384, 512]),
            "projection_dim": random.choice([64, 128, 256]),
            "fusion_type": random.choice(["concat", "attention", "bilinear", "cross"]),
            "temperature": round(random.uniform(0.01, 0.5), 2),
            "use_contrastive": random.choice([True, False]),
        }
    
    def evolve(
        self, 
        fitness_function: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the evolution process.
        
        Args:
            fitness_function: Optional custom fitness function.
                Signature: (architecture: Dict) -> float
            callback: Optional callback function called each generation.
                Signature: (stats: Dict) -> None
            
        Returns:
            Dictionary with evolution results including:
                - best_fitness: Best fitness achieved
                - best_architecture: Best architecture found
                - generations: Number of generations run
                - history: List of generation statistics
                - total_time: Total evolution time
                - cache_stats: Cache statistics (if cache enabled)
        """
        # Initialize population if not already done
        if not self.population:
            self._initialize_population()
        
        logger.info(f"Starting evolution for {self.config.generations} generations")
        self.start_time = time.time()
        
        for gen in range(self.config.generations):
            self.generation = gen + 1
            gen_start = time.time()
            
            # Evaluate population (with cache integration)
            self._evaluate_population(fitness_function)
            
            # Log current generation stats
            fitnesses = [g.fitness for g in self.population if g.fitness is not None]
            if fitnesses:
                best_gen_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                
                logger.info(
                    f"Generation {gen + 1}/{self.config.generations}: "
                    f"Best={best_gen_fitness:.2f}, "
                    f"Avg={avg_fitness:.2f}, "
                    f"Cache hits={self.cache_hits}, "
                    f"Cache misses={self.cache_misses}"
                )
                
                # Update best overall
                if best_gen_fitness > self.best_fitness:
                    self.best_fitness = best_gen_fitness
                    best_gene = max(self.population, key=lambda g: g.fitness or 0)
                    self.best_architecture = best_gene.architecture.copy()
                    self.last_improvement_time = time.time()
            
            # Check for early stopping
            if self._check_early_stopping():
                logger.info(f"Early stopping triggered at generation {gen + 1}")
                break
            
            # Selection
            parents = self.selector.tournament_selection(
                self.population, 
                tournament_size=min(self.config.population_size, 2)
            )
            
            # Create next generation
            next_gen = []
            for i in range(0, len(self.population) - 1, 2):
                parent1 = parents[i] if i < len(parents) else parents[0]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                
                # Crossover
                child_arch = self.crossover.crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.config.mutation_rate:
                    child_gene = ArchitectureGene(child_arch)
                    mutated = self.mutator.mutate(child_gene)
                    child_arch = mutated.architecture if hasattr(mutated, 'architecture') else mutated
                
                child_gene = ArchitectureGene(child_arch)
                next_gen.append(child_gene)
            
            # Ensure population size is maintained
            while len(next_gen) < self.config.population_size:
                # Add random individual if population is too small
                arch = self._random_transformer() if self.config.task_type == "language" else self._random_cnn()
                next_gen.append(ArchitectureGene(arch))
            
            # Replace population (keeping elites if configured)
            if self.config.elite_size > 0 and fitnesses:
                sorted_pop = sorted(self.population, key=lambda g: g.fitness or 0, reverse=True)
                elites = sorted_pop[:self.config.elite_size]
                next_gen = elites + next_gen[:self.config.population_size - self.config.elite_size]
            
            self.population = next_gen
            
            # Record history
            self.history.append({
                "generation": gen + 1,
                "best_fitness": self.best_fitness,
                "avg_fitness": avg_fitness if fitnesses else 0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "timestamp": time.time() - self.start_time
            })
            
            # Callback
            if callback:
                callback({
                    "generation": gen + 1,
                    "best_fitness": self.best_fitness,
                    "best_architecture": self.best_architecture,
                    "population_size": len(self.population),
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                })
        
        total_time = time.time() - self.start_time
        logger.info(
            f"Evolution completed in {total_time:.2f}s. "
            f"Best fitness: {self.best_fitness:.2f}"
        )
        
        result = {
            "best_fitness": self.best_fitness,
            "best_architecture": self.best_architecture,
            "generations": self.generation,
            "history": self.history,
            "total_time": total_time,
        }
        
        # Add cache statistics if available
        if self.cache:
            result["cache_stats"] = self.cache.get_statistics()
        
        return result
    
    def _evaluate_population(
        self, 
        fitness_function: Optional[Callable] = None
    ) -> None:
        """
        Evaluate fitness of all individuals in the population.
        
        Uses cache to avoid redundant evaluations when possible.
        Cache lookup order:
        1. Check cache for existing result
        2. If not cached, evaluate using fitness_function, surrogate, or random
        3. Store new results in cache
        """
        for gene in self.population:
            arch = gene.architecture
            
            # Try cache first
            if self.cache:
                cached_result = self.cache.lookup(arch)
                if cached_result is not None:
                    # Cache hit
                    gene.fitness = cached_result.get("fitness") or cached_result.get("accuracy")
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for architecture: {arch.get('type', 'unknown')}")
                    continue
                else:
                    # Cache miss
                    self.cache_misses += 1
            
            # Evaluate fitness
            eval_start = time.time()
            
            if fitness_function:
                # Use custom fitness function
                gene.fitness = fitness_function(arch)
            elif self.surrogate and self.surrogate.is_fitted:
                # Use surrogate model
                prediction = self.surrogate.predict(arch)
                gene.fitness = prediction
            else:
                # Fallback to random fitness (for testing)
                gene.fitness = random.uniform(50, 80)
            
            eval_time = time.time() - eval_start
            
            # Store result in cache
            if self.cache and gene.fitness is not None:
                metrics = {
                    "fitness": gene.fitness,
                    "accuracy": gene.fitness,  # Also store as accuracy for compatibility
                    "evaluation_time": eval_time
                }
                self.cache.store(arch, metrics, evaluation_time=eval_time)
                logger.debug(f"Cached new result for architecture: {arch.get('type', 'unknown')}")
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping should be triggered."""
        if self.generation < self.config.early_stopping_patience:
            return False
        
        # Calculate improvement over recent generations
        if len(self.history) < self.config.early_stopping_patience:
            return False
        
        recent_history = self.history[-self.config.early_stopping_patience:]
        recent_best = max(h["best_fitness"] for h in recent_history)
        older_best = self.history[-self.config.early_stopping_patience - 1]["best_fitness"] if len(self.history) > self.config.early_stopping_patience else recent_best
        
        if older_best == 0:
            return False
        
        improvement = (recent_best - older_best) / abs(older_best) * 100
        
        logger.debug(f"Early stopping check: improvement={improvement:.2f}%")
        
        return improvement < self.config.min_improvement_threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evolution statistics.
        
        Returns:
            Dictionary containing:
                - generation: Current generation
                - population_size: Current population size
                - best_fitness: Best fitness achieved
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - cache_stats: Detailed cache statistics (if cache enabled)
                - surrogate_stats: Surrogate model statistics (if surrogate enabled)
        """
        stats = {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) 
                if (self.cache_hits + self.cache_misses) > 0 else 0.0,
        }
        
        if self.cache:
            cache_stats = self.cache.get_statistics()
            stats["cache_stats"] = cache_stats
        
        if self.surrogate:
            stats["surrogate_stats"] = {
                "is_fitted": self.surrogate.is_fitted,
                "model_type": self.surrogate.model_type,
            }
        
        return stats
    
    def close(self) -> None:
        """Clean up resources."""
        if self.cache:
            self.cache.close()
            logger.info("Cache connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EvolutionEngine("
            f"generation={self.generation}, "
            f"population_size={len(self.population)}, "
            f"best_fitness={self.best_fitness:.2f}, "
            f"cache_enabled={self.cache is not None})"
        )
