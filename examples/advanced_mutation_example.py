"""
Example: Using Advanced Mutation Strategies for Small-Scale Language Models

This example demonstrates the advanced mutation strategies optimized for small-scale
language models (10M-100M parameters).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_ml_evolution.advanced_mutation import AdvancedMutationStrategy
import numpy as np


def print_architecture(arch, strategy=None):
    """Pretty print architecture"""
    print("\n" + "="*60)
    print("Architecture:")
    for key, value in arch.items():
        print(f"  {key:15s}: {value}")
    
    if strategy and arch.get("type") == "transformer":
        params = strategy.estimate_parameters(arch)
        print(f"\n  Estimated Parameters: {params:,}")
        print(f"  Model Size: {params / 1_000_000:.1f}M")
    print("="*60)


def main():
    print("🧬 Advanced Mutation Strategy Example")
    print("="*60)
    
    # Initialize strategy
    strategy = AdvancedMutationStrategy(ucb_alpha=1.0)
    
    # Example 1: Small-scale language model architecture
    print("\n\n📊 Example 1: Small Language Model (≈20M parameters)")
    print("-"*60)
    
    small_arch = {
        "type": "transformer",
        "num_layers": 4,
        "hidden_size": 256,
        "num_heads": 4,
        "ffn_dim": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "vocab_size": 10000,
        "max_seq_len": 256
    }
    
    print_architecture(small_arch, strategy)
    
    # Apply mutations with different phases
    print("\n\n🔄 Mutation Examples:")
    print("-"*60)
    
    phases = [
        ("Exploration (Gen 10)", 10, 0.8),
        ("Balanced (Gen 40)", 40, 0.5),
        ("Exploitation (Gen 80)", 80, 0.3)
    ]
    
    for phase_name, generation, mutation_rate in phases:
        print(f"\n{phase_name}:")
        mutated, desc = strategy.mutate_transformer_advanced(
            architecture=small_arch,
            base_mutation_rate=mutation_rate,
            individual_fitness=70.0,
            individual_age=3,
            population_diversity=0.5,
            generation=generation,
            best_fitness=100.0,
            max_parameters=50_000_000
        )
        
        print(f"  Mutation: {desc}")
        if desc != "no_change":
            print_architecture(mutated, strategy)
    
    # Example 2: Adaptive mutation rates
    print("\n\n📈 Example 2: Adaptive Mutation Rates")
    print("-"*60)
    
    scenarios = [
        ("High fitness, old individual", 95.0, 15, 0.7),
        ("Low fitness, young individual", 50.0, 2, 0.3),
        ("Medium fitness, low diversity", 75.0, 5, 0.2)
    ]
    
    for scenario_name, fitness, age, diversity in scenarios:
        rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=0.2,
            individual_fitness=fitness,
            individual_age=age,
            population_diversity=diversity,
            generation=30,
            best_fitness=100.0
        )
        print(f"\n{scenario_name}:")
        print(f"  Fitness: {fitness}, Age: {age}, Diversity: {diversity}")
        print(f"  → Adaptive Mutation Rate: {rate:.3f} (base: 0.2)")
    
    # Example 3: Evolution simulation
    print("\n\n🚀 Example 3: Simulated Evolution (50 generations)")
    print("-"*60)
    
    # Reset statistics
    strategy = AdvancedMutationStrategy(ucb_alpha=1.0)
    
    current_arch = small_arch.copy()
    best_fitness = 70.0
    
    print("\nGeneration | Best Fitness | Diversity | Phase        | Mutation Rate | Avg Params")
    print("-" * 85)
    
    for gen in range(0, 50, 5):
        # Simulate evolution
        diversity = 0.8 - gen * 0.01  # Diversity decreases
        fitness = best_fitness + gen * 0.5  # Fitness improves
        
        # Calculate adaptive rate
        rate = strategy.calculate_adaptive_mutation_rate(
            base_rate=0.3,
            individual_fitness=fitness,
            individual_age=min(gen, 10),
            population_diversity=diversity,
            generation=gen,
            best_fitness=best_fitness + gen * 0.5
        )
        
        # Mutate
        mutated, desc = strategy.mutate_transformer_advanced(
            architecture=current_arch,
            base_mutation_rate=rate,
            individual_fitness=fitness,
            individual_age=min(gen, 10),
            population_diversity=diversity,
            generation=gen,
            best_fitness=best_fitness + gen * 0.5,
            max_parameters=50_000_000
        )
        
        phase = strategy.get_mutation_phase(gen)
        params = strategy.estimate_parameters(mutated) / 1_000_000
        
        print(f"{gen:9d} | {fitness:12.1f} | {diversity:8.2f} | {phase:12s} | {rate:13.3f} | {params:10.1f}M")
        
        current_arch = mutated
    
    # Example 4: Mutation statistics
    print("\n\n📊 Example 4: Mutation Operation Statistics")
    print("-"*60)
    
    # Run many mutations
    for _ in range(200):
        mutated, _ = strategy.mutate_transformer_advanced(
            architecture=small_arch,
            base_mutation_rate=0.5,
            generation=30,
            max_parameters=50_000_000
        )
    
    stats = strategy.get_mutation_statistics()
    
    print(f"\nTotal Mutations Attempted: {stats['total_mutations']}")
    print(f"Successful Mutations: {stats['total_successes']}")
    print(f"Overall Success Rate: {stats['overall_success_rate']:.2%}")
    
    print("\nOperation-wise Statistics:")
    print(f"{'Operation':<15} | {'Count':>6} | {'Successes':>9} | {'Rate':>6}")
    print("-" * 50)
    
    for op, op_stats in sorted(stats['operation_stats'].items()):
        print(f"{op:<15} | {op_stats['count']:6d} | {op_stats['successes']:9d} | {op_stats['success_rate']:6.2%}")
    
    print("\n\n✅ Example completed successfully!")
    print("\nKey Takeaways:")
    print("1. Adaptive mutation rates adjust based on fitness, age, diversity, and generation")
    print("2. Layered strategies (exploration → exploitation) guide evolution phases")
    print("3. UCB-based operation selection learns from mutation success rates")
    print("4. Parameter budget prevents oversized models")
    print("5. Small model bias prefers efficient architectures")


if __name__ == "__main__":
    main()
