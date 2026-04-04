# Proxy Models for Accelerating Neural Architecture Search

> A practical guide to using surrogate (proxy) models in `genetic-ml-evolution` to dramatically reduce the cost of architecture evaluation during NAS.

## Table of Contents

- [What Are Proxy Models?](#what-are-proxy-models)
- [Why Use Proxy Models in NAS?](#why-use-proxy-models-in-nas)
- [How Proxy Models Work in This Project](#how-proxy-models-work-in-this-project)
- [Configuration Reference](#configuration-reference)
- [Code Examples](#code-examples)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## What Are Proxy Models?

Proxy models (also called **surrogate models** or **performance predictors**) are lightweight machine learning models that **predict the performance of a neural architecture without actually training it**.

Instead of spending hours or days training each candidate architecture, you:

1. **Train a small number** of architectures to collect real performance data
2. **Fit a surrogate model** on that data
3. **Use the surrogate** to predict performance for the remaining candidates
4. **Periodically validate** predictions with real training (warm-starting the surrogate)

This approach is the foundation of efficient Neural Architecture Search (NAS).

### Analogy

Think of it like hiring: instead of giving every candidate a 3-month trial, you use a structured interview process (the proxy) to predict who will perform well, then only do full trials for the top candidates.

---

## Why Use Proxy Models in NAS?

### The Problem

In evolutionary NAS, each generation requires evaluating **every individual** in the population:

| Scenario | Population | Generations | Time per Evaluation | Total Time |
|----------|-----------|-------------|--------------------|----|
| Small CNN on CIFAR-10 | 20 | 50 | 10 min | ~167 hours |
| Transformer on IMDB | 20 | 50 | 30 min | ~500 hours |
| Multimodal model | 20 | 50 | 60 min | ~1000 hours |

**That's weeks to months of GPU time for a single experiment.**

### The Solution

With proxy models, you only train a fraction of architectures:

| Scenario | Real Evaluations | Surrogate Predictions | Speedup |
|----------|-----------------|----------------------|---------|
| Small CNN on CIFAR-10 | ~100 (initial) | ~900 | **~10×** |
| Transformer on IMDB | ~100 (initial) | ~900 | **~10×** |
| Multimodal model | ~100 (initial) | ~900 | **~10×** |

Combined with the cache system, the effective speedup can reach **90%+ reduction in total training time**.

---

## How Proxy Models Work in This Project

### Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Evolution Engine                      │
│                                                       │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │  Population  │───▶│  Fitness Fn  │                 │
│  │  (Archs)     │    │              │                 │
│  └─────────────┘    └──────┬───────┘                 │
│                             │                          │
│                    ┌────────▼────────┐                │
│                    │  Cache Lookup   │─── Hit? ──▶ Return cached result
│                    └────────┬────────┘                │
│                             │ Miss                     │
│                    ┌────────▼────────┐                │
│                    │ Surrogate Model │─── Fitted? ──▶ Return prediction
│                    └────────┬────────┘                │
│                             │ Unfitted / Low confidence │
│                    ┌────────▼────────┐                │
│                    │  Real Training  │───▶ Store in cache + surrogate
│                    └─────────────────┘                │
└──────────────────────────────────────────────────────┘
```

### Key Components

1. **`SurrogateModel`** (`genetic_ml_evolution/surrogate_model.py`)
   - Ensemble of ML models: Random Forest, Gradient Boosting, MLP
   - Automatically selects the best model via cross-validation
   - Encodes architectures into feature vectors (handles Transformer, CNN, Multimodal)

2. **`ArchitectureCache`** (`genetic_ml_evolution/cache_system.py`)
   - SQLite-based persistent cache
   - Stores architecture → performance mappings
   - Avoids re-evaluating identical or previously-seen architectures

3. **`EvolutionEngine`** (`genetic_ml_evolution/evolution_engine.py`)
   - Orchestrates the evolution loop
   - Integrates surrogate predictions and cache lookups
   - Manages the training data lifecycle for the surrogate

### Architecture Encoding

The surrogate model encodes each architecture type into a fixed-length feature vector:

| Architecture Type | Feature Count | Features |
|------------------|---------------|----------|
| **Transformer** | 11 | num_layers, hidden_size, num_heads, ffn_dim, dropout, vocab_size, max_seq_len, activation (one-hot) |
| **CNN** | 14 | num_blocks, base_channels, kernel_size, stride, use_batch_norm, num_classes, input_channels, input_size, activation + pooling (one-hot) |
| **Multimodal** | 12 | vision blocks/channels, text layers/hidden_size, fusion_dim, projection_dim, temperature, use_contrastive, fusion_type (one-hot) |

---

## Configuration Reference

### EvolutionConfig

```python
from genetic_ml_evolution import EvolutionEngine, EvolutionConfig

config = EvolutionConfig()

# Surrogate model settings
config.use_surrogate = True                    # Enable/disable surrogate model
config.surrogate_model_type = "ensemble"       # "rf", "gb", "mlp", or "ensemble"

# Cache settings
config.use_cache = True                        # Enable/disable caching
config.cache_db_path = "architecture_cache.db" # SQLite cache file path

# Evolution settings
config.population_size = 20                    # Number of individuals per generation
config.generations = 50                        # Maximum number of generations
config.mutation_rate = 0.2                     # Probability of mutation
config.crossover_rate = 0.7                    # Probability of crossover
config.elite_size = 3                          # Number of elites to preserve
```

### SurrogateModel Direct Configuration

```python
from genetic_ml_evolution import SurrogateModel

surrogate = SurrogateModel(
    model_type="ensemble",    # "rf" | "gb" | "mlp" | "ensemble"
    cache_db_path="cache.db"  # Optional: enable caching
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `str` | `"ensemble"` | Which ML model(s) to use |
| `cache_db_path` | `str \| None` | `None` | Path to SQLite cache database |

### Model Type Comparison

| Model | Speed | Accuracy | Data Needed | Best For |
|-------|-------|----------|-------------|----------|
| `rf` (Random Forest) | Fast | Medium | ~10 samples | Quick iteration |
| `gb` (Gradient Boosting) | Medium | High | ~20 samples | Final optimization |
| `mlp` (MLP Regressor) | Medium | Medium-High | ~30 samples | Complex patterns |
| `ensemble` | Slow (trains all) | Best | ~20 samples | Production use |

---

## Code Examples

### Example 1: Basic Surrogate-Accelerated Evolution

```python
from genetic_ml_evolution import GeneticAlgorithm, SurrogateModel
import tempfile

# Create a surrogate model with cache
surrogate = SurrogateModel(model_type="ensemble", cache_db_path="cache.db")

# Create a GA with surrogate model
ga = GeneticAlgorithm(
    population_size=20,
    mutation_rate=0.3,
    task_type="language",
    surrogate_model=surrogate
)

# Fitness function that uses surrogate predictions
def fitness_fn(architecture):
    # Try surrogate prediction first
    if surrogate.is_fitted:
        prediction = surrogate.predict(architecture)
        if prediction is not None:
            return prediction  # No training needed!

    # Fallback: real evaluation
    real_fitness = expensive_training(architecture)

    # Feed back to surrogate
    surrogate.add_training_point(architecture, real_fitness)
    if len(surrogate.training_data) >= 5:
        surrogate.fit()

    return real_fitness

# Run evolution
best = ga.run(fitness_function=fitness_fn, max_generations=50, verbose=True)
print(f"Best architecture: {best.architecture}")
print(f"Best fitness: {best.fitness:.2f}")
```

### Example 2: EvolutionEngine with Built-in Surrogate

```python
from genetic_ml_evolution import EvolutionEngine, EvolutionConfig

config = EvolutionConfig(
    population_size=20,
    generations=50,
    task_type="language",
    dataset="imdb",
    use_surrogate=True,
    surrogate_model_type="ensemble",
    use_cache=True,
    cache_db_path="nas_cache.db",
    max_gpu_memory_gb=16.0
)

engine = EvolutionEngine(config)
best_model = engine.evolve()
```

### Example 3: Manual Surrogate Workflow

```python
from genetic_ml_evolution import SurrogateModel
import numpy as np

surrogate = SurrogateModel(model_type="ensemble")

# Step 1: Collect initial training data (warm-start)
initial_architectures = [
    {"type": "transformer", "num_layers": 4, "hidden_size": 256, "num_heads": 4,
     "ffn_dim": 1024, "dropout": 0.1, "activation": "gelu", "vocab_size": 50000, "max_seq_len": 512},
    {"type": "transformer", "num_layers": 6, "hidden_size": 512, "num_heads": 8,
     "ffn_dim": 2048, "dropout": 0.1, "activation": "gelu", "vocab_size": 50000, "max_seq_len": 512},
    {"type": "transformer", "num_layers": 8, "hidden_size": 384, "num_heads": 6,
     "ffn_dim": 1536, "dropout": 0.15, "activation": "silu", "vocab_size": 50000, "max_seq_len": 256},
    # ... more architectures
]

# Step 2: Evaluate each architecture (real training) and record results
for arch in initial_architectures:
    fitness = evaluate_architecture(arch)  # Your real training function
    surrogate.add_training_point(arch, fitness)

# Step 3: Fit the surrogate model
success = surrogate.fit()
print(f"Model fitted: {success}")
print(f"Best model: {surrogate.best_model_name} (MSE: {surrogate.best_score:.4f})")

# Step 4: Use surrogate to screen candidates
candidate_architectures = generate_candidates(1000)  # Generate many candidates
predictions = surrogate.predict_batch(candidate_architectures)

# Step 5: Only train the top-k candidates
top_k = 10
top_indices = np.argsort(predictions)[-top_k:][::-1]
for idx in top_indices:
    arch = candidate_architectures[idx]
    real_fitness = evaluate_architecture(arch)
    surrogate.add_training_point(arch, real_fitness)

# Step 6: Re-fit with new data
surrogate.fit()
```

### Example 4: CNN Architecture Search with Surrogate

```python
from genetic_ml_evolution import GeneticAlgorithm, SurrogateModel

surrogate = SurrogateModel(model_type="gb", cache_db_path="cnn_cache.db")

ga = GeneticAlgorithm(
    population_size=30,
    mutation_rate=0.3,
    task_type="image",  # CNN search space
    surrogate_model=surrogate
)

def cnn_fitness(architecture):
    if surrogate.is_fitted:
        pred = surrogate.predict(architecture)
        if pred is not None:
            return pred

    # Real CIFAR-10 evaluation
    accuracy = train_and_evaluate_cnn(architecture)
    surrogate.add_training_point(architecture, accuracy)
    if len(surrogate.training_data) >= 5:
        surrogate.fit()
    return accuracy

best_cnn = ga.run(fitness_function=cnn_fitness, max_generations=30, verbose=True)
```

---

## Performance Comparison

### Estimated Speedup by Scenario

| Task | Without Surrogate | With Surrogate | Speedup | GPU Hours Saved |
|------|-------------------|---------------|---------|-----------------|
| CNN on CIFAR-10 (30 gen × 20 pop) | ~100 hrs | ~10-15 hrs | **7-10×** | ~85-90 hrs |
| Transformer on IMDB (50 gen × 20 pop) | ~500 hrs | ~50-75 hrs | **7-10×** | ~425-450 hrs |
| Multimodal (50 gen × 20 pop) | ~1000 hrs | ~100-150 hrs | **7-10×** | ~850-900 hrs |

### Accuracy Impact

| Phase | Surrogate Accuracy | Notes |
|-------|-------------------|-------|
| First 5 generations | N/A (real eval) | Warm-start phase, no predictions used |
| Generations 6-15 | ~70-80% correlation | Surrogate learning architecture-performance relationship |
| Generations 16+ | ~85-92% correlation | High confidence, most evaluations are predictions |
| Final validation | N/A (real eval) | Always validate top candidates with real training |

### Cache Hit Rate (Typical)

| Generation | Cache Hit Rate | Surrogate Hit Rate | Real Evaluations |
|------------|---------------|-------------------|-----------------|
| 1-3 | 0% | 0% | 100% |
| 4-10 | 20-40% | 40-60% | 20-40% |
| 11-30 | 40-60% | 60-80% | 10-25% |
| 31-50 | 50-70% | 70-90% | 5-15% |

> **Note**: These are estimates based on typical NAS workloads. Actual performance depends on search space diversity, population size, and task complexity.

---

## Best Practices

### 1. Warm-Start with Seed Architectures

```python
seed_archs = [
    {"type": "transformer", "num_layers": 6, "hidden_size": 512, ...},
    {"type": "transformer", "num_layers": 4, "hidden_size": 256, ...},
]

ga.initialize_population(seed_architectures=seed_archs)
```

Start with known-good architectures to give the surrogate quality training data from the start.

### 2. Use Ensemble Mode for Production

```python
surrogate = SurrogateModel(model_type="ensemble")  # Best accuracy
```

Ensemble mode trains all three models and picks the best via cross-validation. Slightly slower to fit but most accurate predictions.

### 3. Always Validate Final Results

The surrogate predicts performance — it doesn't guarantee it. **Always** re-train and evaluate the top architectures from the final generation with real training.

### 4. Combine Surrogate + Cache

```python
surrogate = SurrogateModel(model_type="ensemble", cache_db_path="cache.db")
```

The cache prevents re-evaluating identical architectures, while the surrogate predicts performance for novel architectures. Together they provide maximum speedup.

### 5. Periodically Re-fit the Surrogate

The surrogate becomes more accurate as more data is collected. The `EvolutionEngine` handles this automatically, but if you're using the `SurrogateModel` directly, call `fit()` after adding new training points.

### 6. Use Constrained Fitness Functions

```python
def constrained_fitness(arch):
    base = real_fitness(arch)
    params = estimate_params(arch)
    if params > 100_000_000:  # 100M parameter limit
        base -= (params / 100_000_000 - 1) * 10
    return max(0, base)
```

This ensures the surrogate learns to predict performance under your actual constraints.

---

## Troubleshooting

### "Insufficient training data: X samples. Need at least 5 samples."

The surrogate needs at least 5 real evaluations before it can be fitted. This is expected in the first few generations.

**Solution**: Let the evolution run for at least 5 evaluations, or pre-populate with seed architectures.

### Surrogate predictions seem inaccurate

Common causes:
1. **Too few training samples** — Collect more data (10-20 minimum for decent accuracy)
2. **Search space too diverse** — Narrow the search space or use seed architectures
3. **Noisy fitness function** — Reduce training noise (fixed seeds, more epochs)

### Cache hit rate is low

This means the search is exploring diverse architectures. This is normal for early generations and not a problem — the surrogate handles novel architectures.

### Memory usage is high

The surrogate models are lightweight (typically < 100MB). If you see high memory usage, check:
1. The cache database size (can grow large over many experiments)
2. The training data stored in `surrogate.training_data`

To reset: `surrogate.training_data.clear()` and `surrogate.is_fitted = False`.

---

## Related Documentation

- [Surrogate Model Guide](./surrogate-model-guide.md) — Detailed API reference for the surrogate model
- [SLM Mutation Optimization](./slm-mutation-optimization-summary.md) — Mutation strategies for SLM NAS
- [Genetic Algorithm Example](../examples/genetic_algorithm_example.py) — Full working examples

---

*Last updated: 2026-04-05*
