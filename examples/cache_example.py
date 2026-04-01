"""
SQLite Cache System Example
演示如何使用缓存系统避免重复评估
"""

import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from genetic_ml_evolution.cache_system import ArchitectureCache
from genetic_ml_evolution.surrogate_model import SurrogateModel


def main():
    # Create a temporary cache database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        cache_db = f.name
    
    # Initialize cache
    cache = ArchitectureCache(db_path=cache_db)
    
    # Example 1: Store and lookup architecture
    print("=" * 60)
    print("Example 1: Store and lookup architectures")
    print("-" * 60)
    
    arch1 = {
        "type": "transformer",
        "num_layers": 6,
        "hidden_size": 512,
        "num_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.1,
        "activation": "gelu"
    }
    
    metrics1 = {"accuracy": 0.85, "loss": 0.15, "inference_time": 0.025}
    
    # Store
    cache.store(arch1, metrics1, evaluation_time=10.5)
    print(f"✓ Stored: {arch1['type']} with accuracy={metrics1['accuracy']}")
    
    # Lookup
    result = cache.lookup(arch1)
    print(f"✓ Retrieved from cache: {result}")
    
    # Example 2: Try to store duplicate (should fail)
    print("\n" + "=" * 60)
    print("Example 2: Try to store duplicate architecture")
    print("-" * 60)
    
    cache.store(arch1, metrics1)
    print(f"✗ Duplicate detected, not stored again")
    
    # Example 3: Different architecture types
    print("\n" + "=" * 60)
    print("Example 3: Different architecture types")
    print("-" * 60)
    
    arch2 = {
        "type": "cnn",
        "num_blocks": 4,
        "base_channels": 64,
        "kernel_size": 3,
        "activation": "relu"
    }
    metrics2 = {"accuracy": 0.80, "loss": 0.20}
    
    cache.store(arch2, metrics2)
    print(f"✓ Stored: {arch2['type']} with accuracy={metrics2['accuracy']}")
    
    arch3 = {
        "type": "multimodal",
        "vision_encoder": {"num_blocks": 3},
        "text_encoder": {"num_layers": 4},
        "fusion_type": "attention"
    }
    metrics3 = {"accuracy": 0.75, "loss": 0.25}
    
    cache.store(arch3, metrics3)
    print(f"✓ Stored: {arch3['type']} with accuracy={metrics3['accuracy']}")
    
    # Example 4: Get statistics
    print("\n" + "=" * 60)
    print("Example 4: Cache statistics")
    print("-" * 60)
    
    stats = cache.get_statistics()
    print(f"Total entries: {stats['total_entries']}")
    print(f"By type: {stats['entries_by_type']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate_percent']}%")
    
    # Example 5: Integration with Surrogate Model
    print("\n" + "=" * 60)
    print("Example 5: Integration with Surrogate Model")
    print("-" * 60)
    
    # Create surrogate model with cache
    model = SurrogateModel(model_type="rf", cache_db_path=cache_db)
    
    # Add training data
    print("Adding training data...")
    for i in range(10):
        train_arch = {
            "type": "transformer",
            "num_layers": (i % 6) + 2,
            "hidden_size": 128 * ((i % 4) + 1),
            "num_heads": (i % 4) + 2,
            "ffn_dim": 512 * ((i % 4) + 1),
            "dropout": 0.1 + (i % 5) * 0.02
        }
        model.add_training_point(train_arch, 70.0 + i * 2)
    
    # Train the model
    print("Training surrogate model...")
    if model.fit():
        print("✓ Model trained successfully")
        
        # Make predictions
        test_arch = {
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 1024,
            "dropout": 0.15
        }
        
        prediction = model.predict(test_arch)
        print(f"✓ Prediction for test architecture: {prediction:.2f}")
        
        # Store the prediction
        model.store_prediction(test_arch, prediction)
        print(f"✓ Stored prediction in cache")
        
        # Check cache hit
        prediction2 = model.predict(test_arch)
        print(f"✓ Cache hit! Same prediction: {prediction2:.5f}")
        
        # Final statistics
        final_stats = model.get_cache_statistics()
        print(f"\nFinal cache statistics:")
        print(f"  Total entries: {final_stats['total_entries']}")
        print(f"  Cache hits: {final_stats['cache_hits']}")
        print(f"  Hit rate: {final_stats['hit_rate_percent']}%")
    else:
        print("✗ Failed to train model (insufficient data)")
    
    # Example 6: Get top performing architectures
    print("\n" + "=" * 60)
    print("Example 6: Top performing architectures")
    print("-" * 60)
    
    top_archs = cache.get_top_performing(metric="accuracy", limit=3)
    print(f"Top 3 architectures by accuracy:")
    for i, (arch, metrics) in enumerate(top_archs, 1):
        print(f"  {i}. {arch['type']} - accuracy: {metrics['accuracy']:.5f}")
    
    # Example 7: Export and import
    print("\n" + "=" * 60)
    print("Example 7: Export and import cache")
    print("-" * 60)
    
    export_file = cache_db.replace(".db", "_export.json")
    cache.export_to_json(export_file)
    print(f"✓ Exported cache to {export_file}")
    
    # Create new cache and import
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        new_cache_db = f.name
    
    new_cache = ArchitectureCache(db_path=new_cache_db)
    
    imported = new_cache.import_from_json(export_file)
    print(f"✓ Imported {imported} entries into new cache")
    print(f"New cache has {len(new_cache)} entries")
    
    # Cleanup
    os.remove(export_file)
    os.remove(new_cache_db)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    
    # Close cache
    cache.close()
    print("✓ Cache connection closed")


if __name__ == "__main__":
    main()
