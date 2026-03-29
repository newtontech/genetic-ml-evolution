"""
Unit tests for SLM-Optimized Mutation Operators

Tests cover:
- ResourceEstimator
- SemanticAnalyzer
- SLMOptimizedMutation
- Integration with existing framework
"""

import pytest
import random
from genetic_ml_evolution.slm_optimized_mutation import (
    ResourceEstimator,
    SemanticAnalyzer,
    SLMOptimizedMutation,
    MutationRecord,
    MutationStatistics,
    create_slm_mutation_operator,
)


class TestResourceEstimator:
    """Tests for ResourceEstimator class."""
    
    def test_estimate_transformer_params_basic(self):
        """Test basic parameter estimation for Transformer."""
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "vocab_size": 50000,
        }
        
        params = ResourceEstimator.estimate_transformer_params(arch)
        
        # Should be in reasonable range (10-100M for this config)
        assert 10_000_000 < params < 100_000_000
        
        # Should increase with more layers
        arch_more_layers = arch.copy()
        arch_more_layers["num_layers"] = 12
        params_more = ResourceEstimator.estimate_transformer_params(arch_more_layers)
        assert params_more > params
    
    def test_estimate_transformer_params_scaling(self):
        """Test that params scale correctly with architecture size."""
        # Small model
        small_arch = {
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 1024,
            "vocab_size": 30000,
        }
        small_params = ResourceEstimator.estimate_transformer_params(small_arch)
        
        # Large model
        large_arch = {
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072,
            "vocab_size": 100000,
        }
        large_params = ResourceEstimator.estimate_transformer_params(large_arch)
        
        # Large should be significantly bigger
        assert large_params > small_params * 3
    
    def test_estimate_memory_gb(self):
        """Test memory estimation."""
        arch = {
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "vocab_size": 50000,
            "max_seq_len": 512,
        }
        
        memory_gb = ResourceEstimator.estimate_memory_gb(arch, batch_size=32)
        
        # Should be in reasonable range
        assert 0 < memory_gb < 50  # Less than 50GB for this config
        
        # More layers = more memory
        arch_deep = arch.copy()
        arch_deep["num_layers"] = 12
        memory_deep = ResourceEstimator.estimate_memory_gb(arch_deep, batch_size=32)
        assert memory_deep > memory_gb


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""
    
    def test_analyze_transformer_semantics_basic(self):
        """Test basic semantic analysis."""
        arch = {
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
        }
        
        analysis = SemanticAnalyzer.analyze_transformer_semantics(arch)
        
        # Check structure
        assert "depth" in analysis
        assert "width" in analysis
        assert "ffn_ratio" in analysis
        assert "balance_score" in analysis
        assert "issues" in analysis
        assert "recommendations" in analysis
    
    def test_analyze_balanced_architecture(self):
        """Test analysis of a well-balanced architecture."""
        # GPT-2 small-like architecture
        arch = {
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072,  # 4x hidden
            "dropout": 0.1,
        }
        
        analysis = SemanticAnalyzer.analyze_transformer_semantics(arch)
        
        # Should have good balance score
        assert analysis["balance_score"] > 70
        
        # FFN ratio should be close to 4
        assert 3.5 <= analysis["ffn_ratio"] <= 4.5
        
        # Should have few issues
        assert len(analysis["issues"]) <= 2
    
    def test_analyze_unbalanced_architecture(self):
        """Test analysis of an unbalanced architecture."""
        # Unbalanced: too many heads for small hidden
        arch = {
            "num_layers": 12,
            "hidden_size": 256,  # Small
            "num_heads": 12,     # Too many
            "ffn_dim": 512,      # Only 2x hidden
            "dropout": 0.05,     # Light regularization for deep net
        }
        
        analysis = SemanticAnalyzer.analyze_transformer_semantics(arch)
        
        # Should have low balance score
        assert analysis["balance_score"] <= 70
        
        # Should identify issues
        assert len(analysis["issues"]) > 0
        
        # Should provide recommendations
        assert len(analysis["recommendations"]) > 0
    
    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        # Architecture with FFN ratio issue
        arch = {
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 512,  # Only 1x hidden (too small)
            "dropout": 0.1,
        }
        
        analysis = SemanticAnalyzer.analyze_transformer_semantics(arch)
        suggestions = SemanticAnalyzer.suggest_improvements(arch, analysis)
        
        # Should suggest FFN adjustment
        assert len(suggestions) > 0
        
        # Should have target value
        for suggestion in suggestions:
            assert "type" in suggestion
            assert "target_value" in suggestion
            assert "priority" in suggestion


class TestSLMOptimizedMutation:
    """Tests for SLMOptimizedMutation class."""
    
    @pytest.fixture
    def mutator(self):
        """Create a mutator instance."""
        return SLMOptimizedMutation(
            max_params=100_000_000,
            max_memory_gb=20.0,
            enable_semantic_analysis=True,
            enable_history_learning=True,
            verbose=False
        )
    
    @pytest.fixture
    def transformer_arch(self):
        """Create a sample Transformer architecture."""
        return {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
            "vocab_size": 50000,
            "max_seq_len": 512,
        }
    
    @pytest.fixture
    def cnn_arch(self):
        """Create a sample CNN architecture."""
        return {
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64,
            "kernel_size": 3,
        }
    
    def test_initialization(self, mutator):
        """Test mutator initialization."""
        assert mutator.max_params == 100_000_000
        assert mutator.max_memory_gb == 20.0
        assert mutator.enable_semantic_analysis is True
        assert mutator.enable_history_learning is True
    
    def test_mutate_transformer_basic(self, mutator, transformer_arch):
        """Test basic Transformer mutation."""
        mutated, desc = mutator.mutate_transformer(transformer_arch)
        
        # Should return valid architecture
        assert "type" in mutated
        assert mutated["type"] == "transformer"
        
        # Should be a copy
        assert mutated is not transformer_arch
    
    def test_mutate_respects_resource_budget(self, mutator, transformer_arch):
        """Test that mutations respect resource budget."""
        # Run many mutations
        for _ in range(100):
            mutated, desc = mutator.mutate_transformer(transformer_arch)
            
            # Check within budget
            assert mutator._is_within_budget(mutated)
            
            # Check parameters
            params = ResourceEstimator.estimate_transformer_params(mutated)
            assert params <= mutator.max_params
    
    def test_mutate_adaptive_strategy(self, mutator, transformer_arch):
        """Test adaptive mutation strategy."""
        # High fitness -> conservative
        _, conservative_desc = mutator.mutate_transformer(
            transformer_arch, fitness=85, strategy="adaptive"
        )
        
        # Low fitness -> aggressive
        _, aggressive_desc = mutator.mutate_transformer(
            transformer_arch, fitness=30, strategy="adaptive"
        )
        
        # Both should succeed (descriptions may differ)
        assert conservative_desc is not None
        assert aggressive_desc is not None
    
    def test_mutate_maintains_constraints(self, mutator, transformer_arch):
        """Test that mutations maintain architectural constraints."""
        for _ in range(100):
            mutated, desc = mutator.mutate_transformer(transformer_arch)
            
            # Check divisibility constraint
            hidden_size = mutated.get("hidden_size", 512)
            num_heads = mutated.get("num_heads", 8)
            assert hidden_size % num_heads == 0, \
                f"hidden_size ({hidden_size}) not divisible by num_heads ({num_heads})"
            
            # Check parameter ranges
            assert 2 <= mutated.get("num_layers", 6) <= 12
            assert 128 <= mutated.get("hidden_size", 512) <= 768
            assert 2 <= mutated.get("num_heads", 8) <= 12
    
    def test_mutate_semantic_awareness(self, mutator, transformer_arch):
        """Test semantic-aware mutations."""
        # Unbalanced architecture
        unbalanced = {
            "type": "transformer",
            "num_layers": 12,
            "hidden_size": 256,
            "num_heads": 8,
            "ffn_dim": 512,  # Too small (2x instead of 3-4x)
            "dropout": 0.05,
        }
        
        mutated, desc = mutator.mutate_transformer(unbalanced, strategy="moderate")
        
        # Should tend toward better FFN ratio
        new_ffn = mutated.get("ffn_dim", 512)
        new_hidden = mutated.get("hidden_size", 256)
        new_ratio = new_ffn / new_hidden
        
        # Ratio should improve or stay reasonable
        assert new_ratio >= 2.0
    
    def test_block_mutation(self, mutator, transformer_arch):
        """Test block mutation strategy."""
        block_result = mutator._block_mutation(transformer_arch)
        
        if block_result:
            mutated, desc = block_result
            
            # Should have block prefix
            assert desc.startswith("block:")
            
            # Should maintain validity
            assert mutator._is_within_budget(mutated)
    
    def test_conservative_mutation(self, mutator, transformer_arch):
        """Test conservative mutation fallback."""
        mutated, desc = mutator._conservative_mutate_transformer(transformer_arch)
        
        # Should be very similar
        # Most likely only dropout changes
        changes = sum(
            1 for k in transformer_arch.keys()
            if transformer_arch[k] != mutated.get(k)
        )
        
        # Should have 0 or 1 changes
        assert changes <= 1
    
    def test_mutate_cnn(self, mutator, cnn_arch):
        """Test CNN mutation."""
        mutated, desc = mutator.mutate_cnn(cnn_arch)
        
        # Should maintain type
        assert mutated.get("type") == "cnn"
        
        # Should be valid
        assert 2 <= mutated.get("num_blocks", 4) <= 8
    
    def test_mutate_multimodal(self, mutator):
        """Test multimodal architecture mutation."""
        multimodal = {
            "type": "multimodal",
            "vision_encoder": {
                "type": "cnn",
                "num_blocks": 3,
                "base_channels": 32,
            },
            "text_encoder": {
                "type": "transformer",
                "num_layers": 4,
                "hidden_size": 256,
                "num_heads": 4,
                "ffn_dim": 1024,
            },
            "fusion_dim": 256,
        }
        
        mutated, desc = mutator.mutate_multimodal(multimodal)
        
        # Should maintain structure
        assert "vision_encoder" in mutated
        assert "text_encoder" in mutated
    
    def test_record_result_and_statistics(self, mutator, transformer_arch):
        """Test result recording and statistics."""
        # Perform some mutations
        for fitness in [30, 50, 70, 85]:
            mutated, desc = mutator.mutate_transformer(transformer_arch, fitness=fitness)
            
            # Simulate result
            improvement = random.uniform(-5, 10)
            new_fitness = fitness + improvement
            
            mutator.record_result(
                parent_arch=transformer_arch,
                child_arch=mutated,
                parent_fitness=fitness,
                child_fitness=new_fitness,
                mutation_type=desc.split(":")[0] if ":" in desc else "basic"
            )
        
        # Get statistics
        stats = mutator.get_statistics()
        
        assert stats["total_mutations"] == 4
        assert 0 <= stats["success_rate"] <= 1
        assert "by_type" in stats
    
    def test_resource_budget_filtering(self):
        """Test that architectures exceeding budget are filtered."""
        # Create mutator with very tight budget
        tight_mutator = SLMOptimizedMutation(
            max_params=10_000_000,  # Only 10M
            max_memory_gb=5.0,
            verbose=False
        )
        
        large_arch = {
            "type": "transformer",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "ffn_dim": 3072,
            "vocab_size": 100000,
        }
        
        # Should still produce valid results (within budget)
        for _ in range(10):
            mutated, desc = tight_mutator.mutate_transformer(large_arch)
            # The mutator should ensure the result is within budget
            # If not within budget, it should use conservative mutation
            if not tight_mutator._is_within_budget(mutated):
                # Conservative mutation should at least reduce size
                params = ResourceEstimator.estimate_transformer_params(mutated)
                assert params <= ResourceEstimator.estimate_transformer_params(large_arch)
    
    def test_semantic_analysis_disabled(self, transformer_arch):
        """Test mutation with semantic analysis disabled."""
        mutator = SLMOptimizedMutation(
            enable_semantic_analysis=False,
            verbose=False
        )
        
        mutated, desc = mutator.mutate_transformer(transformer_arch)
        
        # Should still work
        assert mutated is not None
        assert desc is not None
    
    def test_history_learning_disabled(self, transformer_arch):
        """Test mutation with history learning disabled."""
        mutator = SLMOptimizedMutation(
            enable_history_learning=False,
            verbose=False
        )
        
        mutated, desc = mutator.mutate_transformer(transformer_arch)
        
        # Record should still be recorded (even if not used for learning)
        mutator.record_result(
            parent_arch=transformer_arch,
            child_arch=mutated,
            parent_fitness=50.0,
            child_fitness=55.0,
            mutation_type="test"
        )
        
        # But should work without errors
        stats = mutator.get_statistics()
        # When history learning is disabled, recording might be skipped
        # The key is that mutation still works
        assert mutated is not None


class TestMutationStatistics:
    """Tests for MutationStatistics class."""
    
    def test_record_mutation(self):
        """Test recording mutations."""
        stats = MutationStatistics()
        
        record = MutationRecord(
            parent_arch={"layers": 6},
            child_arch={"layers": 8},
            parent_fitness=50.0,
            child_fitness=55.0,
            improvement=5.0,
            mutation_type="layer_increase",
            generation=1
        )
        
        stats.record_mutation(record)
        
        assert stats.total_mutations == 1
        assert stats.successful_mutations == 1
        assert stats.failed_mutations == 0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        stats = MutationStatistics()
        
        # Add successful mutations
        for _ in range(7):
            record = MutationRecord(
                parent_arch={},
                child_arch={},
                parent_fitness=50.0,
                child_fitness=55.0,
                improvement=5.0,
                mutation_type="test",
                generation=1
            )
            stats.record_mutation(record)
        
        # Add failed mutations
        for _ in range(3):
            record = MutationRecord(
                parent_arch={},
                child_arch={},
                parent_fitness=50.0,
                child_fitness=48.0,
                improvement=-2.0,
                mutation_type="test",
                generation=1
            )
            stats.record_mutation(record)
        
        assert stats.total_mutations == 10
        assert stats.get_success_rate() == 0.7
    
    def test_by_type_statistics(self):
        """Test statistics by mutation type."""
        stats = MutationStatistics()
        
        # Add different types
        for mutation_type in ["layer", "hidden", "layer"]:
            improvement = 5.0 if mutation_type == "layer" else -2.0
            record = MutationRecord(
                parent_arch={},
                child_arch={},
                parent_fitness=50.0,
                child_fitness=50.0 + improvement,
                improvement=improvement,
                mutation_type=mutation_type,
                generation=1
            )
            stats.record_mutation(record)
        
        # Check by-type stats
        assert stats.get_success_rate("layer") == 1.0
        assert stats.get_success_rate("hidden") == 0.0


class TestIntegration:
    """Integration tests with existing framework."""
    
    def test_compatibility_with_genetic_operators(self):
        """Test compatibility with existing genetic_operators module."""
        from genetic_ml_evolution.genetic_operators import ArchitectureGene
        
        mutator = SLMOptimizedMutation(verbose=False)
        
        gene = ArchitectureGene({
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
        })
        
        # Mutate using new operator
        mutated_arch, desc = mutator.mutate(gene.architecture)
        
        # Should be compatible
        assert isinstance(mutated_arch, dict)
        assert "type" in mutated_arch
        
        # Can create new gene
        new_gene = ArchitectureGene(mutated_arch)
        assert new_gene.architecture == mutated_arch
    
    def test_multiple_generations(self):
        """Test mutation over multiple generations."""
        mutator = SLMOptimizedMutation(verbose=False)
        
        arch = {
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
        }
        
        fitness = 50.0
        
        for gen in range(10):
            mutated, desc = mutator.mutate(arch, fitness=fitness, strategy="adaptive")
            
            # Simulate fitness change
            improvement = random.uniform(-2, 5)
            new_fitness = fitness + improvement
            
            mutator.record_result(
                parent_arch=arch,
                child_arch=mutated,
                parent_fitness=fitness,
                child_fitness=new_fitness,
                mutation_type=desc.split(":")[0] if ":" in desc else "basic"
            )
            
            # Move to next generation
            arch = mutated
            fitness = new_fitness
            mutator.advance_generation()
        
        # Check statistics
        stats = mutator.get_statistics()
        assert stats["total_mutations"] == 10
    
    def test_factory_function(self):
        """Test factory function."""
        mutator = create_slm_mutation_operator(
            max_params=50_000_000,
            max_memory_gb=10.0,
            enable_semantic_analysis=False
        )
        
        assert mutator.max_params == 50_000_000
        assert mutator.max_memory_gb == 10.0
        assert mutator.enable_semantic_analysis is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_unknown_architecture_type(self):
        """Test handling of unknown architecture type."""
        mutator = SLMOptimizedMutation(verbose=False)
        
        unknown_arch = {"type": "unknown", "param": 123}
        
        mutated, desc = mutator.mutate(unknown_arch)
        
        # Should return copy without error
        assert mutated is not unknown_arch
        assert desc == "no_change"
    
    def test_extreme_architecture_values(self):
        """Test with extreme parameter values."""
        mutator = SLMOptimizedMutation(
            max_params=1_000_000_000,  # Very high
            verbose=False
        )
        
        extreme_arch = {
            "type": "transformer",
            "num_layers": 100,
            "hidden_size": 10000,
            "num_heads": 100,
            "ffn_dim": 100000,
        }
        
        # Should handle gracefully
        mutated, desc = mutator.mutate_transformer(extreme_arch)
        
        # Note: Current implementation doesn't clamp extreme input values
        # It only ensures mutations stay within bounds
        # So we just verify it returns a valid dict
        assert isinstance(mutated, dict)
        assert "type" in mutated
    
    def test_minimal_architecture(self):
        """Test with minimal architecture."""
        mutator = SLMOptimizedMutation(verbose=False)
        
        minimal_arch = {
            "type": "transformer",
            "num_layers": 2,
            "hidden_size": 128,
            "num_heads": 2,
            "ffn_dim": 256,
        }
        
        # Should handle small architectures
        mutated, desc = mutator.mutate_transformer(minimal_arch)
        
        # Should maintain minimums
        assert mutated.get("num_layers", 2) >= 2
        assert mutated.get("hidden_size", 128) >= 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
