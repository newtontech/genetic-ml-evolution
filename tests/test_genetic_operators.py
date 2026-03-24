"""
Unit tests for genetic operators optimized for Small Language Models.

Tests cover:
- ArchitectureGene class
- SLMutationOperators for all architecture types
- SLMCrossoverOperators for all architecture types
- SLMSelectionOperators
- SLM constraints enforcement
"""

import pytest
import random
from genetic_ml_evolution.genetic_operators import (
    ArchitectureGene,
    SLMutationOperators,
    SLMCrossoverOperators,
    SLMSelectionOperators,
    SLM_CONSTRAINTS,
    SLM_MUTATION_RATES,
)


class TestArchitectureGene:
    """Tests for ArchitectureGene class."""
    
    def test_initialization(self):
        """Test basic initialization of ArchitectureGene."""
        arch = {"type": "transformer", "num_layers": 6}
        gene = ArchitectureGene(arch)
        
        assert gene.architecture == arch
        assert gene.fitness is None
        assert gene.age == 0
    
    def test_copy(self):
        """Test that copy creates a deep copy."""
        arch = {"type": "transformer", "num_layers": 6}
        gene = ArchitectureGene(arch)
        gene.fitness = 0.85
        gene.age = 3
        
        copied = gene.copy()
        
        # Verify copy is independent
        assert copied.architecture == gene.architecture
        assert copied.fitness == gene.fitness
        assert copied.age == gene.age
        
        # Modify original and verify copy is unchanged
        gene.architecture["num_layers"] = 12
        assert copied.architecture["num_layers"] == 6
    
    def test_repr(self):
        """Test string representation."""
        arch = {"type": "transformer"}
        gene = ArchitectureGene(arch)
        gene.fitness = 0.9
        
        repr_str = repr(gene)
        assert "transformer" in repr_str
        assert "0.9" in repr_str


class TestSLMutationOperators:
    """Tests for SLM-optimized mutation operators."""
    
    @pytest.fixture
    def mutator(self):
        """Create a mutator instance."""
        return SLMutationOperators(
            mutation_rate=0.2,
            mutation_strength="moderate",
            respect_constraints=True
        )
    
    @pytest.fixture
    def transformer_gene(self):
        """Create a sample transformer gene."""
        return ArchitectureGene({
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
            "max_seq_len": 256,
        })
    
    @pytest.fixture
    def cnn_gene(self):
        """Create a sample CNN gene."""
        return ArchitectureGene({
            "type": "cnn",
            "num_blocks": 4,
            "base_channels": 64,
            "kernel_size": 3,
            "stride": 1,
        })
    
    @pytest.fixture
    def multimodal_gene(self):
        """Create a sample multimodal gene."""
        return ArchitectureGene({
            "type": "multimodal",
            "vision_encoder": {
                "num_blocks": 3,
                "base_channels": 32,
            },
            "text_encoder": {
                "num_layers": 4,
                "hidden_size": 256,
            },
            "fusion_dim": 256,
            "projection_dim": 128,
            "fusion_type": "attention",
            "temperature": 0.1,
        })
    
    def test_initialization(self, mutator):
        """Test mutator initialization."""
        assert mutator.mutation_rate == 0.2
        assert mutator.mutation_strength == "moderate"
        assert mutator.respect_constraints is True
        assert mutator.effective_rate == SLM_MUTATION_RATES["moderate"]
    
    def test_mutation_strength_levels(self):
        """Test different mutation strength levels."""
        conservative = SLMutationOperators(mutation_strength="conservative")
        moderate = SLMutationOperators(mutation_strength="moderate")
        aggressive = SLMutationOperators(mutation_strength="aggressive")
        
        assert conservative.effective_rate == SLM_MUTATION_RATES["conservative"]
        assert moderate.effective_rate == SLM_MUTATION_RATES["moderate"]
        assert aggressive.effective_rate == SLM_MUTATION_RATES["aggressive"]
    
    def test_transformer_mutation_respects_constraints(self, mutator, transformer_gene):
        """Test that transformer mutations respect SLM constraints."""
        # Run multiple mutations to increase chance of mutation occurring
        for _ in range(100):
            mutated = mutator.mutate_transformer(transformer_gene)
            arch = mutated.architecture
            
            # Check all parameters are within constraints
            if "num_layers" in arch:
                assert SLM_CONSTRAINTS["transformer"]["num_layers"][0] <= arch["num_layers"]
                assert arch["num_layers"] <= SLM_CONSTRAINTS["transformer"]["num_layers"][1]
            
            if "hidden_size" in arch:
                assert SLM_CONSTRAINTS["transformer"]["hidden_size"][0] <= arch["hidden_size"]
                assert arch["hidden_size"] <= SLM_CONSTRAINTS["transformer"]["hidden_size"][1]
            
            if "num_heads" in arch:
                assert SLM_CONSTRAINTS["transformer"]["num_heads"][0] <= arch["num_heads"]
                assert arch["num_heads"] <= SLM_CONSTRAINTS["transformer"]["num_heads"][1]
    
    def test_transformer_divisibility_constraint(self, mutator):
        """Test that hidden_size is divisible by num_heads after mutation."""
        # Create gene with non-divisible values
        gene = ArchitectureGene({
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 500,  # Not divisible by 8
            "num_heads": 8,
            "ffn_dim": 2048,
        })
        
        # Force mutation
        mutator.effective_rate = 1.0
        mutated = mutator.mutate_transformer(gene)
        
        # Check divisibility
        assert mutated.architecture["hidden_size"] % mutated.architecture["num_heads"] == 0
    
    def test_transformer_ffn_ratio(self, mutator):
        """Test that FFN dimension maintains reasonable ratio to hidden size."""
        gene = ArchitectureGene({
            "type": "transformer",
            "hidden_size": 512,
            "ffn_dim": 5000,  # Too large ratio
        })
        
        mutator.effective_rate = 1.0
        mutated = mutator.mutate_transformer(gene)
        
        # Check ratio is reasonable (2-4x)
        ratio = mutated.architecture["ffn_dim"] / mutated.architecture["hidden_size"]
        assert 1.5 <= ratio <= 4.5
    
    def test_cnn_mutation_respects_constraints(self, mutator, cnn_gene):
        """Test that CNN mutations respect SLM constraints."""
        for _ in range(100):
            mutated = mutator.mutate_cnn(cnn_gene)
            arch = mutated.architecture
            
            if "num_blocks" in arch:
                assert SLM_CONSTRAINTS["cnn"]["num_blocks"][0] <= arch["num_blocks"]
                assert arch["num_blocks"] <= SLM_CONSTRAINTS["cnn"]["num_blocks"][1]
            
            if "base_channels" in arch:
                assert SLM_CONSTRAINTS["cnn"]["base_channels"][0] <= arch["base_channels"]
                assert arch["base_channels"] <= SLM_CONSTRAINTS["cnn"]["base_channels"][1]
    
    def test_cnn_kernel_size_odd(self, mutator):
        """Test that kernel size remains odd after mutation."""
        gene = ArchitectureGene({
            "type": "cnn",
            "kernel_size": 4,  # Even
        })
        
        mutator.effective_rate = 1.0
        mutated = mutator.mutate_cnn(gene)
        
        assert mutated.architecture["kernel_size"] % 2 == 1
    
    def test_cnn_channels_power_of_two(self, mutator):
        """Test that base_channels is rounded to power of 2."""
        gene = ArchitectureGene({
            "type": "cnn",
            "base_channels": 50,  # Not power of 2
        })
        
        mutator.effective_rate = 1.0
        mutated = mutator.mutate_cnn(gene)
        
        channels = mutated.architecture["base_channels"]
        # Check if it's a power of 2
        assert (channels & (channels - 1)) == 0 or channels == 0
    
    def test_multimodal_mutation_respects_constraints(self, mutator, multimodal_gene):
        """Test that multimodal mutations respect SLM constraints."""
        for _ in range(100):
            mutated = mutator.mutate_multimodal(multimodal_gene)
            arch = mutated.architecture
            
            # Check vision encoder
            if "vision_encoder" in arch:
                vision = arch["vision_encoder"]
                if "num_blocks" in vision:
                    assert SLM_CONSTRAINTS["multimodal"]["vision_num_blocks"][0] <= vision["num_blocks"]
                    assert vision["num_blocks"] <= SLM_CONSTRAINTS["multimodal"]["vision_num_blocks"][1]
            
            # Check text encoder
            if "text_encoder" in arch:
                text = arch["text_encoder"]
                if "num_layers" in text:
                    assert SLM_CONSTRAINTS["multimodal"]["text_num_layers"][0] <= text["num_layers"]
                    assert text["num_layers"] <= SLM_CONSTRAINTS["multimodal"]["text_num_layers"][1]
    
    def test_multimodal_fusion_dimension_constraint(self, mutator):
        """Test that fusion_dim >= projection_dim."""
        gene = ArchitectureGene({
            "type": "multimodal",
            "fusion_dim": 100,
            "projection_dim": 200,  # Larger than fusion_dim
        })
        
        mutator.effective_rate = 1.0
        mutated = mutator.mutate_multimodal(gene)
        
        assert mutated.architecture["fusion_dim"] >= mutated.architecture["projection_dim"]
    
    def test_mutation_creates_copy(self, mutator, transformer_gene):
        """Test that mutation creates a new gene, not modifying original."""
        original_layers = transformer_gene.architecture["num_layers"]
        
        mutator.effective_rate = 1.0  # Force mutation
        mutated = mutator.mutate_transformer(transformer_gene)
        
        # Original should be unchanged
        assert transformer_gene.architecture["num_layers"] == original_layers
        # Mutated might be different
        assert mutated is not transformer_gene
    
    def test_no_mutation_when_rate_zero(self, transformer_gene):
        """Test that no mutation occurs when mutation rate is 0."""
        mutator = SLMutationOperators(mutation_rate=0.0)
        mutator.effective_rate = 0.0
        
        mutated = mutator.mutate_transformer(transformer_gene)
        
        # Should be identical
        assert mutated.architecture == transformer_gene.architecture
    
    def test_unknown_architecture_type(self, mutator):
        """Test handling of unknown architecture types."""
        gene = ArchitectureGene({"type": "unknown"})
        
        # Should return a copy without error
        mutated = mutator.mutate(gene)
        assert mutated is not gene
        assert mutated.architecture == gene.architecture


class TestSLMCrossoverOperators:
    """Tests for SLM-optimized crossover operators."""
    
    @pytest.fixture
    def crossover(self):
        """Create a crossover instance."""
        return SLMCrossoverOperators(crossover_rate=0.7)
    
    @pytest.fixture
    def transformer_parent1(self):
        """Create first transformer parent."""
        return ArchitectureGene({
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
            "ffn_dim": 1024,
        })
    
    @pytest.fixture
    def transformer_parent2(self):
        """Create second transformer parent."""
        return ArchitectureGene({
            "type": "transformer",
            "num_layers": 8,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
        })
    
    @pytest.fixture
    def cnn_parent1(self):
        """Create first CNN parent."""
        return ArchitectureGene({
            "type": "cnn",
            "num_blocks": 3,
            "base_channels": 32,
            "kernel_size": 3,
        })
    
    @pytest.fixture
    def cnn_parent2(self):
        """Create second CNN parent."""
        return ArchitectureGene({
            "type": "cnn",
            "num_blocks": 6,
            "base_channels": 64,
            "kernel_size": 5,
        })
    
    @pytest.fixture
    def multimodal_parent1(self):
        """Create first multimodal parent."""
        return ArchitectureGene({
            "type": "multimodal",
            "vision_encoder": {"num_blocks": 2, "base_channels": 16},
            "text_encoder": {"num_layers": 3, "hidden_size": 128},
            "fusion_dim": 128,
            "projection_dim": 64,
        })
    
    @pytest.fixture
    def multimodal_parent2(self):
        """Create second multimodal parent."""
        return ArchitectureGene({
            "type": "multimodal",
            "vision_encoder": {"num_blocks": 4, "base_channels": 32},
            "text_encoder": {"num_layers": 6, "hidden_size": 256},
            "fusion_dim": 256,
            "projection_dim": 128,
        })
    
    def test_initialization(self, crossover):
        """Test crossover initialization."""
        assert crossover.crossover_rate == 0.7
    
    def test_transformer_crossover_divisibility(self, crossover, transformer_parent1, transformer_parent2):
        """Test that crossover maintains divisibility constraint."""
        child1, child2 = crossover.crossover_transformer(
            transformer_parent1, transformer_parent2
        )
        
        # Check both children
        for child in [child1, child2]:
            assert child.architecture["hidden_size"] % child.architecture["num_heads"] == 0
    
    def test_transformer_crossover_arithmetic(self, crossover, transformer_parent1, transformer_parent2):
        """Test that arithmetic crossover produces intermediate values."""
        child1, child2 = crossover.crossover_transformer(
            transformer_parent1, transformer_parent2
        )
        
        # Children should have values between parents
        for child in [child1, child2]:
            # Check num_layers is between parents
            assert transformer_parent1.architecture["num_layers"] <= child.architecture["num_layers"]
            assert child.architecture["num_layers"] <= transformer_parent2.architecture["num_layers"]
    
    def test_cnn_crossover_kernel_odd(self, crossover, cnn_parent1, cnn_parent2):
        """Test that CNN crossover maintains odd kernel size."""
        child1, child2 = crossover.crossover_cnn(cnn_parent1, cnn_parent2)
        
        for child in [child1, child2]:
            assert child.architecture["kernel_size"] % 2 == 1
    
    def test_cnn_crossover_channels_power_of_two(self, crossover, cnn_parent1, cnn_parent2):
        """Test that CNN crossover maintains power of 2 channels."""
        child1, child2 = crossover.crossover_cnn(cnn_parent1, cnn_parent2)
        
        for child in [child1, child2]:
            channels = child.architecture["base_channels"]
            # Check if it's a power of 2
            assert (channels & (channels - 1)) == 0 or channels == 0
    
    def test_multimodal_crossover(self, crossover, multimodal_parent1, multimodal_parent2):
        """Test multimodal crossover."""
        child1, child2 = crossover.crossover_multimodal(
            multimodal_parent1, multimodal_parent2
        )
        
        # Check both children have valid structure
        for child in [child1, child2]:
            assert "vision_encoder" in child.architecture
            assert "text_encoder" in child.architecture
            assert "fusion_dim" in child.architecture
    
    def test_crossover_rate_zero(self, transformer_parent1, transformer_parent2):
        """Test that no crossover occurs when rate is 0."""
        crossover = SLMCrossoverOperators(crossover_rate=0.0)
        
        child1, child2 = crossover.crossover_transformer(
            transformer_parent1, transformer_parent2
        )
        
        # Children should be copies of parents
        assert child1.architecture == transformer_parent1.architecture
        assert child2.architecture == transformer_parent2.architecture
    
    def test_crossover_different_types(self, crossover, transformer_parent1, cnn_parent1):
        """Test crossover between different architecture types."""
        # Should return copies without error
        child1, child2 = crossover.crossover(transformer_parent1, cnn_parent1)
        
        assert child1.architecture == transformer_parent1.architecture
        assert child2.architecture == cnn_parent1.architecture


class TestSLMSelectionOperators:
    """Tests for SLM-optimized selection operators."""
    
    @pytest.fixture
    def population(self):
        """Create a sample population with fitness values."""
        genes = []
        for i in range(10):
            gene = ArchitectureGene({"type": "transformer", "id": i})
            gene.fitness = i * 0.1  # 0.0 to 0.9
            genes.append(gene)
        return genes
    
    def test_tournament_selection(self, population):
        """Test tournament selection."""
        selector = SLMSelectionOperators()
        
        # Run multiple selections
        selected = [
            selector.tournament_selection(population, tournament_size=3)
            for _ in range(100)
        ]
        
        # All selected should be valid genes
        for gene in selected:
            assert isinstance(gene, ArchitectureGene)
            assert gene.fitness is not None
    
    def test_tournament_selection_pressure(self, population):
        """Test that higher pressure favors better individuals."""
        selector = SLMSelectionOperators()
        
        # High pressure should favor better individuals
        high_pressure_selections = [
            selector.tournament_selection(population, tournament_size=3, selection_pressure=0.9)
            for _ in range(100)
        ]
        
        # Calculate average fitness of selected
        avg_fitness_high = sum(g.fitness for g in high_pressure_selections) / len(high_pressure_selections)
        
        # Low pressure should be more random
        low_pressure_selections = [
            selector.tournament_selection(population, tournament_size=3, selection_pressure=0.5)
            for _ in range(100)
        ]
        
        avg_fitness_low = sum(g.fitness for g in low_pressure_selections) / len(low_pressure_selections)
        
        # High pressure should have higher average (with some tolerance for randomness)
        # This test might occasionally fail due to randomness, but statistically should pass
        assert avg_fitness_high >= avg_fitness_low * 0.8  # Allow some variance
    
    def test_rank_selection(self, population):
        """Test rank-based selection."""
        selector = SLMSelectionOperators()
        
        selected = [
            selector.rank_selection(population, selection_pressure=1.5)
            for _ in range(100)
        ]
        
        # All selected should be valid
        for gene in selected:
            assert isinstance(gene, ArchitectureGene)
            assert gene.fitness is not None
    
    def test_elitism_selection(self, population):
        """Test elitism selection."""
        selector = SLMSelectionOperators()
        
        elites = selector.elitism_selection(population, elite_size=3)
        
        # Should get top 3
        assert len(elites) == 3
        
        # Should be sorted by fitness (descending)
        for i in range(len(elites) - 1):
            assert elites[i].fitness >= elites[i + 1].fitness
        
        # Should be the best
        assert elites[0].fitness == max(g.fitness for g in population)
    
    def test_elitism_size_larger_than_population(self, population):
        """Test elitism when elite_size >= population size."""
        selector = SLMSelectionOperators()
        
        elites = selector.elitism_selection(population, elite_size=20)
        
        # Should return entire population
        assert len(elites) == len(population)


class TestSLMConstraints:
    """Tests for SLM-specific constraints."""
    
    def test_constraints_defined(self):
        """Test that all architecture types have constraints."""
        assert "transformer" in SLM_CONSTRAINTS
        assert "cnn" in SLM_CONSTRAINTS
        assert "multimodal" in SLM_CONSTRAINTS
    
    def test_transformer_constraints_reasonable(self):
        """Test that transformer constraints are reasonable for SLM."""
        transformer = SLM_CONSTRAINTS["transformer"]
        
        # Check layer count
        assert transformer["num_layers"][0] >= 2
        assert transformer["num_layers"][1] <= 12
        
        # Check hidden size
        assert transformer["hidden_size"][0] >= 128
        assert transformer["hidden_size"][1] <= 768
        
        # Check num_heads
        assert transformer["num_heads"][0] >= 2
        assert transformer["num_heads"][1] <= 12
    
    def test_cnn_constraints_reasonable(self):
        """Test that CNN constraints are reasonable for SLM."""
        cnn = SLM_CONSTRAINTS["cnn"]
        
        # Check blocks
        assert cnn["num_blocks"][0] >= 2
        assert cnn["num_blocks"][1] <= 8
        
        # Check channels
        assert cnn["base_channels"][0] >= 16
        assert cnn["base_channels"][1] <= 128
    
    def test_mutation_rates_reasonable(self):
        """Test that mutation rates are reasonable."""
        assert SLM_MUTATION_RATES["conservative"] < SLM_MUTATION_RATES["moderate"]
        assert SLM_MUTATION_RATES["moderate"] < SLM_MUTATION_RATES["aggressive"]
        
        # All rates should be between 0 and 1
        for rate in SLM_MUTATION_RATES.values():
            assert 0 < rate < 1


class TestIntegration:
    """Integration tests for genetic operators."""
    
    def test_full_mutation_cycle(self):
        """Test complete mutation cycle for transformer."""
        mutator = SLMutationOperators(mutation_rate=1.0, mutation_strength="moderate")
        
        gene = ArchitectureGene({
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512,
            "num_heads": 8,
            "ffn_dim": 2048,
            "dropout": 0.1,
        })
        
        # Run multiple mutations
        for _ in range(10):
            mutated = mutator.mutate(gene)
            
            # Verify constraints
            assert mutated.architecture["hidden_size"] % mutated.architecture["num_heads"] == 0
            assert SLM_CONSTRAINTS["transformer"]["num_layers"][0] <= mutated.architecture["num_layers"]
    
    def test_full_crossover_cycle(self):
        """Test complete crossover cycle."""
        crossover = SLMCrossoverOperators(crossover_rate=1.0)
        
        parent1 = ArchitectureGene({
            "type": "transformer",
            "num_layers": 4,
            "hidden_size": 256,
            "num_heads": 4,
        })
        
        parent2 = ArchitectureGene({
            "type": "transformer",
            "num_layers": 8,
            "hidden_size": 512,
            "num_heads": 8,
        })
        
        # Run multiple crossovers
        for _ in range(10):
            child1, child2 = crossover.crossover(parent1, parent2)
            
            # Verify constraints
            for child in [child1, child2]:
                assert child.architecture["hidden_size"] % child.architecture["num_heads"] == 0
    
    def test_mutation_then_crossover(self):
        """Test mutation followed by crossover."""
        mutator = SLMutationOperators(mutation_rate=0.5)
        crossover = SLMCrossoverOperators(crossover_rate=0.7)
        
        parent1 = ArchitectureGene({"type": "transformer", "num_layers": 4})
        parent2 = ArchitectureGene({"type": "transformer", "num_layers": 8})
        
        # Mutate parents
        mutated1 = mutator.mutate(parent1)
        mutated2 = mutator.mutate(parent2)
        
        # Crossover
        child1, child2 = crossover.crossover(mutated1, mutated2)
        
        # Verify children are valid
        assert isinstance(child1, ArchitectureGene)
        assert isinstance(child2, ArchitectureGene)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
