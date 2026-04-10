"""
Tests for enhanced SLM mutation features: GQA, PEFT, quantization, progressive scheduling, multi-objective scoring.
"""

import pytest
from genetic_ml_evolution.slm_optimized_mutation import (
    SLMOptimizedMutation,
    ResourceEstimator,
    MutationStatistics,
    MutationStrategy,
    QuantizationMode,
    AttentionType,
)


# ---- Fixtures ----

@pytest.fixture
def transformer_arch():
    return {
        "type": "transformer",
        "num_layers": 6,
        "hidden_size": 512,
        "num_heads": 8,
        "ffn_dim": 1536,
        "dropout": 0.1,
        "vocab_size": 50000,
        "max_seq_len": 256,
    }


@pytest.fixture
def mutator():
    return SLMOptimizedMutation(
        max_params=100_000_000,
        max_memory_gb=20.0,
        max_latency_ms=500.0,
        enable_gqa=True,
        enable_peft_mutation=True,
        enable_quant_aware=True,
        enable_progressive=True,
        total_generations=50,
    )


# ---- GQA Tests ----

class TestGQAMutation:
    def test_gqa_mutates_kv_heads(self, mutator, transformer_arch):
        arch = transformer_arch.copy()
        mutations = mutator._mutate_gqa(arch)
        # Not guaranteed every run, but after many runs should see changes
        changed = False
        for _ in range(100):
            a = transformer_arch.copy()
            m = mutator._mutate_gqa(a)
            if m:
                changed = True
                assert "num_kv_heads" in a
                assert "attention_type" in a
                break
        assert changed

    def test_gqa_disabled(self, transformer_arch):
        mutator = SLMOptimizedMutation(enable_gqa=False)
        arch = transformer_arch.copy()
        mutations = mutator._mutate_gqa(arch)
        assert mutations == []

    def test_gqa_respects_divisibility(self, mutator, transformer_arch):
        for _ in range(50):
            arch = transformer_arch.copy()
            mutator._mutate_gqa(arch)
            if "num_kv_heads" in arch:
                assert arch["num_heads"] % arch["num_kv_heads"] == 0


# ---- PEFT Tests ----

class TestPEFTMutation:
    def test_peft_mutates_mode(self, mutator, transformer_arch):
        changed = False
        for _ in range(100):
            arch = transformer_arch.copy()
            mutations = mutator._mutate_peft_config(arch)
            if mutations:
                changed = True
                assert arch.get("peft_mode") in ("none", "lora", "qlora")
                break
        assert changed

    def test_peft_disabled(self, transformer_arch):
        mutator = SLMOptimizedMutation(enable_peft_mutation=False)
        arch = transformer_arch.copy()
        mutations = mutator._mutate_peft_config(arch)
        assert mutations == []

    def test_lora_rank_mutation(self, mutator, transformer_arch):
        arch = transformer_arch.copy()
        arch["peft_mode"] = "lora"
        arch["lora_rank"] = 8
        changed = False
        for _ in range(100):
            a = arch.copy()
            mutations = mutator._mutate_peft_config(a)
            if "lora_rank" in str(mutations):
                changed = True
                assert a["lora_rank"] in [4, 8, 16, 32, 64]
                break
        assert changed

    def test_qlora_forces_int4(self, mutator):
        arch = {"peft_mode": "qlora", "quantization": "none"}
        # When mutation triggers, it should force int4
        changed = False
        for _ in range(100):
            a = arch.copy()
            m = mutator._mutate_quantization(a)
            if m:  # mutation actually happened
                changed = True
                assert a.get("quantization") == "int4"
                break
        assert changed, "Quantization mutation should have triggered at least once"


# ---- Quantization Tests ----

class TestQuantizationMutation:
    def test_quant_mutates(self, mutator, transformer_arch):
        changed = False
        for _ in range(100):
            arch = transformer_arch.copy()
            mutations = mutator._mutate_quantization(arch)
            if mutations:
                changed = True
                assert arch.get("quantization") in ("none", "int8", "int4")
                break
        assert changed

    def test_quant_disabled(self, transformer_arch):
        mutator = SLMOptimizedMutation(enable_quant_aware=False)
        arch = transformer_arch.copy()
        assert mutator._mutate_quantization(arch) == []


# ---- Progressive Scheduling ----

class TestProgressiveScheduling:
    def test_early_phase_aggressive(self):
        m = SLMOptimizedMutation(enable_progressive=True, total_generations=100)
        m.generation = 5
        assert m._get_progressive_strategy() == "aggressive"

    def test_mid_phase_moderate(self):
        m = SLMOptimizedMutation(enable_progressive=True, total_generations=100)
        m.generation = 35
        assert m._get_progressive_strategy() == "moderate"

    def test_late_phase_conservative(self):
        m = SLMOptimizedMutation(enable_progressive=True, total_generations=100)
        m.generation = 85
        assert m._get_progressive_strategy() == "conservative"

    def test_disabled(self):
        m = SLMOptimizedMutation(enable_progressive=False)
        assert m._get_progressive_strategy() == "moderate"


# ---- Multi-Objective Scoring ----

class TestMultiObjectiveScoring:
    def test_score_with_fitness(self, mutator, transformer_arch):
        score = mutator._multi_objective_score(transformer_arch, fitness=75.0)
        assert 0 < score <= 1.0

    def test_score_without_fitness(self, mutator, transformer_arch):
        score = mutator._multi_objective_score(transformer_arch)
        assert 0 < score <= 1.0

    def test_smaller_arch_scores_higher_efficiency(self, mutator):
        large = {"type": "transformer", "num_layers": 12, "hidden_size": 768,
                 "num_heads": 12, "ffn_dim": 3072, "vocab_size": 100000}
        small = {"type": "transformer", "num_layers": 2, "hidden_size": 128,
                 "num_heads": 2, "ffn_dim": 256, "vocab_size": 10000}
        # Without fitness, efficiency should favor smaller
        assert mutator._multi_objective_score(small) > mutator._multi_objective_score(large)


# ---- Resource Estimator Enhancements ----

class TestResourceEstimatorEnhanced:
    def test_gqa_reduces_params(self):
        mha = {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ffn_dim": 1536,
               "vocab_size": 50000, "num_kv_heads": 8}
        gqa = {**mha, "num_kv_heads": 2}
        assert ResourceEstimator.estimate_transformer_params(gqa) < ResourceEstimator.estimate_transformer_params(mha)

    def test_swiglu_more_params(self):
        gelu = {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ffn_dim": 1536,
                "vocab_size": 50000, "activation": "gelu", "num_kv_heads": 8}
        swiglu = {**gelu, "activation": "swiglu"}
        assert ResourceEstimator.estimate_transformer_params(swiglu) > ResourceEstimator.estimate_transformer_params(gelu)

    def test_quantization_reduces_memory(self):
        arch = {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ffn_dim": 1536,
                "max_seq_len": 256, "vocab_size": 50000, "num_kv_heads": 8}
        fp16_mem = ResourceEstimator.estimate_memory_gb(arch)
        int4_mem = ResourceEstimator.estimate_memory_gb({**arch, "quantization": "int4"})
        assert int4_mem < fp16_mem

    def test_peft_reduces_memory(self):
        arch = {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ffn_dim": 1536,
                "max_seq_len": 256, "vocab_size": 50000, "num_kv_heads": 8}
        full_mem = ResourceEstimator.estimate_memory_gb(arch)
        lora_mem = ResourceEstimator.estimate_memory_gb({**arch, "peft_mode": "lora", "lora_rank": 8})
        assert lora_mem < full_mem

    def test_latency_estimate(self):
        arch = {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ffn_dim": 1536,
                "num_kv_heads": 8}
        latency = ResourceEstimator.estimate_latency_ms(arch, seq_len=128)
        assert latency > 0

    def test_gqa_latency_faster(self):
        mha = {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ffn_dim": 1536, "num_kv_heads": 8}
        gqa = {**mha, "num_kv_heads": 2}
        # GQA should have lower latency (fewer KV operations)
        assert ResourceEstimator.estimate_latency_ms(gqa) < ResourceEstimator.estimate_latency_ms(mha)


# ---- Mutation Statistics Enhanced ----

class TestMutationStatisticsEnhanced:
    def test_recent_trend(self):
        from genetic_ml_evolution.slm_optimized_mutation import MutationRecord
        stats = MutationStatistics()
        for i in range(10):
            stats.record_mutation(MutationRecord(
                parent_arch={}, child_arch={}, parent_fitness=50, child_fitness=55,
                improvement=5.0, mutation_type="test", generation=1
            ))
        assert stats.get_recent_trend() > 0

    def test_by_generation_tracking(self):
        from genetic_ml_evolution.slm_optimized_mutation import MutationRecord
        stats = MutationStatistics()
        stats.record_mutation(MutationRecord(
            parent_arch={}, child_arch={}, parent_fitness=50, child_fitness=55,
            improvement=5.0, mutation_type="test", generation=1
        ))
        assert stats.by_generation[1]["count"] == 1
        assert stats.by_generation[1]["avg_improvement"] == 5.0


# ---- Budget Checks ----

class TestBudgetChecks:
    def test_latency_budget_enforced(self, mutator, transformer_arch):
        # Very tight latency budget
        mutator.max_latency_ms = 0.01
        assert not mutator._is_within_budget(transformer_arch)

    def test_all_budgets_pass(self, mutator, transformer_arch):
        assert mutator._is_within_budget(transformer_arch)

    def test_param_budget_enforced(self, mutator):
        large = {"num_layers": 24, "hidden_size": 1024, "num_heads": 16, "ffn_dim": 4096,
                 "vocab_size": 100000, "num_kv_heads": 16}
        mutator.max_params = 1_000_000
        assert not mutator._is_within_budget(large)


# ---- Integration ----

class TestIntegrationEnhanced:
    def test_full_mutation_with_all_features(self, mutator, transformer_arch):
        mutator.generation = 10
        mutated, desc = mutator.mutate_transformer(transformer_arch, fitness=60, strategy="adaptive")
        assert isinstance(mutated, dict)
        assert isinstance(desc, str)
        assert mutated["type"] == "transformer"

    def test_qlora_pipeline(self, mutator, transformer_arch):
        arch = transformer_arch.copy()
        arch["peft_mode"] = "qlora"
        arch["quantization"] = "int4"
        arch["lora_rank"] = 16
        assert mutator._is_within_budget(arch)
        params = ResourceEstimator.estimate_transformer_params(arch)
        assert params > 0

    def test_backward_compatible(self, transformer_arch):
        """Old code without new fields should still work."""
        old_mutator = SLMOptimizedMutation(
            max_params=100_000_000,
            max_memory_gb=20.0,
            enable_gqa=False,
            enable_peft_mutation=False,
            enable_quant_aware=False,
            enable_progressive=False,
        )
        mutated, desc = old_mutator.mutate_transformer(transformer_arch)
        assert mutated["type"] == "transformer"
