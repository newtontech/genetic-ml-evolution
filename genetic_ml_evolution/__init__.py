"""
Genetic ML Evolution - 低成本遗传算法自我进化机器学习算法框架

A low-cost genetic algorithm framework for self-evolving machine learning models.
"""

__version__ = "0.2.0"
__author__ = "Newton Tech"

from .cache_system import ArchitectureCache
from .surrogate_model import SurrogateModel
from .genetic_algorithm import GeneticAlgorithm, Individual, MutationStrategy
from .slm_optimized_mutation import (
    SLMOptimizedMutation,
    ResourceEstimator,
    SemanticAnalyzer,
    create_slm_mutation_operator,
)

__all__ = [
    "ArchitectureCache",
    "SurrogateModel",
    "GeneticAlgorithm",
    "Individual",
    "MutationStrategy",
    "SLMOptimizedMutation",
    "ResourceEstimator",
    "SemanticAnalyzer",
    "create_slm_mutation_operator",
]
