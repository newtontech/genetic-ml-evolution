"""
Genetic ML Evolution Framework
低成本遗传算法自我进化机器学习算法框架
"""

__version__ = "0.1.0"
__author__ = "OpenClaw"

from .evolution_engine import EvolutionEngine
from .surrogate_model import SurrogateModel
from .cache_system import CacheSystem
from .model_evaluator import ModelEvaluator

__all__ = [
    "EvolutionEngine",
    "SurrogateModel",
    "CacheSystem",
    "ModelEvaluator",
]
