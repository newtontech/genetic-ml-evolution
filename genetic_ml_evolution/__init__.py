"""
Genetic ML Evolution - 低成本遗传算法自我进化机器学习算法框架

A low-cost genetic algorithm framework for self-evolving machine learning models.
"""

__version__ = "0.1.0"
__author__ = "Newton Tech"

from .cache_system import ArchitectureCache
from .surrogate_model import SurrogateModel

__all__ = ["ArchitectureCache", "SurrogateModel"]
