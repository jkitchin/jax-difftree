"""Differentiable decision trees in JAX with built-in uncertainty quantification.

Public API:
    DifferentiableTreeRegressor
    DifferentiableTreeClassifier
    difftree.export: leaf polytopes and jax-ldt JSON spec export
"""
from . import export
from .classifier import DifferentiableTreeClassifier
from .regressor import DifferentiableTreeRegressor

__all__ = ["DifferentiableTreeRegressor", "DifferentiableTreeClassifier"]
__version__ = "0.1.0"
