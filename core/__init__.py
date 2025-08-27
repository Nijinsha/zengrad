"""
ZenGrad - A minimal automatic differentiation engine.

This package provides a lightweight implementation of automatic differentiation
similar to PyTorch's autograd, with support for building and training neural networks.
"""

from .value import Value
from .neuron import Neuron
from .layer import Layer
from .mlp import MLP
from .visualization import trace, draw_dot

__all__ = [
    'Value',
    'Neuron', 
    'Layer',
    'MLP',
    'trace',
    'draw_dot'
]

__version__ = '0.1.0'
