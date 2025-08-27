# Zengrad 🐣

_Named after my daughter Zenha_ ❤️

A minimal automatic differentiation engine built from scratch in Python. Zengrad implements backpropagation and automatic gradient computation for neural networks, inspired by PyTorch's autograd but designed for educational purposes.

## What is Zengrad?

Zengrad is a tiny scalar-valued autograd engine that can automatically compute gradients of mathematical expressions. It's perfect for understanding how automatic differentiation works under the hood in modern deep learning frameworks.

### Core Features

- **Automatic Differentiation**: Computes gradients automatically using backpropagation
- **Dynamic Computation Graphs**: Builds computational graphs on-the-fly
- **Neural Network Building Blocks**: Supports neurons, MLPs, and common activation functions
- **Zero Dependencies**: Pure Python implementation with minimal external dependencies

## Quick Start

```python
from zengrad import Value

# Create values
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Build computation graph
d = a * b          # d = -6.0
e = d + c          # e = 4.0
f = Value(-2.0, label='f')
L = e * f          # L = -8.0

# Compute gradients automatically
L.backward()

print(f"dL/da = {a.grad}")  # 6.0
print(f"dL/db = {b.grad}")  # -4.0
print(f"dL/dc = {c.grad}")  # -2.0
```

## Experiments

The `experiments/` directory contains Jupyter notebooks demonstrating key concepts:

- **`neuron.ipynb`** - Single neuron implementation with manual backprop
- **`backprop.ipynb`** - Understanding backpropagation from first principles
- **`auto-backprop.ipynb`** - Automatic backpropagation implementation
- **`mlp.ipynb`** - Multi-layer perceptron using zengrad
- **`actfun.ipynb`** - Activation functions and their derivatives

## How It Works

Zengrad implements reverse-mode automatic differentiation:

1. **Forward Pass**: Computes the output while building a computation graph
2. **Backward Pass**: Traverses the graph in reverse, applying chain rule to compute gradients
3. **Value Class**: Wraps scalars and tracks operations for gradient computation

```python
class Value:
    def __init__(self, val, _children=(), _op='', label=''):
        self.val = val              # The actual value
        self.grad = 0.0             # Gradient
        self._prev = set(_children) # Previous nodes in computation graph
        self._backward = lambda: None # Backward function
```

## Educational Purpose

This project is designed for learning:

- How automatic differentiation works
- The mathematics behind backpropagation
- How to implement neural networks from scratch
- Understanding computation graphs

Perfect for students, researchers, or anyone curious about the foundations of modern deep learning frameworks.

## Inspiration

Built following the principles demonstrated in Andrej Karpathy's "Building GPT" series, focusing on understanding fundamentals rather than performance optimization.

---

_"Understanding the building blocks makes you a better architect"_
