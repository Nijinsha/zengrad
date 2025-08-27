#!/usr/bin/env python3
"""
Basic Automatic Differentiation Example

This example demonstrates the core automatic differentiation capabilities
of ZenGrad's Value class, showing how gradients are computed automatically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Value


def basic_operations():
    """Demonstrate basic mathematical operations with automatic differentiation."""
    print("=== Basic Operations ===")
    
    # Create some input values
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    
    # Build a computational graph: L = (a * b + c) * f
    d = a * b
    d.label = 'd'
    
    e = d + c
    e.label = 'e'
    
    f = Value(-2.0, label='f')
    L = e * f
    L.label = 'L'
    
    print(f"Forward pass result: L = {L.data}")
    
    # Compute gradients
    L.backward()
    
    print(f"∂L/∂a = {a.grad}")  # Should be 6.0
    print(f"∂L/∂b = {b.grad}")  # Should be -4.0
    print(f"∂L/∂c = {c.grad}")  # Should be -2.0
    print(f"∂L/∂f = {f.grad}")  # Should be 4.0
    print()


def activation_functions():
    """Demonstrate activation functions and their gradients."""
    print("=== Activation Functions ===")
    
    # Tanh activation
    x = Value(0.5, label='x')
    y = x.tanh()
    y.label = 'tanh(x)'
    
    print(f"tanh({x.data}) = {y.data:.4f}")
    
    y.backward()
    print(f"d/dx tanh(x) at x={x.data} = {x.grad:.4f}")
    
    # Reset gradients for next computation
    x.grad = 0.0
    
    # Exponential function
    z = x.exp()
    z.label = 'exp(x)'
    
    print(f"exp({x.data}) = {z.data:.4f}")
    
    z.backward()
    print(f"d/dx exp(x) at x={x.data} = {x.grad:.4f}")
    print()


def polynomial_example():
    """Example with a polynomial function: f(x) = 3x² - 4x + 5"""
    print("=== Polynomial Example ===")
    
    x = Value(3.0, label='x')
    
    # f(x) = 3x² - 4x + 5
    x_squared = x * x
    term1 = 3 * x_squared
    term2 = -4 * x
    f = term1 + term2 + 5
    f.label = 'f(x)'
    
    print(f"f(3) = 3(3)² - 4(3) + 5 = {f.data}")
    
    # Compute derivative: f'(x) = 6x - 4
    f.backward()
    print(f"f'(3) = 6(3) - 4 = {x.grad} (computed: {x.grad})")
    print()


def chain_rule_example():
    """Demonstrate the chain rule with nested functions."""
    print("=== Chain Rule Example ===")
    
    # Let's compute: h(x) = tanh(x²)
    x = Value(2.0, label='x')
    x_squared = x * x
    x_squared.label = 'x²'
    h = x_squared.tanh()
    h.label = 'tanh(x²)'
    
    print(f"h(2) = tanh(2²) = tanh(4) = {h.data:.4f}")
    
    # The derivative should be: h'(x) = 2x * (1 - tanh²(x²))
    h.backward()
    
    # Manual calculation for verification
    import math
    tanh_4 = math.tanh(4)
    expected_derivative = 2 * 2.0 * (1 - tanh_4**2)
    
    print(f"h'(2) = 2(2)(1 - tanh²(4)) = {expected_derivative:.6f}")
    print(f"Computed gradient: {x.grad:.6f}")
    print(f"Difference: {abs(x.grad - expected_derivative):.10f}")
    print()


if __name__ == "__main__":
    print("ZenGrad - Basic Automatic Differentiation Examples\n")
    
    basic_operations()
    activation_functions()
    polynomial_example()
    chain_rule_example()
    
    print("All examples completed! 🎉")
