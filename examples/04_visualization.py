#!/usr/bin/env python3
"""
Computational Graph Visualization Example

This example demonstrates how to visualize computational graphs using
ZenGrad's built-in visualization tools with graphviz.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Value, Neuron, MLP, draw_dot, trace


def simple_expression_graph():
    """Create and visualize a simple mathematical expression."""
    print("=== Simple Expression Graph ===")
    
    # Create a simple expression: f = (a * b + c) ** 2
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    
    d = a * b
    d.label = 'a*b'
    
    e = d + c
    e.label = 'a*b+c'
    
    f = e ** 2
    f.label = '(a*b+c)²'
    
    print(f"Expression: f = (a * b + c)² = ({a.data} * {b.data} + {c.data})² = {f.data}")
    
    # Compute gradients
    f.backward()
    
    print(f"Gradients:")
    print(f"  ∂f/∂a = {a.grad}")
    print(f"  ∂f/∂b = {b.grad}")
    print(f"  ∂f/∂c = {c.grad}")
    
    # Visualize
    print(f"\nVisualizing computational graph...")
    return visualize_graph(f, "simple_expression")


def activation_function_graph():
    """Create and visualize activation function graphs."""
    print("\n=== Activation Function Graph ===")
    
    # Create a more complex expression with activations
    x = Value(0.5, label='x')
    w = Value(2.0, label='w')
    b = Value(-1.0, label='b')
    
    # Linear transformation
    linear = x * w + b
    linear.label = 'x*w+b'
    
    # Apply tanh activation
    activated = linear.tanh()
    activated.label = 'tanh(x*w+b)'
    
    # Square the result
    output = activated ** 2
    output.label = 'output'
    
    print(f"Expression: output = tanh(x*w + b)²")
    print(f"           = tanh({x.data}*{w.data} + {b.data})²")
    print(f"           = {output.data:.4f}")
    
    # Compute gradients
    output.backward()
    
    print(f"Gradients:")
    print(f"  ∂output/∂x = {x.grad:.4f}")
    print(f"  ∂output/∂w = {w.grad:.4f}")
    print(f"  ∂output/∂b = {b.grad:.4f}")
    
    # Visualize
    print(f"\nVisualizing activation function graph...")
    return visualize_graph(output, "activation_function")


def neuron_graph():
    """Create and visualize a single neuron's computational graph."""
    print("\n=== Single Neuron Graph ===")
    
    # Create a neuron with 3 inputs
    neuron = Neuron(3)
    
    # Create some input data
    inputs = [Value(0.5, label='x1'), Value(-0.3, label='x2'), Value(0.8, label='x3')]
    
    # Forward pass
    output = neuron(inputs)
    output.label = 'neuron_output'
    
    print(f"Neuron inputs: {[x.data for x in inputs]}")
    print(f"Neuron output: {output.data:.4f}")
    
    # Compute gradients
    output.backward()
    
    print(f"Input gradients:")
    for i, inp in enumerate(inputs):
        print(f"  ∂output/∂x{i+1} = {inp.grad:.4f}")
    
    print(f"Parameter gradients:")
    params = neuron.parameters()
    for i, param in enumerate(params[:-1]):  # weights
        print(f"  ∂output/∂w{i+1} = {param.grad:.4f}")
    print(f"  ∂output/∂bias = {params[-1].grad:.4f}")
    
    # Visualize
    print(f"\nVisualizing neuron computational graph...")
    return visualize_graph(output, "single_neuron")


def mlp_graph():
    """Create and visualize a small MLP's computational graph."""
    print("\n=== Small MLP Graph ===")
    
    # Create a very small MLP to keep the graph manageable
    mlp = MLP(2, [2, 1])  # 2 inputs → 2 hidden → 1 output
    
    # Create input data
    inputs = [Value(1.0, label='x1'), Value(-0.5, label='x2')]
    
    # Forward pass
    output = mlp(inputs)
    if isinstance(output, list):
        output = output[0]
    output.label = 'mlp_output'
    
    print(f"MLP inputs: {[x.data for x in inputs]}")
    print(f"MLP output: {output.data:.4f}")
    print(f"MLP has {len(mlp.parameters())} parameters")
    
    # Compute gradients
    output.backward()
    
    print(f"Input gradients:")
    for i, inp in enumerate(inputs):
        print(f"  ∂output/∂x{i+1} = {inp.grad:.4f}")
    
    # Visualize
    print(f"\nVisualizing small MLP computational graph...")
    print("Warning: MLP graphs can be quite large!")
    return visualize_graph(output, "small_mlp")


def visualize_graph(root_node, name):
    """Helper function to visualize a computational graph."""
    try:
        # Create the graph
        dot = draw_dot(root_node)
        
        if dot is None:
            print("❌ Graphviz not available. Install with: pip install graphviz")
            return False
        
        print(f"✅ Graph created successfully!")
        print(f"   To view the graph, you can:")
        print(f"   1. Save it: dot.render('{name}', format='png')")
        print(f"   2. View it: dot.view() (opens in default viewer)")
        print(f"   3. Get SVG: dot.pipe(format='svg')")
        
        # Show graph statistics
        nodes, edges = trace(root_node)
        print(f"   Graph stats: {len(nodes)} nodes, {len(edges)} edges")
        
        return dot
        
    except ImportError:
        print("❌ Graphviz not available. Install with: pip install graphviz")
        return False
    except Exception as e:
        print(f"❌ Error creating graph: {e}")
        return False


def graph_analysis_example():
    """Demonstrate how to analyze computational graphs programmatically."""
    print("\n=== Graph Analysis Example ===")
    
    # Create a moderately complex expression
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    
    # f = x²y + xy² + xy
    x2 = x * x
    x2.label = 'x²'
    
    y2 = y * y  
    y2.label = 'y²'
    
    term1 = x2 * y
    term1.label = 'x²y'
    
    term2 = x * y2
    term2.label = 'xy²'
    
    term3 = x * y
    term3.label = 'xy'
    
    f = term1 + term2 + term3
    f.label = 'f'
    
    print(f"Expression: f = x²y + xy² + xy")
    print(f"At x={x.data}, y={y.data}: f = {f.data}")
    
    # Analyze the graph structure
    nodes, edges = trace(f)
    
    print(f"\nGraph structure analysis:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    
    # Count different types of operations
    ops = {}
    for node in nodes:
        op = node._op if node._op else 'input'
        ops[op] = ops.get(op, 0) + 1
    
    print(f"  Operation counts:")
    for op, count in sorted(ops.items()):
        print(f"    {op}: {count}")
    
    # Compute gradients and show
    f.backward()
    
    print(f"\nGradients:")
    print(f"  ∂f/∂x = {x.grad} (expected: 2xy + y² + y = {2*x.data*y.data + y.data**2 + y.data})")
    print(f"  ∂f/∂y = {y.grad} (expected: x² + 2xy + x = {x.data**2 + 2*x.data*y.data + x.data})")
    
    return visualize_graph(f, "complex_expression")


if __name__ == "__main__":
    print("ZenGrad - Computational Graph Visualization Examples\n")
    
    # Run all visualization examples
    graphs = []
    
    graphs.append(simple_expression_graph())
    graphs.append(activation_function_graph())
    graphs.append(neuron_graph())
    graphs.append(mlp_graph())
    graphs.append(graph_analysis_example())
    
    print(f"\n" + "="*60)
    print("Summary:")
    successful_graphs = sum(1 for g in graphs if g)
    print(f"Successfully created {successful_graphs}/{len(graphs)} graphs")
    
    if successful_graphs > 0:
        print(f"\nTo save and view graphs in Python:")
        print(f"  import subprocess")
        print(f"  dot = draw_dot(your_value)")
        print(f"  dot.render('graph_name', format='png')  # Saves as graph_name.png")
        print(f"  subprocess.run(['open', 'graph_name.png'])  # Opens on macOS")
    
    print(f"\nVisualization examples completed! 🎉")
