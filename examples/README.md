# ZenGrad Examples

This directory contains practical examples demonstrating how to use ZenGrad for automatic differentiation and neural network training.

## Examples Overview

### 1. Basic Automatic Differentiation (`01_basic_autograd.py`)

**What it demonstrates:**

- Core automatic differentiation with the `Value` class
- Mathematical operations (+, -, \*, /, \*\*)
- Activation functions (tanh, exp)
- Chain rule in action
- Polynomial and nested function examples

**Run it:**

```bash
python examples/01_basic_autograd.py
```

**Key concepts:**

- Forward pass computation
- Backward pass gradient calculation
- Chain rule application

### 2. Single Neuron Training (`02_single_neuron.py`)

**What it demonstrates:**

- Creating and training a single neuron
- Binary classification on a simple 2D dataset
- Parameter updates with gradient descent
- Model evaluation and testing

**Run it:**

```bash
python examples/02_single_neuron.py
```

**Key concepts:**

- Neuron architecture (weights + bias)
- Training loop (forward → loss → backward → update)
- Decision boundaries for classification

### 3. MLP XOR Problem (`03_mlp_xor.py`)

**What it demonstrates:**

- Multi-layer perceptron (MLP) architecture
- Solving the classic XOR problem (non-linear classification)
- Training a deep network with multiple layers
- Network analysis and parameter statistics

**Run it:**

```bash
python examples/03_mlp_xor.py
```

**Key concepts:**

- Multi-layer networks
- Non-linear function approximation
- Hidden layer representations
- Deep learning fundamentals

### 4. Computational Graph Visualization (`04_visualization.py`)

**What it demonstrates:**

- Visualizing computational graphs with graphviz
- Understanding network structure
- Graph analysis and statistics
- Different complexity levels (expressions → neurons → MLPs)

**Run it:**

```bash
python examples/04_visualization.py
```

**Key concepts:**

- Computational graph structure
- Visual debugging of networks
- Understanding data flow

## Prerequisites

### Basic Requirements

All examples work with just the core ZenGrad library (no external dependencies).

### Optional: Graph Visualization

For the visualization example (`04_visualization.py`), install graphviz:

```bash
pip install graphviz
```

**Note:** You may also need to install the graphviz system package:

- **macOS:** `brew install graphviz`
- **Ubuntu/Debian:** `sudo apt-get install graphviz`
- **Windows:** Download from https://graphviz.org/download/

## Running the Examples

### Individual Examples

```bash
# From the project root directory
python examples/01_basic_autograd.py
python examples/02_single_neuron.py
python examples/03_mlp_xor.py
python examples/04_visualization.py
```

### All Examples at Once

```bash
# Run all examples sequentially
for example in examples/*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

## Example Progression

The examples are designed to build upon each other:

1. **Start with `01_basic_autograd.py`** to understand the fundamental automatic differentiation engine
2. **Move to `02_single_neuron.py`** to see how individual neurons work and train
3. **Progress to `03_mlp_xor.py`** to understand multi-layer networks and complex problems
4. **Finish with `04_visualization.py`** to visualize and debug your understanding

## Expected Output

Each example includes:

- ✅ Clear progress indicators
- 📊 Training metrics (loss, accuracy)
- 📈 Gradient information
- 🎯 Final results and analysis
- 🎉 Success confirmation

## Learning Objectives

After running these examples, you should understand:

- **Automatic Differentiation:** How gradients are computed automatically
- **Neural Networks:** How neurons, layers, and networks work together
- **Training Process:** Forward pass, loss computation, backpropagation, parameter updates
- **Problem Solving:** How to approach different types of machine learning problems
- **Debugging:** How to visualize and analyze your networks

## Customization Ideas

Try modifying the examples:

1. **Change architectures:** Different layer sizes, more/fewer layers
2. **Modify datasets:** Create your own classification problems
3. **Experiment with learning rates:** See how it affects convergence
4. **Add new activation functions:** Implement ReLU, sigmoid, etc.
5. **Try different loss functions:** Mean absolute error, cross-entropy
6. **Add regularization:** Weight decay, dropout concepts

## Troubleshooting

### Common Issues

**Import errors:**

- Make sure you're running from the project root directory
- Check that the `core` module is in the correct location

**Slow training:**

- Try different learning rates (0.01, 0.1, 0.5)
- Increase/decrease the number of epochs
- Modify network architecture

**Visualization not working:**

- Install graphviz: `pip install graphviz`
- Install system graphviz package
- Examples will still run without visualization

## Next Steps

After completing these examples:

1. **Implement new features:** Add more activation functions, optimizers
2. **Solve new problems:** Try regression, multi-class classification
3. **Build larger networks:** Experiment with deeper architectures
4. **Compare with PyTorch:** Understand the similarities and differences
5. **Contribute:** Add your own examples to help others learn!

---

Happy learning with ZenGrad! 🚀
