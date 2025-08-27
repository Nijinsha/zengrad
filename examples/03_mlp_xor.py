#!/usr/bin/env python3
"""
MLP XOR Training Example

This example demonstrates training a Multi-Layer Perceptron (MLP) to solve
the classic XOR problem, which requires a non-linear solution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Value, MLP, draw_dot
import random


def create_xor_dataset():
    """Create the XOR dataset."""
    # XOR truth table
    dataset = [
        ([0.0, 0.0], 0.0),  # 0 XOR 0 = 0
        ([0.0, 1.0], 1.0),  # 0 XOR 1 = 1
        ([1.0, 0.0], 1.0),  # 1 XOR 0 = 1
        ([1.0, 1.0], 0.0),  # 1 XOR 1 = 0
    ]
    
    # Convert to Values and normalize outputs to [-1, 1] for better training
    normalized_dataset = []
    for inputs, output in dataset:
        # Convert 0/1 to -1/1 for better neural network training
        norm_output = 2.0 * output - 1.0  # 0 → -1, 1 → 1
        normalized_dataset.append((inputs, norm_output))
    
    return normalized_dataset


def train_mlp_on_xor():
    """Train an MLP to solve the XOR problem."""
    print("=== MLP XOR Training ===")
    
    # Create XOR dataset
    dataset = create_xor_dataset()
    print("XOR Dataset (normalized to [-1, 1]):")
    for inputs, output in dataset:
        print(f"  {inputs} → {output}")
    
    # Create MLP: 2 inputs → 4 hidden → 4 hidden → 1 output
    mlp = MLP(2, [4, 4, 1])
    print(f"\nCreated MLP with architecture: 2 → 4 → 4 → 1")
    print(f"Total parameters: {len(mlp.parameters())}")
    
    # Training parameters
    learning_rate = 0.05
    epochs = 500
    
    print(f"\nTraining for {epochs} epochs with learning rate {learning_rate}")
    print("-" * 60)
    
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = Value(0.0)
        
        # Process each sample in the dataset multiple times per epoch
        for _ in range(10):  # Repeat dataset 10 times per epoch
            for inputs, target in dataset:
                # Convert to Values
                x = [Value(val) for val in inputs]
                y_target = Value(target)
                
                # Forward pass
                y_pred = mlp(x)
                
                # Make sure we get a scalar output
                if isinstance(y_pred, list):
                    y_pred = y_pred[0]
                
                # Compute loss (squared error)
                loss = (y_pred - y_target) ** 2
                total_loss = total_loss + loss
        
        # Backward pass
        # Zero gradients
        for param in mlp.parameters():
            param.grad = 0.0
        
        # Compute gradients
        total_loss.backward()
        
        # Update parameters
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
        
        loss_history.append(total_loss.data)
        
        # Print progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.data:.6f}")
            
            # Show current predictions
            print("  Current predictions:")
            for inputs, target in dataset:
                x = [Value(val) for val in inputs]
                pred = mlp(x)
                if isinstance(pred, list):
                    pred = pred[0]
                print(f"    {inputs} → {pred.data:6.3f} (target: {target:4.1f})")
            print()
    
    return mlp, dataset, loss_history


def test_mlp_accuracy(mlp, dataset):
    """Test the trained MLP's accuracy on the XOR problem."""
    print("=== Final Results ===")
    
    correct = 0
    print("Final predictions:")
    
    for inputs, target in dataset:
        x = [Value(val) for val in inputs]
        prediction = mlp(x)
        
        # Handle both scalar and list outputs
        if isinstance(prediction, list):
            prediction = prediction[0]
        
        # Convert back to binary classification
        predicted_binary = 1 if prediction.data > 0 else 0
        target_binary = 1 if target > 0 else 0
        
        is_correct = predicted_binary == target_binary
        if is_correct:
            correct += 1
        
        # Convert target back to original 0/1 scale for display
        original_target = 1 if target > 0 else 0
        
        print(f"  Input: {inputs} → Output: {prediction.data:6.3f} → "
              f"Binary: {predicted_binary} (Target: {original_target}) "
              f"{'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(dataset) * 100
    print(f"\nAccuracy: {correct}/{len(dataset)} = {accuracy:.1f}%")
    
    if accuracy == 100:
        print("🎉 Perfect! The MLP has learned the XOR function!")
    elif accuracy >= 75:
        print("👍 Good! The MLP has mostly learned the XOR function.")
    else:
        print("🤔 The MLP needs more training or a different architecture.")


def analyze_network(mlp):
    """Analyze what the network has learned."""
    print("\n=== Network Analysis ===")
    
    # Create a sample computation graph
    x = [Value(1.0, label='x1'), Value(0.0, label='x2')]
    output = mlp(x)
    
    if isinstance(output, list):
        output = output[0]
    output.label = 'mlp_output'
    
    print(f"Sample computation: [1.0, 0.0] → {output.data:.4f}")
    
    # Show parameter statistics
    params = mlp.parameters()
    param_values = [p.data for p in params]
    
    print(f"\nParameter statistics:")
    print(f"  Total parameters: {len(params)}")
    print(f"  Parameter range: [{min(param_values):.3f}, {max(param_values):.3f}]")
    print(f"  Average parameter: {sum(param_values)/len(param_values):.3f}")
    
    # Try to visualize if possible
    try:
        dot = draw_dot(output)
        if dot:
            print("\nComputational graph created (use dot.view() to visualize)")
        else:
            print("\nInstall graphviz to see computational graph: pip install graphviz")
    except Exception as e:
        print(f"\nVisualization not available: {e}")


if __name__ == "__main__":
    print("ZenGrad - MLP XOR Training Example\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Train the MLP
    trained_mlp, xor_data, losses = train_mlp_on_xor()
    
    # Test accuracy
    test_mlp_accuracy(trained_mlp, xor_data)
    
    # Analyze the network
    analyze_network(trained_mlp)
    
    print(f"\nExample completed! 🎉")
    print(f"The MLP learned to solve XOR, demonstrating non-linear function approximation.")
