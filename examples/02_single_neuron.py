#!/usr/bin/env python3
"""
Single Neuron Training Example

This example shows how to create and train a single neuron to learn
a simple binary classification task.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Value, Neuron, draw_dot
import random


def create_dataset():
    """Create a simple 2D dataset for binary classification."""
    # Generate some 2D points with labels
    # Positive class: points where x1 + x2 > 0
    # Negative class: points where x1 + x2 <= 0
    
    dataset = []
    for _ in range(20):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        
        # Simple linear decision boundary
        label = 1.0 if x1 + x2 > 0 else -1.0
        
        dataset.append(([x1, x2], label))
    
    return dataset


def train_neuron():
    """Train a single neuron on the dataset."""
    print("=== Single Neuron Training ===")
    
    # Create dataset
    dataset = create_dataset()
    print(f"Created dataset with {len(dataset)} samples")
    
    # Create a neuron with 2 inputs
    neuron = Neuron(2)
    print(f"Created neuron with {len(neuron.parameters())} parameters")
    
    # Training parameters
    learning_rate = 0.1
    epochs = 100
    
    print(f"\nTraining for {epochs} epochs with learning rate {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        total_loss = Value(0.0)
        correct_predictions = 0
        
        # Process each sample in the dataset
        for inputs, target in dataset:
            # Convert inputs to Values
            x = [Value(val) for val in inputs]
            y_target = Value(target)
            
            # Forward pass
            y_pred = neuron(x)
            
            # Compute loss (squared error)
            loss = (y_pred - y_target) ** 2
            total_loss = total_loss + loss
            
            # Check if prediction is correct (same sign as target)
            if (y_pred.data > 0 and target > 0) or (y_pred.data < 0 and target < 0):
                correct_predictions += 1
        
        # Backward pass
        # Zero gradients
        for param in neuron.parameters():
            param.grad = 0.0
        
        # Compute gradients
        total_loss.backward()
        
        # Update parameters
        for param in neuron.parameters():
            param.data -= learning_rate * param.grad
        
        # Calculate accuracy
        accuracy = correct_predictions / len(dataset) * 100
        
        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.data:.4f}, Accuracy = {accuracy:.1f}%")
    
    return neuron, dataset


def test_neuron(neuron, dataset):
    """Test the trained neuron on some examples."""
    print(f"\n=== Testing Trained Neuron ===")
    
    # Test on training data
    correct = 0
    print("Sample predictions:")
    
    for i, (inputs, target) in enumerate(dataset[:5]):  # Show first 5 examples
        x = [Value(val) for val in inputs]
        prediction = neuron(x)
        
        predicted_class = "+" if prediction.data > 0 else "-"
        actual_class = "+" if target > 0 else "-"
        is_correct = predicted_class == actual_class
        
        if is_correct:
            correct += 1
        
        print(f"  Input: [{inputs[0]:5.2f}, {inputs[1]:5.2f}] → "
              f"Pred: {predicted_class} (Actual: {actual_class}) "
              f"{'✓' if is_correct else '✗'}")
    
    # Test on some new points
    print(f"\nTesting on new points:")
    test_points = [
        [1.5, 1.0],   # Should be positive
        [-1.0, -1.5], # Should be negative  
        [2.0, -1.8],  # Should be positive
        [-0.5, 0.3]   # Should be negative
    ]
    
    for point in test_points:
        x = [Value(val) for val in point]
        prediction = neuron(x)
        predicted_class = "+" if prediction.data > 0 else "-"
        expected = "+" if sum(point) > 0 else "-"
        
        print(f"  Input: [{point[0]:5.2f}, {point[1]:5.2f}] → "
              f"Pred: {predicted_class} (Expected: {expected}) "
              f"Output: {prediction.data:.3f}")


def visualize_neuron(neuron):
    """Create a simple computational graph visualization."""
    print(f"\n=== Neuron Visualization ===")
    
    # Create a simple forward pass to visualize
    x = [Value(1.0, label='x1'), Value(0.5, label='x2')]
    output = neuron(x)
    output.label = 'neuron_output'
    
    print(f"Sample computation: inputs [1.0, 0.5] → output {output.data:.4f}")
    
    # Try to draw the computational graph
    try:
        dot = draw_dot(output)
        if dot:
            print("Computational graph created (use dot.view() to visualize)")
        else:
            print("Install graphviz to see computational graph: pip install graphviz")
    except Exception as e:
        print(f"Visualization not available: {e}")


if __name__ == "__main__":
    print("ZenGrad - Single Neuron Training Example\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Train the neuron
    trained_neuron, training_data = train_neuron()
    
    # Test the neuron
    test_neuron(trained_neuron, training_data)
    
    # Show visualization
    visualize_neuron(trained_neuron)
    
    print(f"\nExample completed! 🎉")
    print(f"The neuron learned to classify points based on whether x1 + x2 > 0")
