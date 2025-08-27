import random
from .value import Value


class Neuron:
    """
    A single neuron with weights, bias, and tanh activation.
    
    This class represents a single artificial neuron that takes multiple inputs,
    applies weights and bias, and outputs the result through a tanh activation function.
    """
    
    def __init__(self, nin):
        """
        Initialize a neuron with random weights and bias.
        
        Args:
            nin (int): Number of input connections
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        """
        Forward pass through the neuron.
        
        Args:
            x (list): Input values
            
        Returns:
            Value: Output after applying weights, bias, and tanh activation
        """
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()
    
    def parameters(self):
        """
        Get all parameters (weights and bias) of the neuron.
        
        Returns:
            list: List of all Value parameters
        """
        return self.w + [self.b]
