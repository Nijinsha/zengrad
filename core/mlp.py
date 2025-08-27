from .layer import Layer


class MLP:
    """
    Multi-Layer Perceptron (Neural Network).
    
    This class represents a complete neural network consisting of multiple layers
    connected in sequence, where the output of one layer becomes the input to the next.
    """
    
    def __init__(self, nin, nouts):
        """
        Initialize an MLP with the specified architecture.
        
        Args:
            nin (int): Number of input features
            nouts (list): List of integers specifying the number of neurons in each layer
        """
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Forward pass through the entire network.
        
        Args:
            x (list): Input values
            
        Returns:
            Value or list: Output from the final layer
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Get all parameters from all layers in the network.
        
        Returns:
            list: List of all Value parameters from all layers
        """
        return [p for layer in self.layers for p in layer.parameters()]
