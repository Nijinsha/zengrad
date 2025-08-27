from .neuron import Neuron


class Layer:
    """
    A layer of neurons in a neural network.
    
    This class represents a fully connected layer containing multiple neurons,
    where each neuron receives the same inputs and produces one output.
    """
    
    def __init__(self, nin, nout):
        """
        Initialize a layer with the specified number of neurons.
        
        Args:
            nin (int): Number of inputs to each neuron
            nout (int): Number of neurons in the layer (outputs)
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x (list): Input values
            
        Returns:
            Value or list: Single Value if one neuron, list of Values otherwise
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """
        Get all parameters from all neurons in the layer.
        
        Returns:
            list: List of all Value parameters from all neurons
        """
        return [p for n in self.neurons for p in n.parameters()]
