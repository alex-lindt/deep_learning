"""
This module implements a multi-layer perceptron (MLP) in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object. 

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.

          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """

        # stack layers:
        self.layers = []

        # n linear layers with ReLU activation
        self.layers = []
        units = [n_inputs]+n_hidden
        
        for i in range(len(units)-1):
            self.layers.append(LinearModule(units[i], units[i+1]))
            self.layers.append(ReLUModule())

        # final linear layer
        self.layers.append(LinearModule(units[-1], n_classes))
        self.layers.append(SoftMaxModule())


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through 
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        
        for layer in self.layers:
            x = layer.forward(x)
    
        return x

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss. 

        Args:
          dout: gradients of the loss

        """
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            
        return
