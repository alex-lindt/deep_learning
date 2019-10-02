"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
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

    super(MLP, self).__init__()

    # stack layers:
    self.layers = []

    # n linear layers with ReLU activation
    self.layers = []
    units = [n_inputs] + n_hidden

    for i in range(len(units) - 1):
      self.layers.append(nn.Linear(units[i], units[i + 1]))
      self.layers.append(nn.ReLU())

    # final linear layer
    self.layers.append(nn.Linear(units[-1], n_classes))
    self.layers.append(nn.Softmax())

    self.sequential = nn.Sequential(*self.layers)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    """
    out = self.sequential(x)

    return out
