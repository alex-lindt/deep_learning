import numpy as np


class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    """

    self.params = {
      # initialize weights using normal distribution with m=0 and std=0.0001
      'weight': np.random.normal(0, 0.0001, (in_features, out_features)),
      # initialize param biases with zeros
      'bias': np.zeros((1, out_features))
    }

    # initialize gradients with zeros
    self.grads = {
      'weight': np.zeros((in_features,out_features)),
      'bias': np.zeros((1,out_features))
    }

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
                                               
    """
    # store input for gradient calculation   
    self.x = x
    
    out = x @ self.params['weight'] + self.params['bias']

    return out


  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    """

    # dL/dx = dL/dx_tilde * W
    dx =  dout @ self.params["weight"].T

    # dL/dW = dL/dx_tilde * x
    self.grads["weight"]= self.x.T @ dout
    
    # dL/db = dL/dx_tilde
    # make mini-batch possible:
    self.grads["bias"] = np.sum(dout,axis=0,keepdims=True)
    
    return dx


class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    """
    # store input for gradient calculation
    self.x_tilde = x
    
    out = np.maximum(x,0)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    """
    
    # dL/dx_tilde = dL/dx * np.diag(x_tilde>0)
    dx =  (self.x_tilde > 0) * dout
 
    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

   """
    
    # compute softmax using max trick:
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    x_max = np.amax(x,axis=1,keepdims=True)
    self.out = np.exp(x-x_max) / np.sum(np.exp(x-x_max),axis=1,keepdims=True)

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    """

    #dL/dx_tilde = dL/dx * dx/dx_tilde
    i  = np.apply_along_axis(lambda x: np.diag(x) - x.reshape(-1, 1) @ x.reshape(-1, 1).T, 1, self.out)
    dx = np.einsum('ij, ijk -> ik', dout, i)

    return dx


class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def __init__(self):
    self.eps = 1e-6

  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    """
    out = np.mean(np.sum(-np.log(x+self.eps)*y, axis=1))
    
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    """

    dx = -(y/(x+self.eps))/x.shape[0]
    
    return dx
