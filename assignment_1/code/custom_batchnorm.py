import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object.

    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    """
    super(CustomBatchNormAutograd, self).__init__()

    self.n_neurons = n_neurons
    self.eps = eps
    self.beta  = nn.Parameter(torch.zeros(n_neurons))
    self.gamma = nn.Parameter(torch.ones(n_neurons))

  def forward(self, input):
    """
    Compute the batch normalization

    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    """

    # Check for the correctness of the shape of the input tensor
    assert self.n_neurons == input.size(1)

    # Implement batch normalization forward pass as given in the assignment
    mean = input.mean(dim=0)
    var  = input.var(dim=0, unbiased=False)
    out = self.gamma * (input-mean)/(var + self.eps).sqrt() + self.beta

    return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization

    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor
    """

    # Implement the forward pass of batch normalization
    mean = input.mean(dim=0)
    var = input.var(dim=0, unbiased=False)
    norm_in = (input - mean) / (var + eps).sqrt()
    out = gamma * norm_in + beta

    # Store necessary tensors for backward pass
    ctx.save_for_backward(input, norm_in, gamma, (var + eps).sqrt())

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.

    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    """
    # retrieve saved tensors
    x, x_hat, gamma, sqrt_var = ctx.saved_tensors

    # compute gradients for inputs
    calc_x_grad, calc_gamma_grad, calc_beta_grad = ctx.needs_input_grad
    grad_gamma = grad_beta = grad_input = None

    if calc_gamma_grad:
      grad_gamma = (grad_output * x_hat).sum(dim=0)

    if calc_beta_grad:
      grad_beta = grad_output.sum(dim=0)

    # gradient of x
    if calc_gamma_grad:
      B = grad_output.shape[0]
      grad_input = (gamma/(B * sqrt_var)) * \
                   (B * grad_output - grad_output.sum(dim=0) - x_hat * (grad_output * x_hat).sum(dim=0))

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.

    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    """
    super(CustomBatchNormManualModule, self).__init__()

    self.n_neurons = n_neurons
    self.eps = eps
    self.beta  = nn.Parameter(torch.zeros(n_neurons))
    self.gamma = nn.Parameter(torch.ones(n_neurons))

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction

    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    """

    # Check for the correctness of the shape of the input tensor
    assert self.n_neurons == input.size(1)

    # Instantiate a CustomBatchNormManualFunction & call it via its .apply() method
    out = CustomBatchNormManualFunction().apply(input, self.gamma, self.beta, self.eps)

    return out