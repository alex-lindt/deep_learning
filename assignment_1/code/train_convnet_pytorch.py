"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import matplotlib.pyplot as plt
import datetime
import torch.nn as nn
from torch.optim import Adam
import torch
import logging

logging.basicConfig(filename='./res/cnn.log', filemode='a', format='%(message)s')
logging.warning('TRAIN THE CNN')

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'Adam'

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  """
  acc = torch.sum((predictions.argmax(dim=1) == targets)).item() / targets.size()[0]
  return acc

def load_batch(batch_data, batch_size):
  s, t = batch_data.next_batch(batch_size)
  return torch.from_numpy(s).to(DEVICE), torch.from_numpy(t).argmax(dim=1).to(DEVICE)

def visualize_loss_acc(loss, acc, max_steps, eval_freq):
    steps = range(0, int(max_steps), int(eval_freq))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.plot(steps, loss['train'])
    plt.plot(steps, loss['test'])
    plt.legend(('Training Loss', 'Test Loss'),
               loc='upper right')
    plt.title('Loss over Time')

    plt.subplot(1,2,2)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.plot(steps, acc['train'])
    plt.plot(steps, acc['test'])
    plt.legend(('Training Accuracy', 'Test Accuracy'),
               loc='lower right')
    plt.title('Accuracy over Time')

    plt.savefig(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_loss_accuracy.png')
    return

def train():
  """
  Performs training and evaluation of ConvNet model.

  """
  # Set the random seeds for reproducibility
  np.random.seed(42)

  # get data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  # read sizeof input and output
  num__samples = cifar10["test"].images.shape[0]

  # define CNN and lossfunction
  cnn  = ConvNet(3, 10).to(DEVICE)
  loss_fn = nn.CrossEntropyLoss()

  # define optimizer
  optimizer = Adam(cnn.parameters(), lr=FLAGS.learning_rate)

  # collect losses+accuracies
  loss = {'train': [], 'test': []}
  acc  = {'train': [], 'test': []}

  for step in range(FLAGS.max_steps):

      # set optimizer gradient to zero
      # (otherwise gradient is accumulated)
      optimizer.zero_grad()

      # load next batch of data
      s_batch, t_batch = load_batch(cifar10['train'], FLAGS.batch_size)

      # pass through model:
      # batch=input -> model forward -> output -> loss -> model backward
      output_batch = cnn.forward(s_batch)
      loss_batch = loss_fn(output_batch, t_batch)

      # update linear layers with SGD
      loss_batch.backward()
      optimizer.step()

      # evaluate mlp performance on test set, document loss & accuracy
      if step % FLAGS.eval_freq == 0:

          s_test, t_test = load_batch(cifar10['test'], FLAGS.batch_size)

          ouput_test = cnn.forward(s_test)
          loss_test =  loss_fn(ouput_test, t_test)

          # document test loss + acc
          loss['test'] += [loss_test.item()]
          acc_test = accuracy(ouput_test, t_test)
          acc['test'] += [acc_test]

          # document train loss+acc
          loss['train'] += [loss_batch.item()]
          acc_train = accuracy(output_batch, t_batch)
          acc['train'] += [acc_train]

          logging.warning('Step {:d}\t\tLoss_TR {:f}\tAcc_TR {:f}\t\t Loss_TE {:f}\tAcc_TE {:f}'
                          .format(step, loss['train'][-1],acc['train'][-1],loss['test'][-1], acc['test'][-1]))

  visualize_loss_acc(loss, acc, FLAGS.max_steps, FLAGS.eval_freq)
  best_tuple = (min(loss['train']), max(acc['train']), min(loss['test']), max(acc['test']))
  logging.warning(
      'BEST: \t\tLoss (Train) {:f}\tAcc (Train) {:f}\t\t Loss (Test){:f}\tAcc (Test) {:f}'.format(*best_tuple))

  return


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()