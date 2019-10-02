"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt
import datetime
import torch.nn as nn
from torch.optim import Adam, SGD
import torch
import logging

logging.basicConfig(filename='./res/mlp.log', filemode='a', format='%(message)s')
logging.warning('OPTIMZE THE MLP')

# Default constants

DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OTIMIZER= 'SGD'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
FLAGS = None

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')

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
    s = s.reshape((batch_size, -1))
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
    Performs training and evaluation of MLP model.

    """

    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    logging.warning("_____________________________________")
    logging.warning(str(dnn_hidden_units) + "_" + str(FLAGS.max_steps) + "_" + str(FLAGS.batch_size) + "_" + str(
        FLAGS.learning_rate))

    # get data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # read input + output size
    _, d, w, h = cifar10["train"].images.shape
    input_size = d * w * h
    _, output_size = cifar10["train"].labels.shape

    # define MLP and lossfunction
    mlp = MLP(input_size, dnn_hidden_units, output_size).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    if OTIMIZER=="SGD":
        optimizer = SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer = Adam(mlp.parameters(), lr=FLAGS.learning_rate)

    # read test data (we use entire test set in every evaluation)
    num_test_samples = cifar10["test"].images.shape[0]
    s_test, t_test = load_batch(cifar10["test"], num_test_samples)

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
        output_batch = mlp.forward(s_batch)
        loss_batch = loss_fn(output_batch, t_batch)
        loss_batch.backward()

        # update linear layers with SGD
        optimizer.step()

        # evaluate mlp performance on test set, document loss & accuracy
        if step % FLAGS.eval_freq == 0:
            ouput_test = mlp.forward(s_test)
            loss_test = loss_fn(ouput_test, t_test)

            # document test loss + acc
            loss['test'] += [loss_test.item()]
            acc_test = accuracy(ouput_test, t_test)
            acc['test']  += [acc_test]

            # document train loss+acc
            print(loss_batch.item())
            loss['train'] += [loss_batch.item()]
            acc_train = accuracy(output_batch, t_batch)
            acc['train'] += [acc_train]

            logging.warning('Step: {:4d}\t\tLoss (Train) {:4f}\tAcc (Train) {:f}\t\t Loss (Test){:4f}\tAcc (Test) {:4f}'.format(
                step, loss_batch.item(), acc_train, loss_test.item(),acc_test
            ))

    #visualize_loss_acc(loss, acc, FLAGS.max_steps, FLAGS.eval_freq)
    best_tuple = (min(loss['train']),max(acc['train']),min(loss['test']),max(acc['test']))
    logging.warning('BEST: \t\tLoss (Train) {:f}\tAcc (Train) {:f}\t\t Loss (Test){:f}\tAcc (Test) {:f}'.format(*best_tuple))

    return

def optimize():

    max_steps = [2000, 5000]
    learning_rates = [2e-5]
    dnn_hidden_units = ['256, 256, 256',
                        '512, 256, 128, 64',
                        '1024, 512, 256, 128']
    batch_sizes = [200, 600, 1000]

    for i in dnn_hidden_units:
        for ii in max_steps:
            for iii in learning_rates:
                for iv in batch_sizes:

                    FLAGS.dnn_hidden_units = i
                    FLAGS.max_steps = ii
                    FLAGS.learning_rate = iii
                    FLAGS.batch_size = iv
                    FLAGS.eval_freq = int(ii/20)

                    train()

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
#    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    train()