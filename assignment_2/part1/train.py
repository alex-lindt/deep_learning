################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

#### HELPERS ###################################################################

def acc_fn(predictions, targets):
    acc = torch.sum((predictions.argmax(dim=1)==targets)).item()/targets.size()[0]
    return acc

################################################################################

def train(config,out_dir = './res/'):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type=='RNN':
        model=VanillaRNN(seq_length=config.input_length,
                         input_dim=config.input_dim,
                         num_hidden=config.num_hidden,
                         num_classes=config.num_classes,
                         batch_size=config.batch_size,
                         device=device)

    if config.model_type=='LSTM':
        model=LSTM(seq_length=config.input_length,
                   input_dim=config.input_dim,
                   num_hidden=config.num_hidden,
                   num_classes=config.num_classes,
                   batch_size=config.batch_size,
                   device=device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(),
                              lr=config.learning_rate)

    # collect loss + acc
    loss = []
    acc  = [0,0,0]

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Data
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_inputs = batch_inputs.view(config.batch_size, config.input_dim, config.input_length)

        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()

        # pass through model:
        # batch=input -> model forward -> output -> loss -> model backward
        batch_outputs = model.forward(batch_inputs)
        loss_batch = loss_fn(batch_outputs, batch_targets)
        loss_batch.backward()

        # update weight matrices
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 100 == 0:
            accuracy = acc_fn(batch_outputs, batch_targets)

            loss.append(loss_batch.item())
            acc.append(accuracy)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss_batch
            ))

            if math.isclose(accuracy,acc[-2],abs_tol=1e-5) and \
                    math.isclose(accuracy,acc[-3],abs_tol=1e-5) and\
                        math.isclose(accuracy,acc[-4],abs_tol=1e-5):
                # If converged before max steps
                break


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    np.save(out_dir+'acc_' + config.model_type+'_'+str(config.input_length), acc[3:])
    np.save(out_dir+'loss_'+ config.model_type+'_'+str(config.input_length), loss)

    return max(acc), min(loss)

def test_different_palindrome_length(config):
    a_s = []
    l_s = []
    for i in range(5,80,3):
        a = l = 0
        for _ in range(3):
            config.input_length = i
            a_, l_  = train(config)
            a += a_
            l += l_
        a_s.append(a/3)
        l_s.append(l/3)
        #print(str(i)+":  "+str(a/3)+", "+str(l/3))

    np.save('./res/'+config.model_type+'_acc', a_s)
    np.save('./res/'+config.model_type+'_loss', l_s)
    return

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)

    # Test differen palindrome lengths
    # test_different_palindrome_length(config)
    # config.model_type = 'LSTM'
    # test_different_palindrome_length(config)

