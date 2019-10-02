# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import time
from datetime import datetime
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel


#### HELPERS ###################################################################

def acc_fn(predictions, targets):
    acc =  torch.sum(predictions.argmax(dim=1)==targets).item()/(targets.size()[0]*targets.size()[1])
    return acc

def save_model(model, step):
    path = './res/model_checkpoints/'+str(step)+'.pt'
    torch.save(model.state_dict(), path)
    return

def one_hot(indices, vocab_size):
    """Convert input into one-hot vectors."""
    out = torch.zeros(size=(*indices.size(),vocab_size),
                      device=indices.device)
    out.scatter_(2, indices.view(*indices.size(),1), 1)
    return out

def sample_sequence(model, dataset,  seq_length, device, sampling="random", temp=0.5):
    """ Sample char sequence from model."""
    with torch.no_grad():
        h_c = None
        s = torch.randint(dataset.vocab_size, (1,1), device=device)
        seq = [s.view(1).item()]

        for i in range(seq_length-1):

            # forward pass
            s_one_hot = one_hot(s,dataset.vocab_size)
            out, h_c = model.forward(s_one_hot, h_c)
            out = out.view(-1)

            # sampling next character
            if sampling=="random":
                softmax = torch.softmax(out/temp, dim=0)
                s = torch.multinomial(softmax, 1).view(1,1)
            else:
                s = out.argmax().view(1,1)

            # adding to seq
            seq.append(s.item())
    return dataset.convert_to_string(seq)

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size,
                                seq_length=config.seq_length,
                                vocabulary_size=dataset.vocab_size,
                                lstm_num_hidden = config.lstm_num_hidden,
                                lstm_num_layers = config.lstm_num_layers,
                                device = device)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.RMSprop(model.parameters(),
                              lr = config.learning_rate)
    # collect loss + acc
    loss = []
    acc  = []

    data_loader_it = iter(data_loader)

    for step in range(int(config.train_steps)):

        batch_inputs, batch_targets = next(data_loader_it)

        # full batch ?
        # if batch is not full, reset iterator over data
        if batch_inputs[0].size()[0] != config.batch_size:
            data_loader_it = iter(data_loader) # reset dataset to beginning
            batch_inputs, batch_targets = next(data_loader_it)

        # Only for time measurement of step through network
        t1 = time.time()

        # update input/output batch
        batch_inputs = torch.stack(batch_inputs, dim=1).to(device)
        batch_inputs = one_hot(batch_inputs, dataset.vocab_size)
        batch_targets = torch.stack(batch_targets, dim=1).to(device)

        optimizer.zero_grad()

        # pass through model:
        # batch=input -> model forward -> output -> loss -> model backward
        batch_outputs, _ = model.forward(batch_inputs)
        batch_outputs = batch_outputs.transpose(2, 1)

        loss_batch = loss_fn(batch_outputs, batch_targets)
        loss_batch.backward()

        # update weight matrices
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            accuracy = acc_fn(batch_outputs, batch_targets)

            loss.append(loss_batch.item())
            acc.append(accuracy)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss_batch.item()
            ))

        if step % config.sample_every == 0 :
            save_model(model, step)

            for _  in range(5):

                sequence = sample_sequence(model=model,
                                           dataset=dataset,
                                           device=device,
                                           seq_length=config.seq_length,
                                           sampling="greedy")
                #print('g   '+sequence)

                for temp in [0.5, 1, 2]:
                    sequence =  sample_sequence(model=model,
                                                dataset=dataset,
                                                device=device,
                                                seq_length=config.seq_length,
                                                sampling="random",
                                                temp=temp)
                    #print(str(temp)+'   '+sequence)


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='./assets/book_EN_grimms_fairy_tails.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    parser.add_argument('--device', type=str, default='cuda', help='Device to used "cuda" or "cpu"')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=200, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=20000, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)

