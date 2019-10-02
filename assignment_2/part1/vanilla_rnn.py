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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size

        # weights
        self.W_hx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self.W_hh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self.W_ph = nn.Parameter(torch.zeros(num_hidden, num_classes))

        # biases
        self.b_h = nn.Parameter(torch.zeros(num_hidden))
        self.b_p = nn.Parameter(torch.zeros(num_classes))

        # tanh layer
        self.tanh = nn.Tanh()

        # initialize weight matrices
        self.init_weights([self.W_hx,self.W_hh,self.W_ph])

        # assign to device
        self.to(device)

    def init_weights(self, weights):
        for w in weights:
            nn.init.xavier_uniform_(w)
        return

    def forward(self, x):
        # init h_0 with zeros
        h = torch.zeros(self.batch_size, self.num_hidden)
        # apply Vanilla RNN on each input of the input sequence (for-loop to step through time)
        for i in range(self.seq_length):
            h = self.tanh(x[:,:,i] @ self.W_hx + h @ self.W_hh + self.b_h)
        # calculate p
        p = h @ self.W_ph + self.b_p
        return p