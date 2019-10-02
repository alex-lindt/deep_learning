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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        self.device = device

        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size

        # weights
        self.W_fx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self.W_fh = nn.Parameter(torch.zeros(num_hidden, num_hidden))

        self.W_gx = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self.W_gh = nn.Parameter(torch.zeros(num_hidden, num_hidden))

        self.W_ix = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self.W_ih = nn.Parameter(torch.zeros(num_hidden, num_hidden))

        self.W_ox = nn.Parameter(torch.zeros(input_dim, num_hidden))
        self.W_oh = nn.Parameter(torch.zeros(num_hidden, num_hidden))

        self.W_ph = nn.Parameter(torch.zeros(num_hidden, num_classes))

        # biases
        self.b_f = nn.Parameter(torch.zeros(num_hidden))
        self.b_g = nn.Parameter(torch.zeros(num_hidden))
        self.b_i = nn.Parameter(torch.zeros(num_hidden))
        self.b_o = nn.Parameter(torch.zeros(num_hidden))
        self.b_p = nn.Parameter(torch.zeros(num_classes))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # initialize weight matrices
        self.init_weights([self.W_fx, self.W_fh, self.W_gx, self.W_gh, self.W_ix,
                           self.W_ih, self.W_ox, self.W_oh, self.W_ph])

        # assign to device
        self.to(device)

    def init_weights(self, weights):
        for w in weights:
            nn.init.xavier_uniform_(w)
        return

    def forward(self, x):
        # init h_0 and c_0 with zeros
        h = torch.zeros(self.batch_size,self.num_hidden,device=self.device)
        c = torch.zeros(self.batch_size, self.num_hidden,device=self.device)

        # for-loop to step through time
        for idx in range(self.seq_length):
            g = self.tanh(h @ self.W_gh + x[:, :, idx] @ self.W_gx + self.b_g)
            f = self.tanh(h @ self.W_fh + x[:, :, idx] @ self.W_fx + self.b_f)
            i = self.tanh(h @ self.W_ih + x[:, :, idx] @ self.W_ix + self.b_i)
            o = self.tanh(h @ self.W_oh + x[:, :, idx] @ self.W_ox + self.b_o)

            c = g*i + c*f
            h = self.tanh(c)*o

        return self.softmax(h @ self.W_ph + self.b_p)