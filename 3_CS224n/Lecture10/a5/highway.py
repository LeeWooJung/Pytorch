#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, e_word, dropout):
        super(Highway, self).__init__()

        self.W_proj = nn.Linear(e_word, e_word, bias = True)
        self.W_gate = nn.Linear(e_word, e_word, bias = True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_conv_out):
        # x_conv_out : [batch size, e_word]
        # x_proj = ReLU(W_proj * x_conv_out + b_proj)
        # W_proj : [e_word, e_word]
        # x_proj : [e_word, batch size]
        x_proj = self.relu(self.W_proj(x_conv_out))

        # x_gate = sigmoid(W_gate * x_conv_out + b_gate)
        # W_gate : [e_word, e_word]
        # x_gate : [e_word, batch size]
        x_gate = self.sigmoid(self.W_gate(x_conv_out))

        # x_highway = x_gate ** x_proj + (1-x_gate) ** x_conv_out
        # x_highway: [e_word, batch size]
        x_highway = x_gate * x_proj + (torch.ones_like(x_gate) - x_gate) * x_conv_out

        # x_word_emb = Dropout(x_highway)
        # x_word_emb : [e_word, batch size]
        x_word_emb = self.dropout(x_highway)

        return x_word_emb

    ### END YOUR CODE

