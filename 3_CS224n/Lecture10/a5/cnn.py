#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, k, e_char, e_word):
        super(CNN,self).__init__()

        self.Conv1D = nn.Conv1d(in_channels = e_char, out_channels = e_word, kernel_size = k, bias = True)
        self.relu = nn.ReLU()

    def forward(self, x_reshaped):
        # x_reshaped : [e_char, m_word, batch_size]
        x_reshaped = x_reshaped.permute(2,0,1)
        # x_reshape : [batch size, e_char, m_word]

        # W : [e_word, e_char, k], b : [e_word] (e_word: number of filters, k : kernerl size)
        # x_conv(i,t) = sum(W[i,:,:] ** x_reshaped[:,t:t+k-1]) + b_i
        # ---> x_conv = Conv1D(x_reshaped)
        x_conv = self.Conv1D(x_reshaped)
        # x_conv : [batch size, e_word, m_word-k+1]
        
        # x_conv_out = MaxPool(ReLU(x_conv))
        x_conv_out, _ = torch.max(self.relu(x_conv),2)
        # x_conv_out : [batch size, e_word]
        
        return x_conv_out

    ### END YOUR CODE

