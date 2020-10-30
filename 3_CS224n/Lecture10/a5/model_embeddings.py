#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"
import torch

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.vocab = vocab
        self.e_word = word_embed_size
        self.e_char = 50
        self.kernel = 5
        self.dropout_prob = 0.3
        self.pad_token = vocab.char2id['<pad>']
        self.embedding = nn.Embedding(num_embeddings = len(vocab.char2id),
				                      embedding_dim = self.e_char,
									  padding_idx = self.pad_token)
        self.cnn = CNN(self.kernel, self.e_char, self.e_word)
        self.highway = Highway(self.e_word, self.dropout_prob)
        self.dropout = nn.Dropout(self.dropout_prob)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        # input : [sentence length, batch size, m_word]
        # x_emb = CharEmbedding(x_padded), x_padded = input

        x_emb = self.embedding(input)
        # x_emb : [sentence length, batch size, m_word, e_char]

        x_reshaped = x_emb.permute(0,3,2,1)
        # x_reshaped = [sentence length, e_char, m_word, batch size]

        sen_len, batch_size, m_word = x_emb.shape[0], x_emb.shape[1], x_emb.shape[2]

        tot_word_emb = []
        for x in x_reshaped:
            # x : [e_char, m_word, batch_size]
            x_conv_out = self.cnn(x) # x_conv_out: [batch size, e_word]
            x_word_emb = self.highway(x_conv_out) # x_word_emb : [batch size, e_word]
            x_word_emb = self.dropout(x_word_emb)
            tot_word_emb.append(x_word_emb)

        x_word_emb = torch.stack(tot_word_emb)

        return x_word_emb
        ### END YOUR CODE

