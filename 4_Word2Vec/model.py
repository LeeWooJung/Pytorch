import torch
import torch.nn as nn
import numpy as np

class Word2Vec(nn.Module):
	def __init__(self, vocab_size = 20000, emb_dim = 300, device = 'cpu', padding_idx = 0):
		super(Word2Vec, self).__init__()

		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.device = device
		self.input = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = padding_idx)
		self.output = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = padding_idx)

		self.input.weight = nn.Parameter(torch.cat([torch.zeros(1, self.emb_dim),
							torch.FloatTensor(self.vocab_size-1, self.emb_dim).uniform_(-0.5/self.emb_dim, 0.5/self.emb_dim)]))
		self.output.weight = nn.Parameter(torch.cat([torch.zeros(1, self.emb_dim),
							torch.FloatTensor(self.vocab_size-1, self.emb_dim).uniform_(-0.5/self.emb_dim, 0.5/self.emb_dim)]))

		self.input.weight.requires_grad = True
		self.output.weight.requires_grad = True


	def forward(self, x):
		return input_forward(x)

	def input_forward(self, x):
		return self.input(torch.LongTensor(x).to(self.device))

	def output_forward(self, x):
		return self.output(torch.LongTensor(x).to(self.device))

class SkipGram_with_NS(nn.Module):
	def __init__(self, word2vec, vocab_size = 20000, num_negs = 20, wordfreq = None):
		super(SkipGram_with_NS, self).__init__()

		self.word2vec = word2vec
		self.vocab_size = vocab_size
		self.num_negs = num_negs

		self.wordfreq = np.power(wordfreq, 0.75)
		self.wordfreq = torch.FloatTensor(self.wordfreq / self.wordfreq.sum())

	def forward(self, center, contexts):
		# center: [batch size]
		# contexts: [batch size, context size]
		batch_size = center.shape[0]
		context_size = contexts.shape[1]

		negative = torch.multinomial(self.wordfreq, batch_size * context_size * self.num_negs, replacement = True).view(batch_size, -1)

		# centerV : [batch size, emb dim]
		# contextV : [batch size, context size, emb dim]
		centerV = self.word2vec.input_forward(center)
		contextV = self.word2vec.output_forward(contexts)
		negativeV = self.word2vec.output_forward(negative)

		return centerV, contextV, negativeV
