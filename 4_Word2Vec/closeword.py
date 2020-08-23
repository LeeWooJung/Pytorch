import argparse
import pickle
import torch
import numpy as np
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--word', default ='woman', help = 'Input the word what you want to know closest words')
args = parser.parse_args()


word2idx = pickle.load(open('word2idx.dat','rb'))
idx2word = pickle.load(open('idx2word.dat','rb'))
embedded = pickle.load(open('idx2vec.dat','rb'))
vocab = pickle.load(open('vocab.dat','rb'))

def closeword(word, topn=5):
	i = word2idx[word]
	word_distance = []
	dist = nn.PairwiseDistance()
	v_i = embedded[i]
	tensor_i = torch.FloatTensor([v_i])
	for j in range(len(vocab)):
		if j != i:
			v_j = embedded[j]
			tensor_j = torch.FloatTensor([v_j])
			word_distance.append((idx2word[j], float(dist(tensor_i, tensor_j))))
	word_distance.sort(key=lambda x: x[1])
	print(word_distance[:topn])
	return

closeword(args.word)
