#-*- coding: utf-8 -*-

import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from preprocess import Preprocess
from model import Word2Vec
from model import SkipGram_with_NS

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default = './data/corpus.txt', help = 'Location of the corpus dataset')
parser.add_argument('--emb_dim', default = 300, help = 'Embedding dimension')
parser.add_argument('--max_vocab', default = 20000, help = 'Maximum vocabulary size')
parser.add_argument('--num_negs', default = 20, help = 'Number of negative samples')
parser.add_argument('--window_size', default = 5, help = 'Window size')
parser.add_argument('--batch_size', default = 4096, help = 'Mini-batch size')
parser.add_argument('--n_epochs', default = 20, help = 'Number of epochs')
parser.add_argument('--sub_sample_t', default = 0.00001, help = 'sub sampling threshold')
parser.add_argument('--preprocess', default = False, help = 'Need preprocess?')

args = parser.parse_args()

def LoadData():
	print("-"*30)
	print("Load data...", end = ' ')
	vocab = pickle.load(open('vocab.dat','rb'))
	word2idx = pickle.load(open('word2idx.dat','rb'))
	idx2word = pickle.load(open('idx2word.dat','rb'))
	wordcount = pickle.load(open('wordcount.dat','rb'))
	training_data = pickle.load(open('training_data.dat','rb'))
	print("DONE")
	return (vocab, word2idx, idx2word, wordcount, training_data)

def SubsampleData(data, wordfreq):
	print("-"*30)
	print("Sub sampling data...", end = ' ')
	t = args.sub_sample_t

	centers, contexts = [], []
	probability = (wordfreq-t)/wordfreq - np.sqrt(t/wordfreq)
	np.clip(probability, 0, 1)
	for center, context in data:
		if random.random() > probability[center]:
			centers.append(center)
			contexts.append(context)

	centers = torch.LongTensor(centers)
	contexts = torch.LongTensor(contexts)
	dataset = TensorDataset(centers, contexts)

	print("DONE")
	return dataset

def train():

	vocab = None
	word2idx = None
	idx2word = None
	wordcount = None
	data = None
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if args.preprocess:
		print("-"*30)
		print("Preprocess step start...")
		preprocess = Preprocess(args.data_path, args.window_size)
		preprocess.build_data(args.max_vocab)
		preprocess.build_training_data()
	
	vocab, word2idx, idx2word, wordcount, data = LoadData()

	wordfreq = np.array([wordcount[word] for word in idx2word])
	wordfreq = wordfreq / wordfreq.sum()

	dataset = SubsampleData(data, wordfreq)
	dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

	word2vec = Word2Vec(args.max_vocab, args.emb_dim, device).to(device)
	model = SkipGram_with_NS(word2vec, args.max_vocab, args.num_negs, wordfreq).to(device)

	optimizer = optim.Adam(model.parameters())

	for X,y in dataloader:
		center, context = model(X,y)
		break


if __name__ == '__main__':
	train()
