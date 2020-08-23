import os
import random
import argparse
import numpy as np

from konlpy.tag import Mecab
from preprocess import Preprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from train import *
from model import Word2Vec
from model import SkipGram_with_NS

parser = argparse.ArgumentParser()
parser.add_argument('--num_negs', default = 20, help = 'Number of negative samples per center word')
parser.add_argument('--data_path', default = './data/kor_corpus.txt', help = 'Korean words dataset')
parser.add_argument('--emb_dim', default = 300, help = 'Embedding dimension')
parser.add_argument('--max_vocab', default = 20000, help = 'Maximum vocabulary size')
parser.add_argument('--window_size', default = 5, help = 'Window size')
parser.add_argument('--batch_size', default = 4096, help = 'Mini-batch size')
parser.add_argument('--n_epochs', default = 100, help = 'Number of epochs')
parser.add_argument('--sub_sample_t', default = 0.00001, help = 'sub sampling threshold')

args = parser.parse_args()

def preprocess(datapath):
	mecab = Mecab()
	cnt = 0
	sentences = []
	f = open('./data/korean_corpus.txt','r',encoding="utf8")
	while True:
		line = f.readline()
		if not line: break
		cnt += 1
		if not (cnt % 1000):
			print("tokenize {}kth line...".format(cnt//1000), end = '\r')
		tokens = mecab.nouns(line)
		if tokens: sentences.append(tokens)
	print("")
	cnt = 0
	with open(datapath, 'w') as f:
		for sentence in sentences:
			cnt += 1
			for idx, word in enumerate(sentence):
				if idx == len(sentence)-1:
					f.write("%s.\n" % word)
				else:
					f.write("%s " % word)
			if not(cnt % 1000):
				print("write {}kth line to the file...".format(cnt//1000), end = '\r')
	print("")

def make_file(datapath):
	if os.path.isfile(datapath):
		print("There is already tokenized file: {} ...".format(datapath))
		return
	else:
		print("There is no tokenized file: {} ...".format(datapath))
		preprocess(datapath)
		print("The tokenized file {} is made...".format(datapath))
	
def train():
	vocab = None
	word2idx = None
	idx2word = None
	wordcount = None
	data = None
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	vocab, word2idx, idx2word, wordcount, data = LoadData()

	wordfreq = np.array([wordcount[word] for word in idx2word])
	wordfreq = wordfreq / wordfreq.sum()

	dataset = SubsampleData(data, wordfreq)

	word2vec = Word2Vec(args.max_vocab, args.emb_dim, device).to(device)
	model = SkipGram_with_NS(word2vec, args.max_vocab, args.num_negs, wordfreq).to(device)
	optimizer = optim.Adam(model.parameters())

	print("-"*30)
	print("Start training word2vec model...")
	for epoch in range(1, args.n_epochs+1):
		dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)
		model.train()
		epoch_loss = 0
		total_batches = int(np.ceil(len(dataset)/args.batch_size))
		pbar = tqdm(dataloader)
		pbar.set_description("[Epoch {}]".format(epoch))
		for center, context in pbar:
			centerV, contextV, negativeV = model(center, context)
			loss = getLoss(centerV, contextV, negativeV)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			pbar.set_postfix(loss=loss.item())
			epoch_loss += loss.item()

		print("Average loss: {0:.4f}".format(epoch_loss/total_batches))
	print("DONE")
	print("-"*30)
	print("Save the model...", end = ' ')
	idx2vec = word2vec.input.weight.data.cpu().numpy()
	pickle.dump(idx2vec, open('idx2vec.dat', 'wb'))
	torch.save(model.state_dict(), 'skipgram_with_negative_sampling.pt')
	torch.save(optimizer.state_dict(), 'optimization.pt')
	print("DONE")


if __name__ == '__main__':
	make_file(args.data_path)
	process = Preprocess(args.data_path, args.window_size)
	process.build_data(args.max_vocab)
	process.build_training_data()
	train()
