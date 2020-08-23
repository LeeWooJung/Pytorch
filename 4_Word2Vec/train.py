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

from tqdm import tqdm
from preprocess import Preprocess
from model import Word2Vec
from model import SkipGram_with_NS

parser = argparse.ArgumentParser()
parser.add_argument('--num_negs', default = 20, help = 'Number of negative samples per center word')
parser.add_argument('--data_path', default = './data/corpus.txt', help = 'Location of the corpus dataset')
parser.add_argument('--emb_dim', default = 300, help = 'Embedding dimension')
parser.add_argument('--max_vocab', default = 20000, help = 'Maximum vocabulary size')
parser.add_argument('--window_size', default = 5, help = 'Window size')
parser.add_argument('--batch_size', default = 4096, help = 'Mini-batch size')
parser.add_argument('--n_epochs', default = 15, help = 'Number of epochs')
parser.add_argument('--sub_sample_t', default = 0.00001, help = 'sub sampling threshold')
parser.add_argument('--preprocess', default = True, help = 'Need preprocess?')

args = parser.parse_args()

def LoadData():
	print("Load data...", end = ' ')
	vocab = pickle.load(open('vocab.dat','rb'))
	word2idx = pickle.load(open('word2idx.dat','rb'))
	idx2word = pickle.load(open('idx2word.dat','rb'))
	wordcount = pickle.load(open('wordcount.dat','rb'))
	training_data = pickle.load(open('training_data.dat','rb'))
	print("DONE")
	return (vocab, word2idx, idx2word, wordcount, training_data)

def SubsampleData(data, wordfreq):
	print("Sub sampling data...", end = ' ')
	t = args.sub_sample_t

	centers, contexts = [], []
	probability = (wordfreq-t)/wordfreq - np.sqrt(t/wordfreq)
	np.clip(probability, 0, 1)
	cnt = 0
	for center, context in data:
		cnt += 1
		if not cnt % 1000:
			print("Sub sampling from {}kth data...".format(cnt//1000), end='\r')
		if random.random() > probability[center]:
			centers.append(center)
			contexts.append(context)

	centers = torch.LongTensor(centers)
	contexts = torch.LongTensor(contexts)
	dataset = TensorDataset(centers, contexts)

	print("")
	return dataset

def getLoss(center, context, negative):
	center = center.unsqueeze(2)

	# center : [batch size, emb dim, 1]
	# context : [batch size, context size, emb dim]
	# negative: [batch size, context size * num negs, emb dim]
	context_size = context.shape[1]
	num_negs = negative.shape[1]//context_size
	correct = F.logsigmoid(torch.bmm(context, center))
	incorrect = F.logsigmoid(-torch.bmm(negative, center))

	# correct : [batch size, context size, 1]
	# incorrect : [batch size, context size * num negs, 1]
	correct = correct.squeeze(2)
	incorrect = incorrect.view(-1, context_size, num_negs)

	# correct : [batch size, context size]
	# incorrect : [batch size, context size, num_negs]
	correct_loss = correct.mean(1)
	incorrect_loss = incorrect.sum(2).mean(1)

	# correct_loss : [batch size]
	# incorrect_loss : [batch size]

	return -(correct_loss + incorrect_loss).mean()


def train():

	vocab = None
	word2idx = None
	idx2word = None
	wordcount = None
	data = None
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if args.preprocess:
		print("Preprocess step start...")
		preprocess = Preprocess(args.data_path, args.window_size)
		preprocess.build_data(args.max_vocab)
		preprocess.build_training_data()
	
	vocab, word2idx, idx2word, wordcount, data = LoadData()

	wordfreq = np.array([wordcount[word] for word in idx2word])
	wordfreq = wordfreq / wordfreq.sum()

	dataset = SubsampleData(data, wordfreq)

	word2vec = Word2Vec(args.max_vocab, args.emb_dim, device).to(device)
	model = SkipGram_with_NS(word2vec, args.max_vocab, args.num_negs, wordfreq).to(device)

	optimizer = optim.Adam(model.parameters())

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
	print("Save the model...", end = ' ')
	idx2vec = word2vec.input.weight.data.cpu().numpy()
	pickle.dump(idx2vec, open('idx2vec.dat', 'wb'))
	torch.save(model.state_dict(), 'skipgram_with_negative_sampling.pt')
	torch.save(optimizer.state_dict(), 'optimization.pt')
	print("DONE")


if __name__ == '__main__':
	train()
