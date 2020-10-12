#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

from utils import preprocess, PrintExample
from model import Encoder, Decoder, Seq2Seq

parser = argparse.ArgumentParser()
parser.add_argument('--min_freq', default = 5, type = int, help = "Minimum frequency in Vocab")
parser.add_argument('--seed', default = 1024, type = int, help = "Seed Value")
parser.add_argument('--batch_size', default = 256, type = int, help = "Number of batch size")
parser.add_argument('--dropout', default = 0.3, type = float, help = "Probability of dropout")
parser.add_argument('--enc_emb_dim', default = 128, type = int, help = "Dimension of encoder embedding")
parser.add_argument('--dec_emb_dim', default = 128, type = int, help = "Dimension of decoder embedding")
parser.add_argument('--hidden_dim', default = 128, type = int, help = "Dimension of hidden states")
parser.add_argument('--n_layers', default = 2, type = int, help = "Number of layers")
parser.add_argument('--learning_rate', default = 0.0001, type = float, help = "Learning rate")
parser.add_argument('--n_epochs', default = 25, type = int, help = "Number of epochs")
parser.add_argument('--clip', default = 1.0, type = float, help = "Gradient Clip")
parser.add_argument('--model', default = 'Seq2Seq-attention.pt', type = str, help = "Trained model name")
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--save_attention_figure', action='store_true')

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train():

	batch_size = args.batch_size
	enc_emb_dim = args.enc_emb_dim
	dec_emb_dim = args.dec_emb_dim
	hidden_dim = args.hidden_dim
	n_layers = args.n_layers
	dropout = args.dropout
	lr = args.learning_rate
	n_epochs = args.n_epochs
	clip = args.clip
	min_freq = args.min_freq
	attention = args.attention
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	best_valid_loss = float('inf')

	pre = preprocess(min_freq = min_freq)
	SRC, TRG, tr, val, ts = pre.build()
	train_iter, valid_iter = BucketIterator.splits(
							 datasets = (tr, val),
							 sort = False,
							 batch_size = batch_size,
							 device = device)

	enc = Encoder(SRC, enc_emb_dim, hidden_dim, n_layers, dropout, device).to(device)
	dec = Decoder(TRG, dec_emb_dim, hidden_dim, n_layers, dropout, attention, device).to(device)
	model = Seq2Seq(enc, dec, n_layers, hidden_dim, attention, device).to(device)

	optimizer = optim.Adam(model.parameters(), lr = lr)
	trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
	criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx).to(device)
	best_valid_loss = float('inf')

	for epoch in range(1, n_epochs + 1):
		model.train()
		epoch_loss = 0
		val_epoch_loss = 0

		pbar = tqdm(train_iter)
		pbar.set_description("[(Train) Epoch{}]".format(epoch))

		for i, batch in enumerate(pbar):

			src = batch.SRC.to(device)
			trg = batch.TRG.to(device)

			optimizer.zero_grad()
			output = model(src, trg).to(device)

			output = output[1:].view(-1, output.shape[-1]) # eliminate <sos>
			trg = trg[1:].view(-1)
			# output: [trg len * batch size, trg vocab size]
			# trg: [trg len * batch size]

			loss = criterion(output, trg)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), clip)

			optimizer.step()
			pbar.set_postfix(loss = loss.item())
			epoch_loss += loss.item()


		epoch_loss = epoch_loss / len(train_iter)

		model.eval()
		pbar = tqdm(valid_iter)
		pbar.set_description("[(Valid) Epoch{}]".format(epoch))

		for j, batch in enumerate(pbar):

			src = batch.SRC.to(device)
			trg = batch.TRG.to(device)

			output = model(src, trg).to(device)

			output = output[1:].view(-1, output.shape[-1])
			trg = trg[1:].view(-1)

			loss = criterion(output, trg)
			pbar.set_postfix(loss = loss.item())
			val_epoch_loss += loss.item()

		val_epoch_loss = val_epoch_loss / len(valid_iter)
		print("[Train] Avg loss: {0:.4f} \n[Valid] Avg loss: {1:.4f}".format(epoch_loss, val_epoch_loss))

		if val_epoch_loss < best_valid_loss:
			best_valid_loss = val_epoch_loss
			torch.save(model, args.model)

def test():
	if not os.path.exists(args.model):
		raise NameError('There is no model with named {}'.format(args.model))

	batch_size = args.batch_size
	enc_emb_dim = args.enc_emb_dim
	dec_emb_dim = args.dec_emb_dim
	hidden_dim = args.hidden_dim
	n_layers = args.n_layers
	dropout = args.dropout
	lr = args.learning_rate
	min_freq = args.min_freq
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	pre = preprocess(min_freq = min_freq)
	SRC, TRG, _, _, ts = pre.build()
	test_iter = BucketIterator(
					dataset = ts,
					sort = False,
					batch_size = batch_size,
					device = device)

	model = torch.load(args.model)

	trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
	criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx).to(device)

	model.eval()
	avg_loss = 0
	index = 0

	with torch.no_grad():
		for i, batch in enumerate(test_iter):
			index += 1

			src = batch.SRC.to(device)
			trg = batch.TRG.to(device)
			output = model(src, trg).to(device)

			if args.save_attention_figure:
				PrintExample(model, SRC, TRG, src, trg, output)

			output = output[1:].view(-1, output.shape[-1])
			trg = trg[1:].view(-1)

			loss = criterion(output, trg)
			avg_loss += loss.item()

	avg_loss = avg_loss / len(test_iter)
	print("[Test] Avg loss: {0:.4f}".format(avg_loss))


def main():
	if args.train:
		train()
	if args.test:
		test()

if __name__ == '__main__':
	main()
