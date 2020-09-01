#-*- coding:utf-8 -*-

import random
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import BucketIterator

from preprocess import Preprocess
from model import Encoder, Decoder, Seq2Seq

parser = argparse.ArgumentParser()
parser.add_argument('--dpath', default = './dataset/', help = 'Location of the dataset')
parser.add_argument('--batch_size', default = 128, help = 'Mini batch size')
parser.add_argument('--enc_emb_dim', default = 256, help = 'Encoder embedding dimension')
parser.add_argument('--dec_emb_dim', default = 256, help = 'Decoder embedding dimension')
parser.add_argument('--hidden_dim', default = 512, help = 'Hidden dimension')
parser.add_argument('--n_layers', default = 2, help = 'Number of layers in Encoder & Decoder')
parser.add_argument('--enc_dropout', default = 0.5, help = 'Probability of dropout in Encoder')
parser.add_argument('--dec_dropout', default = 0.5, help = 'Probability of dropout in Decoder')
parser.add_argument('--lr', default = 0.00001, help = 'Learning rate')
parser.add_argument('--lstm', default = True, help = 'Use LSTM? or GRU?')
parser.add_argument('--n_epochs', default = 10, help = 'Number of Epochs')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def LoadData():
	print("Preprocess the dataset...", end = ' ')
	preprocess = Preprocess()
	SRC, TRG, tr, valid, ts = preprocess.Build()
	print("DONE")
	
	return SRC,TRG, tr, valid, ts

def init_weights(model):
	for name, parameters in model.named_parameters():
		nn.init.uniform_(parameters.data, -0.08, 0.08)

def train(model, iterator, optimizer, criterion, clip, epoch):
	model. train()
	epoch_loss = 0

	pbar = tqdm(iterator)
	pbar.set_description("[(Train) Epoch {}]".format(epoch))

	for i, batch in enumerate(pbar):

		src = batch.SRC.to(device)
		trg = batch.TRG.to(device)

		optimizer.zero_grad()
		output = model(src, trg).to(device)

		output = output[1:].view(-1, output.shape[-1]) # eliminate <sos>
		trg = trg[1:].view(-1)

		loss = criterion(output, trg)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), clip)

		optimizer.step()
		pbar.set_postfix(loss = loss.item())
		epoch_loss += loss.item()

	return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, epoch):
	model.eval()
	epoch_loss = 0

	with torch.no_grad():
		pbar = tqdm(iterator)
		pbar.set_description("[(Validation) Epoch {}]".format(epoch))

		for i, batch in enumerate(pbar):

			src = batch.SRC.to(device)
			trg = batch.TRG.to(device)

			output = model(src, trg, 0).to(device)

			output = output[1:].view(-1, output.shape[-1]) # eliminate <sos>
			trg = trg[1:].view(-1)

			loss = criterion(output, trg)
			pbar.set_postfix(loss = loss.item())
			epoch_loss += loss.item()

	return epoch_loss / len(iterator)


def trainModel():

	SRC, TRG, train_data, valid_data, test_data = LoadData()

	train_iter, valid_iter, test_iter = BucketIterator.splits(
										(train_data, valid_data, test_data),
										batch_size = args.batch_size,
										sort = False,
										device = device)

	enc = Encoder(len(SRC.vocab), args.enc_emb_dim, args.hidden_dim,
					args.n_layers, args.enc_dropout, args.lstm)
	dec = Decoder(len(TRG.vocab), args.dec_emb_dim, args.hidden_dim,
					args.n_layers, args.dec_dropout, args.lstm)
	model = Seq2Seq(enc, dec, device, args.lstm).to(device)

	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

	criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx).to(device)

	best_valid_loss = float('inf')
	clip = 1
	for epoch in range(1, args.n_epochs+1):
		train_loss = train(model, train_iter, optimizer, criterion, clip, epoch)
		print("Average train loss: {0:.4f}".format(train_loss))
		valid_loss = evaluate(model, valid_iter, criterion, epoch)
		print("Average validation loss: {0:.4f}".format(valid_loss))
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			if args.lstm:
				torch.save(model.state_dict(), 'Seq2Seq-LSTM.pt')
			else:
				torch.save(model.state_dict(), 'Seq2Seq-GRU.pt')

	return

if __name__ == "__main__":
	trainModel()
