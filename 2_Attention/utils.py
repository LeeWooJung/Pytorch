#-*- coding:utf-8 -*-

import spacy
from konlpy.tag import Mecab

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchtext.data import Field, TabularDataset


class preprocess():

	def __init__(self, dpath = './dataset/', min_freq = 5):

		self.dpath = dpath
		self.train_path = 'train_Corpus.csv'
		self.valid_path = 'valid_Corpus.csv'
		self.test_path = 'test_Corpus.csv'
		self.min_freq = min_freq

		self.EnTokenizer = spacy.load('en')
		self.KrTokenizer = Mecab()

	def src_tokenizer(self, text):
		return [tok.text for tok in self.EnTokenizer.tokenizer(text)]

	def trg_tokenizer(self, text):
		return [tok for tok in self.KrTokenizer.morphs(text)]

	def build(self):
		SRC = Field(tokenize = self.src_tokenizer,
					init_token = '<sos>',
					eos_token = '<eos>',
					lower = True)
		TRG = Field(tokenize = self.trg_tokenizer,
					init_token = '<sos>',
					eos_token = '<eos>')

		tr, val, ts = TabularDataset.splits(
				path = self.dpath, train = self.train_path, validation = self.valid_path,
				test = self.test_path, format = 'csv',
				fields = [("SRC",SRC), ("TRG",TRG)])

		SRC.build_vocab(tr, min_freq = self.min_freq)
		TRG.build_vocab(tr, min_freq = self.min_freq)

		return SRC, TRG, tr, val, ts

def Print(field, sent, From):

	index = None
	print("{} sentence: ".format(From), end='')
	for i, s in enumerate(sent):
		if field.vocab.itos[s] in [field.eos_token, field.pad_token]:
			index = i
			break
		elif field.vocab.itos[s] == field.init_token:
			continue
		elif field.vocab.itos[s] == field.unk_token:
			print("<unk> ", end='')
		else:
			print("{} ".format(field.vocab.itos[s]), end='')
	print("")
	return index

def SaveAttFigure(attention, src, trg):
	fig = plt.figure(figsize=(len(src), len(trg)))

	ax = fig.add_subplot(1, 1, 1)
	cax = ax.matshow(attention.cpu().detach().numpy(), cmap='bone')

	plt.savefig('yes.png')

	return

def AttentionFigure(model, randN, SRC, TRG, src, trg, src_label, trg_label):

	model.eval()
	with torch.no_grad():

		enc_hiddens, hidden, cell = model.encoder(src)
		src_len = enc_hiddens.shape[0]
		batch_size = 1
		enc_masks = model.make_masks(src, src_len, batch_size)

		hidden = model.enc_hidden_proj(torch.cat([hidden[0], hidden[1], hidden[2], hidden[3]], dim=1))
		cell = model.enc_cell_proj(torch.cat([cell[0], cell[1], cell[2], cell[3]], dim=1))
		enc_hiddens_att = model.attention_proj(enc_hiddens)

		_, attention_weight = model.decoder(trg, enc_hiddens, enc_hiddens_att, enc_masks, (hidden, cell))

		attention_weight = attention_weight[randN, :len(src_label), :len(trg_label)]

		SaveAttFigure(attention_weight, src_label, trg_label)

	return


def PrintExample(model, SRC, TRG, src, trg, output):

	batch_size = src.shape[1]
	randN = random.randint(0, batch_size-1)

	_src = list(src[:, randN].cpu().numpy())
	_trg = list(trg[1:, randN].cpu().numpy())
	_out = list(output[1:, randN, :].squeeze(1).argmax(1).cpu().numpy())

	src_idx = Print(SRC, _src, "Source")
	trg_idx = Print(TRG, _trg, "Target")
	_ = Print(TRG, _out, "Predicted")
	print("")

	src_label = []
	trg_label = []
	for s in _src:
		if SRC.vocab.itos[s] in [SRC.eos_token, SRC.pad_token]:
			break
		elif SRC.vocab.itos[s] == SRC.init_token:
			continue
		elif SRC.vocab.itos[s] == SRC.unk_token:
			src_label.append("<unk>")
		else:
			src_label.append(SRC.vocab.itos[s])
	for t in _trg:
		if TRG.vocab.itos[t] in [TRG.eos_token, TRG.pad_token]:
			break
		elif TRG.vocab.itos[t] == TRG.init_token:
			continue
		elif TRG.vocab.itos[t] == TRG.unk_token:
			trg_label.append("<unk>")
		else:
			trg_label.append(TRG.vocab.itos[t])
	
	# attention figure
	AttentionFigure(model, randN, SRC, TRG, src, trg, src_label, trg_label)
	return
