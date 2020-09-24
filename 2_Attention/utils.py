#-*- coding:utf-8 -*-

import spacy
from konlpy.tag import Mecab
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
