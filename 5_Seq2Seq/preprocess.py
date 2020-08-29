#-*- coding:utf-8 -*-

import os
import spacy

from torchtext.datasets import Multi30k
from torchtext.data import Field, TabularDataset

class Preprocess(object):

	def __init__(self, dpath = './dataset/',
				 src_tokenizer = None, trg_tokenizer = None):

		self.dpath = dpath
		self.train_path = 'train_Corpus.csv'
		self.valid_path = 'valid_Corpus.csv'
		self.test_path = 'test_Corpus.csv'
		self.src_tokenizer = src_tokenizer
		self.trg_tokenizer = trg_tokenizer

	def Build(self):

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
				fields = [('SRC',SRC), ('TRG', TRG)])

		return tr, val, ts 
