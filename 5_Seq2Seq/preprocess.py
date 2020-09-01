#-*- coding:utf-8 -*-

import os
import spacy
import dill
from konlpy.tag import Mecab

from torchtext.datasets import Multi30k
from torchtext.data import Field, TabularDataset

spacy_en = spacy.load('en')
mecab = Mecab()

def eng_tokenizer(text):
	return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

def kor_tokenizer(text):
	return [tok for tok in mecab.morphs(text)]

class Preprocess(object):

	def __init__(self, src_tokenizer = eng_tokenizer,
				 trg_tokenizer = kor_tokenizer, dpath = './dataset/',
				 min_freq = 5):

		self.dpath = dpath
		self.train_path = 'train_Corpus.csv'
		self.valid_path = 'valid_Corpus.csv'
		self.test_path = 'test_Corpus.csv'
		self.src_tokenizer = src_tokenizer
		self.trg_tokenizer = trg_tokenizer
		self.min_freq = min_freq

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
				fields = [("SRC",SRC), ("TRG",TRG)])

		SRC.build_vocab(tr, min_freq = self.min_freq)
		TRG.build_vocab(tr, min_freq = self.min_freq)

		return SRC, TRG, tr, val, ts
