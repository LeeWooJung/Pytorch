# -*- coding: utf-8 -*-

import os
import re
import codecs
import pickle
import argparse

def checkCorpus(datapath):
	corpus = open(datapath, 'r', encoding ='utf-8')
	cnt = 0
	while True:
		line = corpus.readline()
		if not line: break
		if cnt >= 2:
			return True
		cnt += 1
	return False


class Preprocess(object):
	def __init__(self, datapath='./data/corpus.txt', window_size=5):
		self.datapath = datapath 
		self.window_size = window_size

	def split_sentences(self):
		sentences = []
		corpus = open(self.datapath, 'r', encoding = 'utf-8')
		if checkCorpus(self.datapath):
			while True:
				line = corpus.readline()
				if not line: break
				sentences.append(re.sub(r"[^a-z0-9]+", " ", line.lower())[:-1])
		else:
			line = corpus.readline()
			sentence = line.split(". ")
			for sent in sentence:
				sentences.append(re.sub(r"[^a-z0-9]+", " ", sent.lower()))
		return sentences

	def build_data(self, max_vocab):
		sentences = self.split_sentences()
		print("-"*30)
		print("Building data...", end = ' ')
		self.unk = '<unk>'
		self.word_count = {self.unk: 1}
		for idx, sentence in enumerate(sentences):
			for word in sentence.split(" "):
				if word not in self.word_count:
					self.word_count[word] = 1
				else:
					self.word_count[word] += 1
			print("'\r{0}th sentence.".format(idx+1), end = '')

		self.idx2word = [self.unk] + [key for key, _ in sorted(self.word_count.items(), key = (lambda x: x[1]), reverse=True)][:max_vocab-1]
		self.word2idx = {word: index for index, word in enumerate(self.idx2word)}
		self.vocab = set(list(self.word2idx))

		print("DONE!")
		print("-"*30)
		print("Save the data...", end = '')

		pickle.dump(self.word_count, open('wordcount.dat','wb'))
		pickle.dump(self.idx2word, open('idx2word.dat', 'wb'))
		pickle.dump(self.word2idx, open('word2idx.dat', 'wb'))
		pickle.dump(self.vocab, open('vocab.dat', 'wb'))

		print("DONE!")

	def skipgram(self, sentence, index):
		center = sentence[index]
		left = sentence[max(0,index - self.window_size):index]
		right = sentence[index+1: min(len(sentence), index + self.window_size)+1]

		contexts = [self.unk for _ in range(self.window_size - index)] + left + right + [self.unk for _ in range(self.window_size + index - len(sentence)+1)]

		center_idx = self.word2idx[center]
		contexts_idx = [self.word2idx[word] for word in contexts]

		return center_idx, contexts_idx

	def build_training_data(self):
		print("-"*30)
		print("Building training data...", end = ' ')
		data = []
		sentences = self.split_sentences()
		for sentence in sentences:
			sent = []
			for idx, word in enumerate(sentence.split(" ")):
				if word not in self.vocab:
					sent.append(self.unk)
				else:
					sent.append(word)
			for idx, _ in enumerate(sent):
				center, contexts = self.skipgram(sent, idx)
				data.append((center, contexts))

		pickle.dump(data, open('training_data.dat','wb'))
		print("DONE!")
		return
