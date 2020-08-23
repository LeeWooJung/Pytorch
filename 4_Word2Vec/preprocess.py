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
	def __init__(self, datapath='./data/corpus.txt', window_size=5, kor = False):
		self.datapath = datapath 
		self.window_size = window_size
		self.kor = kor

	def split_sentences(self):
		sentences = []
		corpus = open(self.datapath, 'r', encoding = 'utf-8')
		if checkCorpus(self.datapath):
			while True:
				line = corpus.readline()
				if not line: break
				if not self.kor:
					sentences.append(re.sub(r"[^a-z0-9]+", " ", line.lower())[:-1])
				else:
					sentences.append(line)
		else:
			line = corpus.readline()
			sentence = line.split(". ")
			for sent in sentence:
				if not self.kor:
					sentences.append(re.sub(r"[^a-z0-9]+", " ", sent.lower()))
				else:
					sentences.append(sent)
		return sentences

	def build_data(self, max_vocab):
		sentences = self.split_sentences()
		self.unk = '<unk>'
		self.word_count = {self.unk: 1}
		for idx, sentence in enumerate(sentences):
			for word in sentence.split(" "):
				if word not in self.word_count:
					self.word_count[word] = 1
				else:
					self.word_count[word] += 1
			if not (idx+1) % 1000:
				print("Building {}kth sentence to data...".format((idx+1)//1000), end = '\r')

		self.idx2word = [self.unk] + [key for key, _ in sorted(self.word_count.items(), key = (lambda x: x[1]), reverse=True)][:max_vocab-1]
		self.word2idx = {word: index for index, word in enumerate(self.idx2word)}
		self.vocab = set(list(self.word2idx))

		print("")
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
		data = []
		cnt = 0
		sentences = self.split_sentences()
		for sentence in sentences:
			cnt += 1
			if not cnt % 1000:
				print("Building {}kth sentence to training data using skipgram...".format(cnt//1000), end ='\r')
			sent = []
			for idx, word in enumerate(sentence.split(" ")):
				if word not in self.vocab:
					sent.append(self.unk)
				else:
					sent.append(word)
			for idx, _ in enumerate(sent):
				center, contexts = self.skipgram(sent, idx)
				data.append((center, contexts))

		pickle.dump(data, open('training_data.dat','wb'), protocol = 4)
		print("")
		return
