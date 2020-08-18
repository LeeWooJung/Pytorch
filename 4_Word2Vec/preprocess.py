# -*- coding: utf-8 -*-

import os
import re
import codecs
import pickle
import argparse

from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default = './data/corpus.txt', help='Location of the corpus dataset')
parser.add_argument('--window_size', default = 5, help = 'Window size')
parser.add_argument('--max_vocab', default= 20000, help = 'Maximum vocabulary size')
args = parser.parse_args()

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
	def __init__(self, datapath, window_size):
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
		print("Building data...")
		self.unk = '<unk>'
		self.word_count = {self.unk: 1}
		for idx, sentence in enumerate(sentences):
			for word in sentence.split(" "):
				if word not in self.word_count:
					self.word_count[word] = 1
				else:
					self.word_count[word] += 1
			print("'\r{0}th sentence.".format(idx+1), end = '')

		self.idx2word = [self.unk] + list(sorted(self.word_count.items(), key=(lambda x: x[1]), reverse=True))[:max_vocab-1]
		self.word2idx = {word: index for index, word in enumerate(self.idx2word)}
		self.vocab = set(list(self.word2idx))

		print("DONE!")
		print("-"*30)
		print("Save the data...", end = '')

		pickle.dump(self.idx2word, open('idx2word.dat', 'wb'))
		pickle.dump(self.word2idx, open('word2idx.dat', 'wb'))
		pickle.dump(self.vocab, open('vocab.dat', 'wb'))

		print("DONE!")

	def skipgram(self, sentence, index):
		center = sentence[index]
		left = sentence[max(0,index - self.window_size):index]
		right = sentence[index+1: min(len(sentence)+1, index + self.window_size)]

		contexts = [self.unk for _ in range(self.window_size - index)] + left + right + [self.unk for _ in range(self.window_size + index - len(sentence))]

		center_idx = self.word2idx[center]
		contexts_idx = [self.word2idx[word] for word in contexts]

		return center_idx, contexts_idx

	def build_training_data(self):
		print("Building training data...")
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
		print("-"*30)
		return

	

if __name__ == '__main__':
	
	corpus = Preprocess(args.datapath, args.window_size)
	corpus.build_data(args.max_vocab)
	corpus.build_training_data()
