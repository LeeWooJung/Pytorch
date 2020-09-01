import random
import torch
import torch.nn as nn

class Encoder(nn.Module):

	def __init__(self, input_dim, emb_dim, hidden_dim, n_layers,
				 dropout, lstm = True):
		super().__init__()

		self.input_dim = input_dim
		self.emb_dim = emb_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.dropout = dropout
		self.lstm = lstm
		self.rnn = None

		self.embedding = nn.Embedding(input_dim, emb_dim)
		if lstm:
			self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout = dropout)
		else: # assume rnn type: GRU
			self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout = dropout)
		self.dropout = nn.Dropout(dropout)

	def forward(self, src):

		emb = self.dropout(self.embedding(src))
		if self.lstm:
			outputs, (hidden, cell) = self.rnn(emb) # initial hidden, cell as zero vector
			return hidden, cell
		else: # assume rnn type: GRU
			outputs, hidden = self.rnn(emb)
			return hidden


class Decoder(nn.Module):

	def __init__(self, output_dim, emb_dim, hidden_dim, n_layers,
				 dropout, lstm = True):
		super().__init__()

		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.dropout = dropout
		self.lstm = lstm
		self.rnn = None

		self.embedding = nn.Embedding(output_dim, emb_dim)
		if lstm:
			self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout = dropout)
		else:
			self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout = dropout)

		self.fc = nn.Linear(hidden_dim, output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, input, hidden, cell = None):

		emb = self.dropout(self.embedding(input))
		if self.lstm:
			output, (hidden, cell) = self.rnn(emb.unsqueeze(0), (hidden, cell))
			prediction = self.fc(output.squeeze(0))
			return prediction, hidden, cell

		else: # assume rnn type: GRU
			output, hidden = self.rnn(emb.unsqueeze(0), hidden)
			prediction = self.fc(output.squeeze(0))
			return prediction, hidden

class Seq2Seq(nn.Module):

	def __init__(self, encoder, decoder, device, lstm = True):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.device = device
		self.lstm = lstm

	def forward(self, src, trg, tf = 0.5):

		batch_size = src.shape[1]
		trg_len = trg.shape[0]
		trg_vocab_size = self.decoder.output_dim

		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

		if self.lstm:
			hidden, cell = self.encoder(src)
		else:
			hidden = self.encoder(src)

		input = trg[0,:]

		for t in range(1, trg_len):
			if self.lstm:
				output, hidden, cell = self.decoder(input, hidden, cell)
			else:
				output, hidden = self.decoder(input, hidden)

			outputs[t] = output
			_tf = random.random() < tf
			top1 = output.argmax(1)

			input = trg[t] if _tf else top1

		return outputs
