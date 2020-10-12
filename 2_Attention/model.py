import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, SRC, emb_dim, hidden_dim, n_layers, dropout, device):
		super(Encoder, self).__init__()

		self.embedding = nn.Embedding(num_embeddings = len(SRC.vocab),
								embedding_dim = emb_dim,
								padding_idx = SRC.vocab['<pad>'])
		self.encoder = nn.LSTM(input_size = emb_dim, hidden_size = hidden_dim,
							   num_layers = n_layers, bidirectional = True, bias = True)
		self.dropout = nn.Dropout(dropout)

	def forward(self, src):

		emb = self.dropout(self.embedding(src))
		outputs, (hidden, cell) = self.encoder(emb)

		# outputs: [src len, batch size, hidden dim * 2(bidirectional)]
		# hidden: [n_layers * 2(bidirectional), batch size, hidden dim]
		# cell: [n_layers * 2(bidirectional), batch size, hidden dim]
		return outputs, hidden, cell

class Decoder(nn.Module):
	def __init__(self, TRG, emb_dim, hidden_dim, n_layers, dropout, device):
		super(Decoder, self).__init__()

		self.n_layers = n_layers
		self.device = device
		self.attention = attention
		self.embedding = nn.Embedding(num_embeddings = len(TRG.vocab),
									  embedding_dim = emb_dim,
									  padding_idx = TRG.vocab['<pad>'])
		self.decoder = nn.LSTMCell(input_size = emb_dim + hidden_dim,
								   hidden_size = hidden_dim, bias = True)
		self.dec_hidden_proj = nn.Linear(hidden_dim * 3, hidden_dim, bias = True)
		self.fc_out = nn.Linear(hidden_dim, len(TRG.vocab))
		self.dropout = nn.Dropout(dropout)

	def forward(self, trg, enc_hiddens, enc_hiddens_att, enc_masks, init_hidden_cell):
		# trg: [trg len, batch size]
		# enc_hiddens: [src len, batch size, hidden dim * 2]
		# enc_hiddens_att: [src len, batch size, hidden dim]
		# enc_masks: [src len, batch size]
		# init_hidden_cell: ([batch size, hidden dim], [batch size, hidden dim])

		outputs = []

		batch_size = enc_hiddens_att.shape[1]
		hidden_dim = enc_hiddens_att.shape[2]
		src_len = enc_hiddens.shape[0]

		hidden, cell = init_hidden_cell
		prev_out = torch.zeros(batch_size, hidden_dim, device = self.device)
		# real_trg: [trg len -1, batch size]
		# prev_out: [batch size, hidden dim]

		trg_emb = self.embedding(trg)
		# trg_emb: [trg len -1, batch size, embedding dim]

		attention_weight = torch.FloatTensor(batch_size, src_len, trg_emb.shape[0])
		# attention_weight: [batch size, source len, target len - 1]

		for i in range(trg_emb.shape[0]):

			target = trg_emb[i]
			# target: [batch size, embedding dim]
			_target = torch.cat((target, prev_out), dim = 1)
			# _target: [batch size, embedding dim + hidden dim]
			hidden, cell = self.decoder(_target, (hidden, cell))
			# hidden: [batch size, hidden dim]
			# cell: [batch size, hidden dim]

			e_t = torch.bmm(enc_hiddens_att.permute(1,0,2), hidden.unsqueeze(2)).squeeze(2)
			# e_t: [batch size, src len]

			alpha_t = F.softmax(e_t, dim=1)
			# alpha_t: [batch size, src len]

			attention_weight[:,:,i] = alpha_t

			a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens.permute(1,0,2)).squeeze(1)
			# a_t: [batch size, 2*hidden]

			U_t = torch.cat((a_t, hidden), dim = 1) # U_t: [batch size, 3*hidden dim]
			V_t = self.dec_hidden_proj(U_t) # V_t: [batch size, hidden_dim]
			O_t = self.dropout(torch.tanh(V_t))

			prev_out = O_t
			outputs.append(O_t)

		outputs = torch.stack(outputs)
		# outputs: [trg len, batch size, hidden dim]
		outputs = self.fc_out(outputs)
		# outputs: [trg len, batch size, trg vocab size]

		return outputs, attention_weight

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, n_layers, hidden_dim, device):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.attention = attention
		self.device = device
		self.enc_hidden_proj = nn.Linear(hidden_dim*4, hidden_dim, bias = False)
		self.enc_cell_proj = nn.Linear(hidden_dim*4, hidden_dim, bias = False)
		self.attention_proj = nn.Linear(hidden_dim*2, hidden_dim, bias = False)

	def make_masks(self, src, src_len, batch_size):

		src_lengths = [len(s) for s in src]
		enc_masks = torch.zeros(src_len, batch_size, dtype = torch.float)
		# enc_masks: [src len, batch size]

		for i, slen in enumerate(src_lengths):
			enc_masks[i, slen:] = 1
		return enc_masks.to(self.device)

	def forward(self, src, trg):
		enc_hiddens, hidden, cell = self.encoder(src)
		# enc_hiddens: [src len, batch size, hidden dim * 2]
		# hidden: [n_layers * 2(bidirectional), batch size, hidden dim]
		# cell: [n_layers * 2(bidirectional), batch size, hidden dim]

		src_len = enc_hiddens.shape[0]
		batch_size = enc_hiddens.shape[1]
		enc_masks = self.make_masks(src, src_len, batch_size)

		hidden = self.enc_hidden_proj(torch.cat([hidden[0], hidden[1], hidden[2], hidden[3]], dim = 1))
		cell = self.enc_cell_proj(torch.cat([cell[0], cell[1], cell[2], cell[3]], dim = 1))
		enc_hiddens_att = self.attention_proj(enc_hiddens)
		# hidden: [batch size, hidden dim]
		# cell: [batch size, hidden dim]
		# enc_hiddens_att = [src len, batch size, hidden dim]

		dec_outputs, attention_weight  = self.decoder(trg, enc_hiddens, enc_hiddens_att, enc_masks, (hidden, cell))

		return dec_outputs
