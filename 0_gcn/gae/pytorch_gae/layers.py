from initializations import *

import torch
import torch.nn as nn

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
	"""
		Helper function, assigns unique layer IDs
	"""
	if layer_name not in _LAYER_UIDS:
		_LAYER_UIDS[layer_name] = 1
		return 1
	else:
		_LAYER_UIDS[layer_name] += 1
		return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
	"""
		Dropout for sparse tensors.
		Currently fails for very large sparse tensors (>1M elements)
	"""
	noise_shape = [num_nonzero_elems]
	random_tensor = keep_prob
	random_tensor += torch.rand(noise_shape)
	dropout_mask = torch.floor(random_tensor).bool()
	""" """
	pre_out = x[dropout_mask]
	""" """
	return pre_out * (1./keep_prob)

class Layer(object):
	"""
		Base layer class. Defines basci API for all layer objects
	
		# Properties
			name: String, defines the variable scope of the layer
		# Methods
			_call(inputs): Defines computation graph of layer
				(i.e. takes input, returns output)
			__call__(inputs): Wrapper for _call()
	"""

	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			layer = self.__class__.__name__.lower()
			name = layer + '_' + str(get_layer_uid(layer))
		self.name = name
		self.vars = {}
		logging = kwargs.get('logging', False)
		self.logging = logging
		self.issparse = False

	def _call(self, inputs):
		return inputs

	def __call__(self, inputs):
		outputs = self._call(inputs)
		return outputs

class GraphConvolution(Layer):
	"""
		Basic graph convolution layer for undirected graph without edge labels.
	"""
	def __init__(self, input_dim, output_dim, adj, dropout=0., act=nn.ReLU(), **kwargs):
		super(GraphConvolution, self).__init__(**kwargs)
		self.vars['weight'] = weight_variable_glorot(input_dim, output_dim, name="weights")
		self.dropout = dropout
		self.adj = adj
		self.act = act

	def _call(self, inputs):
		x = inputs
		x = nn.Dropout(x, p=1-self.dropout)
		x = torch.matmul(x, self.vars['weights'])
		x = torch.sparse.mm(self.adj, x)
		outputs = self.act(x)
		return outputs

class GraphConvolutionSparse(Layer):
	"""
		Graph convolution layer for sparse inputs.
	"""
	def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout, act=nn.ReLU(), **kwargs):
		super(GraphConvolutionSparse, self).__init__(**kwargs)
		self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
		self.dropout = dropout
		self.adj = adj
		self.act = act
		self.issparse = True
		self.features_nonzero = features_nonzero

	def _call(self, inputs):
		x = inputs
		x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
		x = torch.sparse.mm(x, self.vars['weights'])
		x = torch.sparse.mm(self.adj, x)
		outputs = self.act(x)
		return outputs

class InnerProductDecoder(Layer):
	"""
		Decoder model layer for link prediction.
	"""
	def __init__(self, input_dim, dropout=0., act=nn.Sigmoid(), **kwargs):
		super(InnerProductDecoder, self).__init__(**kwargs)
		self.dropout = dropout
		self.act = act

	def _call(self, inputs):
		inputs = nn.Dropout(inputs, p=1-self.dropout)
		x = torch.t(inputs)
		x = torch.matmul(inputs, x)
		x = x.view(1,-1)
		outputs = self.act(x)
		return outputs
