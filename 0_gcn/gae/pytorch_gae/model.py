from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder

import torch
import torch.nn.functional as F

class Model(object):
	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

		name = kwargs.get('name')
		if not name:
			name = self.__class__.__name__.lower()
		self.name = name
		print(kwargs.keys())
		print(self.name)
		self.vars = {}

	def _build(self):
		raise NotImplementedError

	def build(self):
		"""
			Wrapper for _build()
		"""
		print(self.name)




class GCNModelAE(Model):
	def __init__(self, num_features, features_nonzero, **kwargs):
		super(GCNModelAE, self).__init__(**kwargs)

		self.input_dim = num_features
		self.features_nonzero = features_nonzero
		self.inputs = 
#		self.build()
