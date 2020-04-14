import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
	""" 
		Create a weight variable with Glorot & Bengio (AISTATS 2010)
		initialization.
	"""
	init_range = np.sqrt(6.0 / (input_dim + output_dim))
	initial = Parameter(torch.FloatTensor(input_dim, output_dim).uniform_(-init_range,init_range))
	return Variable(initial, name=name, requires_grad=True)
