import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
	index = []
	for line in open(filename):
		index.append(int(line.strip()))
	return index

def load_data(dataset):
	# load the data: x, tx, allx, graph
	names = ['x', 'tx', 'allx', 'graph']
	objects = []
	for name in names:
		with open("data/ind.{}.{}".format(dataset, name), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				obejcts.append(pkl.load(f))
	x, tx, allx, graph = tuple(objects)
	test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
	test_idx_range = np.sort(test_idx_reorder)

	# citeseer
	if dataset == 'citeseer':
		pass


	features = sp.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	return adj, features
