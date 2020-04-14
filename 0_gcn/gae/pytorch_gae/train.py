from __future__ import division
from __future__ import print_function

import time
import os
import argparse

import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from input_data import load_data
from preprocessing import mask_test_edges, preprocess_graph, sparse_to_tuple#, construct_feed_dict

from model import GCNModelAE#, GCNModelVAE

# Setting
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--weight_decay', type=float, default=0.,help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--dropout', type=float, default=0.,help='Dropout rate (1 - keep prob.).')

parser.add_argument('--model', type=str, default='gcn_ae', help='Model string')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset string')
parser.add_argument('--features', type=int, default=1, help='Whether to use features (1) or not (0).')

args = parser.parse_args()

model_str = args.model
dataset_str = args.dataset

# Load data
adj, features = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis,:],[0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if args.features == 0:
	features = sp.identiy(features.shape[0]) # featureless

adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
model = None
if model_str == 'gcn_ae':
	model = GCNModelAE(num_features, features_nonzero)
elif model_str == 'gcn_vae':
	model = GCNModelVAE(num_features, num_nodes, features_nonzero)
