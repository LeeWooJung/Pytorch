import numpy as np

# Adjacency matrix
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)

# Generate 2 integer features for every node
X = np.matrix([
	[i, -i]
	for i in range(A.shape[0])
	], dtype=float)

"""
	A*X: the representation of each node is now
	     a sum of its neighbors features, without itself.
"""

# sum its feature
I = np.matrix(np.eye(A.shape[0]))

A_hat = A + I

# Normalizing the feature representations
D_hat = np.array(np.sum(A_hat, axis = 0))[0]
D_hat = np.matrix(np.diag(D_hat))

W = np.matrix([
	[1,-1],
	[-1,1]
    ])

"""
	relu(D_hat**-1 * A_hat * X * W) : GCN layer
"""
