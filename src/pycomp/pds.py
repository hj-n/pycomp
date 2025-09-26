"""
	Pairwise distance shift (PDS)

"""

import numpy as np
import cupy as cp
from .helpers.distance import distance_matrix

def pds(data: np.ndarray | cp.ndarray) -> float:
	"""
	Pairwise distance shift: complexity metric targeting global structure
	
	INPUT:
	- data: numpy array of shape (n_samples, n_features) representing the high-dimensional data

	OUTPUT:
	- float: pairwise distance shift value
	"""
	dist_matrix = distance_matrix(data)
	dist_matrix = dist_matrix.flatten()
	dist_matrix = dist_matrix[dist_matrix > 0]
	return -np.log(np.std(dist_matrix) / np.mean(dist_matrix))