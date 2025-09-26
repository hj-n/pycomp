import cupy as cp
import numpy as np


def distance_matrix(data: cp.ndarray) -> cp.ndarray:
    """
    Compute full pairwise Euclidean distance matrix using CuPy.

    Parameters
    ----------
    data : np.ndarray or cp.ndarray of shape (N, D)
        Input dataset.

    Returns
    -------
    dist_matrix : cp.ndarray of shape (N, N)
        Pairwise distance matrix.
    """
    if isinstance(data, np.ndarray):
        data = cp.asarray(data)

    sq_norms = cp.sum(data ** 2, axis=1, keepdims=True)  # (N, 1)
    dists = sq_norms + sq_norms.T - 2 * data @ data.T
    return cp.sqrt(dists)
