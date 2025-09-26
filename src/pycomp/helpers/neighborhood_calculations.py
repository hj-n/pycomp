import cupy as cp
from cuml.neighbors import NearestNeighbors
from cupyx.scipy.sparse import csr_matrix
import numpy as np


class NeighborhoodComputations:
    """
    Compute k-Nearest Neighbors (kNN) and Shared Nearest Neighbors (SNN) graphs.
    """

    def __init__(self, k: int = 20):
        self.k = k

    def knn(self, data: cp.ndarray):
        """
        Compute kNN indices for data.

        Parameters
        ----------
        data : np.ndarray or cp.ndarray of shape (N, D)
            Dataset with N samples and D features.

        Returns
        -------
        knn_indices : cp.ndarray of shape (N, k)
            Indices of k nearest neighbors for each sample.
        """
        if isinstance(data, np.ndarray):
            data = cp.asarray(data)

        knn = NearestNeighbors(n_neighbors=self.k + 1)
        knn.fit(data)
        # discard self-neighbor at index 0
        # @Hyeon this is a potential bottleneck since moving data between GPU and host is unnecessary, I would prefer to
        # keep everything on gpu (also MNC and PDS calculations), but I did not want to mess with your tests
        return knn.kneighbors(data, return_distance=False)[:, 1:].get()

    @staticmethod
    def snn(knn_indices: np.ndarray | cp.ndarray, weighted: bool = True):
        """
        Compute the SNN graph from kNN indices.

        Parameters
        ----------
        knn_indices : cp.ndarray of shape (N, k)
            Indices of k nearest neighbors for each sample.
        weighted : bool, default=False
            If True, weight edges by rank. If False, binary edges.

        Returns
        -------
        snn_graph : csr_matrix of shape (N, N)
            Shared Nearest Neighbor graph.
        """
        if isinstance(knn_indices, np.ndarray):
            knn_indices = cp.asarray(knn_indices)

        n, k = knn_indices.shape

        rows = cp.arange(n)[:, None]
        rows = cp.broadcast_to(rows, (n, k))
        cols = knn_indices

        if weighted:
            vals = cp.tile(cp.arange(k, 0, -1) + 1, n).astype(cp.float32)
        else:
            vals = cp.ones(n * k, dtype=cp.float32)

        knn_graph = csr_matrix((vals, (rows.ravel(), cols.ravel())), shape=(n, n))
        snn_graph = knn_graph @ knn_graph.T

        snn_graph.setdiag(cp.zeros(n, dtype=cp.float32))
        # @Hyeon this is a potential bottleneck since moving data between GPU and host is unnecessary, I would prefer to
        # keep everything on gpu (also MNC and PDS calculations), but I did not want to mess with your tests
        return snn_graph.toarray().get()