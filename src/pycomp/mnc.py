import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from helpers.neighborhood_calculations import NeighborhoodComputations
import numpy as np

def mnc(data: cp.ndarray | np.ndarray, k: int) -> float:
    """
    Mutual Neighborhood Consistency (MNC) fully GPU-optimized, sparse.
    
    Parameters
    ----------
    data : np.ndarray or cp.ndarray, shape (n_samples, n_features)
        Input dataset.
    k : int
        Number of nearest neighbors.
        
    Returns
    -------
    float
        Mutual Neighbor Consistency value.
    """
    
    # @Hyeon this is a potential bottleneck since moving data between GPU and host is unnecessary,
    # I would prefer to keep everything on gpu (also MNC and PDS calculations), but I did not want to mess with your tests
    if isinstance(data, np.ndarray):
        data = cp.asarray(data)

    n = data.shape[0]
    ksnn = NeighborhoodComputations(k)

    knn_indices = cp.asarray(ksnn.knn(data))
    snn_results = cp.asarray(ksnn.snn(knn_indices))

    rows = cp.repeat(cp.arange(n), k)
    cols = knn_indices.ravel()
    vals = cp.tile((k - cp.arange(k)) / k, n)

    knn_sparse = cpx_sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    snn_sparse = cpx_sparse.csr_matrix(snn_results)

    knn_norms = cp.sqrt(knn_sparse.multiply(knn_sparse).sum(axis=1)).ravel()
    snn_norms = cp.sqrt(snn_sparse.multiply(snn_sparse).sum(axis=1)).ravel()

    dot_products = knn_sparse.multiply(snn_sparse).sum(axis=1).ravel()

    cos_sims = dot_products / (knn_norms * snn_norms)

    return float(cp.mean(cos_sims))