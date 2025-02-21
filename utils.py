import numpy as np
from scipy import stats

def deflate(X, w, type='projection'):
    """
    Deflate data matrix used in multiview canonical correlation analysis 
    Currently, only projection deflation is supported. The variables are:
    
    :param X: data matrix
    :param w: scca weights
    """

    if type == 'projection':
        X = X - np.outer(X.dot(w), w)
    else:
        raise ValueError('invalid type specified')
    return X

def compute_loadings(scores, X):
    """
    Compute loadings from a set of scores (or cross-loadings)
    
    :param scores: n_samples x rank matrix containing scores from a CCA analysis
    :param X: n_samples x dimension data matrix
    """
    
    n,d = X.shape
    
    rank = scores.shape[1]
    
    RR = np.zeros((rank,d))
    for rr in range(rank):
        RR[rr,:] = stats.pearsonr(np.tile(scores[:,rr], (d,1)).T, X).statistic
        
    return RR