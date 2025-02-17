import numpy as np

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

