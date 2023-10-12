import numpy as np


def deflate(X, w, type='projection'):
    if type == 'projection':
        X = X - np.outer(X.dot(w), w)
    else:
        raise ValueError('invalid type specified')
    return X