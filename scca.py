from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import optimize , linalg
from scipy.linalg import LinAlgError
from six import with_metaclass
from abc import ABCMeta, abstractmethod
from utils import deflate

class CCABase(with_metaclass(ABCMeta)):
    """ 
    Base class for CCA algorithms
        All Warps must define the following methods::

            CCA.fit() - fit the model
            CCA.transform() - compute tha canonical scores
            Warp.warp_predictions() - compute predictive distribution

    """
    def __init__(self, **kwargs):
        # parse arguments
        self.max_iter = kwargs.get('max_iter', 1000)
        self.n_components = kwargs.get('n_components', 1)
        self.n_views = kwargs.get('n_views',2)

    @abstractmethod
    def fit(self, X, param):
        """ fit the scca method using param
        """

    @abstractmethod
    def transform(self, x, param):
        """ apply the scca method to new data """

class SCCA(CCABase):
    """Sparse canonical correlation analysis 

    Computes a sparse canonical correlation analysis betweeen two sets of
    variables

    Basic usage::

        scca = SCCA()
        scca.fit(X, Y)
        Y_pred = scca.predict(X)

    where the variables are:
    
    :param X: N x D1 data array (view 1)
    :param Y: N x D2 data array (view 2)
   
    :returns: Y_pred: prediction for Y

    References:
        
    Witten, D. et al 2009. A Penalized Matrix Decomposition, with Applications 
    to Sparse Principal Components and Canonical Correlation Analysis.
    Biostatistics 10 (3): 515--34.
    
    Ing, A, et al. 2019. Identification of Neurobehavioural Symptom Groups 
    Based on Shared Brain Mechanisms. Nature Human Behaviour 3 (12): 1306--18.

    Written by A. Marquand
    """
        
    def fit(self, X, Y, **kwargs):
        """ Fit a binary sCCA model

        Basic usage::

            fit(X, Y, [extra_arguments])
        
        where the variables are:
    
        :param X: N x D1 data array (view 1)
        :param Y: N x D2 data array (view 2)
        :param l1_x: sparsity parameter for view 1 (0..1, default=0.5)
        :param l1_y: sparsity parameter for view 2 (0..1, default=0.5)    
        :param sign_x: constraint for view 1 (-1:neg, 0:none(default), 1:pos)
        :param sign_y: constraint for view 1 (-1:neg, 0:none(default), 1:pos)
        """
        
        # get parameters
        l1_x = kwargs.get('l1_x',0.5)
        l1_y = kwargs.get('l1_y',0.5)
        sign_x = kwargs.get('sign_x', 0)
        sign_y = kwargs.get('sign_y', 0)
        verbose = kwargs.get('verbose', False)
        
        #initialise weights
        self.Wx = np.zeros((X.shape[1], self.n_components))
        self.Wy = np.zeros((Y.shape[1], self.n_components))
        self.x_scores = np.zeros((X.shape[0], self.n_components))
        self.y_scores = np.zeros((Y.shape[0], self.n_components))
        self.r = np.zeros(self.n_components)

        for k in range(self.n_components):
            if verbose: 
                print('Component', k, 'of', self.n_components,':')

            # initialise w1 using svd
            U, S, V = np.linalg.svd(X.T.dot(Y), full_matrices=False)
            wx = U[:,0] / np.linalg.norm(U[:,0])
            
            # set up sparsity constraints and ensure cx > 1
            cx = np.round(max(np.sqrt(X.shape[1]) * l1_x, 1.0), decimals=2)
            cy = np.round(np.sqrt(Y.shape[1]) * l1_y, decimals=2)
            
            for i in range(self.max_iter):
                if i > 0:
                    old_wx = wx
                    old_wy = wy
                                
                # compute w2            
                self.ay = Y.T.dot(X).dot(wx)
                if sign_y > 0:
                    self.ay = np.maximum(self.ay,0)
                elif sign_y < 0: 
                    self.ay = -np.maximum(-self.ay,0)
                
                sign_ay = (self.ay > 0) * 2 - 1
                Sy_p = np.abs(self.ay)
                Sy = sign_ay * (Sy_p * (Sy_p > 0))
                self.ay = Sy
                wy = self.ay / np.linalg.norm(self.ay)
                l1_norm_wy_r = np.linalg.norm(wy, 1).round(decimals=2)
                
                if l1_norm_wy_r >= cy:
                    delta_max = max(np.abs(self.ay))/2
                    delta_tmp = 0
                    while l1_norm_wy_r != cy:
                        sign_delta = (np.linalg.norm(wy, 1) > cy)*2 -1
                        delta = delta_tmp + sign_delta * delta_max 
                        Sy_p = np.abs(self.ay) - delta
                        Sy = sign_ay * (Sy_p * (Sy_p > 0))
                        wy = Sy / np.linalg.norm(Sy)
                        l1_norm_wy_r = np.linalg.norm(wy, 1).round(decimals=2)
                        delta_tmp = delta
                        delta_max = delta_max/2

                # compute w1
                self.ax = X.T.dot(Y).dot(wy)
                if sign_x > 0:
                    self.ax = np.maximum(self.ax,0)
                elif sign_x < 0:
                    self.ax = -np.maximum(-self.ax,0)
                
                sign_ax = (self.ax > 0) * 2 - 1
                Sx_p = np.abs(self.ax)
                Sx = sign_ax * Sx_p * (Sx_p > 0)
                self.ax = Sx
                wx = self.ax / np.linalg.norm(self.ax)
                l1_norm_wx_r = np.linalg.norm(wx, 1).round(decimals=2)
                
                if l1_norm_wy_r >= cx:
                    delta_max = max(np.abs(self.ax))/2
                    delta_tmp = 0
                    while l1_norm_wx_r != cx: 
                        sign_delta = (np.linalg.norm(wx, 1) > cx)*2 -1 
                        delta = delta_tmp + sign_delta*delta_max
                        Sx_p = np.abs(self.ax)-delta
                        Sx = sign_ax * (Sx_p * (Sx_p > 0))
                        wx = Sx / np.linalg.norm(Sx)
                        l1_norm_wx_r = np.linalg.norm(wx, 1).round(decimals=2)
                        delta_tmp = delta
                        delta_max = delta_max / 2        
                
                if verbose:
                    if i == 0:
                        print('iter:', i)
                    else:
                        x_diff = wx - old_wx
                        y_diff = wy - old_wy
                        print('iter:', i,
                            'd(x) =', np.dot(x_diff, x_diff),
                            'd(y) =', np.dot(y_diff, y_diff))

            x_scores = X.dot(wx)
            y_scores = Y.dot(wy)
            
            self.r[k] = np.corrcoef(x_scores, y_scores)[0][1]

            if self.n_components > 1:
                if verbose: 
                    print('deflating...')
                X = deflate(X, wx)
                Y = deflate(Y, wy)

            # save useful quantities
            self.Wx[:,k] = wx
            self.Wy[:,k] = wy
            self.x_scores[:,k] = x_scores
            self.y_scores[:,k] = y_scores

            # save principal canonical weights (for backward compatibility)
            if k == 0:
                self.wx = wx
                self.wy = wy
        
    def transform(self, Xs, Ys=None):
        xs_scores = Xs.dot(self.Wx)

        
        if Ys is not None:
            ys_scores = Ys.dot(self.Wy)
            return xs_scores, ys_scores
        else:
            return xs_scores
        
class MSCCA(CCABase):

    def scca_vec(x, w):
        
        return a

    def fit(self, X, **kwargs):
        """ Fit a multi-way sCCA model

        Basic usage::

            fit(X, [extra_arguments])
        
        where the variables are:
    
        :param X: list of N x D_m data arrays (M = number of views)
        :param l1: list of sparsity parameters (0..1, default=0.5)
        :param sign: 
        """
        self.n_views = len(X)

        # get parameters
        l1 = kwargs.get('l1', [0.5] * self.n_views)
        sign = kwargs.get('sign', [0] * self.n_views)
        verbose = kwargs.get('verbose', False)
        
        # initialise weights, sparsity constraints and scores
        self.W = list()
        self.c = list()
        self.scores = list()
        for v in range(self.n_views):
            if len(X[v].shape) == 1 or X[v].shape[1] > 1:
                w_tmp = np.random.normal(size=X[v].shape[1])
            else:
                w_tmp = sign[v]
            self.W.append(w_tmp/np.linalg.norm(w_tmp))

            self.c.append(np.round(max(np.sqrt(X[v].shape[1]) * l1[v], 1.0), decimals=2))
            self.scores.append(np.zeros((X[v].shape[0], self.n_components)))
        
        for v in range(self.n_views):
            if self.c[v] <= 1:
                self.c[v] = 1

        for k in range(self.n_components):
            if verbose: 
                print('Component', k, 'of', self.n_components,':')

            scores = np.zeros((X[1].shape[0], self.n_views))

    def transform(self, Xs):
        xs_scores = list()
        for v in range(self.n_views):
            xs_scores.append(Xs[v].dot(self.W[v]))

        return xs_scores