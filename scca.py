from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import optimize , linalg
from scipy.linalg import LinAlgError

class SCCA:
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

    def __init__(self, **kwargs):
        # parse arguments
        self.max_iter = kwargs.get('max_iter', 1000)
        self.n_components = kwargs.get('n_components', 1)
        
    def fit(self, X, Y, **kwargs):
        """ Fit a binary sCCA model

        Basic usage::

            fit(X, Y, [extra_arguments])
        
        where the variables are:
    
        :param X: N x D1 data array (view 1)
        :param Y: N x D2 data array (view 2)
        :param l1_x: sparsity parameter for view 1 (0..1, default=0.5)
        :param l1_y: sparsity parameter for view 2 (0..1, default=0.5)    
        """
        
        # get parameters
        l1_x = kwargs.get('l1_x',0.5)
        l1_y = kwargs.get('l1_y',0.5)
        sign_x = kwargs.get('sign_x', 0)
        sign_y = kwargs.get('sign_y', 0)
        verbose = kwargs.get('verbose', False)
        
        # initialise w1 using svd
        U, S, V = np.linalg.svd(X.T.dot(Y))
        self.wx = U[:,0] / np.linalg.norm(U[:,0])
        
        # set up sparsity constraints and ensure cx > 1
        cx = np.round(max(np.sqrt(X.shape[1]) * l1_x, 1.0), decimals=2)
        cy = np.round(np.sqrt(Y.shape[1]) * l1_y, decimals=2)
                
        for i in range(self.max_iter):
            if verbose:
                print('iter:', i)
                
            # compute w2            
            self.ay = Y.T.dot(X).dot(self.wx)
            if sign_y > 0:
                self.ay = np.maximum(self.ay,0)
            elif sign_y < 0: 
                self.ay = -np.maximum(-self.ay,0)
            
            sign_ay = (self.ay > 0) * 2 - 1
            Sy_p = np.abs(self.ay)
            Sy = sign_ay * (Sy_p * (Sy_p > 0))
            self.ay = Sy
            self.wy = self.ay / np.linalg.norm(self.ay)
            l1_norm_wy_r = np.linalg.norm(self.wy, 1).round(decimals=2)
            
            if l1_norm_wy_r >= cy:
                delta_max = max(np.abs(self.ay))/2
                delta_tmp = 0
                while l1_norm_wy_r != cy:
                    sign_delta = (np.linalg.norm(self.wy, 1) > cy)*2 -1
                    delta = delta_tmp + sign_delta * delta_max 
                    Sy_p = np.abs(self.ay) - delta
                    Sy = sign_ay * (Sy_p * (Sy_p > 0))
                    self.wy = Sy / np.linalg.norm(Sy)
                    l1_norm_wy_r = np.linalg.norm(self.wy, 1).round(decimals=2)
                    delta_tmp = delta
                    delta_max = delta_max/2

            # compute w1
            self.ax = X.T.dot(Y).dot(self.wy)
            if sign_x > 0:
                self.ax = np.maximum(self.ax,0)
            elif sign_x < 0:
                self.ax = -np.maximum(-self.ax,0)
            
            sign_ax = (self.ax > 0) * 2 - 1
            Sx_p = np.abs(self.ax)
            Sx = sign_ax * Sx_p * (Sx_p > 0)
            self.ax = Sx
            self.wx = self.ax / np.linalg.norm(self.ax)
            l1_norm_wx_r = np.linalg.norm(self.wx, 1).round(decimals=2)
            
            if l1_norm_wy_r >= cx:
                delta_max = max(np.abs(self.ax))/2
                delta_tmp = 0
                while l1_norm_wx_r != cx: 
                    sign_delta = (np.linalg.norm(self.wx, 1) > cx)*2 -1 
                    delta = delta_tmp + sign_delta*delta_max
                    Sx_p = np.abs(self.ax)-delta
                    Sx = sign_ax * (Sx_p * (Sx_p > 0))
                    self.wx = Sx / np.linalg.norm(Sx)
                    l1_norm_wx_r = np.linalg.norm(self.wx, 1).round(decimals=2)
                    delta_tmp = delta
                    delta_max = delta_max / 2

        self.x_scores = X.dot(self.wx)
        self.y_scores = Y.dot(self.wy)
        
        self.r = np.corrcoef(self.x_scores, self.y_scores)[0][1]