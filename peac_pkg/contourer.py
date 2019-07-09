# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:01:34 2019

@author: ecramer
"""

import numpy as np
from scipy import interpolate
from skimage.feature import peak_local_max

class Contourer():
    """ TODO: Full writeup of class documentation here. 
    
    Steps:
        1. generate the contours for each factor
        2. find the peak coordinates and values
        3. transform the peak coordinates into the manifold embedding's space
    
    """
    def __init__(self):
        self._interp_method = 'linear'
        self._resolution = 50 # default resolution
        self._peak_distance = np.floor(self._resolution/10.0).astype(np.uint8)
        self._peak_threshold = 0.5
        self._factor_names = [] # list to contain the factors of each column
        
        # storage structures for the results
        self.contours_ = {}
        self.peak_values_ = {}
        self.peak_coors_ = {}
        self.transformed_peaks_ = {}
        self.all_transformed_peaks_ = []
        self.peak_names_ = []
    
    def _check_input_dims(self, X, Y):
        return (X.shape[1] == 2) and (len(Y.shape) > 0)
        
    def fit(self, X, Y, **kwargs):
        """
        X = the manifold embeddings for a 2D space as a numpy array. Each column must be a 
        Y = a pandas dataframe with the values for a series of predictors/factors
        kwargs = extra parameters to feed to the contouring and peak finding algorithms
        """ 
        if self._check_input_dims(X, Y):
            self.X_ = X
            self.Y_ = Y
            
            # get the names of columns if Y is a pandas dataframe, otherwise assign numbers
            if hasattr(Y, 'columns'):
                self._factor_names = Y.columns
            elif Y.ndim > 1: 
                self._factor_names = np.arange(Y.shape[1])
            else: 
                self._factor_names = [0]
            
            # unpack the dictionary to populate the fields in the class
            for key, value in kwargs.items():
                setattr(self, key, value)
            
            # run the algorithms
            self._gen_contours()
            self._find_peaks()
            self._transform_peaks()
            
            return self
        else:
            print('Please double check input for correct dimensions. See documentation for details.')
            return False
        
    def _gen_contour(self, x1, x2, z):
        """ 
        Generates a contour from the manifold embeddings and factor levels
        """
        x_lin = np.linspace(min(x1), max(x1), self._resolution)
        y_lin = np.linspace(min(x2), max(x2), self._resolution)
        # create a grid of points
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        z_grid = interpolate.griddata((x1, x2), z, (x_grid, y_grid), method=self._interp_method)
        return x_grid, y_grid, z_grid
        
        pass
    
    def _gen_contours(self):
        """
        Step 1
        Generate the contours for each factor in Y
        """
        # check to see if the number of factors to contour is > 1, otherwise 
        if self.Y_.ndim < 2:
            z = np.asarray(self.Y_)
            # get the values of the manifold embedding
            x1 = self.X_[:, 0]
            x2 = self.X_[:, 1]
            x1g, x2g, zg = self._gen_contour(x1, x2, z)
            self.contours_[0] = np.nan_to_num(zg)
        else:
            col = 0
            while col < self.Y_.shape[self.Y_.ndim-1]:
                z = np.asarray(self.Y_)[:, col]
                # get the values of the manifold embedding
                x1 = self.X_[:, 0]
                x2 = self.X_[:, 1]
                x1g, x2g, zg = self._gen_contour(x1, x2, z)
                self.contours_[col] = np.nan_to_num(zg) # zero out the non-contoured points in the 2D space
                col += 1 # go to the next column
            
    def _find_peaks(self):
        """
        Step 2
        Find the local peaks in each contour.
        """
        # find the peaks for each contour
        for key, contour in self.contours_.items():
            # find the peaks such that they are not within _peak_distance 'pixels' of each other
            # and the peaks are above the _peak_threshold
            peaks = peak_local_max(contour, 
                                   min_distance=self._peak_distance, 
                                   threshold_rel=self._peak_threshold)
            self.peak_coors_[key] = peaks
            # get the value of each peak found
            self.peak_values_[key] = [contour[i[0], i[1]] for i in peaks] 
    
    def _transform_peaks(self):
        """
        Step 3
        Transform the peaks into the same space as the manifold embedding
        """
        x = np.arange(0, self._resolution+1, 1)
        x = np.interp(x, (x.min(), x.max()), (self.X_[:, 0].min(), self.X_[:, 0].max()))
        y = np.arange(0, self._resolution+1, 1)
        y = np.interp(y, (y.min(), y.max()), (self.X_[:, 1].min(), self.X_[:, 1].max()))   
        xx, yy = np.meshgrid(x, y)
        for key in self.peak_coors_.keys():
            self.transformed_peaks_[key] = np.column_stack(([x[a[0]] for a in self.peak_coors_[key]], 
                                                      [y[a[1]] for a in self.peak_coors_[key]]))
        self.all_transformed_peaks_ = np.concatenate(tuple(self.transformed_peaks_.values()))
        self.peak_names_ = np.concatenate([[self._factor_names[k]]*len(v) for k, v in self.peak_coors_.items()])
    