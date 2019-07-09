# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:58:58 2019

@author: ecramer
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def plot_clustering(contourer, cluster_labels, **kwargs):
    """
    TODO: fully document
    contourer = the contouring object
    cluster_labels = the labels from the clustering algorithm
    """
    fs = (10,10) if 'figsize' not in kwargs else kwargs['figsize']
    
    # match the contouring space back to the manifold space
    x = np.arange(0, contourer._resolution+1, 1)
    x = np.interp(x, (x.min(), x.max()), (contourer.X_[:, 0].min(), contourer.X_[:, 0].max()))
    y = np.arange(0, contourer._resolution+1, 1)
    y = np.interp(y, (y.min(), y.max()), (contourer.X_[:, 1].min(), contourer.X_[:, 1].max()))   
    xx, yy = np.meshgrid(x, y)
    
    # create the borders within the manifold space with KNN
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(contourer.all_transformed_peaks_, cluster_labels)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = Z.reshape(xx.shape)
    
    # set up the plot
    fig, ax = plt.subplots(figsize=fs)
    
    # plot the cluster areas a s a filled contour
    ax.contourf(xx, yy, zz, cmap='hsv', alpha=0.1, zorder=1)
    ax.contour(xx, yy, zz, colors='black', linewidths=1, zorder=2)
    
    # plot the manifold embedding
    ax.scatter(contourer.X_[:, 0], contourer.X_[:, 1],
       alpha=0.3, s=0.1, c='gray', zorder=3)
    
    # plot the peaks    
    for fact in contourer._factor_names:
        ix = np.where(contourer.peak_names_ == fact)
        ax.scatter(contourer.all_transformed_peaks_[ix, 0], contourer.all_transformed_peaks_[ix, 1],
                        label=fact, s=30, cmap='nipy_spectral', marker='^', zorder=4)

    # titles and axes
    ax.set_title('Clustered Maxima')
    ax.set_xlabel('UMAP 1')
    ax.set_xticks([],[])
    ax.set_ylabel('UMAP 2')
    ax.set_yticks([], [])
    
    # add the legend
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='center', ncol=3)
    
    # display and closing
    plt.show() 
    return fig
    