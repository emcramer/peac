# -*- coding: utf-8 -*-
"""
Ranks the markers within each cluster based on their relative peak height.

Created on Tue Jul  9 15:55:53 2019

@author: ecramer
"""

import numpy as np

def intra_cluster_ranking(contourer, cluster_labels, method='height', **kwargs):
    """
    Wrapper function for ranking peaks within their clusters. 
    
    contourer = the contourer object to rank with
    cluter_labels = labels for each peak to indicate cluster
    method = which method to use when ranking within a cluster
    """
    if method == 'height': 
        return rank_by_height(contourer, cluster_labels)
    elif method == 'momentum':
        print('Sorry, that functionality is not implemented yet.')
    else:
        print('Sorry, that option is not recognized. Please check the documentation for supported options.')

def rank_by_height(contourer, cluster_labels):
    """
    Ranks the peaks in each cluster by their relative height.
    """
    # zip the cluster label to the peak label to the peak's height
    cluster_list = list(zip(cluster_labels, 
             contourer.peak_names_, 
             np.concatenate(tuple(contourer.peak_values_.values()))))
    intraclust = {}
    # populate assign each peak label and height to a cluster
    for tup in cluster_list:
        if tup[0] in intraclust:
            intraclust[tup[0]].append((tup[1], tup[2]))
        else:
            intraclust[tup[0]] = [(tup[1], tup[2])]
    # sort the markers within each clustrer
    for k in intraclust.keys():
         intraclust[k] = sorted(intraclust[k], reverse=True, key=lambda a: a[1])
    return intraclust