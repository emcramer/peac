# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:36:44 2019

Example walkthrough for a more complex or real world data sets.

@author: ecramer
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from contourer import Contourer
import plotter

# all viable markers from the dataset
all_markers = ['CD90.2','CD45.2', 'Ter119','CD11b','Ly6GC',
 'pCDK1','CD69','pBRCA1','CD4','pATM','pH2AX','CyclinB1','KLRG1','CD27',
 'Ki67','CD3','CD45.1','TIM3','OX40','RGS1','pRad51','Foxp3','PD1','tbet',
 'pATR','p21','pBRCA2','CD62L','NK1.1','CD19','Rad51','CD8','TCRb','CD137',
 'CD44','CD86','CTLA4','CD223','pHH3','B220','MHCII']

# filters for markers of interest
prolif_markers = ['Ki67', 'pHH3', 'CyclinB1', 'pCDK1']
ddr_markers = ['pBRCA1', 'pH2AX', 'pRad51', 'pATR','p21', 'pBRCA2', 
               'Rad51', 'RGS1', 'pATM']
tex_markers = ['OX40', 'CD137', 'PD1', 'TIM3', 'CTLA4', 'CD223']
predictor_markers = prolif_markers + ddr_markers + tex_markers

# isolate the non-predictor markers
nonpredictor_markers = list(
    set(all_markers).symmetric_difference(
        set(predictor_markers)))

# load the data
all_data = pd.read_csv("data/27207_lymphoma_panel_1_concatenated.csv", index_col=0)
# isolate data that does not contain the tumor cells
nontumor_data = all_data.loc[all_data['celltype']!='tumor', :]
nontumor_data.columns = [a.split('_')[0] for a in nontumor_data.columns]

expr_data = nontumor_data[prolif_markers]
scaler = StandardScaler().fit(expr_data)
scaled_expr_data = pd.DataFrame(scaler.transform(expr_data))
scaled_expr_data.columns = prolif_markers

umap = joblib.load('data/all_umap_0.2_15_euclidean.joblib')

# do the contouring
cc = Contourer()
cc.fit(umap, scaled_expr_data, **{'_resolution':100})

# do kmeans clustering on the peaks
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0).fit(cc.all_transformed_peaks_)

# plot the clustering with the peak analysis
f = plotter.plot_clustering(cc, kmeans.labels_)
