#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:35:50 2020

@author: David Kroon
"""

from GenerateData import FictiveData
import scipy.stats as ss
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
# import hdbscan as HDBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
# see https://scikit-learn.org/stable/modules/clustering.html for other algorithms


def run_algorithms(algorithms):
    for algo in algorithms:
        if algo['run'] is True:
            model = algo['model'](*algo['args'])
            model.fit(X)
            labels = model.labels_
            # algo['labels'] = labels
            df[algo['col_name']] = labels


cluster_stats = dict(
    c1=dict(means=[1, 1], cov=[[0.2, 0], [0, 0.2]], size=500),
    c2=dict(means=[5, 1], cov=[[0.2, 0], [0, 0.2]], size=500),
    c3=dict(means=[1, 5], cov=[[0.2, 0], [0, 0.2]], size=500),
    c4=dict(means=[5, 5], cov=[[0.2, 0], [0, 0.2]], size=500)
)

fict_data = FictiveData(cluster_stats)
fict_data.generate()
fict_data.plot()
X, cluster_label = fict_data.get_dataset()

var_names = ['x1', 'x2']
df = pd.DataFrame(data=X, columns=var_names)

# Plot correlations between needs
corr_coeff, p = ss.pearsonr(X[:, 0], X[:, 1])
print("correlation coefficient is %.3f, with a p-value of %.2f" % (corr_coeff, p))

# k-means
n_clusters = 4
kmeans = dict(model=KMeans, args=[n_clusters], run=False, col_name='KM_4')

# dbscan
min_samples = 10  # 5 is default
eps = 0.55  # 0.5 is default
dbscan = dict(model=DBSCAN, args=[eps, min_samples], run=True, col_name='DB_10')

# hdbscan
min_cluster_size = 100
hdbscan = dict(model=HDBSCAN, args=[min_cluster_size], run=False, col_name='HDB_100')

# optics
min_samples = 100
optics = dict(model=OPTICS, args=[min_samples], run=False, col_name='OPT_100')

# affinity propagation
damping = 0.5
affin_prop = dict(model=AffinityPropagation, args=[damping], run=False, col_name='AP_0.5')

# mean shift
bin_seeding = True
mean_shift = dict(model=MeanShift, args=[bin_seeding], run=False, col_name='MS')

# spectral clustering
n_clusters = 4
spect_clust = dict(model=SpectralClustering, args=[n_clusters], run=False, col_name='SC_4')

# agglomerative clustering
n_clusters = 4
agg_clust = dict(model=AgglomerativeClustering, args=[n_clusters], run=False, col_name='AC_4')

algorithms = [kmeans, dbscan, hdbscan, optics, mean_shift, spect_clust, agg_clust]

run_algorithms(algorithms)

df.plot.scatter(x='x1', y='x2', c='DB_10', colormap='viridis')
