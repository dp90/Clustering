#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:35:50 2020

@author: David Kroon
"""

from GenerateData import FictiveData
import numpy as np
import matplotlib.pyplot as plt
# see https://scikit-learn.org/stable/modules/clustering.html for other algorithms

cluster_stats = dict(
    c1=dict(means=[1, 2],     cov=[[0.1, 0], [0, 0.4]],  size=50),
    c2=dict(means=[4, 8],     cov=[[0.5, 0], [0, 0.5]],  size=100),
    c3=dict(means=[2.3, 5.1], cov=[[0.45, 0], [0, 0.9]], size=125)
)

fict_data = FictiveData(cluster_stats)
fict_data.generate()
fict_data.plot()
X, cluster_label = fict_data.get_dataset()
