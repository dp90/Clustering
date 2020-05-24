import numpy as np
import matplotlib.pyplot as plt


class FictiveData(object):
    def __init__(self, cluster_stats):
        self.cluster_stats = cluster_stats
        self.n_clusters = len(cluster_stats)
        self.clusters = []

    def generate(self):
        for i in range(self.n_clusters):
            stats = self.cluster_stats['c' + str(i + 1)]
            self.clusters.append(Cluster(stats['means'], stats['cov'], stats['size']))

    def plot(self):
        plt.figure()
        for i in range(self.n_clusters):
            c = self.clusters[i]
            # Plots first 2 dimensions of data
            plt.scatter(c.x[:, 0], c.x[:, 1])

    def get_dataset(self):
        # gather all data from the clusters in self.clusters
        X = self.clusters[0].x
        c_label = np.ones(X.shape[0])
        if self.n_clusters > 1:
            for i in range(1, self.n_clusters):
                X_i = self.clusters[i].x
                X = np.vstack((X, X_i))
                c_label = np.concatenate((c_label, (i + 1)*np.ones(X_i.shape[0])))
        return X, c_label


class Cluster(object):
    def __init__(self, mean, cov, size):
        self.mean = mean
        self.cov = cov
        self.size = size
        self.x = None
        self.generate()

    def generate(self):
        self.x = np.random.multivariate_normal(self.mean, self.cov, self.size)
