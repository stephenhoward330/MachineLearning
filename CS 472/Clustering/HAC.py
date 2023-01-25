import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):  # # add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.X = None
        self.clusters = None

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        if y is not None:
            X = np.append(X, y, 1)

        # MAKE DISTANCE MATRIX
        dist_matrix = np.empty((len(X), len(X)))
        for idx, row in enumerate(X):
            dist_matrix[idx] = np.linalg.norm(row - X, axis=1)

        # INITIALIZE CLUSTERS
        clusters = {}
        for i in range(len(X)):
            clusters[i] = [i]

        # LOOP AND COMBINE CLUSTERS
        while len(clusters) > self.k:
            min_val = np.inf
            min_pair = []
            for i, row in enumerate(dist_matrix):
                for j, num in enumerate(row):
                    if i != j:
                        if dist_matrix[i][j] < min_val:
                            min_val = dist_matrix[i][j]
                            min_pair = [i, j]
            clusters[min_pair[0]].extend(clusters[min_pair[1]])
            clusters.pop(min_pair[1])

            # set i column to the min (or max) of the i column and j column, and repeat for rows
            if self.link_type == "single":
                new_nums = np.amin([dist_matrix[min_pair[0]], dist_matrix[min_pair[1]]], axis=0)
            elif self.link_type == "complete":
                new_nums = np.amax([dist_matrix[min_pair[0]], dist_matrix[min_pair[1]]], axis=0)
            else:
                raise Exception("invalid link type")
            dist_matrix[:, min_pair[0]] = new_nums
            dist_matrix[min_pair[0]] = new_nums
            dist_matrix[:, min_pair[1]] = np.full(len(dist_matrix), np.nan)
            dist_matrix[min_pair[1]] = np.full(len(dist_matrix), np.nan)

        # SAVE X AND CLUSTERS
        self.X = X
        self.clusters = clusters

        return self

    def calc_centroid_sse(self):
        if self.X is None:
            raise Exception("You must call fit() first!")

        centroids = []
        lengths = []
        sses = []
        for key, cluster in self.clusters.items():
            centroid = np.mean([self.X[i] for i in cluster], axis=0).tolist()
            centroids.append(centroid)
            lengths.append(len(cluster))
            sse = 0
            for i in cluster:
                for j in range(len(self.X[0])):
                    sse += (self.X[i][j] - centroid[j]) ** 2
            sses.append(sse)

        return centroids, lengths, sses

    def save_clusters(self, filename):
        centroids, lengths, sses = self.calc_centroid_sse()

        if filename is None:
            print("HAC", self.link_type, "link")
            print("k:", self.k)
            print("Total SSE:", np.sum(sses))
            for i in range(len(centroids)):
                print()
                print(centroids[i])
                print("Size of cluster:", lengths[i])
                print("SSE of cluster:", sses[i])
            print()
            print()
        else:
            with open(filename, "w+") as f:
                # Used for grading.
                f.write("{:d}\n".format(self.k))
                f.write("{:.4f}\n\n".format(np.sum(sses)))  # total SSE
                for i in range(len(centroids)):  # for each cluster and centroid:
                    f.write(np.array2string(np.array(centroids[i]), precision=4, separator=","))
                    f.write("\n")
                    f.write("{:d}\n".format(lengths[i]))
                    f.write("{:.4f}\n\n".format(sses[i]))
