import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False):  # # add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids
                        otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.X = None
        self.clusters = None
        self.centroids = None

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

        # INITIALIZE CENTROIDS
        if self.debug:
            centroids = X[0:self.k, :]
        else:
            indices = np.random.choice(len(X), size=self.k, replace=False)
            centroids = X[indices, :]

        # LOOP -- GROUP POINTS INTO CLUSTERS, RECALCULATE CENTROIDS
        changed = True
        clusters = None
        while changed:
            clusters = [[] for i in range(self.k)]
            changed = False

            # GROUP POINTS INTO CLUSTERS
            for point in X:
                min_dist = np.inf
                best_centroid = None
                for i in range(len(centroids)):
                    dist = np.linalg.norm(point - centroids[i])
                    if dist < min_dist:
                        min_dist = dist
                        best_centroid = i
                clusters[best_centroid].append(point)

            # CALCULATE CENTROIDS
            new_centroids = []
            for i in range(self.k):
                new_centroid = np.mean(clusters[i], axis=0)
                new_centroids.append(new_centroid)
                if not np.array_equal(new_centroid, centroids[i]):
                    changed = True
            centroids = new_centroids

        self.X = X
        self.clusters = clusters
        self.centroids = centroids

        return self

    def calc_sses(self):
        if self.X is None:
            raise Exception("Call fit() first!")

        sses = []
        lengths = []
        for i in range(self.k):
            sse = 0
            lengths.append(len(self.clusters[i]))
            for j in range(len(self.clusters[i])):
                for k in range(len(self.X[0])):
                    sse += (self.clusters[i][j][k] - self.centroids[i][k]) ** 2
            sses.append(sse)

        return lengths, sses

    def save_clusters(self, filename):
        lengths, sses = self.calc_sses()

        if filename is None:
            print("K-Means")
            print("k:", self.k)
            print("Total SSE:", np.sum(sses))
            for i in range(self.k):
                print()
                print(self.centroids[i])
                print("Size of cluster:", lengths[i])
                print("SSE of cluster:", sses[i])
            print()
            print()
        else:
            with open(filename, "w+") as f:
                # Used for grading.
                f.write("{:d}\n".format(self.k))
                f.write("{:.4f}\n\n".format(np.sum(sses)))  # total SSE
                for i in range(len(self.centroids)):  # for each cluster and centroid:
                    f.write(np.array2string(np.array(self.centroids[i]), precision=4, separator=","))
                    f.write("\n")
                    f.write("{:d}\n".format(lengths[i]))
                    f.write("{:.4f}\n\n".format(sses[i]))
