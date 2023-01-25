from HAC import HACClustering
from Kmeans import KMEANSClustering
from arff import Arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


if __name__ == '__main__':
    MINE = False
    DEBUG = False
    EVAL = False
    IRIS = False
    SK_IRIS = False
    SK_DIABETES = True

    if MINE:
        mat = Arff("datasets/mine.arff", label_count=0)  # # label_count = 0 because clustering is unsupervised.

        raw_data = mat.data
        data = raw_data

        # ## Normalize the data ###
        scaler = MinMaxScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

        # # ## K-MEANS ###
        KMEANS = KMEANSClustering(k=2, debug=True)
        KMEANS.fit(norm_data)
        KMEANS.save_clusters("debug_kmeans.txt")

        # ## HAC SINGLE LINK ###
        # HAC_single = HACClustering(k=1, link_type='single')
        # HAC_single.fit(norm_data)
        # HAC_single.save_clusters("debug_hac_single.txt")

        # # ## HAC COMPLETE LINK ###
        # HAC_complete = HACClustering(k=1, link_type='complete')
        # HAC_complete.fit(norm_data)
        # HAC_complete.save_clusters("debug_hac_complete.txt")

    if DEBUG:
        mat = Arff("datasets/abalone.arff", label_count=0)  # # label_count = 0 because clustering is unsupervised.

        raw_data = mat.data
        data = raw_data

        # ## Normalize the data ###
        scaler = MinMaxScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

        # # ## K-MEANS ###
        KMEANS = KMEANSClustering(k=5, debug=True)
        KMEANS.fit(norm_data)
        KMEANS.save_clusters("debug_kmeans.txt")

        # ## HAC SINGLE LINK ###
        HAC_single = HACClustering(k=5, link_type='single')
        HAC_single.fit(norm_data)
        HAC_single.save_clusters("debug_hac_single.txt")

        # # ## HAC COMPLETE LINK ###
        HAC_complete = HACClustering(k=5, link_type='complete')
        HAC_complete.fit(norm_data)
        HAC_complete.save_clusters("debug_hac_complete.txt")

    if EVAL:
        mat = Arff("datasets/seismic-bumps_train.arff", label_count=0)  # # clustering is unsupervised.

        raw_data = mat.data
        data = raw_data

        # ## Normalize the data ###
        scaler = MinMaxScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

        # # ## K-MEANS ###
        KMEANS = KMEANSClustering(k=5, debug=True)
        KMEANS.fit(norm_data)
        KMEANS.save_clusters("evaluation_kmeans.txt")

        # ## HAC SINGLE LINK ###
        HAC_single = HACClustering(k=5, link_type='single')
        HAC_single.fit(norm_data)
        HAC_single.save_clusters("evaluation_hac_single.txt")

        # # ## HAC COMPLETE LINK ###
        HAC_complete = HACClustering(k=5, link_type='complete')
        HAC_complete.fit(norm_data)
        HAC_complete.save_clusters("evaluation_hac_complete.txt")

    if IRIS:
        mat = Arff("datasets/iris.arff", label_count=0)

        raw_data = mat.data
        # data = raw_data[:, :-1]
        # labels = raw_data[:, -1]
        data = raw_data

        # ## Normalize the data ###
        scaler = MinMaxScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

        k = 4

        # # ## K-MEANS ###
        KMEANS = KMEANSClustering(k=k, debug=False)
        KMEANS.fit(norm_data)
        KMEANS.save_clusters(None)

        # ## HAC SINGLE LINK ###
        HAC_single = HACClustering(k=k, link_type='single')
        HAC_single.fit(norm_data)
        HAC_single.save_clusters(None)

        # # ## HAC COMPLETE LINK ###
        HAC_complete = HACClustering(k=k, link_type='complete')
        HAC_complete.fit(norm_data)
        HAC_complete.save_clusters(None)

    if SK_IRIS:
        mat = Arff("datasets/iris.arff", label_count=0)  # # clustering is unsupervised.

        raw_data = mat.data
        data = raw_data

        # ## Normalize the data ###
        scaler = MinMaxScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

        k = 7

        # # ## K-MEANS ###
        clusterer = KMeans(n_clusters=k, init="random")
        c_labels = clusterer.fit_predict(norm_data)
        print("K-Means:", davies_bouldin_score(norm_data, c_labels))

        # ## HAC SINGLE LINK ###
        single = AgglomerativeClustering(n_clusters=k, linkage='single')
        c_labels = single.fit_predict(norm_data)
        print("HAC single:", davies_bouldin_score(norm_data, c_labels))

        # # ## HAC COMPLETE LINK ###
        complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
        c_labels = complete.fit_predict(norm_data)
        print("HAC complete:", davies_bouldin_score(norm_data, c_labels))

    if SK_DIABETES:
        mat = Arff("datasets/diabetes.arff", label_count=0)  # # clustering is unsupervised.

        raw_data = mat.data
        data = raw_data

        # ## Normalize the data ###
        scaler = MinMaxScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

        k = 7

        # # ## K-MEANS ###
        clusterer = KMeans(n_clusters=k, init="random")
        c_labels = clusterer.fit_predict(norm_data)
        print("K-Means:", silhouette_score(norm_data, c_labels))

        # # HAC SINGLE LINK ###
        single = AgglomerativeClustering(n_clusters=k, linkage='single')
        c_labels = single.fit_predict(norm_data)
        print("HAC single:", silhouette_score(norm_data, c_labels))

        # # ## HAC COMPLETE LINK ###
        complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
        c_labels = complete.fit_predict(norm_data)
        print("HAC complete:", silhouette_score(norm_data, c_labels))
