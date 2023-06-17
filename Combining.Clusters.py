import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from numpy import reshape
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def load_data(train_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    loads train/test features with image labels.
    """
    if train_data:
        data = np.load(f'train_data.npz')
    else:
        data = np.load(f'test_data.npz')
    features = data['features']
    img_labels = data['img_labels']

    return features, img_labels


def load_data_with_domain_label() -> Tuple[np.ndarray, np.ndarray]:
    """
    loads portion of training features with domain label
    """
    data = np.load(f'train_data_w_label.npz')
    train_features = data['features']
    domain_labels = data['domain_labels']

    return train_features, domain_labels


def plot(data, labels, dimension):
    if dimension == "2D":
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(data)
        print(z.shape)
        print(labels.shape)
        plt.scatter(z[:, 0], z[:, 1], c=labels)
        plt.title('2.plot')
        plt.show()
    elif dimension == "3D":
        tsne = TSNE(n_components=3, verbose=1, random_state=123)
        z = tsne.fit_transform(data)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=labels)
        plt.title('3D plot')
        plt.show()


if __name__ == '__main__':
    train_features, train_labels = load_data(True)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(train_features)
    KMeansLabels = kmeans.labels_
    # silhouetteScore = silhouette_score(train_features, KMeansLabels)
    # calinskiHarabaszScore = calinski_harabasz_score(train_features, KMeansLabels)
    # daviesBouldinScore = davies_bouldin_score(train_features, KMeansLabels)
    # print("This is k-means with k = 10:")
    # print("\tsilhouette score = " + str(silhouetteScore))
    # print("\tcalinski harabasz score = " + str(calinskiHarabaszScore))
    # print("\tdavies bouldin score = " + str(daviesBouldinScore))
    # plot(train_features, KMeansLabels, "2D")
    # first way to combine clusters -----------------------------------------------
    # agg = AgglomerativeClustering(n_clusters=5).fit(train_features)
    # labels = agg.labels_
    # for i in range(len(labels)):
    #   if labels[i] == 0:
    #     labels[i] = KMeansLabels[i]
    #   elif labels[i] == 1:
    #     labels[i] = KMeansLabels[i] + 10
    #   elif labels[i] == 2:
    #     labels[i] = KMeansLabels[i] + 20
    #   elif labels[i] == 3:
    #     labels[i] = KMeansLabels[i] + 30
    #   else:
    #     labels[i] = KMeansLabels[i] + 40
    # silhouetteScore = silhouette_score(train_features, labels)
    # calinskiHarabaszScore = calinski_harabasz_score(train_features, labels)
    # daviesBouldinScore = davies_bouldin_score(train_features, labels)
    # print("This is Agglomerative with k = 5 by adding to results before:")
    # print("\tsilhouette score = " + str(silhouetteScore))
    # print("\tcalinski harabasz score = " + str(calinskiHarabaszScore))
    # print("\tdavies bouldin score = " + str(daviesBouldinScore))
    # plot(train_features, labels, "2D")
    # second way to combine clusters-----------------------------------------------
    # centroids = kmeans.cluster_centers_
    # map = [0 for x in range(10)]
    # agg = AgglomerativeClustering(n_clusters=5).fit(centroids)
    # labels = agg.labels_
    # for i in range(10):
    #   map[kmeans.predict(centroids[i].reshape(1,-1))[0]] = labels[i]
    # print(map)
    # for i in range(len(KMeansLabels)):
    #   KMeansLabels[i] = map[KMeansLabels[i]]
    # silhouetteScore = silhouette_score(train_features, KMeansLabels)
    # calinskiHarabaszScore = calinski_harabasz_score(train_features, KMeansLabels)
    # daviesBouldinScore = davies_bouldin_score(train_features, KMeansLabels)
    # print("This is Agglomerative with k = 5 by using center of clusters:")
    # print("\tsilhouette score = " + str(silhouetteScore))
    # print("\tcalinski harabasz score = " + str(calinskiHarabaszScore))
    # print("\tdavies bouldin score = " + str(daviesBouldinScore))
    # plot(train_features, KMeansLabels, "2D")