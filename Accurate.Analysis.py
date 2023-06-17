import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
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
    for i in range(5,11):
      kmeans = KMeans(n_clusters=i, random_state=0).fit(train_features)
      labels = kmeans.labels_
      silhouetteScore = silhouette_score(train_features, labels)
      calinskiHarabaszScore = calinski_harabasz_score(train_features, labels)
      daviesBouldinScore = davies_bouldin_score(train_features, labels)
      print("This is k-means with k = " + str(i) + ":")
      print("\tsilhouette score = " + str(silhouetteScore))
      print("\tcalinski harabasz score = " + str(calinskiHarabaszScore))
      print("\tdavies bouldin score = " + str(daviesBouldinScore))
      plot(train_features, labels, "2D")
      plot(train_features, labels, "3D")