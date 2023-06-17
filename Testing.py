import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score
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
    test_features, test_labels = load_data(False)
    domain_features, domain_labels = load_data_with_domain_label()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(train_features)
    predict = kmeans.predict(domain_features)
    plot(domain_features, predict, "2D")
    map = [[0, -1] for i in range(5)]
    for i in range(len(predict)):
      idx = predict[i]
      if map[idx][1] == -1:
        map[idx][0] = 1 
        map[idx][1] = domain_labels[i]
      else:
        if map[idx][1] == domain_labels[i]:
          map[idx][0] += 1
        else:
          if map[idx][0] == 1:
            map[idx][0] = 0
            map[idx][1] = -1
          else:
            map[idx][0] -= 1
    print(map)
    sum = 0
    for i in range(5):
      sum += map[i][0]
    accuracy = int((sum / len(predict)) * 100)
    print(accuracy_score(domain_labels, predict))
    print("your clustering is {} % equal to the domain labels".format(accuracy))