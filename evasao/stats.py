from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from scripts_ufes.docs.data.gen_discrete_student import Students

# k= # of variables
# AIC= 2k - 2ln(sse)

# Alternatively, for BIC:
#
# k = number of variables
# n = number of observations
# BIC = n*ln(sse/n) + k*ln(n)

def compute_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters
    Credit to https://stats.stackexchange.com/a/146539 and https://stats.stackexchange.com/a/251169

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, d = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
                                                           'euclidean') ** 2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d + 1)

    BIC = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                  ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

    return BIC


def example_iris(kmax=10):
    # IRIS DATA
    iris = sklearn.datasets.load_iris()
    X = iris.data[:, :4]  # extract only the features
    X = StandardScaler().fit_transform(X)
    Y = iris.target

    ks = range(1, kmax)

    # run 9 times kmeans and save each result in the KMeans object
    KMeans = [cluster.KMeans(n_clusters=i, init="k-means++", max_iter=1000, n_init=10).fit(X) for i in tqdm(ks)]

    # now run for each cluster the BIC computation
    BIC = [compute_bic(kmeansi, X) for kmeansi in KMeans]

    print(BIC)

    print(F"Optimal K: {np.argmax(BIC)+1}")

    plt.plot(ks, BIC)
    plt.xlabel('k')
    plt.show()


def example_evasao(kmax):
    # Evasao DATA
    config_file= "/scripts_ufes/docs/data/students.json"
    students = Students(config_file=config_file)
    students.init_data()
    students.stats()


    X = students.dados[students.config["states"]].fillna(0)
    X = StandardScaler().fit_transform(X)

    ks = range(1, kmax)

    # run ks times kmeans and save each result in the KMeans object
    KMeans = [cluster.KMeans(n_clusters=i, init="k-means++", max_iter=1000, n_init=10).fit(X) for i in tqdm(ks)]

    # now run for each cluster the BIC computation
    BIC = [compute_bic(kmeansi, X) for kmeansi in KMeans]

    print(BIC)
    print(F"Optimal K: {np.argmax(BIC)+1}")

    plt.plot(ks, BIC)
    plt.xlabel('k')
    plt.show()


if __name__ == "__main__":
    example_evasao(1000)
